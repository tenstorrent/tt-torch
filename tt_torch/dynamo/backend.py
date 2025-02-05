# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

from torch._dynamo.backends.common import aot_autograd
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.compile_utils import strip_overloads
import operator

from tt_torch.dynamo.passes import pass_pipeline
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    Op,
    Tensor,
    OpCompilationStatus,
    calculate_atol,
    calculate_pcc,
)

import tt_mlir
from tt_mlir import is_runtime_debug_enabled
import torch_mlir
from torch_mlir.ir import Context, Location
from torch_mlir.extras.fx_importer import FxImporter, ContextCache

from torch_mlir.dialects import torch as torch_dialect

from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)
from typing import List, Tuple, Union, Optional
import os
import multiprocessing as mp
import time
import faulthandler
from pathlib import Path
import re
import sys
import tempfile


class TTContextCache(ContextCache):
    def get_node_location(self, node: torch.fx.Node) -> Optional[Location]:
        return Location.name(node.name, context=self._c)


def import_graph(graph: torch.fx.GraphModule):
    context = Context()
    torch_dialect.register_dialect(context)
    importer = FxImporter(context=context)
    importer._cc = TTContextCache(
        importer._c, py_attr_tracker=importer._py_attr_tracker
    )
    importer.import_stateless_graph(graph)
    return importer.module


def lower_to_stable_hlo(module, op=None, enable_ir_printing=False):
    run_pipeline_with_repro_report(
        module,
        f"builtin.module(torchdynamo-export-to-torch-backend-pipeline)",
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing,
    )
    if op is not None:
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_TORCH_BACKEND_IR

    lower_mlir_module(False, OutputType.STABLEHLO, module)
    if op is not None:
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_STABLE_HLO


def compile_process(receiver, sender, ttir_event, ttnn_event, json_event):
    obj = receiver.get()
    faulthandler.disable()
    asm = obj["asm"]
    ttir = tt_mlir.compile_stable_hlo_to_ttir(asm)
    sender.put({"ttir": ttir})
    ttir_event.wait()
    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
    sender.put({"binary": binary, "ttnn": ttnn})
    ttnn_event.wait()
    sender.put({"json": tt_mlir.bytestream_to_json(binary)})
    json_event.wait()
    sys.exit(0)


def execute_process(receiver, sender, exec_event):
    while 1:
        obj = receiver.get()
        faulthandler.disable()
        binary = obj["binary"]
        inputs = obj["inputs"]
        file_name = obj["dump_file"]
        file_stderr = open(file_name, "w")
        old_stderr = sys.stderr
        sys.stderr = file_stderr
        old_stdout = sys.stdout
        sys.stdout = file_stderr
        outputs = tt_mlir.run(inputs, binary)
        sys.stderr = old_stderr
        sys.stdout = old_stdout
        file_stderr.close()
        sender.put({"outputs": outputs})
        exec_event.wait()
    sys.exit(0)


class Executor:
    def __init__(
        self,
        gm,
        graph_constants,
        compiler_config=None,
        required_pcc=0.99,
        required_atol=1e-2,
    ):
        self.gm = gm
        self.binary = None
        self.graph_constants = tuple(graph_constants)
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self.compiler_config = compiler_config
        self.required_atol = required_atol
        self.required_pcc = required_pcc
        # Dictionary to keep track of the type conversion for unsupported hardware
        # types and use it to convert the input arguments to supported types.
        self.type_conversion = {
            torch.bool: torch.bfloat16,
            torch.int64: torch.int32,
            torch.float64: torch.float32,
        }
        self.intermediate_callbacks = {}

        # Opening a device in a new process is very slow as the pcie device needs to be initializes
        # So we keep the process alive and reuse it. If the process dies, the next call will create a new process
        self.execute_process = None
        self.execute_sender = None
        self.execute_receiver = None

        # Create temp file at start of execution of first op and pass the name
        # of temp file to subprocess which will be used to redirect the stderr
        # to capture runtime stack dump.
        self.stderror_redirected = False
        self.file_stderr = None

    def register_intermediate_callback(self, callback):
        if not is_runtime_debug_enabled():
            raise RuntimeError(
                "Runtime debug is required to use intermediate callbacks. Please recompile this project with -DTT_RUNTIME_DEBUG=ON."
            )
        tt_mlir.DebugHooks.get_debug_hooks(callback)

    def set_binary(self, binary):
        self.binary = binary

    def compile_op(self, node, *inputs, **kwargs):
        input_shapes_and_constants = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                input_shapes_and_constants.append(inp.shape)
            elif isinstance(inp, (list, tuple)):
                sub = []
                for sub_inp in inp:
                    if isinstance(sub_inp, torch.Tensor):
                        sub.append(sub_inp.shape)
                    else:
                        sub.append(sub_inp)
                input_shapes_and_constants.append(sub)
            elif isinstance(inp, (int, float, bool)):
                input_shapes_and_constants.append(inp)
            elif isinstance(inp, torch.dtype):
                input_shapes_and_constants.append(inp.__str__())
            elif inp is None:
                input_shapes_and_constants.append(None)
            else:
                raise ValueError(f"Unexpected input type: {type(inp)}")

        name = node.target.name() if hasattr(node.target, "name") else node.name
        if not isinstance(node.target, torch._ops.OpOverload):
            if "getitem" not in name:
                raise ValueError(f"Node target is not an OpOverload: {name}")
            return None, None

        op = Op(name, input_shapes_and_constants, self.compiler_config.model_name)
        if op.unique_key() not in self.compiler_config.unique_ops:
            self.compiler_config.unique_ops[op.unique_key()] = op
        else:
            self.compiler_config.unique_ops[op.unique_key()].num_ops += 1
            return None, None

        graph = torch.fx.Graph()
        placeholders = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                placeholders.append(graph.placeholder("input"))
            elif isinstance(inp, (list, tuple)):
                inps = torch.fx.immutable_collections.immutable_list(
                    [
                        graph.placeholder(f"input_{idx}")
                        if isinstance(sub_inp, torch.Tensor)
                        else sub_inp
                        for idx, sub_inp in enumerate(inp)
                    ]
                )
                placeholders.append(inps)
            else:
                placeholders.append(inp)

        if len(placeholders) != len(node.args):
            # are any of the args duplicates? If so, we need to duplicate the placeholders
            for idx, arg in enumerate(node.args):
                if arg in node.args[idx + 1 :]:
                    placeholders.append(placeholders[idx])

        placeholders = tuple(placeholders)
        for placeholder, arg in zip(placeholders, node.args):
            if isinstance(placeholder, torch.fx.node.Node):
                placeholder.meta["tensor_meta"] = arg.meta["tensor_meta"]
            elif isinstance(placeholder, (list, tuple)):
                for sub_placeholder, sub_arg in zip(placeholder, arg):
                    if isinstance(sub_placeholder, torch.fx.node.Node):
                        sub_placeholder.meta["tensor_meta"] = sub_arg.meta[
                            "tensor_meta"
                        ]

        graph_node = graph.call_function(node.target, placeholders, kwargs)
        graph_node.meta["tensor_meta"] = node.meta["tensor_meta"]

        # if the node has multiple outputs, add a getitem for each and append to graph
        if not isinstance(
            node.meta["tensor_meta"], torch.fx.passes.shape_prop.TensorMetadata
        ):
            getitem_nodes = []
            graph_node.meta["val"] = node.meta["val"]

            for idx, tensor_meta in enumerate(node.meta["tensor_meta"]):
                # filter out unused outputs that do not exist in the reduced graph
                users = self.gm.graph.find_nodes(
                    op="call_function", target=operator.getitem
                )
                if not any(user_node.args == (node, idx) for user_node in users):
                    continue

                getitem_node = graph.call_function(
                    operator.getitem, args=(graph_node, idx)
                )
                getitem_nodes.append(getitem_node)
                getitem_node.meta["tensor_meta"] = tensor_meta
            out = graph.output(tuple(getitem_nodes))
            if len(node.users) != len(graph_node.users):
                raise ValueError(
                    f"Op Node {node} has different number of users({len(graph_node.users)}) from global graph({len(node.users)})"
                )
        else:
            out = graph.output((graph_node,))
        if "tensor_meta" not in node.meta:
            raise ValueError(f"Node {node} does not have tensor_meta")

        op.compilation_status = OpCompilationStatus.CREATED_GRAPH
        out.meta["tensor_meta"] = node.meta["tensor_meta"]

        out_meta = out.meta["tensor_meta"]
        if isinstance(out_meta, torch.fx.passes.shape_prop.TensorMetadata):
            out_meta = (out_meta,)
        for out in out_meta:
            op.output_shapes.append([dim for dim in out.shape])

        module = import_graph(graph)
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_TORCH_IR
        op.add_torch_ir_graph(module.operation.get_asm())
        lower_to_stable_hlo(module, op=op)
        op.add_stable_hlo_graph(module.operation.get_asm())

        sender = mp.Queue()
        receiver = mp.Queue()
        ttir_event = mp.Event()
        ttnn_event = mp.Event()
        json_event = mp.Event()
        obj = {"asm": module.operation.get_asm()}
        process = mp.Process(
            target=compile_process,
            args=(sender, receiver, ttir_event, ttnn_event, json_event),
        )
        process.start()
        sender.put(obj)
        start = time.time()
        binary = None
        while True:
            try:
                result = receiver.get_nowait()
                if "ttir" in result:
                    op.compilation_status = OpCompilationStatus.CONVERTED_TO_TTIR
                    op.add_ttir_graph(result["ttir"])
                    ttir_event.set()
                if "binary" in result:
                    binary = result["binary"]
                    op.binary = binary
                    op.add_ttnn_graph(result["ttnn"])
                    ttnn_event.set()
                if "json" in result:
                    op.json = result["json"]
                    json_event.set()
                    op.parse_json()
                    op.compilation_status = OpCompilationStatus.CONVERTED_TO_TTNN
                    break
            except mp.queues.Empty:
                pass
            except Exception as e:
                process.terminate()
                raise e
            if time.time() - start > self.compiler_config.single_op_timeout:
                process.terminate()
                break
            if not process.is_alive():
                break
            time.sleep(0.01)
        process.join()
        print(f"json len {len(op.json)}")
        return binary, op

    def pre_process_inputs(self, *inputs):
        # Remove scalar constants as they're absorbed into the binary
        # Convert torch.nn.Parameter to torch.Tensor
        processed_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.nn.Parameter):
                if not inp.data.is_contiguous():
                    inp.data = inp.data.contiguous()
                processed_inputs.append(inp.data)
            elif isinstance(inp, torch.Tensor):
                if not inp.is_contiguous():
                    inp = inp.contiguous()
                processed_inputs.append(inp)

        # Typecast the unsupported data types to hardware supported types.
        supported_inputs = ()
        for input in processed_inputs:
            # Handle scalar inputs.
            if not hasattr(input, "dtype"):
                assert (
                    type(input) is not bool
                ), "Conversion for scalar boolean is not supported."
                supported_inputs = supported_inputs + ((input),)
                continue

            # Apply type conversion if required.
            input_type = input.dtype
            if input_type in self.type_conversion.keys():
                supported_inputs = supported_inputs + (
                    (input.to(dtype=self.type_conversion[input_type])),
                )
                continue

            # No conversion required.
            supported_inputs = supported_inputs + ((input),)

        return supported_inputs

    def run_op(self, binary, *inputs):
        inputs = self.pre_process_inputs(*inputs)
        if not self.stderror_redirected:
            self.file_stderr = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
            self.stderror_redirected = True

        obj = {"binary": binary, "inputs": inputs, "dump_file": self.file_stderr.name}

        exec_event = mp.Event()
        if self.execute_process is None:
            self.execute_sender = mp.Queue()
            self.execute_receiver = mp.Queue()
            self.execute_process = mp.Process(
                target=execute_process,
                args=(self.execute_sender, self.execute_receiver, exec_event),
            )
            self.execute_process.start()
        self.execute_sender.put(obj)
        result = {}
        start = time.time()
        outputs = [None]
        while True:
            if not self.execute_process.is_alive():
                self.execute_process = None
                break
            try:
                result = self.execute_receiver.get_nowait()
                outputs = result["outputs"]
                exec_event.set()
                break
            except mp.queues.Empty:
                pass
            if time.time() - start > self.compiler_config.single_op_timeout:
                self.execute_process.terminate()
                self.execute_process = None
                break

        if len(outputs) == 1:
            outputs = outputs[0]

        stderr_data = ""
        if outputs is None:
            file_stderr = open(self.file_stderr.name, "r")
            stderr_data = file_stderr.read()
            stderr_data = stderr_data.replace("\n", "\\n")
            stderr_data = re.sub(r"[^\x20-\x7E]", "", stderr_data)
            file_stderr.close()

        return outputs, stderr_data

    def run_gm_op_by_op(self, *inputs):
        node_to_tensor = {}
        input_index = 0
        outputs = []
        num_nodes = len(self.gm.graph.nodes)
        out_degree = {}
        for idx, node in enumerate(self.gm.graph.nodes):
            print(f"Compiling {idx}/{num_nodes}: {node.target}")
            out_degree[node] = len(node.users)
            if node.op == "placeholder":
                node_to_tensor[node] = inputs[input_index]
                input_index += 1
            elif node.op == "get_attr":
                for buffer in self.gm.named_buffers():
                    if buffer[0] == node.target:
                        node_to_tensor[node] = buffer[1]
                        break
            elif node.op == "call_function":
                args = []
                for arg in node.args:
                    if isinstance(arg, torch.fx.node.Node):
                        args.append(node_to_tensor[arg])
                    elif isinstance(arg, list):
                        args.append(
                            [
                                node_to_tensor[a]
                                if isinstance(a, torch.fx.node.Node)
                                else a
                                for a in arg
                            ]
                        )
                    else:
                        args.append(arg)
                try:
                    binary, op = self.compile_op(node, *args, **node.kwargs)
                except Exception as e:
                    binary = None
                    print(f"Failed to compile {idx}/{num_nodes}: {node.target}: {e}")

                if (
                    self.compiler_config.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
                    and binary is not None
                ):
                    try:
                        calculated, runtime_stack_dump = self.run_op(binary, *args)
                        self.compiler_config.unique_ops[
                            op.unique_key()
                        ].runtime_stack_dump = runtime_stack_dump

                        print(f"Ran: {idx}/{num_nodes}: {node.target}")
                        if calculated is None:
                            raise ValueError("Failed to execute")
                        op.compilation_status = OpCompilationStatus.EXECUTED
                        tensor = node.target(*args, **node.kwargs)
                        if self.compiler_config.verify_op_by_op:
                            atol = calculate_atol(calculated, tensor)
                            op.atol = atol
                            if atol > self.required_atol:
                                print(f"atol too high for {idx}: {atol}")
                            pcc = calculate_pcc(calculated, tensor)
                            op.pcc = pcc
                            if pcc < self.required_pcc:
                                print(f"pcc too low for {idx}: {pcc}")
                    except Exception as e:
                        print(
                            f"Failed to execute {idx}/{num_nodes}: {node.target}: {e}"
                        )
                        tensor = node.target(*args, **node.kwargs)
                else:
                    tensor = node.target(*args, **node.kwargs)
                node_to_tensor[node] = tensor
            elif node.op == "output":
                args = node.args[0]
                output_tensors = [node_to_tensor[arg] for arg in args]
                outputs = output_tensors
            args_set = set()
            for arg in node.args:
                if arg in args_set:
                    continue
                args_set.add(arg)
                if isinstance(arg, torch.fx.node.Node):
                    out_degree[arg] -= 1
                    if out_degree[arg] == 0 and arg.op != "output":
                        del node_to_tensor[arg]
                        out_degree.pop(arg)

        self.compiler_config.save_unique_ops()
        if self.execute_process is not None:
            self.execute_process.terminate()
            self.execute_process = None
        if self.stderror_redirected:
            os.unlink(self.file_stderr.name)
            self.stderror_redirected = False

        return outputs

    def __call__(self, *inputs):
        new_inputs = ()
        for input in inputs:
            # Handle scalar inputs.
            if not hasattr(input, "dtype"):
                assert (
                    type(input) is not bool
                ), "Conversion for scalar boolean is not supported."
                new_inputs = new_inputs + ((input),)
                continue

            # Apply type conversion if required.
            input_type = input.dtype
            if input_type in self.type_conversion.keys():
                new_inputs = new_inputs + (
                    (input.to(dtype=self.type_conversion[input_type])),
                )
                continue

            # No conversion required.
            new_inputs = new_inputs + ((input),)

        inputs = new_inputs

        if self.compiler_config.compile_depth == CompileDepth.EXECUTE:
            assert self.binary is not None, "Binary must be set for EXECUTE mode"
            return tt_mlir.run(inputs + self.graph_constants, self.binary)
        elif self.compiler_config.compile_depth in (
            CompileDepth.EXECUTE_OP_BY_OP,
            CompileDepth.COMPILE_OP_BY_OP,
        ):
            return self.run_gm_op_by_op(*(inputs + self.graph_constants))
        else:
            return self.gm(*inputs)


def verify_golden_callback(binary, callback_context, op_context):
    # Using these parameters, we should be able to query information
    # about the op described by op_context, and its output. I.e. location:
    location = tt_mlir.get_op_loc_info(op_context)
    # ...

    # We will need to provide the bindings necesarry in this frontend.
    # Those bindings will interact with the runtime API


def _base_backend(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    # Apply environment overrides at start of compilation to allow overriding what was set in the test
    compiler_config.apply_environment_overrides()
    with torch.no_grad():
        gm, graph_constants = pass_pipeline(gm, example_inputs, compiler_config)
    executor = Executor(gm, graph_constants, compiler_config)
    if compiler_config.compile_depth in (
        CompileDepth.EXECUTE_OP_BY_OP,
        CompileDepth.COMPILE_OP_BY_OP,
        CompileDepth.TORCH_FX,
    ):
        return executor

    dump_intermediates = os.environ.get("TT_TORCH_IR_LOG_LEVEL")
    dump_info = False
    dump_debug = False
    if dump_intermediates:
        dump_debug = dump_intermediates == "DEBUG"
        dump_info = dump_debug or dump_intermediates == "INFO"

    module = import_graph(gm.graph)
    if dump_info:
        print("Torch module", file=sys.stderr)
        module.dump()

    if compiler_config.profile_ops:
        compiler_config.set_torch_mlir_module(module.operation.get_asm())
    if compiler_config.compile_depth == CompileDepth.TORCH_MLIR:
        return executor

    lower_to_stable_hlo(module, enable_ir_printing=dump_debug)
    if dump_info:
        print("StableHLO module", file=sys.stderr)
        module.dump()

    if compiler_config.profile_ops:
        compiler_config.set_stablehlo_mlir_module(module.operation.get_asm())
    if compiler_config.compile_depth == CompileDepth.STABLEHLO:
        return executor

    # Need to set enable_debug_info=True to get the location information for the ops in the asm string
    ttir = tt_mlir.compile_stable_hlo_to_ttir(
        module.operation.get_asm(enable_debug_info=True)
    )
    if dump_info:
        print("TTIR module", file=sys.stderr)
        print(ttir, file=sys.stderr)

    if compiler_config.enable_intermediate_verification:
        executor.register_intermediate_callback(verify_golden_callback)

    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
    if dump_info:
        print("TTNN module", file=sys.stderr)
        print(ttnn, file=sys.stderr)

    executor.set_binary(binary)
    return executor


def backend(gm, example_inputs, options=None):
    if options is None:
        options = CompilerConfig()

    concrete_inputs = [
        x.view(x.shape) if isinstance(x, torch.Tensor) else x for x in example_inputs
    ]
    # fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(example_inputs)
    # fake_tensor_mode.allow_non_fake_inputs = True
    # aten = make_fx(gm, tracing_mode="symbolic", decomposition_table={}, _allow_non_fake_inputs=True)(*example_inputs)
    # return _base_backend(aten, example_inputs)
    return _base_backend(gm, example_inputs, compiler_config=options)


# backend = aot_autograd(fw_compiler=_base_backend)
