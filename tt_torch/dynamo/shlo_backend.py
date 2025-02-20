# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_mlir
import torch
from mlir.ir import Context, Location, Module
import numpy as np
import faulthandler
import multiprocessing as mp
import time
import sys
import tempfile
import re
import os
import mlir.dialects.stablehlo as stablehlo

from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    Op,
    OpCompilationStatus,
    calculate_atol,
    calculate_pcc,
)

from tt_torch.dynamo.executor import Executor


def generate_random_inputs_for_shlo(module_str):
    tensor_shapes = re.findall(r"tensor<([\dx]+)xf32>", module_str)
    inputs = []
    for shape_str in tensor_shapes:
        shape = [int(dim) for dim in shape_str.split("x")]
        inputs.append(torch.randn(shape, dtype=torch.float32))
    return inputs


def parse_module_from_str(module_str):
    module = None
    with Context() as ctx:
        stablehlo.register_dialect(ctx)
        module = Module.parse(module_str)
    return module


def get_input_shapes_and_constants(*inputs):
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
    return input_shapes_and_constants

def get_input_shapes_and_constants(*inputs):
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
    return input_shapes_and_constants


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


class StablehloOp(Op):
    def __init__(
        self,
        model_name,
        op_id,
        original_shlo,
    ):
        super().__init__("", [], model_name)
        self.op_id = op_id
        self.compilation_status = OpCompilationStatus.CREATED_GRAPH
        self.original_shlo = original_shlo


class StablehloExecutor(Executor):
    def __init__(
        self,
        module_str=None,
        parsed_module=None,
        compiler_config=None,
        required_pcc=0.99,
        required_atol=1e-2,
    ):
        super().__init__(
            compiler_config=compiler_config,
            required_pcc=required_pcc,
            required_atol=required_atol,
        )
        if module_str:
            self.parsed_module = parse_module_from_str(module_str)
        elif parsed_module:
            self.parsed_module = parsed_module
        else:
            print(
                "Either module_str or parsed_module should be provided", file=sys.stderr
            )
            exit(1)
        self.sub_ops = []
        self.get_ops_in_module(self.parsed_module)
        self.gm = None
        self.graph_constants = None

    def add_gm(self, gm: torch.fx.GraphModule, graph_constants):
        assert (
            self.compiler_config.compile_depth
            == CompileDepth.COMPILE_STABLEHLO_OP_BY_OP
        ), "gm can only be added in COMPILE_STABLEHLO_OP_BY_OP mode"
        self.gm = gm
        self.graph_constants = tuple(graph_constants)

    def gm_op_by_op(self, *inputs):
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
                tensor = node.target(*args, **node.kwargs)
                node_to_tensor[node] = tensor
                name = node.target.name() if hasattr(node.target, "name") else node.name
                input_shapes_and_constants = get_input_shapes_and_constants(args)
                op = Op(
                    name, input_shapes_and_constants, self.compiler_config.model_name
                )
                if op.unique_key() not in self.compiler_config.unique_ops:
                    self.compiler_config.unique_ops[op.unique_key()] = op
                else:
                    self.compiler_config.unique_ops[op.unique_key()].num_ops += 1
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
        return outputs

    def get_ops_in_module(self, module):
        for func_op in module.body.operations:
            for block in func_op.regions[0].blocks:
                for op in block.operations:
                    if op.name.startswith(("func.", "return")):
                        continue

                    inputs = {
                        operand.get_name(): str(operand.type) for operand in op.operands
                    }
                    args_str = ", ".join(f"{key}: {typ}" for key, typ in inputs.items())

                    # Handle multiple results in the operation
                    result_names = [str(result.get_name()) for result in op.results]
                    result_types = [str(result.type) for result in op.results]

                    # Construct the function signature based on the number of results
                    if len(result_names) == 1:
                        result_str = f"{result_types[0]}"
                        return_stmt = f"return {result_names[0]} : {result_types[0]}"
                    else:
                        result_str = f"({', '.join(result_types)})"
                        return_stmt = f"return ({', '.join(result_names)}) : ({', '.join(result_types)})"
                    # Build the new module string
                    new_module_str = f"""module {{
        func.func @main({args_str}) -> {result_str} {{
            {str(op)}
            {return_stmt}
        }}
    }}"""

                # Create the StablehloOp object and add it to the sub_ops list
                opObj = StablehloOp(
                    model_name=self.compiler_config.model_name,
                    op_id=", ".join(result_names),
                    original_shlo=str(op),
                )
                opObj.add_stable_hlo_graph(new_module_str)
                self.sub_ops.append(opObj)
        breakpoint()

    def compile_op(self, op):
        parsed = parse_module_from_str(op.stable_hlo_graph)
        asm = parsed.operation.get_asm()
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_STABLE_HLO
        op.add_stable_hlo_graph(asm)

        obj = {"asm": asm}

        sender = mp.Queue()
        receiver = mp.Queue()
        ttir_event = mp.Event()
        ttnn_event = mp.Event()
        json_event = mp.Event()

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
                    op.compilation_status = OpCompilationStatus.CONVERTED_TO_TTNN
                    ttnn_event.set()

                if "json" in result:
                    op.json = result["json"]
                    json_event.set()
                    # op.parse_json()
                    op.compilation_status = OpCompilationStatus.CONVERTED_TO_TTNN

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
        return binary

    def compile_shlo_op_by_op(self):
        num_ops = len(self.sub_ops)
        for idx, node in enumerate(self.sub_ops):
            print(f"Compiling {idx}/{num_ops}: {node.stable_hlo_graph}")
            binary = self.compile_op(node)
        self.set_binary(binary)
        return binary

    def print_op(self, op):
        print(op.op_id)
        print(op.stable_hlo_graph)
        print(op.ttir_graph)
        print(op.ttnn_graph)
        print(op.json)

    def print_ops(self):
        for op in self.sub_ops:
            self.print_op(op)

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
            return tt_mlir.run(inputs, self.binary)
        elif self.compiler_config.compile_depth == CompileDepth.EXECUTE_OP_BY_OP:
            self.compile_shlo_op_by_op()
            return tt_mlir.run(inputs, self.binary)
        elif (
            self.compiler_config.compile_depth
            == CompileDepth.COMPILE_STABLEHLO_OP_BY_OP
        ):
            # assuming input is a torch graph
            # if input is stablehlo graph, compile depth should have
            # been reset to COMPILE_OP_BY_OP
            self.compile_shlo_op_by_op()
            return self.gm_op_by_op(*(inputs + self.graph_constants))
        elif self.compiler_config.compile_depth == CompileDepth.COMPILE_OP_BY_OP:
            self.compile_shlo_op_by_op()
            return  # return nothing
        else:
            print("Invalid compile depth", file=sys.stderr)
            exit(1)