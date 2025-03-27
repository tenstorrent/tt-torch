# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.export.graph_signature import InputKind
import operator
import tt_mlir
import torch_mlir
import os
import re
import copy
import ml_dtypes
import numpy as np
from torch_mlir.dialects import torch as torch_dialect

from typing import Optional, Any
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    Op,
    OpCompilationStatus,
    calculate_atol,
    calculate_pcc,
)
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)
from torch_mlir.extras.fx_importer import (
    FxImporter,
    ContextCache,
    FxImporterHooks,
    InputInfo,
    GraphNodeImporter,
    TORCH_DTYPE_TO_NPY_TYPE,
    TORCH_DTYPE_TO_MLIR_TYPE,
)
from torch_mlir.ir import Context, Location, DenseElementsAttr, Operation
from tt_torch.dynamo.executor import (
    Executor,
    OpByOpExecutor,
)
from torch_xla.stablehlo import exported_program_to_stablehlo, StableHLOExportOptions
from mlir.ir import Context, Module
import mlir.dialects.stablehlo as stablehlo

#########################################################
# Helper functions
#########################################################


def verify_ir(module):
    def verify_op(op):
        if hasattr(op, "verify"):
            op.verify()
        return torch_mlir.ir.WalkResult.ADVANCE

    module.operation.walk(verify_op)


class TTContextCache(ContextCache):
    def get_node_location(self, node: torch.fx.Node) -> Optional[Location]:
        stack_trace = node.meta.get("stack_trace")
        if stack_trace is None:
            return None

        stack_trace = node.stack_trace
        if stack_trace:
            stack_frames = re.findall(
                r"""File "([^"]+)", line ([0-9]+),""", stack_trace
            )
            locations = []
            for filename, line in stack_frames:
                if filename:
                    locations.append(
                        Location.file(filename, line, col=0, context=self._c)
                    )
            return Location.fused(locations, context=self._c)
        return Location.unknown(context=self._c)


def import_graph(graph: torch.fx.GraphModule):
    with Context() as context:
        torch_dialect.register_dialect(context)
        importer = FxImporter(context=context)
        importer._cc = TTContextCache(
            importer._c, py_attr_tracker=importer._py_attr_tracker
        )
        importer.import_stateless_graph(graph)
        return importer.module


class TTFxImporterHooks(FxImporterHooks):
    def resolve_literal(
        self, gni: GraphNodeImporter, tensor: Any, info: Optional[InputInfo]
    ):
        # This implementation is a near exact copy of the default implementation of
        # torch_mlir.extras.fx_importer._make_vtensor_literal_op. The difference is
        # that we wish to use DenseElementsAttr at all times and never DenseResourceElementsAttr.
        # This is because the use of DenseResourceElementsAttr is causing the GIL to block
        # Mentioned in this IREE issue: https://github.com/iree-org/iree/issues/20102
        if not isinstance(tensor, torch.Tensor):
            return None

        assert not (
            tensor.dtype == torch.bfloat16 and ml_dtypes is None
        ), f"torch.bfloat16 requires the ml_dtypes package, please run:\n\npip install ml_dtypes\n"
        # Resolve the attribute.
        npy_dtype = TORCH_DTYPE_TO_NPY_TYPE.get(tensor.dtype)
        assert (
            npy_dtype is not None
        ), f"Can not create literal tensor for unsupported datatype: {tensor.dtype}"

        np_tensor = np.array(tensor.tolist()).astype(npy_dtype)

        try:
            dtype = tensor.dtype
            element_type = TORCH_DTYPE_TO_MLIR_TYPE[dtype]()
        except KeyError:
            raise TypeError(f"Could not map Torch dtype {dtype} to an MLIR type")
        elements_attr = DenseElementsAttr.get(
            type=element_type, array=np_tensor, shape=np_tensor.shape
        )
        return Operation.create(
            name="torch.vtensor.literal",
            results=[gni._cc.tensor_to_vtensor_type(tensor)],
            attributes={"value": elements_attr},
        ).result


def import_program(program: torch.export.ExportedProgram):
    with Context() as context:
        context.enable_multithreading(False)
        torch_dialect.register_dialect(context)
        importer = FxImporter(context=context, hooks=TTFxImporterHooks())
        importer._cc = TTContextCache(
            importer._c, py_attr_tracker=importer._py_attr_tracker
        )
        importer.import_program(program)
        return importer.module


def lower_to_stable_hlo(
    program: torch.export.ExportedProgram, op=None, enable_ir_printing=False
):
    stablehlo_graph_module = exported_program_to_stablehlo(
        program, StableHLOExportOptions(export_weights=False)
    )
    module_str = stablehlo_graph_module.get_stablehlo_text()
    with Context() as ctx:
        stablehlo.register_dialect(ctx)
        module = Module.parse(module_str)

    if op is not None:
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_STABLE_HLO

    return module, stablehlo_graph_module


##################################################################
# TorchExecutor covers all CompileDepth options except for EXECUTE
##################################################################


class TorchExecutor(OpByOpExecutor):
    def __init__(
        self,
        program,
        compiler_config=None,
        required_pcc=0.99,
        required_atol=1e-2,
    ):
        super().__init__(
            compiler_config=compiler_config,
            required_pcc=required_pcc,
            required_atol=required_atol,
        )
        self.program = program
        if self.compiler_config is None:
            compiler_config = CompilerConfig()
        self.compiler_config = compiler_config

    def is_node_valid(self, node):
        if not isinstance(node.target, torch._ops.OpOverload):
            if "getitem" not in name:
                raise ValueError(f"Node target is not an OpOverload: {name}")
            return False
        return True

    def get_node_name(self, node):
        name = node.target.name() if hasattr(node.target, "name") else node.name
        return name

    def get_stable_hlo_graph(self, node, inputs, **kwargs):

        input_shapes_and_constants = self.get_input_shapes_and_constants(*inputs)

        name = node.target.name() if hasattr(node.target, "name") else node.name
        if not isinstance(node.target, torch._ops.OpOverload):
            if "getitem" not in name:
                raise ValueError(f"Node target is not an OpOverload: {name}")
            return None, None, None, None

        op = Op(name, input_shapes_and_constants, self.compiler_config.model_name)
        if op.unique_key() not in self.compiler_config.unique_ops:
            self.compiler_config.unique_ops[op.unique_key()] = op
        else:
            self.compiler_config.unique_ops[op.unique_key()].num_ops += 1
            return None, None, None, None

        graph = torch.fx.Graph()
        placeholders = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                placeholders.append(graph.placeholder(f"input_{i}"))
            elif isinstance(inp, (list, tuple)):
                inps = torch.fx.immutable_collections.immutable_list(
                    [
                        graph.placeholder(f"input_{idx}_{i}")
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

            # if the output of the getitem node is not used, we don't append it to the graph
            unused_output = [len(user.users) == 0 for user in node.users]
            for idx, tensor_meta in enumerate(node.meta["tensor_meta"]):
                if unused_output[idx]:
                    continue

                getitem_node = graph.call_function(
                    operator.getitem, args=(graph_node, idx)
                )
                getitem_nodes.append(getitem_node)
                getitem_node.meta["tensor_meta"] = tensor_meta
            out = graph.output(tuple(getitem_nodes))
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

        mod = copy.deepcopy(self.program.graph_module)
        mod.graph = graph

        # If there are lists/tuples of tensors as inputs, each element of that sequence has been
        # created as a separate placeholder in the graph. So we must flattedn the inputs.
        def flatten(x):
            if isinstance(x, torch.Tensor):
                return [x]

            flat = []
            if isinstance(x, (list, tuple)):
                for i in x:
                    flat.extend(flatten(i))
            return flat

        inputs = flatten(inputs)
        program = torch.export.export(mod, tuple(inputs))
        module, stablehlo_graph_module = lower_to_stable_hlo(program, op=op)
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_STABLE_HLO
        op.add_stable_hlo_graph(module.operation.get_asm())
        return module, stablehlo_graph_module, inputs, op

    def run_gm_op_by_op(self, *inputs):

        fx_target_to_program_target = {}
        for input_spec in self.program.graph_signature.input_specs:
            fx_target_to_program_target[input_spec.arg.name] = {
                "target": input_spec.target,
                "kind": input_spec.kind,
            }

        buffers = dict(self.program.named_buffers())
        parameters = dict(self.program.named_parameters())
        constants = self.program.tensor_constants

        node_to_tensor = {}
        input_index = 0
        outputs = []
        num_nodes = len(self.program.graph_module.graph.nodes)
        out_degree = {}
        for idx, node in enumerate(self.program.graph_module.graph.nodes):
            print(f"Compiling {idx}/{num_nodes}: {node.target}")
            out_degree[node] = len(node.users)
            if node.op == "placeholder":
                # if node.target in self.program.graph_signature.
                program_target = fx_target_to_program_target[node.target]
                if program_target["kind"] == InputKind.PARAMETER:
                    node_to_tensor[node] = parameters[program_target["target"]]
                elif program_target["kind"] == InputKind.CONSTANT_TENSOR:
                    node_to_tensor[node] = constants[program_target["target"]]
                elif program_target["kind"] == InputKind.BUFFER:
                    node_to_tensor[node] = buffers[program_target["target"]]
                elif program_target["kind"] == InputKind.USER_INPUT:
                    node_to_tensor[node] = inputs[input_index]
                    input_index += 1
                else:
                    raise ValueError(f"Unexpected input kind: {program_target['kind']}")

            elif node.op == "get_attr":
                for buffer in self.program.graph_module.named_buffers():
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
                    binary, stablehlo_graph_module, flat_args, op = self.compile_op(
                        node, *args, **node.kwargs
                    )
                except Exception as e:
                    binary = None
                    print(f"Failed to compile {idx}/{num_nodes}: {node.target}: {e}")

                if (
                    self.compiler_config.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
                    and binary is not None
                ):
                    try:
                        # Only want to pass in tensors.
                        call_args = self._extract_call_args(
                            [arg for arg in flat_args if isinstance(arg, torch.Tensor)],
                            stablehlo_graph_module,
                        )
                        calculated, runtime_stack_dump = self.run_op(binary, call_args)
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
        inputs = self.typecast_inputs(inputs)
        if self.compiler_config.compile_depth in (
            CompileDepth.EXECUTE_OP_BY_OP,
            CompileDepth.COMPILE_OP_BY_OP,
        ):
            return self.run_gm_op_by_op(*inputs)
        else:
            return self.program.graph_module(*inputs)
