# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import operator
import tt_mlir
import torch_mlir
import os
from torch_mlir.dialects import torch as torch_dialect

from typing import Optional
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
from torch_mlir.extras.fx_importer import FxImporter, ContextCache
from torch_mlir.ir import Context, Location
from tt_torch.dynamo.executor import (
    Executor,
    OpByOpExecutor,
)

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


##################################################################
# TorchExecutor covers all CompileDepth options except for EXECUTE
##################################################################


class TorchExecutor(OpByOpExecutor):
    def __init__(
        self,
        gm,
        graph_constants,
        compiler_config=None,
        required_pcc=0.99,
        required_atol=1e-2,
    ):
        super().__init__(
            compiler_config=compiler_config,
            required_pcc=required_pcc,
            required_atol=required_atol,
        )
        self.gm = gm
        self.graph_constants = (
            (graph_constants,)
            if isinstance(graph_constants, (int, float))
            else tuple(graph_constants)
        )
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

        input_shapes_and_constants = self.get_input_shapes_and_constants(inputs)

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
        return module, op

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
        inputs = self.typecast_inputs(inputs)
        if self.compiler_config.compile_depth in (
            CompileDepth.EXECUTE_OP_BY_OP,
            CompileDepth.COMPILE_OP_BY_OP,
        ):
            return self.run_gm_op_by_op(*(inputs + self.graph_constants))
        else:
            return self.gm(*inputs)
