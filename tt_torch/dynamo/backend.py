import torch
from torch._dynamo.backends.common import aot_autograd
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.compile_utils import strip_overloads
import operator

from tt_torch.dynamo.passes import pass_pipeline
from tt_torch.tools.utils import CompilerConfig, CompileDepth, Op, OpCompilationStatus

import tt_mlir
from torch_mlir.ir import Context
from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir.dialects import torch as torch_dialect

from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)
from typing import List, Tuple, Union
import os
import multiprocessing as mp
import time
import faulthandler
import sys


def run_shape_prop(gm, example_inputs):
    shape_prop = torch.fx.passes.shape_prop.ShapeProp(gm)
    if shape_prop.fake_mode is not None:
        fake_args = [
            shape_prop.fake_mode.from_tensor(act, static_shapes=True) if isinstance(act, torch.Tensor) else act
            for act in example_inputs
        ]
    else:
        fake_args = example_inputs
    shape_prop.run(*fake_args)

def reduce_graph(module_or_graph: Union[torch.fx.Graph, torch.fx.GraphModule]):
    # Reduce the graph to only the nodes that are used

    # Traverse up the graph from output nodes to populate consumed nodes set
    graph = module_or_graph.graph if isinstance(module_or_graph, torch.fx.GraphModule) else module_or_graph
    consumed = set()
    working_nodes = []
    for node in graph.nodes:
        if node.op == "output":
            working_nodes.append(node)
            consumed.add(node)

    while len(working_nodes) > 0:
        node = working_nodes.pop(0)
        if not isinstance(node, torch.fx.Node):
            continue
        for arg in node.all_input_nodes:
            if arg not in consumed:
                consumed.add(arg)
                working_nodes.append(arg)

    for node in reversed(graph.nodes):
        if node not in consumed:
            graph.erase_node(node)

    if len(graph.nodes) == 1:
        for node in graph.nodes:
            if node.op == "output":
                # Remove the output node if it's the only one
                graph.erase_node(node)

def import_graph(graph: torch.fx.GraphModule):
    context = Context()
    torch_dialect.register_dialect(context)
    importer = FxImporter(context=context)
    importer.import_stateless_graph(graph)
    return importer.module

def lower_to_stable_hlo(module, op=None):
    run_pipeline_with_repro_report(
        module,
        f"builtin.module(torchdynamo-export-to-torch-backend-pipeline)",
        "Lowering TorchFX IR -> Torch Backend IR",
    )
    if op is not None:
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_TORCH_BACKEND_IR
    lower_mlir_module(False, OutputType.STABLEHLO, module)
    if op is not None:
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_STABLE_HLO

def compile_process(receiver, sender):
    obj = receiver.get()
    faulthandler.disable()
    asm = obj["asm"]
    binary = tt_mlir.compile(asm)
    result = {"binary": binary.as_json()}
    sender.put({"binary": result})
    sys.exit(0)

class Executor():
    def __init__(self, gm, compiler_config=None):
        self.gm = gm
        self.binary = None
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self.compiler_config = compiler_config
    
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
        op = Op(name, input_shapes_and_constants)
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
                inps = torch.fx.immutable_collections.immutable_list([graph.placeholder(f"input_{idx}") if isinstance(sub_inp, torch.Tensor) else sub_inp for idx, sub_inp in enumerate(inp)])
                placeholders.append(inps)
            else:
                placeholders.append(inp)
        placeholders = tuple(placeholders)

        if len(placeholders) != len(node.args):
            raise ValueError (f"Placeholders and args must be the same length: {len(placeholders)} != {len(node.args)}")

        for placeholder, arg in zip(placeholders, node.args):
            if isinstance(placeholder, torch.fx.node.Node):
                placeholder.meta["tensor_meta"] = arg.meta["tensor_meta"]
            elif isinstance(placeholder, (list, tuple)):
                for sub_placeholder, sub_arg in zip(placeholder, arg):
                    if isinstance(sub_placeholder, torch.fx.node.Node):
                        sub_placeholder.meta["tensor_meta"] = sub_arg.meta["tensor_meta"]
            
        graph_node = graph.call_function(node.target, placeholders, kwargs)
        graph_node.meta["tensor_meta"] = node.meta["tensor_meta"]

        # if the node has multiple outputs, add a getitem for each and append to graph
        if not isinstance(node.meta["tensor_meta"], torch.fx.passes.shape_prop.TensorMetadata):
            getitem_nodes = []
            graph_node.meta["val"] = node.meta["val"]
            for idx, _ in enumerate(node.meta["tensor_meta"]):
                getitem_node = graph.call_function(operator.getitem, args=(graph_node, idx))
                # getitem_node.meta["val"] = graph_node.meta["val"]
                getitem_nodes.append(getitem_node)
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
        
        module = import_graph(graph)
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_TORCH_IR
        lower_to_stable_hlo(module, op=op)
        op.add_stable_hlo_graph(module.operation.get_asm())

        sender = mp.Queue()
        receiver = mp.Queue()
        obj = {"asm": module.operation.get_asm()}
        process = mp.Process(target=compile_process, args=(sender, receiver))
        process.start()
        sender.put(obj)
        start = time.time()
        result = {}
        result["binary"] = ""
        while True:
            if not process.is_alive():
                break
            try:
                result = receiver.get_nowait()
                op.compilation_status = OpCompilationStatus.CONVERTED_TO_TTNN_IR
                break
            except mp.queues.Empty:
                pass
            if time.time() - start > 5:
                process.terminate()
                break
            time.sleep(0.01)
        process.join()
        return result["binary"], op
    
    def run_op(self, binary, *inputs):
        pid = os.fork()
        if pid == 0:
            outputs = tt_mlir.run(inputs, binary)
            if len(outputs) == 1:
                outputs = outputs[0]
        else:
            pid, status = os.wait()
        return outputs

        
    def run_gm_op_by_op(self, *inputs):
        node_to_tensor = {}
        input_index = 0
        outputs = []
        num_nodes = len(self.gm.graph.nodes)
        for idx, node in enumerate(self.gm.graph.nodes):
            print(f"Compiling {idx}/{num_nodes}: {node.target}")
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
                        args.append([node_to_tensor[a] if isinstance(a, torch.fx.node.Node) else a for a in arg])
                    else:
                        args.append(arg)
                # if idx == 103:
                #     breakpoint()
                #     binary, op = self.compile_op(node, *args, **node.kwargs)
                try:
                    binary, op = self.compile_op(node, *args, **node.kwargs)
                    if self.compiler_config.compile_depth == CompileDepth.EXECUTE_OP_BY_OP and binary is not None:
                        tensor = self.run_op(binary, *args)
                        op.compilation_status = OpCompilationStatus.EXECUTED
                    else:
                        tensor = node.target(*args, **node.kwargs)
                except Exception as e:
                    print(f"Failed to compile {idx}/{num_nodes}: {node.target}: {e}")
                    tensor = node.target(*args, **node.kwargs)

                node_to_tensor[node] = tensor
            elif node.op == "output":
                args = node.args[0]
                output_tensors = [node_to_tensor[arg] for arg in args]
                outputs = output_tensors
        
        self.compiler_config.save_unique_ops()
        return outputs
    
    def __call__(self, *inputs):
        if self.compiler_config.compile_depth == CompileDepth.EXECUTE:
            assert self.binary is not None, "Binary must be set for EXECUTE mode"
            return tt_mlir.run(inputs, self.binary)
        elif self.compiler_config.compile_depth in (CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.COMPILE_OP_BY_OP):
            return self.run_gm_op_by_op(*inputs)
        else:
            return self.gm(*inputs)   
        
def _base_backend(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    gm = pass_pipeline(gm, example_inputs)
    reduce_graph(gm)
    gm.graph.print_tabular()
    run_shape_prop(gm, example_inputs)
    executor = Executor(gm, compiler_config)
    if compiler_config.compile_depth in (CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.COMPILE_OP_BY_OP):
        return executor

    
    module = import_graph(gm.graph)
    if compiler_config.profile_ops:
        compiler_config.set_torch_mlir_module(module.operation.get_asm())
    if compiler_config.compile_depth == CompileDepth.TORCH_MLIR:
        return executor
    
    lower_to_stable_hlo(module)
    if compiler_config.profile_ops:
        compiler_config.set_stablehlo_mlir_module(module.operation.get_asm())
    if compiler_config.compile_depth == CompileDepth.STABLEHLO:
        return executor

    binary = tt_mlir.compile(module.operation.get_asm())
    executor.set_binary(binary)
    return executor


def backend(gm, example_inputs, options=None):
    if options is None:
        options = CompilerConfig()
    # fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(example_inputs)
    # fake_tensor_mode.allow_non_fake_inputs = True
    # aten = make_fx(gm, tracing_mode="symbolic", decomposition_table={}, _allow_non_fake_inputs=True)(*example_inputs)
    # return _base_backend(aten, example_inputs)
    return _base_backend(gm, example_inputs, compiler_config=options)
# backend = aot_autograd(fw_compiler=_base_backend)
