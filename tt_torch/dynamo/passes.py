# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import sys
import traceback
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental import const_fold
from torch._decomp import get_decompositions
from torch.func import functionalize
from typing import List, Optional, Union
from torch.export.graph_signature import InputKind

from .decompositions import (
    CUSTOM_DECOMPOSITION_TABLE,
)


def run_shape_prop(gm, example_inputs):
    shape_prop = torch.fx.passes.shape_prop.ShapeProp(gm)
    if shape_prop.fake_mode is not None:
        fake_args = [
            shape_prop.fake_mode.from_tensor(act, static_shapes=True)
            if isinstance(act, torch.Tensor)
            else act
            for act in example_inputs
        ]
    else:
        fake_args = example_inputs
    shape_prop.run(*fake_args)


def reduce_graph(module_or_graph: Union[torch.fx.Graph, torch.fx.GraphModule]):
    # Reduce the graph to only the nodes that are used

    # Traverse up the graph from output nodes to populate consumed nodes set
    graph = (
        module_or_graph.graph
        if isinstance(module_or_graph, torch.fx.GraphModule)
        else module_or_graph
    )
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


def bypass_redundant_getitem(gm):
    for node in gm.graph.nodes:
        if node.op == "call_function" and "getitem" in node.name:
            if isinstance(node.args[0], tuple):
                idx = node.args[1]
                if isinstance(idx, int):
                    node.replace_all_uses_with(node.args[0][idx])
    return gm


def constant_fold(gm):
    gm = const_fold.split_const_subgraphs(gm)
    gm.run_folding()

    gm.graph.eliminate_dead_code()
    return gm

def outplace_index_copy_(gm):
    # Rewrite the graph to replace in-place torch.ops.aten.copy_ operations with out-of-place equivalents
    output_cache = []
    for node in gm.graph.nodes:
        print(node.target, file=sys.stderr)
        if node.op == "call_function" and node.target == torch.ops.aten.copy_.default:
            source_node = node.args[1]
            output_cache.append(source_node)
            print(f"[DEBUG] Erased inplace copy_ node {node.name}", file=sys.stderr)
            gm.graph.erase_node(node)
            
    output_node = [node for node in gm.graph.nodes if node.op=='output'][0]
    current_output = output_node.args[0] # one node reference
    
    new_args = current_output + tuple(output_cache)
    new_args = (new_args,)
    output_node.args = new_args
    return gm

def pass_pipeline(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    print(f"GM Graph at start of pass pipeline{type(gm.graph)}", file=sys.stderr)
    gm.graph.print_tabular()
    # print("", file=sys.stderr)
    decompositions = torch.export.default_decompositions()
    decompositions.update(CUSTOM_DECOMPOSITION_TABLE)
    
    # gm = outplace_index_copy_(gm)
    
    # we use the export API to run the decompositions, as this maintains the
    # soruce locations in stack_trace
    gm = (
        torch.export.export_for_training(gm, tuple(example_inputs), strict=False)
        .run_decompositions(decompositions)
        .module()
    )
    
    print(f"GM Graph after first export {type(gm.graph)}", file=sys.stderr)
    gm.graph.print_tabular()

    if compiler_config.enable_consteval:
        gm = constant_fold(gm)
    elif compiler_config.consteval_parameters:
        raise Exception("consteval_parameters is enabled but enable_consteval is not")

    gm = bypass_redundant_getitem(gm)
    gm = outplace_index_copy_(gm)

    # Proceed with exporting the graph
    program = torch.export.export(gm, tuple(example_inputs), strict=False)
    # The proper order of inputs when outlining everything is constants + parameters + buffers + example_inputs
    if not compiler_config.inline_parameters:
        constant_inputs = (
            list(program.tensor_constants.values())
            + [
                param.contiguous() if not param.is_contiguous() else param
                for param in program.parameters()
            ]
            + list(program.buffers())
        )
        for i in range(len(program._graph_signature.input_specs)):
            if program._graph_signature.input_specs[i].kind != InputKind.USER_INPUT:
                program._graph_signature.input_specs[i].kind = InputKind.USER_INPUT
    else:
        constant_inputs = []

    # Need to run shape_prop again to populate tensor_meta
    run_shape_prop(program.graph_module, constant_inputs + example_inputs)

    # # Rewrite the graph to replace in-place torch.ops.aten.copy_ operations with out-of-place equivalents
    # output_cache = []
    #     # Ideally also check that the args[1] comes from an index_put which comes from a getattr buffers
    # for node in program.graph_module.graph.nodes:
    #     if node.op == "call_function" and node.target == torch.ops.aten.copy_.default:
    #         source_node = node.args[1]
    #         output_cache.append(source_node)
    #         program.graph_module.graph.erase_node(node)
            
    # output_node = [node for node in program.graph_module.graph.nodes if node.op=='output'][0]
    # current_output = output_node.args[0] # one node reference
    
    # new_args = current_output + tuple(output_cache)
    # new_args = (new_args,)
    
    # output_node.args = new_args
    # print(f"[JAMES] Trying to add {new_args} to graph output", file=sys.stderr)
    # program.graph_module.recompile() # is this necessary?
    
    print(f"GM Graph after second export {type(gm.graph)}", file=sys.stderr)
    program.graph_module.graph.print_tabular()

    return program, constant_inputs
