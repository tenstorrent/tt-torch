# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
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


def pass_pipeline(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    decompositions = torch.export.default_decompositions()
    decompositions.update(CUSTOM_DECOMPOSITION_TABLE)

    # we use the export API to run the decompositions, as this maintains the
    # soruce locations in stack_trace
    gm = (
        torch.export.export_for_training(gm, tuple(example_inputs), strict=False)
        .run_decompositions(decompositions)
        .module()
    )

    if compiler_config.enable_consteval:
        gm = constant_fold(gm)
    elif compiler_config.consteval_parameters:
        raise Exception("consteval_parameters is enabled but enable_consteval is not")

    gm = bypass_redundant_getitem(gm)

    # reduce_graph(gm) - ISSUE: https://github.com/tenstorrent/tt-torch/issues/513
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
    return program, constant_inputs
