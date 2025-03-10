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
    DecompositionTable,
    DEFAULT_DECOMPOSITION_TABLE,
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


def apply_decompositions(
    gm: torch.fx.GraphModule,
    example_inputs,
    decompositions: Optional[DecompositionTable] = None,
):
    concrete_inputs = [
        x.view(tuple(int(dim) for dim in x.shape)) if isinstance(x, torch.Tensor) else x
        for x in example_inputs
    ]
    if decompositions is None:
        return gm

    with torch.no_grad():
        gm = make_fx(
            functionalize(gm),
            decomposition_table=decompositions,
        )(*example_inputs)

    return gm


def bypass_redundant_getitem(gm):
    for node in gm.graph.nodes:
        if node.op == "call_function" and "getitem" in node.name:
            if isinstance(node.args[0], tuple):
                idx = node.args[1]
                if isinstance(idx, int):
                    node.replace_all_uses_with(node.args[0][idx])
    return gm


def sanitize_floating_point_tensors(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
        return tensor.to(torch.float32)
    return tensor


def run_folding(gm):
    # If there's no const subgraph module or attr output names to use, return
    # early as there is no const folding to perform.
    if gm.const_subgraph_module is None or gm.fx_const_folded_attrs_name is None:
        return

    assert not gm.has_folding_been_run
    gm.has_folding_been_run = True

    # Actually run const folding subgraph. Note that single attr const fold
    # subgraphs output a single Tensor while multiple outputs are returned as
    # Tuple[Tensor,].
    folded_attrs = gm.const_subgraph_module()

    def _create_param(i):
        return torch.nn.Parameter(
            sanitize_floating_point_tensors(i.detach())
            if not isinstance(i, int)
            else torch.Tensor([i]).to(device=gm.device_for_folded_attrs),
            requires_grad=i.requires_grad if isinstance(i, torch.Tensor) else False,
        )

    params = (
        torch.nn.ParameterList([_create_param(i) for i in folded_attrs])
        if isinstance(folded_attrs, tuple)
        else _create_param(folded_attrs)
    )
    setattr(gm, gm.fx_const_folded_attrs_name, params)


def constant_fold(gm):
    gm = const_fold.split_const_subgraphs(gm)
    gm.run_folding()
    graph_constants = {}

    gm.graph.eliminate_dead_code()
    return gm, graph_constants


def order_constant_inputs(gm, parameters, constants):
    constant_inputs = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.target in parameters:
                constant_inputs.append(parameters[node.target])
            elif node.target in constants:
                constant_inputs.append(constants[node.target])
    return constant_inputs


def pass_pipeline(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    decompositions = DEFAULT_DECOMPOSITION_TABLE
    decompositions.update(CUSTOM_DECOMPOSITION_TABLE)

    # we use the export API to run the decompositions, as this maintains the
    # soruce locations in stack_trace
    gm = (
        torch.export.export(gm, tuple(example_inputs), strict=False)
        .run_decompositions(decompositions)
        .module()
    )

    if compiler_config.enable_consteval:
        gm, constants = constant_fold(gm)
    elif compiler_config.consteval_parameters:
        raise Exception("consteval_parameters is enabled but enable_consteval is not")
    else:
        constants = {}
    gm = bypass_redundant_getitem(gm)
    parameters = {}

    constant_inputs = order_constant_inputs(gm, parameters, constants)

    # some constant folding operations are preformed by changing tensor strides, we
    # want all the strides to be 1, so make them contiguous
    for (i, t) in enumerate(constant_inputs):
        if not t.is_contiguous():
            constant_inputs[i] = t.contiguous()

    reduce_graph(gm)
    run_shape_prop(gm, example_inputs + constant_inputs)
    program = torch.export.export(
        gm, tuple(example_inputs + constant_inputs), strict=False
    )
    return program, constant_inputs
