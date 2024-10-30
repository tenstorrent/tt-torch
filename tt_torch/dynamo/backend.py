# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch._dynamo.backends.common import aot_autograd
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.compile_utils import strip_overloads
from tt_torch.dynamo.passes import pass_pipeline

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


def execute(gm, inputs):
    return gm(*inputs)


class Executor:
    def __init__(self, binary):
        self.binary = binary

    def __call__(self, *inputs):
        return tt_mlir.run(inputs, self.binary)


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

    if len(graph.nodes) == 1:
        for node in graph.nodes:
            if node.op == "output":
                # Remove the output node if it's the only one
                graph.erase_node(node)


def _base_backend(gm: torch.fx.GraphModule, example_inputs):
    gm.graph.print_tabular()
    gm = pass_pipeline(gm, example_inputs)
    gm.graph.print_tabular()
    reduce_graph(gm)
    gm.graph.print_tabular()

    context = Context()
    torch_dialect.register_dialect(context)
    importer = FxImporter(context=context)
    importer.import_graph_module(gm)

    run_pipeline_with_repro_report(
        importer.module,
        f"builtin.module(torchdynamo-export-to-torch-backend-pipeline)",
        "Lowering TorchFX IR -> Torch Backend IR",
    )
    lower_mlir_module(True, OutputType.STABLEHLO, importer.module)
    binary = tt_mlir.compile(importer.module.operation.get_asm())
    executor = Executor(binary)
    return executor


def backend(gm, example_inputs):
    # fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(example_inputs)
    # fake_tensor_mode.allow_non_fake_inputs = True
    # aten = make_fx(gm, tracing_mode="symbolic", decomposition_table={}, _allow_non_fake_inputs=True)(*example_inputs)
    # return _base_backend(aten, example_inputs)
    return _base_backend(gm, example_inputs)


# backend = aot_autograd(fw_compiler=_base_backend)
