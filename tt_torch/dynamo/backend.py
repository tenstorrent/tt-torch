import torch
from torch._dynamo.backends.common import aot_autograd
from torch.fx.experimental.proxy_tensor import make_fx
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

def execute(gm, inputs):
    return gm(*inputs)

class Executor():
    def __init__(self, binary):
        self.binary = binary
    
    def __call__(self, *inputs):
        return tt_mlir.run(inputs, self.binary)

def _base_backend(gm: torch.fx.GraphModule, example_inputs):
    gm.graph.print_tabular()
    gm = pass_pipeline(gm, example_inputs)
    gm.graph.print_tabular()

    context = Context()
    torch_dialect.register_dialect(context)
    importer = FxImporter(context=context)
    importer.import_graph_module(gm)

    lower_mlir_module(True, OutputType.STABLEHLO, importer.module)
    binary = tt_mlir.compile(importer.module.__str__())
    executor = Executor(binary)
    return executor


def backend(gm, example_inputs):
    aten = make_fx(gm, tracing_mode="symbolic", decomposition_table={})(*example_inputs)
    return _base_backend(aten, example_inputs)
# backend = aot_autograd(fw_compiler=_base_backend)
