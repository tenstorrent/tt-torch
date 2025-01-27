# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

# import torch.nn as nn
# import torch_xla
import torch_xla.core.xla_model as xm

# import torch_xla.debug.metrics as met
import onnx

from torch.export import export, load
from torch_xla.stablehlo import exported_program_to_stablehlo

# from torch.utils.tensorboard.

from torch_mlir.tools.import_onnx import __main__ as import_onnx
import torch_mlir

from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)


def load_module_from_file(filenm) -> torch_mlir._mlir_libs._mlir.ir.Module:
    src = open(filenm, "r").read()
    with torch_mlir.ir.Context() as ctx:
        torch_mlir.dialects.torch.register_dialect(ctx)
        with torch_mlir.ir.Location.unknown() as loc:
            module = torch_mlir.ir.Module.parse(src)
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(module)
    jit_module = backend.load(compiled)
    return jit_module


# Load your ONNX model
# filename = "/localdev/achoudhury/tt-torch/tests/onnx/mnist_custom.onnx"
# "/localdev/achoudhury/tt-torch/tests/onnx/mnist-12-int8.onnx"
# filename = "/localdev/achoudhury/tt-torch/tests/onnx/mnist-1.onnx"
filename = "/localdev/achoudhury/tt-torch/tests/onnx/distilbert.onnx"
tmpfilename = filename[:-5] + "_tmp.mlir"
# onnx_model = onnx.load(filename)

# Convert ONNX model to PyTorch
# torch_model, _ = torch.onnx.import_model(onnx_model)
main_args = import_onnx.parse_arguments([filename, "-o", tmpfilename])
import_onnx.main(main_args)
# torch_model = load(tmpfilename)
module = load_module_from_file(tmpfilename)

# try compile with torch_mlir
# run_pipeline_with_repro_report(
#     module,
#     "builtin.module(torch-onnx-to-torch-backend-pipeline)",
#     "Lowering Torch Onnx IR -> Torch Backend IR",
#     enable_ir_printing=True,
# )
# lower_mlir_module(True, OutputType.STABLEHLO, module)
# print(tt_mlir.compile(module.operation.get_asm()))

# # # Prepare sample input (replace with your actual input)
# sample_input = torch.randn(1, 3, 224, 224)

# # # Export the PyTorch model using torch.export
# exported_model = export(torch_model.to(xm.xla_device()), (sample_input.to(xm.xla_device()),))

# # # Convert the exported model to StableHLO
# stablehlo_representation = exported_program_to_stablehlo(exported_model)

# # # Print or further process the StableHLO representation
# # print(stablehlo_representation)
