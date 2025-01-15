# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### PRE-REQUISITES
#   pip install --pre torch-mlir torchvision
#   pip install stablehlo -f https://github.com/openxla/stablehlo/releases/expanded_assets/dev-wheels

import torch_mlir
import tt_mlir
import mlir.ir as ir
import mlir.dialects.stablehlo as stablehlo
from mlir.ir import Context, Location, Module


def get_module_from_str(module_str: str):
    module = None
    with ir.Context() as ctx:
        stablehlo.register_dialect(ctx)
        module = Module.parse(module_str)
    return module


def extract_op_by_op(file_content: str):
    module = get_module_from_str(file_content)
    i = 0
    for op in module.operation:
        print(i)
        print(op)
        i += 1


def lower_stablehlo_to_ttnn(stablehlo_ir: str):
    try:
        with ir.Context() as ctx:
            stablehlo.register_dialect(ctx)
            module = ir.Module.parse(stablehlo_ir)
            ttir = tt_mlir.compile_stable_hlo_to_ttir(module.operation.get_asm())
            binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
            return ttnn
    except Exception as e:
        print("Error: ", e)
        return None


MODULE_STRING = """
module {
  func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
    return %2 : tensor<1x128xf32>
  }
}
"""

## TODO: Update stablehlo_output.txt to use most recent stablehlo conventions
file_path = "tests/models/mmdetection/stablehlo_output.txt"
file_content = ""
with open(file_path, "r") as file:
    file_content = file.read()


def test_op_by_op_shlo(input_str: str):
    extract_op_by_op(input_str)
    # ttnn = lower_stablehlo_to_ttnn(input_str)


test_op_by_op_shlo(MODULE_STRING)
