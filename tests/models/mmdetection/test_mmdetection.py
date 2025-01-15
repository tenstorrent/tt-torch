# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### PRE-REQUISITES
#   pip install --pre torch-mlir torchvision
#   pip install stablehlo -f https://github.com/openxla/stablehlo/releases/expanded_assets/dev-wheels

import mlir
import tt_mlir
import mlir.ir as ir
from mlir.ir import Context, Location, Module
import mlir.dialects.stablehlo as stablehlo
from typing import List, Dict, Any

modules = {}


def get_module_from_str(module_str: str):
    module = None
    with ir.Context() as ctx:
        stablehlo.register_dialect(ctx)
        module = Module.parse(module_str)
    return module


def lower_stablehlo_to_ttnn(stablehlo_ir: str):
    module = get_module_from_str(stablehlo_ir)
    try:
        ttir = tt_mlir.compile_stable_hlo_to_ttir(module.operation.get_asm())
        print("ttir done")
        try:
            binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
            print("ttnn done")
            return ttnn
        except Exception as e:
            print("Error: ", e)
            return None
    except Exception as e:
        print("Error: ", e)
        return None


def get_ops_in_module(module: mlir.ir.Module):
    """
    Get all operations in a module, excluding func.func and return ops.

    Args:
        module: MLIR module
    Returns:
        List of operation names
    """
    # Iterate through all functions in the module
    for func_op in module.body.operations:
        # Iterate through all blocks in the function
        for block in func_op.regions[0].blocks:
            # Iterate through all operations in the block
            for op in block.operations:
                # Skip return operations
                inputs = {}
                result_type = None
                if not op.name.startswith(("func.", "return")):
                    if (
                        op.name == "stablehlo.pad"
                        or op.name == "stablehlo.reduce_window"
                    ):
                        continue
                    for operand in op.operands:
                        inputs[operand.get_name()] = str(operand.type)
                    args_str = ", ".join(f"{key}: {typ}" for key, typ in inputs.items())
                    result_type = str(
                        op.result.type
                    )  # assuming there is only one return value
                    result_name = str(op.result.get_name())
                    new_module_str = f"""module {{ \n\tfunc.func @main({args_str}) -> {result_type} {{ \n\t\t{str(op)} \n\t\treturn {result_name} : {result_type} \n\t}} \n}}"""
                    modules[result_name] = new_module_str


mlir_code = """
module {
  func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
    return %2 : tensor<1x128xf32>
  }
}
"""

file_path = "tests/models/mmdetection/stablehlo_output.txt"
file_content = ""
with open(file_path, "r") as file:
    file_content = file.read()

if __name__ == "__main__":
    module = get_module_from_str(file_content)
    get_ops_in_module(module)
    for key in modules.keys():
        print(f"\nCompiling {key}")
        print(modules[key])
        ttnn = lower_stablehlo_to_ttnn(modules[key])
        if ttnn == None:
            print(f"Error compiling {key}")
            print(modules[key])
        else:
            print(f"Successfully compiled {key}")
            print(ttnn)
