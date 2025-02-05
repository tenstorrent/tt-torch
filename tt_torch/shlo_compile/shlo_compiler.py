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
import mlir.execution_engine as ex
import numpy as np


def parse_module_from_str(module_str: str):
    module = None
    with ir.Context() as ctx:
        stablehlo.register_dialect(ctx)
        module = Module.parse(module_str)
    return module


class StableHLOOp:
    def __init__(self, op_id, shlo):
        self.op_id = op_id
        self.shlo = shlo
        self.result = None
        self.ttir = ""
        self.ttnn = ""


class ShloCompiler:
    def __init__(self, top_level_module_str):
        self.top_level_module_str = top_level_module_str
        self.parsed_top_level = parse_module_from_str(self.top_level_module_str)
        self.sub_modules = []
        self.get_ops_in_module(self.parsed_top_level)

    def get_ops_in_module(self, module: mlir.ir.Module):
        # Iterate through all functions in the module
        for func_op in module.body.operations:
            for block in func_op.regions[0].blocks:
                for op in block.operations:
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
                        args_str = ", ".join(
                            f"{key}: {typ}" for key, typ in inputs.items()
                        )
                        result_type = str(
                            op.result.type
                        )  # assuming there is only one return value
                        result_name = str(op.result.get_name())
                        new_module_str = f"""module {{ \n\tfunc.func @main({args_str}) -> {result_type} {{ \n\t\t{str(op)} \n\t\treturn {result_name} : {result_type} \n\t}} \n}}"""
                        shlo_op = StableHLOOp(result_name, new_module_str)
                        self.sub_modules.append(shlo_op)

    def compile(self):
        for shlo_op in self.sub_modules:
            parsed = parse_module_from_str(shlo_op.shlo)
            try:
                shlo_op.ttir = tt_mlir.compile_stable_hlo_to_ttir(
                    parsed.operation.get_asm()
                )
                __, shlo_op.ttnn = tt_mlir.compile_ttir_to_bytestream(shlo_op.ttir)
            except:
                print("Error in compilation")
                with ir.Context() as ctx:
                    stablehlo.register_dialect(ctx)
                    module = parse_module_from_str(shlo_op.shlo)
                    print(module)
                    print(module.operation.get_asm())


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
shlo_compiler = ShloCompiler(mlir_code)
shlo_compiler.compile()
for shlo_op in shlo_compiler.sub_modules:
    print(f"Op: {shlo_op.op_id}")
    print(f"TTIR: {shlo_op.ttir}")
    print(f"TTNN: {shlo_op.ttnn}")
    print("\n")
