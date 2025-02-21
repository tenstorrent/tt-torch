# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_mlir
from torch_mlir.compiler_utils import run_pipeline_with_repro_report, OutputType, lower_mlir_module
from torch_mlir.ir import Context, Module
from torch_mlir.dialects import torch as torch_dialect
import torch
import sys
import tempfile

def capture_failure(func):
    """Helper to capture failure messages"""
    f_stderr = tempfile.TemporaryFile(mode="w+t")
    old_stderr = sys.stderr
    sys.stderr = f_stderr
    
    try:
        result = func()
        success = True
    except Exception as e:
        result = str(e)
        success = False
    
    sys.stderr = old_stderr
    f_stderr.seek(0)
    stderr_data = f_stderr.read()
    f_stderr.close()
    
    return success, result, stderr_data

def try_torch_ir_path():
    print("\n=== Testing Torch IR Path ===")
    torch_ir = """
module {
  func.func @main(%arg0: !torch.vtensor<[10,5120],bf16>, %arg1: !torch.vtensor<[5120,152064],bf16>) -> !torch.vtensor<[10,152064],bf16> {
      %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[10,5120],bf16>, !torch.vtensor<[5120,152064],bf16> -> !torch.vtensor<[10,152064],bf16>
        return %0 : !torch.vtensor<[10,152064],bf16>
    }
}
          
"""
    stablehlo_ir_expected = """
    'module {\n  func.func @main(%arg0: tensor<10x5120xbf16>, %arg1: tensor<5120x152064xbf16>) -> tensor<10x152064xbf16> {\n    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<10x5120xbf16>, tensor<5120x152064xbf16>) -> tensor<10x152064xbf16>\n    return %0 : tensor<10x152064xbf16>\n  }\n}\n'
    """
    context = Context()
    torch_dialect.register_dialect(context)
    module = Module.parse(torch_ir, context)
    
    # Lower to StableHLO
    run_pipeline_with_repro_report(
        module,
        "builtin.module(torchdynamo-export-to-torch-backend-pipeline)",
        "Lowering TorchFX IR -> Torch Backend IR",
    )
    lower_mlir_module(False, OutputType.STABLEHLO, module)
    print("\n=== Lowered to StableHLO ===")
    print(str(module))
    stablehlo_asm = module.operation.get_asm()
    
    # Convert and run
    ttir = tt_mlir.compile_stable_hlo_to_ttir(stablehlo_asm)
    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
    
    input1 = torch.randn(10, 5120, dtype=torch.bfloat16)
    input2 = torch.randn(5120, 152064, dtype=torch.bfloat16)
    outputs = tt_mlir.run([input1, input2], binary)
    return outputs

# Try Torch IR path
success, result, stderr = capture_failure(try_torch_ir_path)
print("Torch IR Path Success:", success)
if not success:
    print("Error:", result)
    print("Stderr:", stderr)