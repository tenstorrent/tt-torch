# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
from tt_torch.dynamo.backend import backend
from tt_torch.dynamo.shlo_backend import generate_random_inputs_for_shlo
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend

os.environ["HF_HOME"] = "/localdev/ddilbaz/cache"


def clear_cache():
    cache_path = "/localdev/ddilbaz/cache/*"
    files = glob.glob(cache_path)
    for file in files:
        try:
            os.remove(file)
            print(f"Removed cache file: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")


def test_openposev2():
    mlir_code = ""
    compile_depths = [
        CompileDepth.COMPILE_OP_BY_OP,
        CompileDepth.EXECUTE,
        CompileDepth.EXECUTE_OP_BY_OP,
    ]

    # Read the MLIR file content
    with open("mlir_tests/OpenPose V2.mlir", "r", encoding="utf-8") as file:
        mlir_code = file.read()

    for compile_depth in compile_depths:
        compiler_config = CompilerConfig()
        compiler_config.compile_depth = compile_depth
        compiler_config.op_by_op_backend = OpByOpBackend.STABLEHLO
        compiler_config.model_name = "OpenPose V2"

        inputs = generate_random_inputs_for_shlo(mlir_code)
        executor = backend(mlir_code, inputs, compiler_config)
        result = executor(*inputs)
        print(f"SUCCESS: {test_name} - {compile_depth}")

    clear_cache()
