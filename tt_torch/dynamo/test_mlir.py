# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import glob
from tt_torch.dynamo.backend import backend
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
)
import re
import torch

os.environ["HF_HOME"] = "/localdev/ddilbaz/cache"


def generate_random_inputs_for_shlo(module_str):
    # Parse tensor shapes from the module string
    import re

    tensor_shapes = re.findall(r"tensor<([\dx]+)xf32>", module_str)
    inputs = []
    for shape_str in tensor_shapes:
        shape = [int(dim) for dim in shape_str.split("x")]
        inputs.append(torch.randn(shape, dtype=torch.float32))
    return inputs


def clear_cache():
    cache_path = "/localdev/ddilbaz/cache/*"
    files = glob.glob(cache_path)
    for file in files:
        try:
            os.remove(file)
            print(f"Removed cache file: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")


if __name__ == "__main__":
    directory = "./mlir_tests"

    mlir_code = ""
    compile_depths = [
        CompileDepth.COMPILE_STABLEHLO_OP_BY_OP,
        CompileDepth.COMPILE_OP_BY_OP,
        CompileDepth.EXECUTE,
        CompileDepth.EXECUTE_OP_BY_OP,
    ]

    for filename in os.listdir(directory):
        print(f"\n==={filename}===")
        for compile_depth in compile_depths:
            print(f"Running with {compile_depth}")
            filepath = os.path.join(directory, filename)

            # Check if it's a file (not a directory)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as file:
                    mlir_code = file.read()
                compiler_config = CompilerConfig()
                compiler_config.compile_depth = compile_depth
                try:
                    inputs = generate_random_inputs_for_shlo(mlir_code)
                    executor = backend(mlir_code, inputs, compiler_config)
                    result = executor(inputs)
                    print(f"SUCESS: {filename} - {compile_depth}")
                except Exception as e:
                    print(f"FAILURE: {filename} - {compile_depth}")
                    print(e)
        clear_cache()
