# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from tt_torch.dynamo.backend import backend
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
)
import re
import torch


def generate_random_inputs_for_shlo(module_str):
    # Parse tensor shapes from the module string
    import re

    tensor_shapes = re.findall(r"tensor<([\dx]+)xf32>", module_str)
    inputs = []
    for shape_str in tensor_shapes:
        shape = [int(dim) for dim in shape_str.split("x")]
        inputs.append(torch.randn(shape, dtype=torch.float32))
    return inputs


if __name__ == "__main__":
    directory = "./mlir_tests"

    mlir_code = ""
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Check if it's a file (not a directory)
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                mlir_code = file.read()
            compiler_config = CompilerConfig()
            compiler_config.compile_depth = CompileDepth.COMPILE_STABLEHLO_OP_BY_OP
            try:
                inputs = generate_random_inputs_for_shlo(mlir_code)
                executor = backend(mlir_code, inputs, compiler_config)
                result = executor(inputs)
                print(f"Executed {filename} successfully")
            except Exception as e:
                print(f"Failed to compile {filename}")
                print(e)
