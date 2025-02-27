# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys

# Define the path to the script
script_path = "third_party/tt-mlir/src/tt-mlir/tools/stablehlo_splitter/shlo_split.py"

# Load the module
spec = importlib.util.spec_from_file_location("shlo_split", script_path)
shlo_split = importlib.util.module_from_spec(spec)
sys.modules["shlo_split"] = shlo_split
spec.loader.exec_module(shlo_split)

mlir_path = "./Autoencoder.mlir"
module_str = ""
with open(mlir_path, "r") as file:
    module_str = file.read()
# Now you can use the StablehloSplitter class
splitter = shlo_split.StablehloSplitter(module_str)
print(splitter.sub_ops)
