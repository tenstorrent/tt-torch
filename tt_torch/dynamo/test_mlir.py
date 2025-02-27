# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import glob
from tt_torch.dynamo.backend import backend
from tt_torch.dynamo.shlo_backend import generate_random_inputs_for_shlo
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    OpByOpBackend,
)
import re
import torch

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


# The tests dictionary with model names and mlir paths
tests = {
    "mgp_str_base": {
        "model_name": "alibaba-damo/mgp-str-base",
        "mlir_path": "mlir_tests/alibaba-damomgp-str-base.mlir",
    },
    "autoencoder_linear": {
        "model_name": "Autoencoder (linear)",
        "mlir_path": "mlir_tests/Autoencoder (linear).mlir",
    },
    "detr": {
        "model_name": "DETR",
        "mlir_path": "mlir_tests/DETR.mlir",
    },
    "distilbert-base-uncased": {
        "model_name": "distilbert-base-uncased",
        "mlir_path": "mlir_tests/distilbert-base-uncased.mlir",
    },
    "glpn-kitti": {
        "model_name": "GLPN-KITTI",
        "mlir_path": "mlir_tests/GLPN-KITTI.mlir",
    },
    "beit-base-patch16-224": {
        "model_name": "microsoft/beit-base-patch16-224",
        "mlir_path": "mlir_tests/microsoftbeit-base-patch16-224.mlir",
    },
    "beit-large-patch16-224": {
        "model_name": "microsoft/beit-large-patch16-224",
        "mlir_path": "mlir_tests/microsoftbeit-large-patch16-224.mlir",
    },
    "MLPMixer": {
        "model_name": "MLPMixer",
        "mlir_path": "mlir_tests/MLPMixer.mlir",
    },
    "MNIST": {
        "model_name": "Mnist",
        "mlir_path": "mlir_tests/MNIST.mlir",
    },
    "MobileNetSSD": {
        "model_name": "MobileNetSSD",
        "mlir_path": "mlir_tests/MobileNetSSD.mlir",
    },
    "MobileNetV2": {
        "model_name": "MobileNetV2",
        "mlir_path": "mlir_tests/MobileNetV2.mlir",
    },
    "OpenPoseV2": {
        "model_name": "OpenPose V2",
        "mlir_path": "mlir_tests/OpenPose V2.mlir",
    },
    "PerceiverIO": {
        "model_name": "Perceiver IO",
        "mlir_path": "mlir_tests/Perceiver IO.mlir",
    },
    "ResNet18": {
        "model_name": "ResNet18",
        "mlir_path": "mlir_tests/ResNet18.mlir",
    },
    "ResNet50": {
        "model_name": "ResNet50",
        "mlir_path": "mlir_tests/ResNet50.mlir",
    },
    "SegFormer": {
        "model_name": "SegFormer",
        "mlir_path": "mlir_tests/SegFormer.mlir",
    },
    "SqueezeBERT": {
        "model_name": "SqueezeBERT",
        "mlir_path": "mlir_tests/SqueezeBERT.mlir",
    },
    "ViLT": {
        "model_name": "ViLT",
        "mlir_path": "mlir_tests/ViLT.mlir",
    },
    "YOLOV3": {
        "model_name": "YOLOV3",
        "mlir_path": "mlir_tests/YOLOV3.mlir",
    },
}


def generate_test_file(model_name, mlir_path, test_name):
    test_name_new = (
        test_name.replace("/", "_").replace(" ", "_").replace("-", "_").lower()
    )
    test_file_name = f"mlir_tests/pytests/test_{test_name_new}.py"
    with open(test_file_name, "w") as f:
        f.write(
            f"""import pytest
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
            print(f"Removed cache file: {{file}}")
        except Exception as e:
            print(f"Error removing {{file}}: {{e}}")

def test_{test_name_new}():
    mlir_code = ""
    compile_depths = [
        CompileDepth.COMPILE_OP_BY_OP,
        CompileDepth.EXECUTE,
        CompileDepth.EXECUTE_OP_BY_OP,
    ]

    # Read the MLIR file content
    with open("{mlir_path}", "r", encoding="utf-8") as file:
        mlir_code = file.read()

    for compile_depth in compile_depths:
        compiler_config = CompilerConfig()
        compiler_config.compile_depth = compile_depth
        compiler_config.op_by_op_backend = OpByOpBackend.STABLEHLO
        compiler_config.model_name = "{model_name}"

        inputs = generate_random_inputs_for_shlo(mlir_code)
        executor = backend(mlir_code, inputs, compiler_config)
        result = executor(*inputs)
        print(f"SUCCESS: {{test_name}} - {{compile_depth}}")

    clear_cache()
"""
        )
    print(f"Generated test file: {test_file_name}")


if __name__ == "__main__":
    # Iterate over the tests dictionary and generate a Pytest file for each model
    for test_name, test_info in tests.items():
        model_name = test_info["model_name"]
        mlir_path = test_info["mlir_path"]
        generate_test_file(model_name, mlir_path, test_name)
