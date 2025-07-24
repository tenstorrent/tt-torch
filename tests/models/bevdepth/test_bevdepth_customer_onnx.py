# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
import requests
import os

import pytest
from tests.utils import OnnxModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTesterPart1(OnnxModelTester):
    def _load_model(self):
        part1_path = "/localdev/../bevdepth_fp32_dummy_v2_part1.onnx"
        
        model = onnx.load(part1_path)
        return model
    
    
    def _load_torch_inputs(self):
        # Create dummy input tensors with the expected shapes for the BEVDepth model (part1)
        # The model expects the following inputs:
        # 1. 'img' with shape [1, 1, 6, 3, 800, 1600]
        # 2. 'sensor2ego' with shape [1, 1, 6, 4, 4]
        # 3. 'intrin' with shape [1, 1, 6, 4, 4]
        # 4. 'ida' with shape [1, 1, 6, 4, 4]
        # 5. 'bda' with shape [1, 4, 4]
        
        img = torch.randn(1, 1, 6, 3, 800, 1600)  # Image input
        sensor2ego = torch.randn(1, 1, 6, 4, 4)    # Sensor to ego transform
        intrin = torch.randn(1, 1, 6, 4, 4)        # Intrinsic parameters
        ida = torch.randn(1, 1, 6, 4, 4)           # Image data augmentation
        bda = torch.randn(1, 4, 4)                 # BEV data augmentation
        
        # Return a tuple of tensors in the same order as the model expects
        return (img, sensor2ego, intrin, ida, bda)


class ThisTesterPart2(OnnxModelTester):
    def _load_model(self):
        # part2_path = "/localdev/achoudhury/confidential_customer_models/quantized_qdq/priorityA/bevdepth/bevdepth_fp32_dummy_v2_part2.onnx"
        part2_path = "/localdev/../bevdepth_fp32_dummy_v2_part2_v2.onnx"
        
        model = onnx.load(part2_path)
        return model
    
    
    def _load_torch_inputs(self):
        # Create dummy input with shape matching output from voxel pooling
        output_features = torch.randn(1, 80, 128, 128)  # Voxel pooling output features
        
        return (output_features,)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, None],
    ids=["op_by_op_stablehlo", "full"],
)
def test_bevdepth_customer_onnx_part1(record_property, mode, op_by_op):
    model_name = "BevDepth"
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    if op_by_op is not None:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        cc.op_by_op_backend = op_by_op

    tester = ThisTesterPart1(
        model_name,
        mode,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()
    if mode == "eval":
        # Print the top 5 predictions
        _, indices = torch.topk(results[0], 5)
        print(f"Top 5 predictions: {indices[0].tolist()}")

    tester.finalize()


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, None],
    ids=["op_by_op_stablehlo", "full"],
)
def test_bevdepth_customer_onnx_part2(record_property, mode, op_by_op):
    model_name = "BevDepth_Part2"
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    if op_by_op is not None:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        cc.op_by_op_backend = op_by_op

    tester = ThisTesterPart2(
        model_name,
        mode,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()
    if mode == "eval":
        # Print information about the outputs
        print(f"Number of outputs: {len(results)}")
        for i, output in enumerate(results):
            print(f"Output {i}: shape {output.shape}, dtype {output.dtype}")

    tester.finalize()
