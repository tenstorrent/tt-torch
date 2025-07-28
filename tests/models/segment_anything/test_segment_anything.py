# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/facebookresearch/segment-anything-2
# Hugging Face version: https://huggingface.co/facebook/sam2-hiera-tiny

import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.segment_anything.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model()

    def _load_inputs(self):
        return self.loader.load_inputs()

    def run_model(self, model, inputs):
        # Custom model runner for SAM2 prediction
        outputs = model.predict(inputs)
        return outputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.skip(
    reason="Failed to install sam2. sam2 requires Python >=3.10.0 but the default version on Ubuntu 20.04 is 3.8. We found no other pytorch implementation of segment-anything."
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_segment_anything(record_property, mode, op_by_op):

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    tester.test_model()

    tester.finalize()
