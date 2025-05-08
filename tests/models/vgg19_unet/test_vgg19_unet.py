# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from tests.models.vgg19_unet.src.vgg19_unet import VGG19UNet


class ThisTester(ModelTester):
    def _load_model(self):
        self.input_shape = (3, 512, 512)
        model = VGG19UNet(input_shape=self.input_shape, out_channels=1)
        return model

    def _load_inputs(self):
        inputs = torch.rand(1, *self.input_shape)
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_vgg19_unet(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "VGG19-Unet"
    model_group = "red"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_RUNTIME",
        reason="Out of Memory: Not enough space to allocate 84213760 B L1 buffer across 64 banks, where each bank needs to store 1315840 B - https://github.com/tenstorrent/tt-torch/issues/729",
        model_group=model_group,
    )

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
    )

    with torch.no_grad():
        results = tester.test_model()
    tester.finalize()
