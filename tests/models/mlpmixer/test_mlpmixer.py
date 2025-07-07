# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from mlp_mixer_pytorch import MLPMixer
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        """
        https://github.com/lucidrains/mlp-mixer-pytorch
        """
        model = MLPMixer(
            image_size=256,
            channels=3,
            patch_size=16,
            dim=512,
            depth=12,
            num_classes=1000,
            expansion_factor_token=1,  # see mlpmixer package issue #17 https://github.com/lucidrains/mlp-mixer-pytorch/issues/17
        )
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        img = torch.randn(1, 3, 256, 256)
        img = img.to(torch.bfloat16)
        return img


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_mlpmixer(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "MLPMixer"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=True,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    tester.test_model()
    tester.finalize()
