# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from mlp_mixer_pytorch import MLPMixer
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth


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
def test_mlpmixer(record_property, mode, nightly):
    if mode == "train":
        pytest.skip()
    model_name = "MLPMixer"
    record_property("model_name", model_name)
    record_property("mode", mode)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if nightly:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
    else:
        cc.compile_depth = CompileDepth.TTNN_IR

    tester = ThisTester(model_name, mode, compiler_config=cc)
    results = tester.test_model()
    record_property("torch_ttnn", (tester, results))
