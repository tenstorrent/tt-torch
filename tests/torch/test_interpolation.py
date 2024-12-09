# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest

import tt_torch
from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig, CompileDepth
import torch.nn.functional as F


@pytest.mark.parametrize("inH", [50, 128, 224, 960])
@pytest.mark.parametrize("inW", [50, 128, 224, 540])
@pytest.mark.parametrize("inC", [3])
@pytest.mark.parametrize("scale_factor", [2, 3])
@pytest.mark.parametrize("align_corners", [False, True])
def test_bilinear_interpolation(inH, inW, inC, scale_factor, align_corners):
    class Interpolate(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return F.interpolate(
                x,
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=align_corners,
            )

    input_shape = (1, inC, inH, inW)
    small = torch.randn(input_shape, dtype=torch.bfloat16)

    cc = CompilerConfig()
    cc.enable_consteval = True
    verify_module(
        Interpolate(),
        inputs=[small],
        compiler_config=cc,
        required_atol=3,
        required_pcc=0.99 - 0.05 * scale_factor,
    )
