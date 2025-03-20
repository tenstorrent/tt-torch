# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

import tt_torch
from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig, CompileDepth
import torch.nn.functional as F
import itertools


def test_bilinear_upsample(inH, inW, scale_factor, align_corners):
    torch.manual_seed(0)

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

    input_shape = (1, 1, inH, inW)
    small = torch.randn(input_shape, dtype=torch.bfloat16)

    cc = CompilerConfig()
    cc.enable_consteval = True
    verify_module(
        Interpolate(),
        inputs=[small],
        compiler_config=cc,
        required_atol=0.07,
    )
    torch._dynamo.reset()


def run_single_shape():
    test_bilinear_upsample(50, 224, 2, False)


def run_all_shapes():
    inHs = [50, 128, 224, 960]
    inWs = [50, 128, 224, 540]
    scale_factors = [0.5, 2]
    align_corners_options = [False, True]

    for inH, inW, scale_factor, align_corners in itertools.product(
        inHs, inWs, scale_factors, align_corners_options
    ):
        test_bilinear_upsample(inH, inW, scale_factor, align_corners)


def main():
    run_all_shapes()
    # run_single_shape()


main()
