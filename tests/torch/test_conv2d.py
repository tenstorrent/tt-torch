# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest

import tt_torch
from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig


@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, padding",
    ((1, 64, 3, 256, 256, 7, 7, 2, 2, 3), (1, 128, 64, 128, 128, 2, 2, 2, 2, 0)),
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_conv2d(
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    padding,
    dtype,
):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2d = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=(filter_height, filter_width),
                stride=(stride_h, stride_w),
                padding=padding,
                bias=False,
                dtype=dtype,
            )

        def forward(self, x):
            return self.conv2d(x)

    verify_module(
        Basic(),
        input_shapes=[(batch_size, input_channels, input_height, input_width)],
        input_data_types=[dtype],
        required_atol=10,
    )
