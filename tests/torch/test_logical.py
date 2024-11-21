# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest

import tt_torch
from tt_torch.tools.verify import verify_module


def test_and():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.logical_and(x, y)

    verify_module(
        Basic(),
        input_shapes=[(256, 256), (256, 256)],
        input_data_types=[torch.bool],
    )


def test_not():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.logical_not(x)

    verify_module(
        Basic(),
        input_shapes=[(256, 256)],
        input_data_types=[torch.bool],
    )


def test_or():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.logical_or(x, y)

    verify_module(
        Basic(),
        input_shapes=[(256, 256), (256, 256)],
        input_data_types=[torch.bool],
    )


def test_xor():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.logical_xor(x, y)

    verify_module(
        Basic(),
        input_shapes=[(256, 256), (256, 256)],
        input_data_types=[torch.bool],
    )
