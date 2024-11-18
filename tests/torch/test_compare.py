# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest

import tt_torch
from tt_torch.tools.verify import verify_module

# TT devices can generate incorrect result for compare operation if the difference
# between the inputs is small. We are using higher ATOL i.e. '1' to handle the
# incorrect output (True instead of False and vice versa).


def test_equal():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x == y

    verify_module(
        Basic(),
        input_shapes=[(64, 64), (64, 64)],
        input_range=(-10, 10),
        required_atol=1,
    )


def test_notEqual():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x != y

    verify_module(
        Basic(),
        input_shapes=[(64, 64), (64, 64)],
        input_range=(-10, 10),
        required_atol=1,
    )


def test_greaterThan():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x > y

    verify_module(
        Basic(),
        input_shapes=[(64, 64), (64, 64)],
        input_range=(-10, 10),
        required_atol=1,
    )


def test_greaterEqual():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x >= y

    verify_module(
        Basic(),
        input_shapes=[(64, 64), (64, 64)],
        input_range=(-10, 10),
        required_atol=1,
    )


def test_lessThan():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x < y

    verify_module(
        Basic(),
        input_shapes=[(64, 64), (64, 64)],
        input_range=(-10, 10),
        required_atol=1,
    )


def test_lessEqual():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x <= y

    verify_module(
        Basic(),
        input_shapes=[(64, 64), (64, 64)],
        input_range=(-10, 10),
        required_atol=1,
    )
