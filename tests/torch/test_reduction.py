# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest

from tt_torch.tools.verify import verify_module


@pytest.mark.parametrize(
    ("input_shape", "dim_arg", "keep_dim", "input_type", "atol"),
    [
        ([(8, 8)], [0], True, [torch.float32], 0.01),
        ([(8, 8)], [0, 1], True, [torch.float32], 0.02),
        ([(8, 8)], [], True, [torch.float32], 0.02),
        ([(4, 32, 64)], [1], False, [torch.float32], 0.02),
        ([(4, 32, 64)], [2], False, [torch.float32], 0.03),
        ([(4, 32, 64)], [1, 2], False, [torch.float32], 0.07),
        ([(4, 2, 32, 32)], [0], False, [torch.bfloat16], 0.02),
        ([(4, 2, 32, 32)], [2], False, [torch.bfloat16], 0.02),
        ([(4, 2, 32, 32)], [3], False, [torch.bfloat16], 0.02),
        ([(4, 2, 32, 32)], [0, 2], False, [torch.bfloat16], 0.035),
        ([(4, 2, 32, 32)], [0, 1, 2, 3], True, [torch.bfloat16], 0.13),
        ([(4, 2, 32, 32)], [0, 2, 3], True, [torch.bfloat16], 0.30),
    ],
)
def test_reduce_sum(input_shape, dim_arg, keep_dim, input_type, atol):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sum(x, dim=dim_arg, keepdim=keep_dim)

    verify_module(
        Basic(),
        input_shapes=input_shape,
        input_data_types=input_type,
        required_atol=atol,
    )


# PyTorch returns a scalar value for full tensor reduction for 'keepDim=False'
# option (default option). tt-metal does not support scalars on other hand; so
# it returns 1D tensor as output for full reduction op. We are reshaping the
# output so that device output tensor shape matches with the golden.
@pytest.mark.parametrize(
    ("input_shape"),
    [
        ([(64, 64)]),
        ([(4, 128, 64)]),
        ([(4, 4, 128, 128)]),
        ([(4, 8, 32, 32)]),
    ],
)
def test_sum_full(input_shape):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.reshape(torch.sum(x), (1,))

    verify_module(Basic(), input_shapes=input_shape, required_atol=0.7)


@pytest.mark.parametrize(
    ("input_shape", "dim_arg", "keep_dim", "input_type"),
    [
        ([(8, 8)], [0], True, [torch.float32]),
        ([(8, 8)], [0, 1], True, [torch.float32]),
        ([(8, 8)], [], True, [torch.float32]),
        ([(4, 32, 64)], [1], False, [torch.float32]),
        ([(4, 32, 64)], [2], False, [torch.float32]),
        ([(4, 32, 64)], [1, 2], False, [torch.float32]),
        ([(4, 2, 32, 32)], [0], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [2], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [3], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [0, 2], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [0, 2, 3], True, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [0, 1, 2, 3], True, [torch.bfloat16]),
    ],
)
def test_reduce_min(input_shape, dim_arg, keep_dim, input_type):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.amin(x, dim=dim_arg, keepdim=keep_dim)

    verify_module(
        Basic(),
        input_shapes=input_shape,
        input_data_types=input_type,
    )


# PyTorch returns a scalar value for full tensor reduction for 'keepDim=False'
# option (default option). tt-metal does not support scalars on other hand; so
# it returns 1D tensor as output for full reduction op. We are reshaping the
# output so that device output tensor shape matches with the golden.
@pytest.mark.parametrize(
    ("input_shape"),
    [
        ([(64, 64)]),
        ([(4, 128, 64)]),
        ([(4, 4, 128, 128)]),
        ([(4, 8, 32, 32)]),
    ],
)
def test_min_full(input_shape):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.reshape(torch.amin(x), (1,))

    verify_module(Basic(), input_shapes=input_shape)


@pytest.mark.parametrize(
    ("input_shape", "dim_arg", "keep_dim", "input_type"),
    [
        ([(8, 8)], [0], True, [torch.float32]),
        ([(8, 8)], [0, 1], True, [torch.float32]),
        ([(8, 8)], [], True, [torch.float32]),
        ([(4, 32, 64)], [1], False, [torch.float32]),
        ([(4, 32, 64)], [2], False, [torch.float32]),
        ([(4, 32, 64)], [1, 2], False, [torch.float32]),
        ([(4, 2, 32, 32)], [0], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [2], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [3], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [0, 2], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [1, 2], True, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [1, 2, 3], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [0, 1, 2, 3], True, [torch.bfloat16]),
    ],
)
def test_reduce_max(input_shape, dim_arg, keep_dim, input_type):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.amax(x, dim=dim_arg, keepdim=keep_dim)

    verify_module(
        Basic(),
        input_shapes=input_shape,
        input_data_types=input_type,
    )


# PyTorch returns a scalar value for full tensor reduction for 'keepDim=False'
# option (default option). tt-metal does not support scalars on other hand; so
# it returns 1D tensor as output for full reduction op. We are reshaping the
# output so that device output tensor shape matches with the golden.
@pytest.mark.parametrize(
    ("input_shape"),
    [
        ([(64, 64)]),
        ([(4, 128, 64)]),
        ([(4, 4, 128, 128)]),
        ([(4, 8, 32, 32)]),
    ],
)
def test_max_full(input_shape):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.reshape(torch.amax(x), (1,))

    verify_module(Basic(), input_shapes=input_shape)


@pytest.mark.parametrize(
    ("input_shape", "dim_arg", "keep_dim", "input_type"),
    [
        ([(8, 8)], [0], True, [torch.float32]),
        ([(8, 8)], [0, 1], True, [torch.float32]),
        ([(8, 8)], [], True, [torch.float32]),
        ([(4, 32, 64)], [1], False, [torch.float32]),
        ([(4, 32, 64)], [2], False, [torch.float32]),
        ([(4, 32, 64)], [1, 2], False, [torch.float32]),
        ([(4, 2, 32, 32)], [0], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [2], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [3], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [0, 2], False, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [1, 2], True, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [0, 1, 2, 3], True, [torch.bfloat16]),
        ([(4, 2, 32, 32)], [1, 2, 3], True, [torch.bfloat16]),
    ],
)
def test_reduce_mean(input_shape, dim_arg, keep_dim, input_type):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.mean(x, dim=dim_arg, keepdim=keep_dim)

    verify_module(
        Basic(),
        input_shapes=input_shape,
        input_data_types=input_type,
    )


# PyTorch returns a scalar value for full tensor reduction for 'keepDim=False'
# option (default option). tt-metal does not support scalars on other hand; so
# it returns 1D tensor as output for full reduction op. We are reshaping the
# output so that device output tensor shape matches with the golden.
@pytest.mark.parametrize(
    ("input_shape"),
    [
        ([(64, 64)]),
        ([(4, 128, 64)]),
        ([(4, 4, 128, 128)]),
        ([(4, 8, 32, 32)]),
    ],
)
def test_mean_full(input_shape):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.reshape(torch.mean(x), (1,))

    verify_module(Basic(), input_shapes=input_shape)
