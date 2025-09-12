# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest

from tt_torch.tools.verify import verify_module

# Examples from https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html

# Example 1:
# src = torch.arange(1, 11).reshape((2, 5))
# src
# tensor([[ 1,  2,  3,  4,  5],
#         [ 6,  7,  8,  9, 10]])
# index = torch.tensor([[0, 1, 2, 0]])
# torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
# tensor([[1, 0, 0, 4, 0],
#         [0, 2, 0, 0, 0],
#         [0, 0, 3, 0, 0]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_scatter_dim0(dtype):
    class ScatterDim0(nn.Module):
        def forward(self, x, index, src):
            return torch.scatter(x, dim=0, index=index, src=src)

    src = torch.arange(1, 11, dtype=dtype).reshape((2, 5))
    index = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.int64)  # shape (1,5)
    x = torch.zeros((3, 5), dtype=dtype)
    verify_module(
        ScatterDim0(),
        inputs=[x, index, src],
        required_atol=1,
    )


# Example 2:
# src = torch.arange(1, 11).reshape((2, 5))
# src
# tensor([[ 1,  2,  3,  4,  5],
#         [ 6,  7,  8,  9, 10]])
# index = torch.tensor([[0, 1, 2], [0, 1, 4]])
# torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
# tensor([[1, 2, 3, 0, 0],
#         [6, 7, 0, 0, 8],
#         [0, 0, 0, 0, 0]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_scatter_dim1(dtype):
    class ScatterDim1(nn.Module):
        def forward(self, x, index, src):
            return torch.scatter(x, dim=1, index=index, src=src)

    src = torch.arange(1, 11, dtype=dtype).reshape((2, 5))
    index = torch.tensor([[0, 1, 2], [0, 1, 4]], dtype=torch.int64)  # shape (2,5)
    x = torch.zeros((3, 5), dtype=dtype)
    verify_module(
        ScatterDim1(),
        inputs=[x, index, src],
        required_atol=1,
    )


# Example 3:
# index = torch.tensor([[0, 1]])
# value = 2
# torch.zeros(3, 5).scatter_(0, index, value)
# tensor([[2., 0., 0., 0., 0.],
#         [0., 2., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_scatter_scalar_value_fixed(dtype):
    class ScatterScalarValueFixed(nn.Module):
        def forward(self, x, index, value):
            src = torch.full_like(index, value, dtype=dtype)
            return torch.scatter(x, dim=0, index=index, src=src)

    index = torch.tensor([[0, 1]], dtype=torch.int64)  # shape (1,2)
    value = torch.tensor(2, dtype=dtype)
    x = torch.zeros((3, 5), dtype=dtype)
    verify_module(
        ScatterScalarValueFixed(),
        inputs=[x, index, value],
        required_atol=1,
    )
