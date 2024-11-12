# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest

import tt_torch
from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig


def test_abs():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.abs(x)

    verify_module(Basic(), input_shapes=[(256, 256)])


def test_add():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.add(x, y)

    verify_module(Basic(), input_shapes=[(256, 256)] * 2)


def test_concat_dim0():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat((x, y), dim=0)

    verify_module(Basic(), input_shapes=[(32, 32), (64, 32)])


def test_concat_dim1():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat((x, y), dim=1)

    verify_module(Basic(), input_shapes=[(32, 32), (32, 64)])


def test_concat_dim2():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat((x, y), dim=2)

    verify_module(Basic(), input_shapes=[(32, 32, 32), (32, 32, 64)])


def test_concat_dim3():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat((x, y), dim=3)

    verify_module(Basic(), input_shapes=[(32, 32, 32, 32), (32, 32, 32, 64)])


@pytest.mark.skip(
    "Torch keeps the 'value' as dialect resource which are not processed."
)
def test_constant_ones():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.tensor([1.0, 1.0, 1.0, 1.0])

    verify_module(Basic(), input_shapes=[(1, 1)])


def test_convert():
    class Basic_toFloat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.to(torch.float32)

    class Basic_toInt(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.to(torch.int32)

    verify_module(
        Basic_toFloat(), input_shapes=[(4, 4)], input_data_types=[torch.int32]
    )
    verify_module(
        Basic_toFloat(), input_shapes=[(4, 4)], input_data_types=[torch.float32]
    )
    verify_module(Basic_toInt(), input_shapes=[(4, 4)], input_data_types=[torch.int32])
    verify_module(
        Basic_toInt(),
        input_shapes=[(4, 4)],
        input_data_types=[torch.float32],
        input_range=(0, 60),
    )


def test_div():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x / y

    verify_module(Basic(), input_shapes=[(2, 2), (2, 2)], required_atol=5e-2)


def test_exp():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.exp(x)

    verify_module(Basic(), input_shapes=[(2, 2)], required_atol=3e-2)


def test_linear():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(32, 64, bias=False)
            self.linear_b = nn.Linear(64, 64, bias=False)

        def forward(self, x):
            x = self.linear_a(x)
            x = self.linear_b(x)
            return x

    verify_module(Basic(), input_shapes=[(32, 32)])


from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType


def test_linear_with_bias():
    pytest.xfail()

    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(32, 32)

        def forward(self, x):
            x = self.linear_a(x)
            return x

    verify_module(Basic(), input_shapes=[(32, 32)])


def test_maximum():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.maximum(x, y)

    verify_module(Basic(), input_shapes=[(32, 32), (32, 32)], input_range=(-6, 6))


def test_multiply():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x * y

    verify_module(Basic(), input_shapes=[(32, 32), (32, 32)])


def test_negate():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return -x

    verify_module(Basic(), input_shapes=[(32, 32)], input_range=(-6, 6))


@pytest.mark.skip("keepdim=False is not supported")
def test_reduce_max():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.max(x)

    verify_module(Basic(), input_shapes=[(32, 32)], input_range=(-6, 6))


@pytest.mark.skip("keepdim=False is not supported")
def test_reduce_sum():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sum(x)

    verify_module(Basic(), input_shapes=[(32, 32)], input_range=(-6, 6))


def test_relu():
    pytest.xfail()

    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.relu(x)

    verify_module(Basic(), input_shapes=[(32, 32)])


def test_rsqrt():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.rsqrt(x)

    verify_module(
        Basic(), input_shapes=[(32, 32)], required_atol=3e-2, input_range=(0.1, 1)
    )


def test_sqrt():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sqrt(x)

    verify_module(
        Basic(), input_shapes=[(32, 32)], required_atol=3e-2, input_range=(0.1, 1)
    )


dim0_cases = []
for begin in torch.arange(10).tolist():
    for end in torch.arange(90, 100).tolist():
        dim0_cases.append((begin, end, 0))

dim1_cases = []
for begin in torch.arange(10).tolist():
    for end in torch.arange(90, 100).tolist():
        dim1_cases.append((begin, end, 1))

dim2_cases = []
for begin in torch.arange(0, 64, 32).tolist():
    for end in torch.arange(64, 128, 32).tolist():
        dim2_cases.append((begin, end, 2))

dim3_cases = []
for begin in torch.arange(0, 64, 32).tolist():
    for end in torch.arange(64, 128, 32).tolist():
        dim3_cases.append((begin, end, 3))


@pytest.mark.parametrize(
    "begin, end, dim", [*dim2_cases, *dim3_cases, *dim0_cases, *dim1_cases]
)
def test_slice(begin, end, dim):
    # Slice test is only working for dim=3; skipping all other tests.
    if dim != 3:
        pytest.skip()

    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            if dim == 0:
                return a[begin:end, :, :, :]
            elif dim == 1:
                return a[:, begin:end, :, :]
            elif dim == 2:
                return a[:, :, begin:end, :]
            else:
                return a[:, :, :, begin:end]

    shape = [10, 10, 10, 10]
    shape[dim] = 128
    verify_module(Basic(), input_shapes=[shape])


def test_subtract():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x - y

    verify_module(Basic(), input_shapes=[(32, 32), (32, 32)], input_range=(-6, 6))


def test_transpose_2d():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.transpose(x, 0, 1)

    verify_module(Basic(), input_shapes=[(4, 8)], input_range=(-6, 6))


@pytest.mark.skip("TTNN does not support transpose for higher ranks/dimensions.")
def test_transpose_3d():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.transpose(x, 0, 1)

    verify_module(Basic(), input_shapes=[(4, 8, 4)], input_range=(-6, 6))


def test_multiple_ops():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            y = x + x
            z = y + y
            z = torch.argmax(z)
            return z

    cc = CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    verify_module(
        Basic(), input_shapes=[(256, 256)], compiler_config=cc, do_assert=False
    )
