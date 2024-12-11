# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import torch.nn.functional as F
import pytest

import tt_torch
from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig, CompileDepth


def test_multiple_ops():
    class ConstantFoldable(nn.Module):
        def __init__(self):
            super().__init__()
            self.attr_1 = torch.nn.Parameter(torch.tensor([[-0.9]]))
            self.attr_2 = torch.nn.Parameter(torch.tensor([[17.1]]))

        def forward(self, x):
            a = self.attr_1 + self.attr_2
            b = torch.arange(0, 11)
            c = b.repeat([1, 1, 4])
            x = x - a
            x = x + c.sum()
            return x

    cc = CompilerConfig()
    verify_module(
        ConstantFoldable(),
        input_shapes=[(256, 256)],
        compiler_config=cc,
        do_assert=False,
    )


def test_interp():
    torch.set_printoptions(linewidth=1000000, threshold=1000000)

    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )

    inH = 5
    inW = 5
    inC = 1
    scale_factor = 3

    input_shape = (1, inC, inH, inW)
    small = (
        (torch.arange(torch.prod(torch.tensor(input_shape))) + 1)
        .reshape(input_shape)
        .float()
    )
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.compile_depth = CompileDepth.TORCH_FX
    verify_module(Basic(), inputs=[small], compiler_config=cc)


def test_linear():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(32, 64, bias=False)
            self.linear_b = nn.Linear(64, 64, bias=False)
            self.mm_c_weight = torch.nn.Parameter(torch.rand(64, 256) - 0.5)
            self.mm_d_weight = torch.nn.Parameter(torch.rand(256, 256) - 0.5)

        def forward(self, x):
            x = self.linear_a(x)
            x = self.linear_b(x)
            x = self.linear_b(x)
            x = torch.mm(x, self.mm_c_weight)
            x = torch.mm(x, self.mm_d_weight)
            x = torch.mm(x, self.mm_d_weight)
            return x

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    verify_module(Basic(), input_shapes=[(32, 32)], compiler_config=cc)
