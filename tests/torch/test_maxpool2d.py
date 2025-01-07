# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest

import tt_torch
from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig, CompileDepth


def test_maxpool2d():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)

    cc = CompilerConfig()
    cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
    verify_module(
        Basic(),
        inputs=[torch.randn(1, 1, 224, 224).to(torch.bfloat16)],
        compiler_config=cc,
    )
