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

def test_nearest_upsample():
    class Padding(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, source):
            result = F.pad(input=source, pad=(0, 1, 1, 1), mode='constant', value=0)
            return result


    input_shape = (5, 10)
    small = torch.randn(input_shape, dtype=torch.bfloat16)

    cc = CompilerConfig()
    cc.enable_consteval = True
    verify_module(Padding(), inputs=[small], compiler_config=cc, required_atol=0.02)
