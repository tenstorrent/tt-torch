# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest

import tt_torch
from tt_torch.tools.verify import verify_module


@pytest.mark.xfail(reason="softmax_kernel_impl not implemented for Bool")
def test_safe_softmax_bool():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch._safe_softmax(x, 1)

    input_tensor = torch.randint(0, 2, (1, 16, 197, 197), dtype=torch.bool)
    verify_module(Basic(), inputs=[input_tensor])
