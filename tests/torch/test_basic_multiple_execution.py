# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import tt_torch


def test_multiple_execution():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32)

        def forward(self, x):
            return self.linear(x)

    model = Basic()
    model = torch.compile(model, backend="tt")
    inputs = torch.randn(32, 32)

    for _ in range(10):
        model(inputs)
