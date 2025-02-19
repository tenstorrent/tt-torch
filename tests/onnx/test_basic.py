# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest

from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig


def test_add():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.add(x, y)

    torch_model = Basic()
    torch_input = (torch.randn(256, 256), torch.randn(256, 256))
    torch.onnx.export(torch_model, torch_input, "add.onnx")

    verify_module("add.onnx")
