# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
from torch import nn
import pytest
import os

from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig, CompileDepth


def test_add():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.add(x, y)

    torch_model = Basic()
    torch_input = (torch.randn(256, 256), torch.randn(256, 256))
    torch.onnx.export(torch_model, torch_input, "add.onnx")
    model_proto = onnx.load("add.onnx")
    os.remove("add.onnx")
    verify_module(model_proto, compiler_config=CompilerConfig())
