# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.dynamo.backend import backend, BackendOptions
from torch import nn
import pytest
import math
import threading

import tt_torch
from tt_torch.tools.utils import CompilerConfig


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

    host_model = Basic()
    options = BackendOptions(
        compiler_config=CompilerConfig(),
    )
    options.compiler_config.compile_depth = tt_torch.tools.utils.CompileDepth.TORCH_FX
    model = torch.compile(host_model, backend=backend, options=options)

    input_data = torch.randn(32, 32)
    output = model(input_data)


def test_erf():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.erf(x)
            return x

    host_model = Basic()
    options = BackendOptions(
        compiler_config=CompilerConfig(),
    )
    options.compiler_config.compile_depth = tt_torch.tools.utils.CompileDepth.TORCH_FX
    model = torch.compile(host_model, backend=backend, options=options)

    input_data = torch.randn(32, 32)
    output = model(input_data)


def test_gelu():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.nn.functional.gelu(x)
            return x

    host_model = Basic()
    options = BackendOptions(
        compiler_config=CompilerConfig(),
    )
    options.compiler_config.compile_depth = tt_torch.tools.utils.CompileDepth.TORCH_FX
    model = torch.compile(host_model, backend=backend, options=options)

    input_data = torch.randn(32, 32)
    output = model(input_data)
