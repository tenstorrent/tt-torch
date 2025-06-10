# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import tt_torch
from tt_torch.tools.verify import verify_against_golden
import torch_xla.core.xla_model as xm


def test_addition_xla_device():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = Model()

    x, y = torch.randn(32, 64), torch.randn(32, 64)
    golden = model(x, y)

    x, y = x.to(xm.xla_device()), y.to(xm.xla_device())
    model = model.to(xm.xla_device())

    calculated = model(x, y).to("cpu")
    verify_against_golden((golden,), (calculated,), True, True, required_atol=0.1)
