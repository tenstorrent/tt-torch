# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_torch
from tt_torch.tools.verify import verify_against_golden
import torch
import torch_xla.core.xla_model as xm
import os


def test_simple_mm():
    os.environ["TT_TORCH_EXPERIMENTAL_USE_XLA"] = "1"

    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 64, bias=False)

        def forward(self, x):
            return self.linear(x)

    device = xm.xla_device()
    input_x = torch.randn(32, 32)

    model = MM()
    golden = model(input_x)

    model = torch.compile(model, backend="tt")

    output = model(input_x)

    verify_against_golden(
        (golden,),
        (output,),
        assert_pcc=True,
        assert_atol=True,
        required_pcc=0.99,
        required_atol=0.01,
    )


def test_simple_mm_eager():
    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32, bias=False)

        def forward(self, x):
            return self.linear(x)

    device = xm.xla_device()
    input_x = torch.randn(32, 32)

    model = MM()
    golden = model(input_x)

    model = model.to(device)
    input_x = input_x.to(device)

    output_device = model(input_x)
    output_host = output_device.to("cpu")

    verify_against_golden(
        (golden,),
        (output_host,),
        assert_pcc=True,
        assert_atol=True,
        required_pcc=0.99,
        required_atol=0.01,
    )
