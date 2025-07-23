# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from tt_torch.tools.verify import verify_against_golden
import torch
import torch_xla

import pytest


def test_bernoulli_transformation():
    input_x = torch.full((1, 16), 0.5, dtype=torch.float32)

    class Bernoulli(torch.nn.Module):
        def forward(self, x):
            return torch.bernoulli(x)

    model = Bernoulli()
    torch.manual_seed(33)
    golden = model(input_x)

    device = torch_xla.device()
    model = torch.compile(model, backend="tt-experimental")
    input_x = input_x.to(device)

    torch.manual_seed(33)
    output = model(input_x)
    output = output.to("cpu")

    verify_against_golden(
        (golden,),
        (output,),
        assert_pcc=True,
        assert_atol=True,
        required_pcc=0.99,
        required_atol=0.00,
    )
