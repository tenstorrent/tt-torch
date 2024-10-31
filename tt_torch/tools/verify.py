# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
from tt_torch.dynamo.backend import backend


def verify_module(
    mod,
    input_shapes,
    input_data_types=[torch.float32],
    required_pcc=0.99,
    required_atol=1e-2,
    input_range=(-0.5, 0.5),
):
    tt_mod = torch.compile(mod, backend=backend)

    if all([dtype.is_floating_point for dtype in input_data_types]):
        low, high = input_range
        # Uniformly distribute random numbers within the input_range
        inputs = [(low - high) * torch.rand(shape) + high for shape in input_shapes]
    else:
        inputs = [
            torch.randint(0, 1000, shape, dtype=torch.int32) for shape in input_shapes
        ]
    ret = tt_mod(*inputs)
    golden = mod(*inputs)

    atol = torch.max(torch.abs(golden - ret)).item()
    assert atol <= required_atol, f"ATOL too high: {atol} vs {required_atol}"
    pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(ret).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
        )
    )
    assert pcc >= required_pcc, f"PCC too low: {pcc} vs {required_pcc}"
