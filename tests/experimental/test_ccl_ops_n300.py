# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_torch

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import os
import copy
from tests.utils import create_device_mesh, setup_xla_environment_for_tp
import pytest

"""
These tests are meant to be run on an N300 (2 devices).
"""

@pytest.mark.parametrize("shard_dim", [0, 1])
def test_all_reduce(shard_dim):
    """Test all_reduce operation with sharding on different dimensions.

    Args:
        shard_dim: Dimension to shard on (0 for batch, 1 for model)
    """
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    setup_xla_environment_for_tp()

    # Create tensor with values that make reduction easy to verify
    t = torch.ones(256, 512)
    t = t.to(torch_xla.device())

    if shard_dim == 0:
        # Shard on batch dimension (dim 0)
        mesh = create_device_mesh((2, 1), ("batch", "model"))
        xs.mark_sharding(t, mesh, ("batch", None))
        groups = [[0, 1]]
    elif shard_dim == 1:
        # Shard on model dimension (dim 1)
        mesh = create_device_mesh((1, 2), ("batch", "model"))
        xs.mark_sharding(t, mesh, (None, "model"))
        groups = [[0, 1]]

    # Perform all_reduce sum operation
    y = xm.all_reduce(xm.REDUCE_SUM, t, groups=groups)

    torch_xla.sync()
    y = y.to("cpu")
    print(f"All-reduce shard dim: {shard_dim}, Y Shape: {y.shape}")
    print(f"y: {y}")

    expected = torch.ones(256, 512) * 2.0

    assert torch.allclose(y, expected, atol=0.001)


@pytest.mark.parametrize("shard_dim", [0, 1])
def test_all_gather(shard_dim):
    """Test all_gather operation with sharding on different dimensions.

    Args:
        shard_dim: Dimension to shard on (0 for batch, 1 for model)
    """
    setup_xla_environment_for_tp()

    # Random inputs between 0 and 0.1
    t = (torch.rand(8192, 784) - 0.0) * 0.1
    golden = copy.deepcopy(t)
    print("Golden shape: ", golden.shape)

    t = t.to(torch_xla.device())

    if shard_dim == 0:
        mesh = create_device_mesh((2, 1), ("batch", "model"))
        xs.mark_sharding(t, mesh, ("batch", None))
        groups = [[0, 1]]
        gather_dim = 0
    else:
        mesh = create_device_mesh((1, 2), ("batch", "model"))
        xs.mark_sharding(t, mesh, (None, "model"))
        groups = [[0, 1]]
        gather_dim = 1

    y = xm.all_gather(t, gather_dim, groups=groups, pin_layout=False)

    y = y.to("cpu")
    print(f"All-gather shard dim: {shard_dim}, Y Shape: {y.shape}")
    chunks = torch.chunk(y, len(groups[0]), dim=gather_dim)
    for i in range(1, len(chunks)):
        assert torch.allclose(chunks[i], chunks[0], atol=0.001)
    assert torch.allclose(chunks[0], golden, atol=0.001)
