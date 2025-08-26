# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This import is unused but needed to register the Tenstorrent PJRT device with XLA
import tt_torch

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import os
import copy
from tests.utils import create_device_mesh
import pytest

"""
These tests are meant to be run on an LLMBox or T3K (8 devices).
"""


@pytest.mark.parametrize("shard_dim", [0, 1])
@pytest.mark.usefixtures("use_xla_spmd_environment")
def test_all_reduce(shard_dim):
    """Test all_reduce operation with sharding on different dimensions.

    Args:
        shard_dim: Dimension to shard on (0 for batch, 1 for model)
    """
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    mesh = create_device_mesh((2, 4), ("batch", "model"))

    # Create tensor with values that make reduction easy to verify
    t = torch.ones(256, 512)
    t = t.to(torch_xla.device())

    if shard_dim == 0:
        # Shard on batch dimension (dim 0)
        xs.mark_sharding(t, mesh, ("batch", None))
        # For all_reduce on batch sharding: pair devices across batch rows
        groups = [[0, 4], [1, 5], [2, 6], [3, 7]]
    else:
        # Shard on model dimension (dim 1)
        xs.mark_sharding(t, mesh, (None, "model"))
        # For all_reduce on model sharding: two groups of 4 devices each
        groups = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Perform all_reduce sum operation
    y = xm.all_reduce(xm.REDUCE_SUM, t, groups=groups)

    torch_xla.sync()
    y = y.to("cpu")
    print(f"All-reduce shard dim: {shard_dim}, Y Shape: {y.shape}")
    print(f"y: {y}")

    # All_reduce sums values across the sharded dimension within each group
    # The result tensor has reduced shape along the sharded dimension
    if shard_dim == 0:
        expected = torch.ones(256, 512) * 2.0
    else:
        expected = torch.ones(256, 512) * 4.0

    assert torch.allclose(y, expected, atol=0.001)


@pytest.mark.parametrize("shard_dim", [0, 1])
@pytest.mark.usefixtures("use_xla_spmd_environment")
def test_all_gather(shard_dim):
    """Test all_gather operation with sharding on different dimensions.

    Args:
        shard_dim: Dimension to shard on (0 for batch, 1 for model)
    """
    mesh = create_device_mesh((2, 4), ("batch", "model"))

    # Random inputs between 0 and 0.1
    t = (torch.rand(8192, 784) - 0.0) * 0.1
    golden = copy.deepcopy(t)
    print("Golden shape: ", golden.shape)

    t = t.to(torch_xla.device())

    if shard_dim == 0:
        # Shard on batch dimension (dim 0)
        xs.mark_sharding(t, mesh, ("batch", None))
        # Correct replica groups for batch sharding: pair devices across batch rows
        groups = [[0, 4], [1, 5], [2, 6], [3, 7]]
        gather_dim = 0
    else:
        # Shard on model dimension (dim 1)
        xs.mark_sharding(t, mesh, (None, "model"))
        # For model sharding: two groups of 4 devices each (one group per batch row)
        groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
        gather_dim = 1

    y = xm.all_gather(t, gather_dim, groups=groups, pin_layout=False)

    y = y.to("cpu")
    print(f"All-gather shard dim: {shard_dim}, Y Shape: {y.shape}")
    chunks = torch.chunk(y, len(groups[0]), dim=gather_dim)
    for i in range(1, len(chunks)):
        assert torch.allclose(chunks[i], chunks[0], atol=0.001)
    assert torch.allclose(chunks[0], golden, atol=0.001)
