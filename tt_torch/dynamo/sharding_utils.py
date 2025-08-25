# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Optional, Dict
import torch
import torch_xla.runtime as xr
import torch_xla
import os

# Type aliases
ShardSpec = Tuple[Optional[str], ...]
TensorKey = Tuple[int, Tuple[int, ...], torch.dtype, str]  # (id, shape, dtype, device)


def _tensor_key(tensor: torch.Tensor) -> TensorKey:
    """Generate a stable key for a tensor based on id, shape, dtype, and device"""
    return (id(tensor), tuple(tensor.shape), tensor.dtype, str(tensor.device))


class ShardingRegistry:
    def __init__(self):
        self.shard_map: Dict[TensorKey, ShardSpec] = {}

    def mark_sharding(self, tensor: torch.Tensor, shard_spec: ShardSpec) -> None:
        key = _tensor_key(tensor)
        assert (
            key not in self.shard_map
        ), f"Source tensor with shape {tensor.shape} is already marked for sharding with shard_spec {self.shard_map[key]}"
        self.shard_map[key] = shard_spec

    def get_sharding(self, tensor: torch.Tensor) -> Optional[ShardSpec]:
        key = _tensor_key(tensor)
        return self.shard_map.get(key)


_sharding_registry = ShardingRegistry()


def mark_sharding(tensor: torch.Tensor, shard_spec: ShardSpec) -> None:
    """Mark sharding for a tensor."""
    _sharding_registry.mark_sharding(tensor, shard_spec)


def get_sharding(tensor: torch.Tensor) -> Optional[ShardSpec]:
    """Get sharding spec for a tensor."""
    return _sharding_registry.get_sharding(tensor)


def get_shard_map_size() -> int:
    """Get the number of tensors in the shard map."""
    return len(_sharding_registry.shard_map)


def setup_xla_spmd_environment():
    """
    Configure XLA environment for SPMD.

    Per torchxla issue https://github.com/pytorch/xla/issues/9578 SPMD enablement
        is irreversible within the same process, so SPMD and non SPMD tests
        should not be mixed.
    """

    # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    # Initialize SPMD - This has some side effects that don't seem reversible https://github.com/pytorch/xla/issues/9578
    # It is unsafe to run pytests using the XLA backend where some tests use SPMD and some don't in the same test process
    xr.use_spmd()

    torch_xla.sync(True, True)

    print("XLA environment configured.")
