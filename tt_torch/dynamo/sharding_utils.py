# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Optional, Dict
import torch

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
