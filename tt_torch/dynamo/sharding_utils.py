from typing import Dict, Tuple, Optional
import torch

# Type aliases
ShardSpec = Tuple[Optional[str], ...]
ShardMap = Dict[torch.Tensor, ShardSpec]

class ShardingRegistry:
    def __init__(self):
        self.shard_map:ShardMap = {}
    
    def mark_sharding(self, tensor:torch.Tensor, shard_spec: ShardSpec) -> None:
        self.shard_map[tensor] = shard_spec
    
    def get_sharding(self, tensor:torch.Tensor) -> Optional[ShardSpec]:
        return self.shard_map[tensor]

_sharding_registry = ShardingRegistry()

def mark_sharding(tensor: torch.Tensor, shard_spec: ShardSpec) -> None:
    """Mark sharding for a tensor."""
    _sharding_registry.mark_sharding(tensor, shard_spec)

def get_sharding(tensor: torch.Tensor) -> Optional[ShardSpec]:
    """Get sharding spec for a tensor."""
    return _sharding_registry.get_sharding(tensor)