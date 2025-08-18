from typing import Dict, Tuple, Optional
import torch
import os

# Type aliases
ShardSpec = Tuple[Optional[str], ...]
# Update ShardMap to use tensor IDs as keys
ShardMap = Dict[int, ShardSpec]

# Debug flag - can be enabled via environment variable
DEBUG_SHARDING = False

class ShardingRegistry:
    def __init__(self):
        self.shard_map:ShardMap = {}
    
    def _debug_log(self, message: str) -> None:
        """Print debug message if debugging is enabled."""
        if DEBUG_SHARDING:
            print(f"[SHARD_DEBUG] {message}", flush=True)
    
    def _print_shard_map_contents(self) -> None:
        """Print current contents of the shard map for debugging."""
        if not DEBUG_SHARDING:
            return
            
        tensor_count = len(self.shard_map)
        self._debug_log(f"=== SHARD MAP CONTENTS ({tensor_count} tensors) ===")
        
        if tensor_count == 0:
            self._debug_log("  (empty)")
        else:
            for i, (tensor_id, shard_spec) in enumerate(self.shard_map.items()):
                self._debug_log(f"  {i+1:2d}. id={tensor_id}, shard_spec: {shard_spec}")

        self._debug_log("=== END SHARD MAP ===")
    
    def mark_sharding(self, tensor:torch.Tensor, shard_spec: ShardSpec) -> None:
        # Use tensor ID as the key
        tensor_id = id(tensor)
        already_exists = tensor_id in self.shard_map
        old_shard_spec = self.shard_map.get(tensor_id) if already_exists else None

        # Add/update the tensor
        self.shard_map[tensor_id] = shard_spec

        # Debug logging
        if DEBUG_SHARDING:
            tensor_info = f"id={tensor_id}, shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
            if already_exists:
                self._debug_log(f"UPDATED tensor: {tensor_info}")
                self._debug_log(f"  old shard_spec: {old_shard_spec}")
                self._debug_log(f"  new shard_spec: {shard_spec}")
            else:
                self._debug_log(f"ADDED tensor: {tensor_info}")
                self._debug_log(f"  shard_spec: {shard_spec}")

    def get_sharding(self, tensor:torch.Tensor) -> Optional[ShardSpec]:
        # Use tensor ID as the key
        tensor_id = id(tensor)
        result = self.shard_map.get(tensor_id)

        if DEBUG_SHARDING:
            tensor_info = f"id={tensor_id}, shape={list(tensor.shape)}, dtype={tensor.dtype}"
            if result is not None:
                self._debug_log(f"GET tensor: {tensor_info} -> shard_spec: {result}")
            else:
                self._debug_log(f"GET tensor: {tensor_info} -> NOT FOUND")

        return result

_sharding_registry = ShardingRegistry()

def mark_sharding(tensor: torch.Tensor, shard_spec: ShardSpec) -> None:
    """Mark sharding for a tensor."""
    _sharding_registry.mark_sharding(tensor, shard_spec)

def get_sharding(tensor: torch.Tensor) -> Optional[ShardSpec]:
    """Get sharding spec for a tensor."""
    return _sharding_registry.get_sharding(tensor)

def print_shard_map_summary() -> None:
    """Print a summary of the current shard map contents."""
    _sharding_registry._print_shard_map_contents()

def get_shard_map_size() -> int:
    """Get the number of tensors in the shard map."""
    return len(_sharding_registry.shard_map)