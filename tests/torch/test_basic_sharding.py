# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_torch.dynamo.sharding_utils as ts
import torch
import torch.nn as nn
from tt_torch.dynamo.backend import CompilerConfig
from tt_torch.tools.verify import verify_module

import pytest
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
import numpy as np
import weakref


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Args:
        num_devices: Total number of devices
        mesh_shape: Shape of the device mesh (batch_dim, model_dim)

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


@pytest.mark.usefixtures("use_xla_environment")
def test_linear_param_sharded():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.sharded_weight = nn.Parameter(torch.randn(32, 128))

        def forward(self, x):
            return x @ self.sharded_weight

    cc = CompilerConfig()
    cc.xla_mesh = create_device_mesh()

    test_class = Basic()
    weight_shard_spec = (None, "model")
    ts.mark_sharding(test_class.sharded_weight, weight_shard_spec)

    # Check that sharding annotation worked
    assert ts.get_sharding(test_class.sharded_weight) == weight_shard_spec

    weak_ref = weakref.ref(test_class.sharded_weight)

    # Check that the PCC between the ondevice sharded matmul and CPU unsharded matmul is the same
    verify_module(test_class, input_shapes=[(32, 32)], compiler_config=cc)

    # Check that cache doesn't hold onto the weight tensor
    del test_class
    import gc

    gc.collect()
    assert (
        weak_ref() is None
    ), "Reference to sharded weight torch.Tensor is still held after deleting the module"
