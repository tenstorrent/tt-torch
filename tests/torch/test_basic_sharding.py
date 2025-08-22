# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_torch.dynamo.sharding_utils as ts
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
from tt_torch.dynamo.backend import CompilerConfig, BackendOptions
from tt_torch.tools.verify import verify_module

import os
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
import torch_xla


class FooModule(nn.Module):
    def __init__(self):
        super(FooModule, self).__init__()
        # Define x1 as a weight parameter
        self.x1 = nn.Parameter(torch.ones((32, 32)))

    def forward(self, x2):
        x2 *= 2
        y1 = self.x1 @ x2
        return y1, x2


def setup_xla_environment():
    """Setup XLA environment for tensor parallelism."""
    print("Setting up XLA environment...")
    num_devices = xr.global_runtime_device_count()

    # Enables the auto parallel pass in tt-mlir
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    # Sets the mesh shape used by the auto parallel pass
    os.environ["MESH_SHAPE"] = f"1,{num_devices}"

    os.environ["TT_TORCH_USE_EXPERIMENTAL_BACKEND"] = "1"
    # Initialize SPMD
    xr.use_spmd()

    torch_xla.sync(True, True)
    print("XLA environment configured.")


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


def test_linear():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.sharded_weight = nn.Parameter(torch.randn(32, 64))

        def forward(self, x):
            return x @ self.sharded_weight

    setup_xla_environment()
    cc = CompilerConfig()
    cc.mesh = create_device_mesh()
    test_class = Basic()
    weight_shard_spec = (None, "model")  # Shard along the model dimension
    ts.mark_sharding(test_class.sharded_weight, weight_shard_spec)

    # This also verifies that the PCC between the ondevice sharded matmul  and CPU unsharded matmul is the same
    verify_module(test_class, input_shapes=[(32, 32)], compiler_config=cc)

    assert ts.get_sharding(test_class.sharded_weight) == weight_shard_spec
