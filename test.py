import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
from tt_torch.dynamo.backend import CompilerConfig, backend, BackendOptions

from transformers.models.llama.modeling_llama import LlamaModel
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

    # Basic XLA configuration
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE" # Enables the auto parallel pass in tt-mlir
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1" # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ["MESH_SHAPE"] = f"1,{num_devices}" # Sets the mesh shape used by the auto parallel pass
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



def main():
    setup_xla_environment()
    mesh = create_device_mesh()
    cc = CompilerConfig()
    cc.mesh = mesh # routed into xlaBackend
    options = BackendOptions()
    options.compiler_config = cc
    
    x = torch.ones((32,16))
    x.shard_spec = (None, 'model')

    # Create an instance of FooModule
    foo_model = FooModule()
    foo_model.x1.shard_spec = ('model', None)
    
    tt_model = torch.compile(
        foo_model, backend='tt-experimental', dynamic=False, options=options
    )
    result = tt_model(x)
    results = [el.to('cpu') for el in result]

main()