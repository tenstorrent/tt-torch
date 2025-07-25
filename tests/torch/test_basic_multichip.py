# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from tt_torch.dynamo.backend import BackendOptions
from torch import nn
import torch
import tt_torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.tools.device_manager import DeviceManager
from tt_torch.tools.verify import verify_against_golden


def test_pipeline_parallel():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(32, 32)
            self.l2 = nn.Linear(64, 32)

        def forward(self, x, y):
            x = self.l1(x)
            y = self.l2(y)
            return x + y

    options = BackendOptions()
    cc = CompilerConfig()
    options.compiler_config = cc
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.device_map = {"l1": 0, "l2": 1}
    parent_device = DeviceManager.create_parent_mesh_device([1, 2])
    device1 = DeviceManager.create_sub_mesh_device(parent_device, (0, 0))
    device2 = DeviceManager.create_sub_mesh_device(parent_device, (0, 1))
    options.devices = [device1, device2]

    host_model = Basic()

    model = torch.compile(host_model, backend="tt", options=options)
    x = torch.rand(32, 32)
    y = torch.rand(32, 64)
    calculated = model(x, y)
    golden = host_model(x, y)
    verify_against_golden((golden,), (calculated,), True, True, required_atol=0.1)
    DeviceManager.release_parent_device(parent_device, True)


def test_pipeline_parallel_topological_sort():
    class L3(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32)

        def forward(self, a, b):
            # Example: add the two inputs, then apply linear
            return self.linear(a + b)

    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(32, 32)
            self.l2 = nn.Linear(64, 32)
            self.l3 = L3()
            self.l4 = nn.Linear(32, 32)

        def forward(self, x, y):
            x = self.l1(x)
            y = self.l2(y)
            z = self.l3(x, y)
            a = self.l4(z)
            return a

    options = BackendOptions()
    cc = CompilerConfig()
    options.compiler_config = cc
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.device_map = {"l1": 0, "l2": 1, "l3": 1, "l4": 0}
    parent_device = DeviceManager.create_parent_mesh_device([1, 2])
    device1 = DeviceManager.create_sub_mesh_device(parent_device, (0, 0))
    device2 = DeviceManager.create_sub_mesh_device(parent_device, (0, 1))
    options.devices = [device1, device2]

    host_model = Basic()

    model = torch.compile(host_model, backend="tt", options=options)
    x = torch.rand(32, 32)
    y = torch.rand(32, 64)
    calculated = model(x, y)
    golden = host_model(x, y)
    verify_against_golden((golden,), (calculated,), True, True, required_atol=0.1)
    DeviceManager.release_parent_device(parent_device, True)
