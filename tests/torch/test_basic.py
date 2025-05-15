# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.dynamo.backend import backend, BackendOptions
from torch import nn
import pytest
import math
import threading

import tt_torch
from tt_torch.tools.utils import CompilerConfig


def test_return_same_tensors():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x, y

    verify_module(Basic(), input_shapes=[(256, 256), (256, 256)])


def test_return_rt_tensor_and_torch_tensors():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y, z):
            output1 = y
            output2 = torch.abs(x)
            return output1, output2, z

    verify_module(
        Basic(),
        input_shapes=[(5, 10), (5, 10), (5, 10)],
    )


def test_abs():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.abs(x)

    verify_module(Basic(), input_shapes=[(256, 256)])


def test_add():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.add(x, y)

    verify_module(Basic(), input_shapes=[(256, 256)] * 2)


# Runs the AddOp on all detected chips in parallel
def test_add_multidevice():
    class AddOp(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.add(x, y)

    num_devices = DeviceManager.get_num_available_devices()
    parent, device_list = DeviceManager.acquire_available_devices()
    assert (
        len(device_list) == num_devices
    ), "Number of devices is not equal to expected."
    threads = []
    compiler_configs = [CompilerConfig() for _ in range(num_devices)]
    for i in range(num_devices):
        cc = compiler_configs[i]
        device = device_list[i]
        mod = AddOp()
        input_shapes = [(256, 256)] * 2
        thread = threading.Thread(
            target=verify_module,
            kwargs={
                "mod": mod,
                "input_shapes": input_shapes,
                "compiler_config": cc,
                "devices": [device],
            },
        )
        threads.append(thread)

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)
    assert (
        len(DeviceManager.get_parent_devices()) == 0
    ), "Some devices are not released."


@pytest.mark.parametrize(
    ("input"),
    [
        (torch.tensor([3.0], dtype=torch.bfloat16)),
        (torch.tensor([6], dtype=torch.int32)),
        (torch.tensor([[1], [2]], dtype=torch.float32)),
    ],
)
def test_broadcast(input):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.expand(2, 8)

    verify_module(Basic(), inputs=[input])


def test_clamp():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.clamp(x, min=-1, max=1)

    verify_module(Basic(), input_shapes=[(32, 32)], input_range=(-5, 5))


def test_clamp_tensor():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, min, max):
            return torch.clamp(x, min=min, max=max)

    verify_module(
        Basic(), input_shapes=[(32, 32), (32, 32), (32, 32)], input_range=(-5, 5)
    )


def test_concat_dim0():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat((x, y), dim=0)

    verify_module(Basic(), input_shapes=[(32, 32), (64, 32)])


def test_concat_dim1():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat((x, y), dim=1)

    verify_module(Basic(), input_shapes=[(32, 32), (32, 64)])


def test_concat_dim2():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat((x, y), dim=2)

    verify_module(Basic(), input_shapes=[(32, 32, 32), (32, 32, 64)])


def test_concat_dim3():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat((x, y), dim=3)

    verify_module(Basic(), input_shapes=[(32, 32, 32, 32), (32, 32, 32, 64)])


@pytest.mark.skip(
    "Torch keeps the 'value' as dialect resource which are not processed."
)
def test_constant_ones():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.tensor([1.0, 1.0, 1.0, 1.0])

    verify_module(Basic(), input_shapes=[(1, 1)])


def test_convert():
    class Basic_toFloat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.to(torch.float32)

    class Basic_toInt(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.to(torch.int32)

    verify_module(
        Basic_toFloat(), input_shapes=[(4, 4)], input_data_types=[torch.int32]
    )
    verify_module(
        Basic_toFloat(), input_shapes=[(4, 4)], input_data_types=[torch.float32]
    )
    verify_module(Basic_toInt(), input_shapes=[(4, 4)], input_data_types=[torch.int32])
    verify_module(
        Basic_toInt(),
        input_shapes=[(4, 4)],
        input_data_types=[torch.float32],
        input_range=(0, 60),
    )


@pytest.mark.parametrize(
    ("input_range", "input_shapes", "input_type"),
    [
        ((-0.5, 0.5), [(2, 2), (2, 2)], [torch.float32, torch.float32]),
        ((1, 10), [(32, 32), (32, 32)], [torch.bfloat16, torch.bfloat16]),
        ((1, 10), [(32, 32), (32, 32)], [torch.float32, torch.float32]),
    ],
)
def test_div(input_range, input_shapes, input_type):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x / y

    verify_module(
        Basic(),
        input_shapes=input_shapes,
        input_data_types=input_type,
        input_range=input_range,
        required_atol=0.1,
    )


def test_div_zero():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x / y

    input1 = torch.tensor([1, -2, 3, -4, 5, 6, -7, 18], dtype=torch.float32)
    input2 = torch.tensor([1, 0, 3, 0, -9, 10, -7, 12], dtype=torch.float32)
    verify_module(Basic(), inputs=[input1, input2])


def test_exp():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.exp(x)

    verify_module(Basic(), input_shapes=[(2, 2)], required_atol=3e-2)


def test_floor():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.floor(x)

    verify_module(Basic(), input_shapes=[(32, 32)], input_data_types=[torch.float32])


def test_linear():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(32, 64, bias=False)
            self.linear_b = nn.Linear(64, 64, bias=False)

        def forward(self, x):
            x = self.linear_a(x)
            x = self.linear_b(x)
            return x

    host_model = Basic()
    options = BackendOptions(
        compiler_config=CompilerConfig(),
    )
    options.compiler_config.compile_depth = tt_torch.tools.utils.CompileDepth.TORCH_FX
    model = torch.compile(host_model, backend=backend, options=options)

    input_data = torch.randn(32, 32)
    output = model(input_data)


def test_erf():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.erf(x)
            return x

    host_model = Basic()
    options = BackendOptions(
        compiler_config=CompilerConfig(),
    )
    options.compiler_config.compile_depth = tt_torch.tools.utils.CompileDepth.TORCH_FX
    model = torch.compile(host_model, backend=backend, options=options)

    input_data = torch.randn(32, 32)
    output = model(input_data)


def test_gelu():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.nn.functional.gelu(x)
            return x

    host_model = Basic()
    options = BackendOptions(
        compiler_config=CompilerConfig(),
    )
    options.compiler_config.compile_depth = tt_torch.tools.utils.CompileDepth.TORCH_FX
    model = torch.compile(host_model, backend=backend, options=options)

    input_data = torch.randn(32, 32)
    output = model(input_data)
