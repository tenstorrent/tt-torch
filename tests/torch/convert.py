# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest
import math
import threading

import tt_torch
from tt_torch.tools.verify import verify_torch_module_async
from tt_torch.tools.utils import CompilerConfig
from tt_torch.tools.device_manager import DeviceManager


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

    # int32 -> float32 succeeds
    verify_torch_module_async(
        Basic_toFloat(), input_shapes=[(4, 4)], input_data_types=[torch.int32]
    )
    # float32 -> float32 fails
    verify_torch_module_async(
        Basic_toFloat(), input_shapes=[(4, 4)], input_data_types=[torch.float32]
    )
    # int32 -> int32 fails
    verify_torch_module_async(
        Basic_toInt(), input_shapes=[(4, 4)], input_data_types=[torch.int32]
    )
    # float32 -> int32 succeeds
    verify_torch_module_async(
        Basic_toInt(),
        input_shapes=[(4, 4)],
        input_data_types=[torch.float32],
        input_range=(0, 60),
    )


if __name__ == "__main__":
    test_convert()
