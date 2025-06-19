# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Process start method must be "forkserver" for op-by-op compilation for the following reasons:
#   - The backend will hang if attempting to retrieve the device descriptor from a forked process if there are multiple chips
#   - torch tensors cannot be made contiguous in a forked process
import multiprocessing as mp
import torch
from typing import Union

if mp.get_start_method() != "forkserver":
    mp.set_start_method("forkserver", force=True)

import os
import importlib.util

from torch_xla.experimental import plugins


class TTPjrtPlugin(plugins.DevicePlugin):
    def library_path(self):
        # This is where the pjrt plugin will be located if you've built and installed tt-torch according to the instructions in README.md
        direct_build_install_path = os.path.join(
            os.path.dirname(__file__), "../install/tt-xla/lib/pjrt_plugin_tt.so"
        )
        if os.path.exists(direct_build_install_path):
            return direct_build_install_path

        # This is where the pjrt plugin will be located if you've installed the tt-torch wheel in another project environment
        env_path = os.path.join(
            os.path.dirname(__file__), "../../../tt-xla/lib/pjrt_plugin_tt.so"
        )
        if os.path.exists(env_path):
            return env_path

        # This is where the pjrt plugin will be located if you've only built and installed the wheel - but you're running your code with the root of the source tree (CI does this)
        source_path = os.path.join(
            os.path.dirname(__file__), "../env/venv/tt-xla/lib/pjrt_plugin_tt.so"
        )
        if os.path.exists(source_path):
            return source_path

        assert False, "Could not find pjrt_plugin_tt.so"


plugins.register_plugin("TT", TTPjrtPlugin())
os.environ["XLA_STABLEHLO_COMPILE"] = "1"
os.environ["PJRT_DEVICE"] = "TT"


def set_tt_metal_home(*, device_api: str = None):
    """
    This function will correctly set the TT_METAL_HOME directory depending on which device API you use ("tt-xla" or "tt-torch").
    Each device api has a different underlying tt-metal build that currently cannot be used interchangeably.
    Parameters:
        device_api: str | Either "tt-torch" or "tt-xla"
    """
    assert device_api in [
        "tt-xla",
        "tt-torch",
    ], f"Invalid device API: {device_api}. Expecting one of ['tt-torch', 'tt-xla']"

    if device_api == "tt-xla":
        spec = importlib.util.find_spec("tt-xla")
        assert spec is not None, "tt-xla directory not found"
        tt_xla_path = os.path.abspath(spec.submodule_search_locations[0])
        os.environ["TT_METAL_HOME"] = f"{tt_xla_path}/tt-metal"
    elif device_api == "tt-torch":
        spec = importlib.util.find_spec("tt-metal")
        assert spec is not None, "tt-metal directory not found"
        tt_metal_path = os.path.abspath(spec.submodule_search_locations[0])
        os.environ["TT_METAL_HOME"] = tt_metal_path
    else:
        assert (
            False
        ), "device_api should have already been asserted to be one of ['tt_torch', 'tt_xla']"


set_tt_metal_home(device_api="tt-torch")
