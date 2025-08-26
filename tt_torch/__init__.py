# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Process start method must be "forkserver" for op-by-op compilation for the following reasons:
#   - The backend will hang if attempting to retrieve the device descriptor from a forked process if there are multiple chips
#   - torch tensors cannot be made contiguous in a forked process
import multiprocessing as mp
import torch

if mp.get_start_method() != "forkserver":
    mp.set_start_method("forkserver", force=True)

import os
import sys
import importlib.util

# find the tt-metal directory, it can either be in the venv if installed from a wheel or in the third_party source tree
package_name = "tt-metal"
spec = importlib.util.find_spec(package_name)
if spec is not None:
    tt_metal_home = os.path.abspath(spec.submodule_search_locations[0])
    os.environ["TT_METAL_HOME"] = tt_metal_home

# Import these modules so backends are registered ("tt", and "tt-experimental")
import tt_torch.dynamo.backend
import tt_torch.dynamo.experimental.xla_backend

from torch_xla.experimental import plugins
from torch_xla.experimental import stablehlo_custom_call


class TTPjrtPlugin(plugins.DevicePlugin):
    def library_path(self):
        # This is where the pjrt plugin will be located if you've built and installed from source
        direct_build_install_path = os.path.join(
            os.path.dirname(__file__), "../install/lib/pjrt_plugin_tt.so"
        )
        if os.path.exists(direct_build_install_path):
            return direct_build_install_path

        # This is where the pjrt plugin will be located if you've installed the tt-torch wheel into a virtual environment
        env_path = os.path.join(os.path.dirname(__file__), "../../../pjrt_plugin_tt.so")
        if os.path.exists(env_path):
            return env_path

        # This is where the pjrt plugin will be located if you've only built and installed the wheel - but you're running your code with
        # the root of the source tree in which an env was already activated (CI does this)
        source_path = os.path.join(
            os.path.dirname(__file__), "../env/venv/lib/pjrt_plugin_tt.so"
        )
        if os.path.exists(source_path):
            return source_path

        # This is where the pjrt plugin will be located if you've installed the tt-torch wheel in a clean virtual environment
        # Use sys.prefix to get the venv root instead of relative paths
        venv_install_path = os.path.join(sys.prefix, "lib/pjrt_plugin_tt.so")
        if os.path.exists(venv_install_path):
            return venv_install_path

        assert False, "Could not find pjrt_plugin_tt.so"


plugins.register_plugin("TT", TTPjrtPlugin())
os.environ["XLA_STABLEHLO_COMPILE"] = "1"
os.environ["PJRT_DEVICE"] = "TT"


@torch.library.custom_op("tt::paged_attention", mutates_args=[], device_types=["xla"])
def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
) -> torch.Tensor:
    return stablehlo_custom_call.stablehlo_custom_call(
        [query, key_cache, value_cache, context_lens, block_tables],
        "tt.paged_attention",
        [[1, query.shape[0], query.shape[1], query.shape[2]]],
        [query.dtype],
    )


@paged_attention.register_fake
def _(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
) -> torch.Tensor:
    fake_output = torch.empty_like(query)
    return fake_output.reshape(1, query.shape[0], query.shape[1], query.shape[2])


torch._dynamo.allow_in_graph(torch.ops.tt.paged_attention)
