# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.fx
from torch.fx.passes.infra.pass_base import PassBase

# Transform: Replace aten.rand.default with a constant random tensor
class ReplaceRandWithConstant(torch.fx.Transformer):
    def call_function(self, target, args, kwargs):
        # TODO: Remove this transformation when TTNN api add support for random
        # number generation.
        if target == torch.ops.aten.rand.default:
            size = args[0]
            dtype = kwargs.get("dtype", torch.float32)
            device = kwargs.get("device", "cpu")
            layout = kwargs.get("layout", None)
            pin_memory = kwargs.get("pin_memory", False)
            # Generate fixed random tensor on cpu
            with torch.no_grad():
                const_tensor = torch.rand(
                    size,
                    dtype=dtype,
                    device="cpu",
                    layout=layout,
                    pin_memory=pin_memory,
                )
            return const_tensor.to("xla")
        return super().call_function(target, args, kwargs)
