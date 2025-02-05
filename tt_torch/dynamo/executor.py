# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tt_torch.tools.utils import CompilerConfig
import torch


class Executor:
    def __init__(
        self,
        compiler_config=None,
        required_pcc=0.99,
        required_atol=1e-2,
    ):

        self.binary = None
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self.compiler_config = compiler_config
        self.required_atol = required_atol
        self.required_pcc = required_pcc

        # Dictionary to keep track of the type conversion for unsupported hardware
        # types and use it to convert the input arguments to supported types.
        self.type_conversion = {
            torch.bool: torch.bfloat16,
            torch.int64: torch.int32,
            torch.float64: torch.float32,
        }

    def set_binary(self, binary):
        self.binary = binary
