#!/usr/bin/env python
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys

from torch_mlir._mlir_libs._mlir.ir import Module
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.ir import Context

from tt_torch.dynamo.backend import lower_to_stable_hlo

if len(sys.argv) < 2:
    exit()

filename = sys.argv[1]
with open(filename) as file:
    torch_fx_ir: str = file.read()

with Context() as context:
    torch_dialect.register_dialect(context)
    try:
        module = Module.parse(torch_fx_ir, context)
        lower_to_stable_hlo(module)
    except:
        # Error details from stderr will be reported
        # Exception contains boilerplate that can be ignored
        exit(1)
