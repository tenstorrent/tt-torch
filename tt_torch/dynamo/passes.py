# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch.func import functionalize
from typing import List, Optional
import traceback

from .decompositions import DEFAULT_DECOMPOSITIONS


def apply_decompositions(
    gm: torch.fx.GraphModule,
    example_inputs,
    decompose_ops: Optional[List[torch._ops.OpOverload]] = None,
):
    concrete_inputs = [
        x.view(tuple(int(dim) for dim in x.shape)) if isinstance(x, torch.Tensor) else x
        for x in example_inputs
    ]
    if decompose_ops is None:
        return gm

    decompositions = get_decompositions(decompose_ops)
    gm = make_fx(
        functionalize(gm),
        decomposition_table=decompositions,
    )(*example_inputs)

    return gm


def pass_pipeline(gm: torch.fx.GraphModule, example_inputs):
    decompose_ops = DEFAULT_DECOMPOSITIONS
    try:
        # Convert SymInt to concrete int if possible
        concrete_inputs = []
        for inp in example_inputs:
            if isinstance(inp, torch.Tensor):
                # Convert any SymInt dimensions to concrete integers
                concrete_shape = tuple(
                    int(dim) if hasattr(dim, "node") else dim for dim in inp.shape
                )
                concrete_inp = inp.view(concrete_shape)
                concrete_inputs.append(concrete_inp)
            else:
                concrete_inputs.append(inp)

        return apply_decompositions(gm, concrete_inputs, decompose_ops)
    except Exception as e:
        print(f"Pass pipeline error: {e}")
        print(traceback.format_exc())
        raise
    # return apply_decompositions(gm, example_inputs, decompose_ops)  # type: ignore
