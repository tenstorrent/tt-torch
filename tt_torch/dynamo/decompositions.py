# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Dict, List, Optional, Sequence, Union

import contextlib
import threading

import torch
from torch._decomp import get_decompositions, remove_decompositions
from torch_mlir.extras.fx_decomp_util import get_decomposition_table
import numpy as np

DecompositionTable = Dict[torch._ops.OperatorBase, Callable]
DecompositionOpsList = Sequence[
    Union[torch._ops.OperatorBase, torch._ops.OpOverloadPacket]
]

# Manages "scopes" for decompositions used. Each unique scope is an attribute on
# the _decomp_local. If the attribute is missing, then the default
# decompositions are used. The scope "aot" is used for all AOT cases.
_decomp_local = threading.local()


def _get_decomp_stack(scope: str) -> List[DecompositionTable]:
    try:
        return getattr(_decomp_local, scope)
    except AttributeError:
        stack: List[DecompositionTable] = []
        setattr(_decomp_local, scope, stack)
        return stack


def _current(scope: str) -> DecompositionTable:
    """Gets the current decomposition table (which may be the default)."""
    stack = _get_decomp_stack(scope)
    if stack:
        return dict(stack[-1])
    else:
        return dict(DEFAULT_DECOMPOSITION_TABLE)


@contextlib.contextmanager
def _extend_context_manager(
    scope: str,
    *,
    from_current: bool = True,
    add_ops: Optional[DecompositionOpsList] = None,
    remove_ops: Optional[DecompositionOpsList] = None
):
    table: DecompositionTable
    if from_current:
        table = dict(_current(scope))
    else:
        table = {}
    if add_ops:
        table.update(get_decompositions(add_ops))
    if remove_ops:
        remove_decompositions(table, remove_ops)  # type: ignore
    stack = _get_decomp_stack(scope)
    stack.append(table)
    try:
        yield table
    finally:
        popped = stack.pop()
        assert (
            popped is table
        ), "contextmanager unbalanced: popped different that pushed"


# This method is derived from the implementation of jax.image.resize in JAX:
#     https://github.com/jax-ml/jax/blob/354bd5271077654af983965c8e01ee462ce4ce91/jax/_src/image/scale.py#L52
#
# I've modified it to use numpy rather than JAX. I've also added the ability
# to generate a weight matrix that allows the matmul to be identical to to
# torch's upsample_bilinear2d when align_corners=True.
# This logic was derived from @brentyi's implementation in:
#    https://github.com/jax-ml/jax/issues/11206#issuecomment-1423140760
def compute_bilinear_weight(input_size, output_size, scale, align_corners, dtype):
    translation = 0
    if align_corners:
        scale = (output_size - 1) / (input_size - 1)
        translation = 0.5 - (scale / 2)

    inv_scale = 1 / scale
    sample_f = (
        (torch.arange(output_size, dtype=torch.float64) + 0.5) * inv_scale
        - translation * inv_scale
        - 0.5
    )
    x = torch.abs(sample_f - torch.arange(input_size, dtype=torch.float64).unsqueeze(1))

    weights = torch.relu(1 - torch.abs(x))

    total_weight_sum = torch.sum(weights, axis=0, keepdims=True)
    weights = torch.divide(
        weights,
        torch.where(total_weight_sum != 0, total_weight_sum, 1),
    )

    weights = torch.where(
        torch.logical_and(sample_f >= -0.5, sample_f <= input_size - 0.5),
        weights,
        0,
    )
    weights = weights.squeeze()
    return weights.to(dtype)


def upsample_bilinear2d(
    input: torch.Tensor,
    output_size: List[int],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    input_size = input.shape[-2:]

    if scales_h is None:
        scales_h = float(output_size[0]) / float(input_size[0])

    if scales_w is None:
        scales_w = float(output_size[1]) / float(input_size[1])

    scales = [scales_h, scales_w]
    if (
        scales_h == scales_w
        and input_size[0] == input_size[1]
        and output_size[0] == output_size[1]
    ):
        weight_w = compute_bilinear_weight(
            input_size[1], output_size[1], scales[1], align_corners, input.dtype
        )
        weight_h = weight_w
    else:
        weight_w = compute_bilinear_weight(
            input_size[1], output_size[1], scales[1], align_corners, input.dtype
        )
        weight_h = compute_bilinear_weight(
            input_size[0], output_size[0], scales[0], align_corners, input.dtype
        )

    res = (input.transpose(-1, -2) @ weight_h).transpose(-1, -2) @ weight_w
    return res


def upsample_nearest2d(
    input,
    output_size,
    scales_h=None,
    scales_w=None,
):
    input_size = input.shape[-2:]

    if scales_h is None or not isinstance(scales_h, int):
        scales_h = int(output_size[0] / input_size[0])

    if scales_w is None or not isinstance(scales_w, int):
        scales_w = int(output_size[1] / input_size[1])

    # To perform a nearest neighbor upsample with matrix dot products, we need to
    # make the right hand side select each element along the columns <scale> times.
    # We can make this right hand size by creating an identity matrix and computing
    # the Kronecker product of that with a row of <scale> ones.
    weight_w = torch.kron(torch.eye(input_size[1]), torch.ones(scales_w)).to(
        input.dtype
    )
    if (
        scales_w == scales_h
        and input_size[0] == input_size[1]
        and output_size[0] == output_size[1]
    ):
        weight_h = weight_w
    else:
        weight_h = torch.kron(torch.eye(input_size[0]), torch.ones(scales_h)).to(
            input.dtype
        )

    res = (input.transpose(-1, -2) @ weight_h).transpose(-1, -2) @ weight_w
    return res


# TODO: DO we ever need this?
def _get_default_decomposition_ops() -> DecompositionOpsList:
    aten = torch.ops.aten
    # default decompositions pulled from SHARK / torch._decomp
    return [
        aten.embedding_dense_backward,
        aten.native_layer_norm_backward,
        aten.slice_backward,
        aten.select_backward,
        aten.norm.ScalarOpt_dim,
        aten.native_group_norm,
        aten.split.Tensor,
        aten.split_with_sizes,
        aten.native_layer_norm,
        aten.masked_fill.Tensor,
        aten.masked_fill.Scalar,
        aten.t,
        aten.addmm,
        # decompositions that aid us in handling nn.BatchNorm2d
        aten._native_batch_norm_legit_functional,
        aten._native_batch_norm_legit_no_training,
        aten._native_batch_norm_legit,
        aten._native_batch_norm_legit.no_stats,
        aten.squeeze.dims,
        # decompositions for miscellaneous ops that are not handled in torch-mlir but have available decompositions
        aten.soft_margin_loss,
        aten.im2col,
        aten._euclidean_dist,
        aten.index_copy,
        aten.index_copy_,
        aten.grid_sampler_2d,
        aten.log_sigmoid_forward,
        aten.unsafe_split.Tensor,
        aten.binary_cross_entropy,
        aten.dot,
        aten._adaptive_avg_pool2d,
        aten._prelu_kernel,
        aten.full,
        aten._log_softmax,
        aten.nll_loss_forward,
        aten.nll_loss_backward,
        aten._to_copy,
        aten._log_softmax_backward_data,
        aten.lift_fresh_copy.default,
        aten._unsafe_index.Tensor,
        aten.unbind.int,
        aten.linspace.Tensor_Tensor,
        aten._scaled_dot_product_flash_attention_for_cpu.default,
        aten.slice_scatter,
    ]


def _get_custom_decopositions() -> DecompositionTable:
    aten = torch.ops.aten
    return {
        # aten.upsample_nearest2d.default: upsample_nearest2d,    #TODO: https://github.com/tenstorrent/tt-torch/issues/145
        aten.upsample_bilinear2d.default: upsample_bilinear2d,
    }


# Some older APIs still use an op list instead of a table.
DEFAULT_DECOMPOSITIONS: DecompositionOpsList = _get_default_decomposition_ops()

# The table of default decompositions.
DEFAULT_DECOMPOSITION_TABLE: DecompositionTable = get_decompositions(
    DEFAULT_DECOMPOSITIONS
)

CUSTOM_DECOMPOSITION_TABLE = _get_custom_decopositions()
