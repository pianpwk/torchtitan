"""Grouped matmul for routed experts on Tokamax's ``gmm_v2`` / ``tgmm_v2``.

Same contract as ``expert_segment_matmul.segment_matmul_pallas_2d``: for
expert ``e``, forward ``Y_e = X_e @ W_e``, backward ``dX_e = dY_e @ W_e.T``
(``gmm_v2`` with an in-kernel rhs transpose) and ``dW_e = X_e.T @ dY_e``
(``tgmm_v2``), FP32 accumulation, outputs rounded once to the input / weight
dtype. ``group_sizes`` is a runtime int32 tensor; rows past its sum are not
read and their output rows are zero.

MegaBlox uses one fixed M tile, so a tile that straddles a group boundary is
computed once per group it touches. ``gmm_v2`` starts tiles at each group's
first row, sizes a group's last tile to its rows, and picks its own K/N tiles,
so no tiling needs to be configured per model.

TODO: ``gmm_v2`` / ``tgmm_v2`` are Tokamax internals
(``tokamax._src.ops.gmm_v2``); switch to a public entry point once Tokamax
exposes one.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.gmm_v2 import gmm_v2 as tokamax_gmm_v2
from tokamax._src.ops.gmm_v2 import tgmm_v2 as tokamax_tgmm_v2
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels.pallas_naming import call_named

# Tokamax's tgmm tiling grows tile_m in steps while its VMEM estimate fits.
# On TPU7x, tiles above 512 rows that also fill the last fifth of VMEM run 3-4x
# slower (Qwen3-235B K=4096, N=1536: tile_m 768 at 12% of peak vs 512 at 52%;
# Qwen3.5-122B K=3072, N=1024: 1024 at 10% vs 512 at 45%), while 1024-row
# tiles with VMEM to spare (Qwen3-30B) and 512-row tiles near the limit
# (DeepSeek-V4-Flash) are fine. Such tiles are re-tiled at 80% of the limit,
# keeping tile_k / tile_n and at least 512 rows.
# TODO: move this into Tokamax's calculate_tgmm_tiling.
_TGMM_MAX_TILE_M_NEAR_VMEM_LIMIT = 512
_TGMM_VMEM_HEADROOM_FRACTION = 0.8


def _tgmm_tiling(
    dims,
    lhs_cfgs,
    rhs_cfgs,
    vmem_limit_bytes,
    out_dtype,
    acc_dtype,
    target_zero_ref_bytes,
):
    tiling = functools.partial(
        tokamax_tgmm_v2.calculate_tgmm_tiling,
        dims,
        lhs_cfgs,
        rhs_cfgs,
        out_dtype=out_dtype,
        acc_dtype=acc_dtype,
        target_zero_ref_bytes=target_zero_ref_bytes,
    )
    tiles = tiling(vmem_limit_bytes=vmem_limit_bytes)
    if tiles.tile_m <= _TGMM_MAX_TILE_M_NEAR_VMEM_LIMIT:
        return tiles
    try:
        tight = tiling(
            vmem_limit_bytes=int(
                vmem_limit_bytes * _TGMM_VMEM_HEADROOM_FRACTION
            )
        )
    except ValueError:  # Even the smallest tiles exceed 80% of the limit.
        return tiles
    if (tight.tile_k, tight.tile_n) != (tiles.tile_k, tiles.tile_n):
        return tiles
    tile_m = max(tight.tile_m, _TGMM_MAX_TILE_M_NEAR_VMEM_LIMIT)
    return dataclasses.replace(tiles, tile_m=tile_m)


def weight_grad(
    lhs: jax.Array,
    dout: jax.Array,
    group_sizes: jax.Array,
    num_groups: int,
    dtype: jnp.dtype,
) -> jax.Array:
    """Per-group ``lhs.T @ dout`` (``tgmm_v2``) in FP32, stored in ``dtype``."""
    # tgmm_v2 views its rows in blocks of the dtype's sublane tiling; zero rows
    # past every group add nothing to dWeight.
    row_block = pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype)
    pad = -lhs.shape[0] % row_block
    if pad:
        lhs, dout = (jnp.pad(x, ((0, pad), (0, 0))) for x in (lhs, dout))
    return call_named(
        tokamax_tgmm_v2.tgmm_v2,
        "moe_tgmm_wgrad",
        lhs,
        dout,
        group_sizes,
        num_groups,
        preferred_element_type=dtype,
        tile_info=_tgmm_tiling,
    )


@pallas.jax_op("tt_experts::grouped_mm_v2")
def grouped_mm_v2(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    transpose_rhs: bool,
) -> jax.Array:
    """Per-expert matmuls of ``lhs [R, K]`` with ``rhs [E, K, N]``.

    ``group_sizes`` is ``int32[E]``. ``transpose_rhs`` takes ``rhs`` as
    ``[E, N, K]`` without a physical transpose.
    """
    return call_named(
        tokamax_gmm_v2.gmm_v2,
        "moe_gmm_fwd",
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=lhs.dtype,
        transpose_rhs=transpose_rhs,
    )


@pallas.jax_op("tt_experts::grouped_mm_v2_bwd")
def _grouped_mm_v2_bwd(
    dout: jax.Array,
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    transpose_rhs: bool,
) -> tuple[jax.Array, jax.Array]:
    grad_lhs = call_named(
        tokamax_gmm_v2.gmm_v2,
        "moe_gmm_dgrad",
        dout,
        rhs,
        group_sizes,
        preferred_element_type=dout.dtype,
        transpose_rhs=not transpose_rhs,
    )
    # dWeight accumulates in FP32 and is stored in the weight's dtype, as in
    # expert_segment_matmul (autograd would round a wider store right away).
    if transpose_rhs:
        grad_rhs = weight_grad(dout, lhs, group_sizes, rhs.shape[0], rhs.dtype)
    else:
        grad_rhs = weight_grad(lhs, dout, group_sizes, rhs.shape[0], rhs.dtype)
    return grad_lhs, grad_rhs


def _setup_context(ctx, inputs, output):
    del output
    lhs, rhs, group_sizes, transpose_rhs = inputs
    ctx.save_for_backward(lhs, rhs, group_sizes)
    ctx.transpose_rhs = transpose_rhs


def _backward(ctx, grad_output):
    lhs, rhs, group_sizes = ctx.saved_tensors
    grad_lhs, grad_rhs = _grouped_mm_v2_bwd(
        grad_output.contiguous(), lhs, rhs, group_sizes, ctx.transpose_rhs
    )
    return grad_lhs, grad_rhs, None, None


grouped_mm_v2.register_autograd(_backward, setup_context=_setup_context)


__all__ = ["grouped_mm_v2", "weight_grad"]
