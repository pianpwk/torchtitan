"""Grouped GLU experts: one grouped matmul with a gated-activation epilogue.

For expert ``e``: ``h1 = X_e @ Wg_e``, ``h3 = X_e @ Wu_e`` (one MXU pass over
both weights, FP32 accumulation) and ``y = act(h1, h3)`` evaluated on the
FP32 accumulators, rounded once to the output dtype. The kernel also writes
``h1`` / ``h3`` (rounded to the output dtype) for the backward. This replaces
two ``gmm_v2`` calls and the activation pass that re-reads their outputs.

Activations (``GluActivation``):

* ``SwiGLU``: ``silu(gate) * up``.
* ``SiTUGLU(beta, linear_beta)``: ``beta * tanh(gate / beta) * sigmoid(gate)``
  times ``up`` (or ``linear_beta * tanh(up / linear_beta)``), Kimi K3's
  ``kimi_k3.moe._situ_glu``.

Built on Tokamax's ``gmm_v2`` (group-aligned M tiles, its tiling, metadata and
zero-fill helpers); the kernel body is a trimmed copy of ``gmm_v2``'s
unquantized path with a multi-output epilogue. The gate / up weights stay two
``[G, K, N]`` arrays (Tokamax's ``FusedWeightsRef`` lane-interleaves the two
tiles in VMEM), so no ``[G, K, 2N]`` copy is made.
TODO: upstream a ``return_preact`` option and separate gate / up weight inputs
to Tokamax ``gmm_v2`` (``tokamax._src.ops.gmm_v2``, a Tokamax internal) and
drop this copy.

Group sizes are runtime int32 values. Rows past ``sum(group_sizes)`` are not
read and their outputs are zero. Group sizes are sanitized on device (negative
sizes count as 0 and the running sum is capped at ``M``), the identity for a
valid table, so a bad routing table cannot make the kernel read or write out
of bounds.

Rounding vs the unfused path (``gmm_v2`` w1, ``gmm_v2`` w3, then ``act``):
- unfused: h1, h3 round to bf16 at the GEMM outputs; the activation runs on
    the bf16 values; y rounds to bf16.
- fused: the activation runs in FP32 on the FP32 accumulators; y rounds to
    bf16 once. The saved h1 / h3 are the same bf16 values the unfused GEMMs
    produce, so the backward (computed from them) matches the unfused backward.
"""

import dataclasses
import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.gmm_v2 import gmm_v2 as tk
import torch
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels.gmm_v2_segment_matmul import weight_grad

# VMEM staging buffer used to zero-fill rows past the groups.
_ZERO_BUFFER_BYTES = 2 * 1024 * 1024


def _check_positive(name, **values):
    for k, v in values.items():
        if not (isinstance(v, (int, float)) and 0 < v < float("inf")):
            raise ValueError(
                f"{name}.{k} must be finite and positive, got {v!r}"
            )


@dataclasses.dataclass(frozen=True)
class SwiGLU:
    """``silu(gate) * up``."""

    def __call__(self, gate, up):
        return jax.nn.silu(gate) * up


@dataclasses.dataclass(frozen=True)
class SiTUGLU:
    """``beta * tanh(gate / beta) * sigmoid(gate) * up'`` (Kimi K3), where
    ``up' = linear_beta * tanh(up / linear_beta)``, or ``up`` without it."""

    beta: float
    linear_beta: float | None = None

    def __post_init__(self):
        _check_positive("SiTUGLU", beta=self.beta)
        if self.linear_beta is not None:
            _check_positive("SiTUGLU", linear_beta=self.linear_beta)

    def __call__(self, gate, up):
        gate = self.beta * jnp.tanh(gate / self.beta) * jax.nn.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * jnp.tanh(up / self.linear_beta)
        return gate * up


GluActivation = SwiGLU | SiTUGLU


def _situ_bwd(gate, up, grad, beta: float, linear_beta: float | None):
    """FP32 d(gate), d(up) of SiTU-GLU in the order of torch autograd's chain
    rule for ``kimi_k3.moe._situ_glu`` (mul, tanh_backward, div,
    sigmoid_backward)."""
    tanh = jnp.tanh(gate / beta)
    scaled_tanh = beta * tanh
    sig = jax.nn.sigmoid(gate)
    up_tanh = None
    if linear_beta is not None:
        up_tanh = jnp.tanh(up / linear_beta)
        up = linear_beta * up_tanh
    d_act = grad * up
    d_up = grad * (scaled_tanh * sig)
    d_gate = ((d_act * sig) * beta * (1 - tanh * tanh)) / beta + (
        d_act * scaled_tanh
    ) * (sig * (1 - sig))
    if linear_beta is not None:
        d_up = (d_up * linear_beta * (1 - up_tanh * up_tanh)) / linear_beta
    return d_gate, d_up


def _validate(lhs, w_gate, w_up, group_sizes):
    if lhs.ndim != 2 or w_gate.ndim != 3:
        raise ValueError(
            "gmm_glu expects lhs [M, K] and weights [G, K, N], got"
            f" {lhs.shape} and {w_gate.shape}"
        )
    if w_up.shape != w_gate.shape:
        raise ValueError(
            f"w_gate {w_gate.shape} and w_up {w_up.shape} must have the same"
            " shape"
        )
    num_groups, size_k, size_n = w_gate.shape
    if lhs.shape[1] != size_k:
        raise ValueError(
            f"lhs has K={lhs.shape[1]} but the weights have K={size_k}"
        )
    if group_sizes.shape != (num_groups,):
        raise ValueError(
            f"group_sizes must have shape ({num_groups},), one size per expert,"
            f" got {group_sizes.shape}"
        )
    if lhs.dtype != w_gate.dtype or w_up.dtype != w_gate.dtype:
        raise ValueError(
            "lhs, w_gate and w_up must share a dtype, got"
            f" {lhs.dtype}, {w_gate.dtype}, {w_up.dtype}"
        )
    if size_n % 128:
        raise ValueError(
            "gmm_glu needs the expert hidden size N to be a multiple of 128,"
            f" got {size_n}"
        )


def sanitize_group_sizes(group_sizes: jax.Array, size_m: int) -> jax.Array:
    """int32 group sizes: negatives at 0, the running sum capped at ``size_m``.

    The identity for a valid table (non-negative, summing to <= ``size_m``).
    """
    gs = jnp.clip(group_sizes.astype(jnp.int32), 0, size_m)
    ends = jnp.minimum(jnp.cumsum(gs), size_m)
    return jnp.diff(ends, prepend=jnp.zeros((1,), jnp.int32))


# ---------------------------------------------------------------------------
# Forward kernel.
# ---------------------------------------------------------------------------


def _inner_kernel(
    tiled_lhs_ref,  # LhsRef [tile_m // sublane, sublane, tile_k]
    tiled_rhs_ref,  # FusedWeightsRef, gate / up [tile_k, tile_n]
    tiled_out_refs,  # (y, h1, h3) each [tile_m // sublane, sublane, tile_n]
    partial_out_refs,  # 3 x [sublane, tile_n]
    acc_ref,  # [tile_m, 2 * tile_n] FP32
    metadata_ref,
    *,
    cfgs,
    act: GluActivation,
):
    gm_id = pl.program_id(1)
    sub = cfgs.dims.size_lhs_sublane
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % sub
    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset
    tile_n = cfgs.tiles.tile_n

    def _matmul(is_first_k_step: bool, is_last_k_step: bool, bucket_m: int):
        mxu_size = pltpu.get_tpu_info().mxu_column_size
        tiled_lhs = tiled_lhs_ref.get_value().reshape(-1, cfgs.tiles.tile_k)[
            :bucket_m
        ]
        tiled_rhs = (
            tiled_rhs_ref.get_weight()
        )  # [tile_k, 2 * tile_n] interleaved
        valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
        if is_last_k_step and valid_k != 0:
            mask_lhs = (
                lax.broadcasted_iota(jnp.int32, tiled_lhs.shape, 1) < valid_k
            )
            tiled_lhs = jnp.where(mask_lhs, tiled_lhs, 0)
            mask_rhs = (
                lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, 0) < valid_k
            )
            tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)
        cols = tiled_rhs.shape[1]
        acc = jnp.concatenate(
            [
                jnp.matmul(
                    tiled_lhs,
                    tiled_rhs[:, s : min(cols, s + mxu_size)],
                    preferred_element_type=jnp.float32,
                )
                for s in range(0, cols, mxu_size)
            ],
            axis=1,
        )
        if not is_first_k_step:
            acc += acc_ref[:bucket_m]
        if not is_last_k_step:
            acc_ref[:bucket_m] = acc
            return
        gate, up = tk.deinterleave_lane(acc)
        outs = (act(gate, up), gate, up)
        iota = lax.broadcasted_iota(jnp.int32, gate.shape, 0)
        mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
        last_row = m_end_local // sub
        for val, out_ref, pout_ref in zip(
            outs, tiled_out_refs, partial_out_refs, strict=False
        ):
            out_2d = out_ref.reshape(-1, tile_n)
            out_2d[:bucket_m] = jnp.where(mask, val, 0).astype(out_ref.dtype)
            zeros = jnp.zeros_like(pout_ref)
            # Rows of this group sharing the first sublane tile with the
            # previous group (gmm_v2's partial-output carry).
            out_ref[0] += jnp.where(gm_id == 0, zeros, pout_ref[...])
            pout_ref[...] = jnp.where(
                m_end_local % sub == 0, zeros, out_ref[last_row]
            )

    def run_matmul_step(bucket_idx: int):
        bucket_m = cfgs.tiles.bucket_base * (bucket_idx + 1)
        k_id, num_k = pl.program_id(2), pl.num_programs(2)
        first, last = k_id == 0, k_id == num_k - 1
        lax.cond(
            first,
            lambda: lax.cond(
                last,
                lambda: _matmul(True, True, bucket_m),
                lambda: _matmul(True, False, bucket_m),
            ),
            lambda: lax.cond(
                last,
                lambda: _matmul(False, True, bucket_m),
                lambda: _matmul(False, False, bucket_m),
            ),
        )

    branches = [
        functools.partial(run_matmul_step, bucket_idx=i)
        for i in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    lax.switch(m_end_local // cfgs.tiles.bucket_base, branches)


def _kernel_main(
    group_sizes_ref,
    group_offset_ref,
    lhs_ref,
    w_gate_ref,
    w_up_ref,
    *refs,
    cfgs,
    act: GluActivation,
    num_outs: int,
):
    out_refs = refs[:num_outs]
    pouts = refs[num_outs : 2 * num_outs]
    acc_ref, metadata_ref, zero_ref, sem_ref = refs[2 * num_outs :]
    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)
    num_gm = tk.fill_metadata(
        group_sizes_ref, group_offset_ref, metadata_ref, cfgs=cfgs
    )
    zero_sizes = [
        tk.zero_out_start(
            out_ref,
            zero_ref,
            sem_ref.at[i : i + 1],
            metadata_ref,
            num_gm,
            dims=cfgs.dims,
        )
        for i, out_ref in enumerate(out_refs)
    ]
    (lhs_spec, rhs_spec), out_spec = tk.generate_block_specs(metadata_ref, cfgs)
    rhs_ref = tk.FusedWeightsRef(
        gate=tk.WeightsRef(weight=w_gate_ref, scale=None, bias=None),
        up=tk.WeightsRef(weight=w_up_ref, scale=None, bias=None),
    )
    rhs_spec = tk.FusedWeightsRef(gate=rhs_spec, up=rhs_spec)
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(_inner_kernel_wrapper, cfgs=cfgs, act=act),
        grid=(num_n, num_gm, num_k),
        in_specs=(lhs_spec, rhs_spec),
        out_specs=tuple([out_spec] * num_outs),
        core_axis_name="core",
        dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY, pltpu.ARBITRARY),
    )
    sub = cfgs.dims.size_lhs_sublane
    lhs_in = tk.LhsRef(
        value=lhs_ref.reshape(-1, sub, lhs_ref.shape[-1]), scale=None
    )
    outs_in = tuple(o.reshape(-1, sub, o.shape[-1]) for o in out_refs)
    pipeline_fn(
        lhs_in,
        rhs_ref,
        *outs_in,
        scratches=[tuple(pouts), acc_ref, metadata_ref],
    )
    for i, (out_ref, zero_size) in enumerate(zip(out_refs, zero_sizes)):
        tk.zero_out_end(
            out_ref, sem_ref.at[i : i + 1], zero_size, dims=cfgs.dims
        )


def _inner_kernel_wrapper(lhs, rhs, *refs, **kw):
    # emit_pipeline passes the output tiles positionally, then the scratches.
    *outs, pouts, acc_ref, metadata_ref = refs
    _inner_kernel(lhs, rhs, tuple(outs), pouts, acc_ref, metadata_ref, **kw)


def gmm_glu(
    lhs: jax.Array,  # [M, K]
    w_gate: jax.Array,  # [G, K, N]
    w_up: jax.Array,  # [G, K, N]
    group_sizes: jax.Array,  # int [G]
    *,
    act: GluActivation,
):
    """``act(lhs @ w_gate[g], lhs @ w_up[g])`` per group, in lhs's dtype.

    Rows past the groups are zero. Returns ``(y, h1, h3)``, each ``[M, N]``.
    """
    _validate(lhs, w_gate, w_up, group_sizes)
    size_m = lhs.shape[0]
    num_groups, size_k, size_n = w_gate.shape
    out_dtype = lhs.dtype
    num_outs = 3
    if size_m == 0:
        outs = tuple(jnp.zeros((0, size_n), out_dtype) for _ in range(num_outs))
        return outs
    group_sizes = sanitize_group_sizes(group_sizes, size_m)
    # pylint: disable-next=protected-access
    sub = tk._get_lhs_sublane_size(lhs.dtype, size_m)
    padded_m = tk.align_to(size_m, sub)
    if padded_m != size_m:
        lhs = jnp.pad(lhs, ((0, padded_m - size_m), (0, 0)))
    group_offset = jnp.zeros((1,), jnp.int32)
    rhs_shape = jax.ShapeDtypeStruct(
        (num_groups, size_k, 2 * size_n), w_gate.dtype
    )
    vmem_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    num_lanes = pltpu.get_tpu_info().num_lanes
    out_bytes = jnp.dtype(out_dtype).itemsize
    tile_zero_m = min(_ZERO_BUFFER_BYTES // num_lanes // out_bytes, padded_m)
    # calculate_tiling budgets gmm_v2's buffers only. Reserve this kernel's
    # extra scratch (zero-fill buffer, per-output partial rows at the widest
    # possible tile_n) so the chosen tiles fit, e.g. K=4096, N=2048.
    extra_vmem = (
        tile_zero_m * num_lanes * out_bytes
        + num_outs * sub * tk.align_to(size_n, num_lanes) * out_bytes
    )
    cfgs = tk.make_gmm_configs(
        lhs,
        rhs_shape,
        None,
        None,
        group_sizes,
        group_offset,
        tile_info=tk.calculate_tiling,
        vmem_limit_bytes=vmem_limit - extra_vmem,
        out_dtype=out_dtype,
        acc_dtype=jnp.float32,
        maybe_quantize_lhs=False,
        zero_initialize=True,
        # Any non-None fuse_act makes the tiling budget for the gate / up pair.
        fuse_act="silu",
    )
    dims, tiles = cfgs.dims, cfgs.tiles
    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
    scratch = [
        *[pltpu.VMEM((sub, tiles.tile_n), cfgs.out_dtype)] * num_outs,
        pltpu.VMEM((tiles.tile_m, 2 * tiles.tile_n), jnp.float32),
        tk.MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
        pltpu.VMEM((tile_zero_m, num_lanes), cfgs.out_dtype),
        pltpu.SemaphoreType.DMA((num_outs,)),
    ]
    aligned_n = tk.align_to(cfgs.out_size_n, num_lanes)
    out_type = tuple(
        jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs.out_dtype)
        for _ in range(num_outs)
    )
    group_sizes = pltpu.with_memory_space_constraint(group_sizes, pltpu.SMEM)
    group_offset = pltpu.with_memory_space_constraint(group_offset, pltpu.SMEM)
    outs = pl.kernel(
        functools.partial(_kernel_main, cfgs=cfgs, act=act, num_outs=num_outs),
        out_type=out_type,
        mesh=pltpu.TensorCoreMesh(axis_name="core"),
        scratch_types=scratch,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit, disable_bounds_checks=True
        ),
        name=(
            f"gmm_glu-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}"
            f"-n_{cfgs.out_size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}"
            f"-tn_{tiles.tile_n}"
        ),
    )(group_sizes, group_offset, lhs, w_gate, w_up)
    outs = tuple(o[:size_m, :size_n] for o in outs)
    return outs


# ---------------------------------------------------------------------------
# Backward.
# ---------------------------------------------------------------------------


def gmm_glu_bwd(
    grad: jax.Array,  # [M, N] dL/dy
    lhs: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    gate: jax.Array,  # saved h1 [M, N]
    up: jax.Array,  # saved h3 [M, N]
    group_sizes: jax.Array,
    *,
    act: GluActivation,
):
    """``(d_lhs, d_w_gate, d_w_up)`` from the forward's saved ``gate`` / ``up``.

    The activation backward runs in FP32 on the saved bf16 pre-activations and
    rounds ``d_gate`` / ``d_up`` to their dtype; dInput is two ``gmm_v2`` calls
    (in-kernel rhs transpose) summed in the lhs dtype, and dWeight is
    ``weight_grad`` (``tgmm_v2``), as in ``gmm_v2_segment_matmul``.
    """
    group_sizes = sanitize_group_sizes(group_sizes, lhs.shape[0])
    gate32, up32, grad32 = (t.astype(jnp.float32) for t in (gate, up, grad))
    if isinstance(act, SiTUGLU):
        d_gate, d_up = _situ_bwd(
            gate32, up32, grad32, act.beta, act.linear_beta
        )
    else:
        _, vjp = jax.vjp(act, gate32, up32)
        d_gate, d_up = vjp(grad32)
    d_gate, d_up = d_gate.astype(gate.dtype), d_up.astype(up.dtype)
    d_lhs = tk.gmm_v2(
        d_gate,
        w_gate,
        group_sizes,
        preferred_element_type=lhs.dtype,
        transpose_rhs=True,
    ) + tk.gmm_v2(
        d_up,
        w_up,
        group_sizes,
        preferred_element_type=lhs.dtype,
        transpose_rhs=True,
    )
    d_w_gate = weight_grad(
        lhs, d_gate, group_sizes, w_gate.shape[0], w_gate.dtype
    )
    d_w_up = weight_grad(lhs, d_up, group_sizes, w_up.shape[0], w_up.dtype)
    return d_lhs, d_w_gate, d_w_up


# ---------------------------------------------------------------------------
# Torch op. The activation travels as (kind, beta, linear_beta) scalars so the
# jax_op compile cache keys on it.
# ---------------------------------------------------------------------------


def _encode_act(act: GluActivation) -> tuple[str, float, float]:
    if isinstance(act, SiTUGLU):
        # 0 stands for "no linear_beta" (it must be positive otherwise).
        return "situ", float(act.beta), float(act.linear_beta or 0.0)
    if isinstance(act, SwiGLU):
        return "swiglu", 0.0, 0.0
    raise ValueError(f"act must be a SwiGLU or SiTUGLU, got {act!r}")


def _decode_act(kind: str, beta: float, linear_beta: float) -> GluActivation:
    return SiTUGLU(beta, linear_beta or None) if kind == "situ" else SwiGLU()


@pallas.jax_op("tt_experts::grouped_glu")
def _glu_fwd_op(
    lhs: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    group_sizes: jax.Array,
    kind: str,
    beta: float,
    linear_beta: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return gmm_glu(
        lhs, w_gate, w_up, group_sizes, act=_decode_act(kind, beta, linear_beta)
    )


@pallas.jax_op("tt_experts::grouped_glu_bwd")
def _glu_bwd_op(
    grad: jax.Array,
    lhs: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    gate: jax.Array,
    up: jax.Array,
    group_sizes: jax.Array,
    kind: str,
    beta: float,
    linear_beta: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return gmm_glu_bwd(
        grad,
        lhs,
        w_gate,
        w_up,
        gate,
        up,
        group_sizes,
        act=_decode_act(kind, beta, linear_beta),
    )


def _setup_context(ctx, inputs, output):
    lhs, w_gate, w_up, group_sizes, *act_args = inputs
    _, gate, up = output
    ctx.save_for_backward(lhs, w_gate, w_up, gate, up, group_sizes)
    ctx.act_args = act_args
    ctx.mark_non_differentiable(gate, up)


def _backward(ctx, grad_y, grad_gate, grad_up):
    del grad_gate, grad_up  # Saved for the backward only.
    lhs, w_gate, w_up, gate, up, group_sizes = ctx.saved_tensors
    d_lhs, d_w_gate, d_w_up = _glu_bwd_op(
        grad_y.contiguous(),
        lhs,
        w_gate,
        w_up,
        gate,
        up,
        group_sizes,
        *ctx.act_args,
    )
    return d_lhs, d_w_gate, d_w_up, None, None, None, None


_glu_fwd_op.register_autograd(_backward, setup_context=_setup_context)


def grouped_glu(
    lhs_RK: torch.Tensor,
    w_gate_EKN: torch.Tensor,
    w_up_EKN: torch.Tensor,
    group_sizes_E: torch.Tensor,
    *,
    act: GluActivation,
) -> torch.Tensor:
    """``act(lhs @ w_gate, lhs @ w_up)`` per expert as one grouped matmul.

    Rows of ``lhs`` are grouped by expert in order: expert ``e`` owns the next
    ``group_sizes_E[e]`` (int32) rows; rows past the groups produce zeros.
    """
    y_RN, _, _ = _glu_fwd_op(
        lhs_RK, w_gate_EKN, w_up_EKN, group_sizes_E, *_encode_act(act)
    )
    return y_RN


__all__ = [
    "GluActivation",
    "SiTUGLU",
    "SwiGLU",
    "gmm_glu",
    "gmm_glu_bwd",
    "grouped_glu",
    "sanitize_group_sizes",
]
