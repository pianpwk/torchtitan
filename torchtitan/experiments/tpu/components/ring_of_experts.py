# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static-shape Ring-of-Experts expert-parallel (EP) transport for TPU.

Each EP rank all-gathers the *un-expanded* local tokens, routing scores, and
expert ids of its EP group, selects the routes owned by its local experts, and
gathers those rows into an expert-major buffer of static size ``max_R``. After
the expert computation, each rank weights and sums its partial outputs per
token and a reduce-scatter returns every token's sum to its owner. This
replaces the two ragged all-to-alls of the standard dispatcher with an
all-gather and a reduce-scatter, which are cheap on the TPU ICI and need no
dynamic shapes. This follows MaxText's ``ring_of_experts`` MoE strategy.

Routes beyond a rank's capacity ``max_R`` are dropped, in local-expert order
and within an expert in global route order. Rows past the valid count are zero
and belong to no expert: the group sizes sum to the valid count, so the
grouped matmuls skip those row tiles instead of computing the capacity
padding, and may leave their outputs unspecified. The combine and its backward
read only the valid rows and mask with selects, so those rows never reach a
result.

The forward and backward of each step are custom autograd functions so that
every token-sized gather and reduction runs in the TensorCore kernels of
``kernels/ragged_gather.py`` and ``kernels/ragged_gather_reduce.py``. Nothing
here depends on the dispatcher class.
"""

from dataclasses import dataclass
import math

import torch
from torch.autograd.function import once_differentiable
import torch.distributed._functional_collectives as funcol

from torchtitan.experiments.tpu.kernels.ragged_gather import (
    inverse_permute_scatter,
    ragged_gather,
    to_token_3d,
)
from torchtitan.experiments.tpu.kernels.ragged_gather_reduce import (
    ragged_gather_reduce,
    ragged_gather_reduce_metadata,
)
from torchtitan.experiments.tpu.kernels.route_score import route_score_grad
from torchtitan.experiments.tpu.components.remat_regions import (
    annotate_region,
    MOE_ROUTING,
)

# The per-rank row capacity is rounded up to a multiple of the grouped-matmul
# row tile.
CAPACITY_TILE = 256


def ring_of_experts_capacity(
    num_routed_tokens: int, ep_size: int, capacity_factor: float | None
) -> int:
  """Static number of expert rows `max_R` per EP rank.

  Args:
    num_routed_tokens: local routes, `T * top_k`.
    ep_size: EP degree.
    capacity_factor: rows per rank as a multiple of `num_routed_tokens`, or
      `None` for drop-free (`ep_size * num_routed_tokens`).

  Returns:
    `max_R`, a multiple of `CAPACITY_TILE`.
  """
  max_rows = ep_size * num_routed_tokens
  if capacity_factor is not None:
    max_rows = min(max_rows, math.ceil(capacity_factor * num_routed_tokens))
  return -(-max(max_rows, 1) // CAPACITY_TILE) * CAPACITY_TILE


@dataclass(frozen=True, slots=True)
class RingOfExpertsMetadata:
  """Dispatch state reused by `ring_of_experts_combine`."""

  token_ids_R: torch.Tensor
  valid_count_1: torch.Tensor
  scores_R: torch.Tensor
  reduce_metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
  num_out_tokens: int
  top_k: int
  ep_mesh: object


def _exclusive_cumsum_int32(x: torch.Tensor) -> torch.Tensor:
  """Exclusive int32 cumsum of a 1-D tensor without `torch.cumsum`.

  `torch.cumsum` lowers to an XLA `kScan`, which XLA mishandles when the
  enclosing while body (the scanned layer stack) is cloned. A log-depth
  Hillis-Steele scan lowers to slices, adds, and concatenates only.
  """
  x = x.to(torch.int32)
  out = x
  step = 1
  while step < out.shape[0]:
    out = torch.cat([out[:step], out[step:] + out[:-step]])
    step *= 2
  return out - x


def _packed_stable_argsort(keys: torch.Tensor, max_key: int) -> torch.Tensor:
  """Stable int32 argsort of non-negative `keys <= max_key`.

  Packs `(key, position)` into one int32 so an unstable single-operand sort is
  stable, which is much cheaper on TPU than a stable key-value sort.
  """
  n = int(keys.shape[0])
  if n <= 1:
    return torch.zeros_like(keys, dtype=torch.int32)
  pos_bits = (n - 1).bit_length()
  if pos_bits + max(1, int(max_key).bit_length()) > 31:
    return torch.argsort(keys, stable=True).to(torch.int32)
  positions = torch.arange(n, device=keys.device, dtype=torch.int32)
  packed = (keys.to(torch.int32) << pos_bits) | positions
  return torch.sort(packed).values & ((1 << pos_bits) - 1)


def _fp8_static_quantize(x: torch.Tensor) -> torch.Tensor:
  """``x`` in e4m3 with the FP8 expert GEMMs' static scale."""
  from torchtitan.experiments.tpu.kernels.fp8_expert_segment_matmul import STATIC_SCALE  # pylint: disable=g-import-not-at-top

  fmax = torch.finfo(torch.float8_e4m3fn).max
  return (
      (x.float() * (1.0 / STATIC_SCALE))
      .clamp(-fmax, fmax)
      .to(torch.float8_e4m3fn)
  )


def _fp8_static_dequantize(x_q: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
  from torchtitan.experiments.tpu.kernels.fp8_expert_segment_matmul import STATIC_SCALE  # pylint: disable=g-import-not-at-top

  return (x_q.float() * STATIC_SCALE).to(dtype)


class _GatherTokens(torch.autograd.Function):
  """EP all-gather of local tokens, then gather this rank's expert rows.

  With ``fp8``, the tokens travel as e4m3 with the FP8 expert GEMMs' static
  scale (half the bytes) and are dequantized once gathered. The scale is a
  power of two, so the GEMM's own quantization of the rows recovers the same
  e4m3 values exactly. (Viewing the FP8 bytes as BF16 pairs to keep the BF16
  row-copy kernel lowers to slow relayouts on TPU, so the dequantization runs
  on the gathered tokens instead.)

  Backward is the transpose: an FP32 per-token sum of the row gradients, then
  a reduce-scatter back to the token owners.
  """

  @staticmethod
  def forward(ctx, x_TD, token_ids_R, valid_count_1, reduce_metadata, top_k,
              ep_mesh, fp8):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(valid_count_1, *reduce_metadata)
    ctx.top_k = top_k
    ctx.ep_mesh = ep_mesh
    ctx.local_shape = tuple(x_TD.shape)
    # Relayout the local shard, not the ep_size-times larger gathered tensor.
    if fp8:
      x_global_T3 = _fp8_static_dequantize(
          funcol.all_gather_tensor(
              to_token_3d(_fp8_static_quantize(x_TD)), gather_dim=0, group=ep_mesh
          ),
          x_TD.dtype,
      )
    else:
      x_global_T3 = funcol.all_gather_tensor(
          to_token_3d(x_TD), gather_dim=0, group=ep_mesh
      )
    ctx.num_out_tokens = int(x_global_T3.shape[0])
    return ragged_gather(x_global_T3, token_ids_R, valid_count_1)

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_RD):
    if grad_RD is None or not ctx.needs_input_grad[0]:
      return (None,) * 7
    valid_count_1, *reduce_metadata = ctx.saved_tensors
    grad_global_T3 = ragged_gather_reduce(
        grad_RD,
        None,
        valid_count_1,
        tuple(reduce_metadata),
        num_out_tokens=ctx.num_out_tokens,
        top_k=ctx.top_k,
    )
    grad_T3 = funcol.reduce_scatter_tensor(
        grad_global_T3, "sum", scatter_dim=0, group=ctx.ep_mesh
    )
    return (grad_T3.reshape(ctx.local_shape),) + (None,) * 6


class _GatherRoutes(torch.autograd.Function):
  """One EP all-gather of every rank's expert ids and routing scores.

  The ids and the FP32 score bits travel in a single int32 buffer, so each
  dispatch issues one small collective instead of separate ones for counts,
  ids, and scores (small collectives queue behind the large token ones).
  """

  @staticmethod
  def forward(ctx, scores_TK, expert_ids_TK, ep_mesh):
    ctx.set_materialize_grads(False)
    ctx.score_shape = tuple(scores_TK.shape)
    ctx.score_dtype = scores_TK.dtype
    ctx.ep_mesh = ep_mesh
    ep_size = ep_mesh.size()
    packed = torch.cat([
        expert_ids_TK.to(torch.int32).reshape(-1),
        scores_TK.to(torch.float32).reshape(-1).view(torch.int32),
    ])
    gathered = funcol.all_gather_tensor(
        packed, gather_dim=0, group=ep_mesh
    ).view(ep_size, 2, -1)
    expert_ids_N = gathered[:, 0].reshape(-1)
    scores_N = gathered[:, 1].contiguous().view(-1).view(torch.float32)
    ctx.mark_non_differentiable(expert_ids_N)
    return expert_ids_N, scores_N

  @staticmethod
  @once_differentiable
  def backward(ctx, _grad_ids_N, grad_scores_N):
    if grad_scores_N is None or not ctx.needs_input_grad[0]:
      return None, None, None
    grad_TK = funcol.reduce_scatter_tensor(
        grad_scores_N, "sum", scatter_dim=0, group=ctx.ep_mesh
    )
    return grad_TK.reshape(ctx.score_shape).to(ctx.score_dtype), None, None


class _SelectScores(torch.autograd.Function):
  """Select this rank's routes from the gathered scores, zero past `valid`."""

  @staticmethod
  def forward(ctx, scores_N, slots_R, perm_N, valid_count_1):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(perm_N, valid_count_1)
    selected_R = torch.index_select(scores_N, 0, slots_R)
    valid_R = (
        torch.arange(slots_R.shape[0], device=slots_R.device)
        < valid_count_1.reshape(1)
    )
    return torch.where(valid_R, selected_R, torch.zeros_like(selected_R))

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_R):
    if grad_R is None or not ctx.needs_input_grad[0]:
      return None, None, None, None
    perm_N, valid_count_1 = ctx.saved_tensors
    # `slots_R` is a prefix of the permutation `perm_N`, so the transpose is a
    # sort keyed on `perm_N` instead of a scalar scatter over all routes.
    return (
        inverse_permute_scatter(perm_N, grad_R, valid_count_1),
        None,
        None,
        None,
    )


class _CombineTokens(torch.autograd.Function):
  """Weight rows by score, sum per token in FP32, reduce-scatter over EP."""

  @staticmethod
  def forward(ctx, routed_RD, scores_R, token_ids_R, valid_count_1,
              reduce_metadata, num_out_tokens, top_k, ep_mesh,
              fuse_route_score_backward, pre_scaled):
    ctx.set_materialize_grads(False)
    ctx.pre_scaled = pre_scaled
    if pre_scaled:
      # Rows already carry their scores: sum them as they are. Nothing of the
      # expert output is needed for backward, so it is not saved (and a
      # rematerializing backward does not recompute the last expert matmul).
      scores_R = torch.ones_like(scores_R)
      ctx.save_for_backward(token_ids_R, valid_count_1)
    else:
      ctx.save_for_backward(routed_RD, scores_R, token_ids_R, valid_count_1)
    ctx.ep_mesh = ep_mesh
    ctx.fuse_route_score_backward = fuse_route_score_backward
    # Same values as `(routed.float() * scores).to(routed.dtype)` followed by
    # the per-token sum; the reduce-scatter runs in the kernels' token layout
    # so only the local shard is relayouted back to [T, D].
    partial_T3 = ragged_gather_reduce(
        routed_RD,
        scores_R,
        valid_count_1,
        reduce_metadata,
        num_out_tokens=num_out_tokens,
        top_k=top_k,
    )
    out_T3 = funcol.reduce_scatter_tensor(
        partial_T3, "sum", scatter_dim=0, group=ep_mesh
    )
    return out_T3.reshape(out_T3.shape[0], -1)

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_TD):
    if grad_TD is None:
      return (None,) * 10
    if ctx.pre_scaled:
      token_ids_R, valid_count_1 = ctx.saved_tensors
    else:
      routed_RD, scores_R, token_ids_R, valid_count_1 = ctx.saved_tensors
    grad_global_T3 = funcol.all_gather_tensor(
        to_token_3d(grad_TD), gather_dim=0, group=ctx.ep_mesh
    )
    grad_weighted_RD = ragged_gather(grad_global_T3, token_ids_R, valid_count_1)
    if ctx.pre_scaled:
      return (grad_weighted_RD,) + (None,) * 9
    grad_routed_RD = None
    if ctx.needs_input_grad[0]:
      grad_routed_RD = (
          grad_weighted_RD.float() * scores_R.reshape(-1, 1)
      ).to(routed_RD.dtype)
    grad_scores_R = None
    if ctx.needs_input_grad[1]:
      if (
          ctx.fuse_route_score_backward
          and routed_RD.dtype == torch.bfloat16
          and scores_R.dtype == torch.float32
          and routed_RD.device.type in ("cpu", "tpu")
      ):
        grad_scores_R = route_score_grad(grad_weighted_RD, routed_RD)
      else:
        grad_scores_R = (
            (grad_weighted_RD.float() * routed_RD.float())
            .sum(dim=1)
            .to(scores_R.dtype)
        )
    return (grad_routed_RD, grad_scores_R) + (None,) * 8


def ring_of_experts_dispatch(
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    *,
    num_experts: int,
    ep_mesh,
    capacity_factor: float | None,
    fp8_tokens: bool = False,
    gather_tokens: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor, RingOfExpertsMetadata]:
  """Gather this EP rank's expert rows from all ranks of `ep_mesh`.

  Args:
    x_TD: local tokens; `D` must be a multiple of 128.
    topk_scores_TK: local routing scores.
    topk_expert_ids_TK: local global expert ids.
    num_experts: number of global experts, divisible by the EP degree.
    ep_mesh: 1-D EP device mesh.
    capacity_factor: see `ring_of_experts_capacity`.
    fp8_tokens: all-gather the tokens in e4m3 with the FP8 expert GEMMs'
      static scale (``TpuGroupedExperts.Config.use_fp8_grouped_mm``). The
      returned rows are those values dequantized, which those GEMMs quantize
      back exactly; other consumers see the FP8 rounding.
    gather_tokens: gather the routed rows. Without, only the routing is
      computed (``routed_RD`` is None), for `ring_of_experts_fp8_experts`.

  Returns:
    `(routed_RD, group_sizes_e, metadata)`: expert-major rows for the local
    experts (rows past the valid count are zero), the local-expert group sizes
    (summing to the valid count, at most `R`), and the state for
    `ring_of_experts_combine`. The expert computation may leave its output
    rows past the valid count unspecified.
  """
  ep_size = ep_mesh.size()
  if num_experts % ep_size:
    raise ValueError(
        f"num_experts ({num_experts}) must be divisible by the EP degree"
        f" ({ep_size})"
    )
  num_local_experts = num_experts // ep_size
  experts_start = ep_mesh.get_local_rank() * num_local_experts
  local_T, top_k = topk_expert_ids_TK.shape
  num_out_tokens = ep_size * local_T
  num_global_routes = num_out_tokens * top_k
  max_R = ring_of_experts_capacity(local_T * top_k, ep_size, capacity_factor)

  with annotate_region(MOE_ROUTING):
    expert_ids_N, scores_N = _GatherRoutes.apply(
        topk_scores_TK, topk_expert_ids_TK, ep_mesh
    )
    local_e_N = expert_ids_N - experts_start
    is_local_N = (local_e_N >= 0) & (local_e_N < num_local_experts)
    sort_key_N = torch.where(
        is_local_N, local_e_N, torch.full_like(local_e_N, num_local_experts)
    )
    # Per-local-expert route counts, from the gathered ids rather than another
    # collective.
    counts_e = (
        sort_key_N.unsqueeze(1)
        == torch.arange(
            num_local_experts, device=sort_key_N.device, dtype=torch.int32
        )
    ).sum(dim=0, dtype=torch.int32)
    # Truncate to `max_R` rows in local-expert order.
    starts_e = _exclusive_cumsum_int32(counts_e)
    valid_e = torch.minimum(counts_e, (max_R - starts_e).clamp(min=0))
    valid_count_1 = valid_e.sum(dtype=torch.int32).reshape(1)
    # Rows past the valid count belong to no expert, so the grouped matmuls
    # skip them rather than compute the capacity padding.
    group_sizes_e = valid_e

    perm_N = _packed_stable_argsort(sort_key_N, num_local_experts)
    if max_R <= num_global_routes:
      slots_R = perm_N[:max_R]
    else:
      slots_R = torch.nn.functional.pad(perm_N, (0, max_R - num_global_routes))
    token_ids_R = slots_R // top_k
    reduce_metadata = ragged_gather_reduce_metadata(
        token_ids_R,
        valid_count_1,
        top_k=top_k,
        num_out_tokens=num_out_tokens,
        row_bytes=x_TD.shape[-1] * x_TD.element_size(),
    )

  routed_RD = None
  if gather_tokens:
    routed_RD = _GatherTokens.apply(
        x_TD, token_ids_R, valid_count_1, reduce_metadata, top_k, ep_mesh,
        fp8_tokens,
    )
  scores_R = _SelectScores.apply(scores_N, slots_R, perm_N, valid_count_1)
  metadata = RingOfExpertsMetadata(
      token_ids_R=token_ids_R,
      valid_count_1=valid_count_1,
      scores_R=scores_R,
      reduce_metadata=reduce_metadata,
      num_out_tokens=num_out_tokens,
      top_k=top_k,
      ep_mesh=ep_mesh,
  )
  return routed_RD, group_sizes_e, metadata


def ring_of_experts_combine(
    routed_output_RD: torch.Tensor,
    metadata: RingOfExpertsMetadata,
    *,
    fuse_route_score_backward: bool = False,
    pre_scaled: bool = False,
) -> torch.Tensor:
  """Score-weight, sum per token, and return each token to its EP owner.

  Args:
    routed_output_RD: expert outputs in the order of `ring_of_experts_dispatch`.
    metadata: state returned by `ring_of_experts_dispatch`.
    fuse_route_score_backward: compute the score gradient with the fused
      `route_score_grad` kernel (BF16 rows and FP32 scores only).
    pre_scaled: the rows were already multiplied by `metadata.scores_R`
      (e.g. before the down projection), so they are summed unweighted and
      the score gradient comes from where the scaling happened.

  Returns:
    `[T, D]` combined local tokens.
  """
  return _CombineTokens.apply(
      routed_output_RD,
      metadata.scores_R,
      metadata.token_ids_R,
      metadata.valid_count_1,
      metadata.reduce_metadata,
      metadata.num_out_tokens,
      metadata.top_k,
      metadata.ep_mesh,
      fuse_route_score_backward,
      pre_scaled,
  )


def _fp8_rowwise_all_gather(x_TD: torch.Tensor, ep_mesh) -> torch.Tensor:
  """All-gather ``x_TD`` over ``ep_mesh`` as e4m3 with one absmax scale per
  row (half the bytes of BF16), dequantized to BF16 ``[N, D // 128, 128]``."""
  fmax = torch.finfo(torch.float8_e4m3fn).max
  x_f32 = x_TD.float()
  scale_T = torch.clamp(
      x_f32.abs().amax(dim=-1) / fmax, min=torch.finfo(torch.float32).tiny
  )
  x_q_T3 = to_token_3d(
      (x_f32 / scale_T.unsqueeze(-1)).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
  )
  x_q_N3 = funcol.all_gather_tensor(x_q_T3, gather_dim=0, group=ep_mesh)
  scale_N = funcol.all_gather_tensor(scale_T, gather_dim=0, group=ep_mesh)
  return (x_q_N3.float() * scale_N.reshape(-1, 1, 1)).to(torch.bfloat16)


class _Fp8Experts(torch.autograd.Function):
  """EP token all-gather, the fused FP8 expert MLP of
  ``kernels/fp8_roe_experts.py`` and the combine reduce-scatter.

  The tokens travel and are gathered by the kernels as e4m3 (static scale),
  so no routed-row buffer is written in the forward. Backward all-gathers the
  output gradient (BF16), runs the fused backward kernels, sums each token's
  row gradients and reduce-scatters them; the expert-weight gradients go to
  the `Fp8ExpertWeights`' ``reduce_joint_grad``.
  """

  @staticmethod
  def forward(ctx, x_TD, scores_R, token_ids_R, valid_count_1, group_sizes_E,
              reduce_metadata, top_k, ep_mesh, fp8_grad_all_gather, w13, w2,
              *weight_shards):
    from torchtitan.experiments.tpu.kernels.fp8_roe_experts import fp8_experts_fwd_op  # pylint: disable=g-import-not-at-top

    ctx.set_materialize_grads(False)
    ctx.top_k = top_k
    ctx.ep_mesh = ep_mesh
    ctx.fp8_grad_all_gather = fp8_grad_all_gather
    ctx.w13, ctx.w2 = w13, w2
    ctx.num_w13 = len(w13.shards)
    x_ag_N3 = funcol.all_gather_tensor(
        to_token_3d(_fp8_static_quantize(x_TD)), gather_dim=0, group=ep_mesh
    )
    y_N3, gate_out_RF2 = fp8_experts_fwd_op(
        x_ag_N3, token_ids_R, group_sizes_E, scores_R, w13.weight_q,
        w2.weight_q,
    )
    ctx.save_for_backward(
        x_ag_N3, scores_R, token_ids_R, valid_count_1, group_sizes_E,
        gate_out_RF2, w13.weight_q, w2.weight_q, *reduce_metadata,
    )
    # In the token layout: the same reduce-scatter on the flattened [N, D]
    # rows crashed a SparseCore (UserFatal) on the 671B run.
    out_T3 = funcol.reduce_scatter_tensor(
        y_N3, "sum", scatter_dim=0, group=ep_mesh
    )
    return out_T3.reshape(out_T3.shape[0], -1).to(x_TD.dtype)

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_TD):
    from torchtitan.experiments.tpu.kernels.fp8_roe_experts import fp8_experts_bwd_op  # pylint: disable=g-import-not-at-top

    (x_ag_N3, scores_R, token_ids_R, valid_count_1, group_sizes_E,
     gate_out_RF2, w13_q, w2_q, *reduce_metadata) = ctx.saved_tensors
    num_weights = ctx.num_w13 + len(ctx.w2.shards)
    if grad_TD is None:
      return (None,) * (11 + num_weights)
    if ctx.fp8_grad_all_gather:
      dy_N3 = _fp8_rowwise_all_gather(grad_TD, ctx.ep_mesh)
    else:
      dy_N3 = funcol.all_gather_tensor(
          to_token_3d(grad_TD.to(torch.bfloat16).contiguous()),
          gather_dim=0,
          group=ctx.ep_mesh,
      )
    dx_R3, dscores_R, dw13, dw2 = fp8_experts_bwd_op(
        dy_N3, x_ag_N3, token_ids_R, group_sizes_E, scores_R, w13_q, w2_q,
        gate_out_RF2,
    )
    grad_global_T3 = ragged_gather_reduce(
        dx_R3.reshape(dx_R3.shape[0], -1),
        None,
        valid_count_1,
        tuple(reduce_metadata),
        num_out_tokens=x_ag_N3.shape[0],
        top_k=ctx.top_k,
    )
    grad_T3 = funcol.reduce_scatter_tensor(
        grad_global_T3, "sum", scatter_dim=0, group=ctx.ep_mesh
    )
    valid_R = (
        torch.arange(dscores_R.shape[0], device=dscores_R.device)
        < valid_count_1.reshape(1)
    )
    dscores_R = torch.where(valid_R, dscores_R, torch.zeros_like(dscores_R))
    return (
        grad_T3.reshape(grad_TD.shape).to(grad_TD.dtype),
        dscores_R.to(scores_R.dtype),
        *(None,) * 9,
        *ctx.w13.reduce_joint_grad(dw13),
        *ctx.w2.reduce_joint_grad(dw2),
    )


def ring_of_experts_fp8_experts(
    x_TD: torch.Tensor,
    group_sizes_E: torch.Tensor,
    metadata: RingOfExpertsMetadata,
    w13,
    w2,
    *,
    fp8_grad_all_gather: bool = False,
) -> torch.Tensor:
  """The routed experts of `ring_of_experts_dispatch(..., gather_tokens=
  False)` routing as the fused FP8 kernels of ``kernels/fp8_roe_experts.py``
  (MaxText ``fp8_full`` numerics): EP token all-gather in e4m3, the W1|W3
  GEMM with the ``silu(g) * u * score`` epilogue, the W2 GEMM fused with the
  per-token combine, and the combine reduce-scatter.

  Args:
    x_TD: local tokens.
    group_sizes_E: local-expert group sizes from the dispatch.
    metadata: the dispatch state.
    w13: `Fp8ExpertWeights` of ``[W1 | W3]`` (``[E, D, 2F]``).
    w2: `Fp8ExpertWeights` of W2 (``[E, F, D]``).
    fp8_grad_all_gather: all-gather the output gradient in backward as e4m3
      with per-token scales instead of BF16 (changes numerics).

  Returns:
    `[T, D]` combined local tokens, the scores applied.
  """
  return _Fp8Experts.apply(
      x_TD,
      metadata.scores_R,
      metadata.token_ids_R,
      metadata.valid_count_1,
      group_sizes_E.to(torch.int32),
      metadata.reduce_metadata,
      metadata.top_k,
      metadata.ep_mesh,
      fp8_grad_all_gather,
      w13,
      w2,
      *w13.shards,
      *w2.shards,
  )


__all__ = [
    "CAPACITY_TILE",
    "RingOfExpertsMetadata",
    "ring_of_experts_capacity",
    "ring_of_experts_combine",
    "ring_of_experts_dispatch",
    "ring_of_experts_fp8_experts",
]
