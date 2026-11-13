# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused FP8 Ring-of-Experts expert MLP on MaxText's gmmv2 kernels.

One rank's routed experts between the EP token all-gather and the combine
reduce-scatter, with MaxText's ``fp8_full`` numerics (static e4m3 scale for
tokens, weights and the SiLU output). Forward:

* ``gather_gate``: the W1|W3 GEMM of the routed rows, DMA'd from the gathered
  FP8 tokens in-kernel, with the ``silu(g) * u * score`` epilogue emitting the
  FP8 hidden; ``gate_out`` (``[g | u]``, BF16) is kept for backward.
* ``linear_combine``: the W2 GEMM, adding each row straight into its token's
  row of the combined ``[N, D // 128, 128]`` output.

Backward:

* ``linear_bwd_gather``: W2 dgrad of the output-gradient rows (gathered
  in-kernel, quantized to e5m2 with power-of-two per-tile scales), the SiLU
  backward, the score gradient and dW2.
* ``gate_bwd``: the fused W1|W3 dgrad + wgrad on the routed tokens.

Neither the routed tokens nor the expert outputs are written to HBM in the
forward, and the backward writes only the routed tokens (FP8) and their
gradient. Rows past ``sum(group_sizes)`` are skipped.

Shape legend: N = gathered tokens, R = routed rows (capacity), E = local
experts, D = model dim, F = expert hidden dim.
"""

import jax
import jax.numpy as jnp
from torch_tpu._internal import pallas

from torchtitan.experiments.tpu.kernels.fp8_expert_segment_matmul import (
    BWD_DTYPE,
    FWD_DTYPE,
    STATIC_SCALE,
)
from torchtitan.experiments.tpu.kernels import ragged_gather
from torchtitan.experiments.tpu.kernels.maxtext_gmmv2 import expert_mlp

LANES = 128


def _scale(value: float) -> jax.Array:
  return jnp.full((1, 1), value, jnp.float32)


def _gmm(d: int) -> expert_mlp.ExpertGmm3d:
  return expert_mlp.make_gmm_fn(d)


def experts_fwd(
    x_ag_N3: jax.Array,
    token_ids_R: jax.Array,
    group_sizes_E: jax.Array,
    scores_R: jax.Array,
    w13_q_EDF2: jax.Array,
    w2_q_EFD: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Returns ``(y_N3, gate_out_RF2)``: the combined expert outputs of every
  gathered token (BF16, zero for tokens without rows) and the W1|W3 GEMM
  output kept for backward."""
  num_tokens = x_ag_N3.shape[0]
  num_rows = token_ids_R.shape[0]
  _, d, f2 = w13_q_EDF2.shape
  gmm = _gmm(d)
  token_ids_R = token_ids_R.astype(jnp.int32)
  s = STATIC_SCALE
  bank, act_RF = gmm.gather_gate(
      x_ag_N3,
      token_ids_R,
      0,
      w13_q_EDF2,
      group_sizes_E,
      jnp.zeros((num_rows, f2), jnp.bfloat16),
      0,
      out_scale=_scale(s * s),
      size_m=num_rows,
      glu_coeffs=scores_R.reshape(-1, 1).astype(jnp.float32),
      glu_out_dtype=FWD_DTYPE,
      glu_out_scale=_scale(s),
  )
  valid_R = jnp.arange(num_rows, dtype=jnp.int32) < jnp.sum(group_sizes_E)
  sorted_tokens_R = jnp.sort(jnp.where(valid_R, token_ids_R, num_tokens))
  y_N3 = gmm.linear_combine(
      act_RF,
      w2_q_EFD,
      group_sizes_E,
      token_ids_R,
      0,
      sorted_tokens_R,
      out_scale=_scale(s * s),
      num_out_tokens=num_tokens,
  )
  return y_N3, bank


def experts_bwd(
    dy_N3: jax.Array,
    x_ag_N3: jax.Array,
    token_ids_R: jax.Array,
    group_sizes_E: jax.Array,
    scores_R: jax.Array,
    w13_q_EDF2: jax.Array,
    w2_q_EFD: jax.Array,
    gate_out_RF2: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Returns ``(dx_R3, dscores_R, dW13, dW2)``: BF16 weight gradients, dW13
  for the concatenated ``[W1 | W3]`` (splitting it here would copy it; split
  its reduced shard instead). Rows of ``dx_R3`` past ``sum(group_sizes)`` are
  unspecified."""
  _, d, _ = w13_q_EDF2.shape
  gmm = _gmm(d)
  token_ids_R = token_ids_R.astype(jnp.int32)
  s = STATIC_SCALE
  dgate_RF2, _, dgate_tile_exps, dscores_R1, dw2 = gmm.linear_bwd_gather(
      dy_N3,
      token_ids_R,
      0,
      w2_q_EFD,
      group_sizes_E,
      gate_out_RF2,
      0,
      scores_R.reshape(-1, 1).astype(jnp.float32),
      w_q=w2_q_EFD,
      x_scale=s,
      x_qtype=FWD_DTYPE,
      dout_qtype=BWD_DTYPE,
      w_scale=s,
      dw_dtype=jnp.bfloat16,
  )
  # The routed FP8 tokens, by row DMA (an XLA gather is ~2.7x slower).
  routed_R3 = ragged_gather._ragged_gather_tc_impl(  # pylint: disable=protected-access
      x_ag_N3,
      token_ids_R,
      jnp.int32(0),
      jnp.sum(group_sizes_E, dtype=jnp.int32),
      token_ids_R.shape[0],
      block_size=2048,
  )
  dx_R3, dw13 = gmm.gate_bwd(
      routed_R3,
      w13_q_EDF2,
      group_sizes_E,
      dgate_RF2,
      None,
      w_q=w13_q_EDF2,
      dgate_tile_exps=dgate_tile_exps,
      dx_scale=_scale(s),
      dw_scale=_scale(s),
      dw_dtype=jnp.bfloat16,
  )
  return dx_R3, dscores_R1.reshape(-1), dw13, dw2


@pallas.jax_op("tt_roe::fp8_experts_fwd")
def fp8_experts_fwd_op(
    x_ag_N3: jax.Array,
    token_ids_R: jax.Array,
    group_sizes_E: jax.Array,
    scores_R: jax.Array,
    w13_q_EDF2: jax.Array,
    w2_q_EFD: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  return experts_fwd(
      x_ag_N3, token_ids_R, group_sizes_E, scores_R, w13_q_EDF2, w2_q_EFD
  )


@pallas.jax_op("tt_roe::fp8_experts_bwd")
def fp8_experts_bwd_op(
    dy_N3: jax.Array,
    x_ag_N3: jax.Array,
    token_ids_R: jax.Array,
    group_sizes_E: jax.Array,
    scores_R: jax.Array,
    w13_q_EDF2: jax.Array,
    w2_q_EFD: jax.Array,
    gate_out_RF2: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  return experts_bwd(
      dy_N3,
      x_ag_N3,
      token_ids_R,
      group_sizes_E,
      scores_R,
      w13_q_EDF2,
      w2_q_EFD,
      gate_out_RF2,
  )


__all__ = [
    "experts_bwd",
    "experts_fwd",
    "fp8_experts_bwd_op",
    "fp8_experts_fwd_op",
]
