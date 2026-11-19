"""Transposed Splash attention forward: causal, packed-document GQA attention.

TPU, Pallas/Mosaic. The forward counterpart of `splash_tbwd` (same shape
suffixes, schedule and supported shapes). It emits the residuals splash_tbwd
and Tokamax's backward take: o, and the log2-domain logsumexp of logits from q
rounded exactly as the Tokamax forward rounds it.

Formulation
-----------
  S^T  = K    [C, D] @ Q2^T [D, Q]   (Q2^T latched; contraction D)
  m    = max(m, colmax(S^T)); P^T = exp2(S^T - m)   (online softmax over C)
  l    = l * exp2(m_old - m) + colsum(P^T)
  O^T  = O^T * exp2(m_old - m) + V^T [D, C] @ P^T [C, Q]   (P^T latched)
  on the q block's last tile: o = O^T / l, lse2 = m + log2(l)

  Q2 = bf16(bf16(Q * scale) * log2(e)), with one more rounding, bf16(Q *
  ratio) first, for a non-default scale (`splash_tbwd_call`'s q_scale /
  q_ratio); computed once per q block.

Per-step costs, Q235 layer on TPU7x (B 16, S 4096, 64 q / 4 kv heads, D 128,
packed C4-like rows): the kernel runs at about Tokamax's per-tile speed; the
non-MXU work (the [C, Q] f32 logits stored, read for the max, read for exp2,
written as bf16 P) and the two matmuls overlap poorly, so the transposed P.V
gains little here. The wins over Tokamax are structural:
  - one call for the whole batch over the tiles with a same-document pair
    (Tokamax skips such tiles only per call, so a vmapped forward keeps every
    tile any example needs, and a per-example loop pays slice / stack copies);
  - all G q heads per step (K / V^T loaded once);
  - the scale multiply of q in the kernel instead of an XLA pass.

Schedule
--------
`splash_tbwd.build_schedule`: grid (N, T) over the active (example, q block,
kv block) tiles in that order. O^T, m and l of one (example, q block)
accumulate in VMEM over its run of steps and are written once; K and V^T of
one (example, kv head) stay resident. A q column's first tiles can be fully
masked (its document starts in a later kv block): their m / l / O^T terms are
exp2(mask - mask) = 1 garbage that the diagonal tile, which always holds the q
token itself, wipes with exp2(m_old - m) = 0.

Supported shapes
----------------
splash_tbwd's (`splash_tbwd.unsupported_reason`, checked by the caller).
VMEM is below splash_tbwd's estimate: per step 14 * G * block * D bytes (q^T
and o^T blocks, bf16, 2 buffers each; Q2 scratch, bf16; f32 O^T accumulator)
versus splash_tbwd's 20 * G * block * D, and per (example, kv head) K and V^T
only, without dK^T / dV^T and their f32 accumulators.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from torchtitan.experiments.tpu.kernels import splash_tbwd

NUM_LANES = splash_tbwd.NUM_LANES
LOG2E = splash_tbwd.LOG2E
# As Tokamax's DEFAULT_MASK_VALUE: finite, so exp2(mask - mask) stays finite.
_MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)


def _round_q(qt_DQ, q_scale: float, q_ratio: float, q_dtype):
  """Q2 = q rounded as the forward (and splash_tbwd) round it before Q.K."""
  if q_ratio != 1.0:
    q1 = (qt_DQ.astype(jnp.float32) * q_ratio).astype(q_dtype)
    qk = (q1.astype(jnp.float32) * (q_scale / q_ratio)).astype(q_dtype)
  elif float(np.asarray(q_scale, q_dtype)) == q_scale:
    qk = (qt_DQ * q_scale).astype(q_dtype)
  else:
    qk = (qt_DQ.astype(jnp.float32) * q_scale).astype(q_dtype)
  return (qk * LOG2E).astype(q_dtype)


def _kernel(
    # scalar prefetch [T] (SMEM)
    b_ref,
    qi_ref,
    kj_ref,
    kind_ref,
    q_bounds_start_ref,
    q_bounds_end_ref,
    b_bounds_start_ref,
    b_bounds_end_ref,
    # inputs (VMEM blocks)
    q_GDQ_ref,
    k_SD_ref,
    v_DS_ref,
    qseg_8Q_ref,
    kvseg_CL_ref,
    # outputs
    o_GDQ_ref,
    lse2_GQ_ref,
    # scratch
    q2_GDQ,
    acc_GDQ,
    m_GQ,
    l_GQ,
    *,
    group_size: int,
    block: int,
    q_scale: float,
    q_ratio: float,
):
  del b_ref, b_bounds_start_ref, b_bounds_end_ref
  step = pl.program_id(1)
  kind = kind_ref[step]
  q_block = qi_ref[step]
  kv_start = pl.multiple_of(kj_ref[step] * block, block)
  q_dtype = q_GDQ_ref.dtype

  @pl.when(q_bounds_start_ref[step].astype(jnp.bool_))
  def _():
    acc_GDQ[...] = jnp.zeros_like(acc_GDQ)
    m_GQ[...] = jnp.full_like(m_GQ, _MASK_VALUE)
    l_GQ[...] = jnp.zeros_like(l_GQ)
    for g in range(group_size):
      q2_GDQ[g] = _round_q(q_GDQ_ref[g], q_scale, q_ratio, q_dtype)

  def tile(masked: bool) -> None:
    kv = pl.ds(kv_start, block)
    k_CD = k_SD_ref[kv, :]
    vt_DC = v_DS_ref[:, kv]
    if masked:
      k_pos = kv_start + lax.broadcasted_iota(jnp.int32, (block, block), 0)
      q_pos = q_block * block + lax.broadcasted_iota(
          jnp.int32, (block, block), 1
      )
      kvseg_CQ = jnp.tile(kvseg_CL_ref[...], (1, block // NUM_LANES))
      mask_CQ = (q_pos >= k_pos) & (kvseg_CQ == qseg_8Q_ref[0:1, :])
    nn = (((1,), (0,)), ((), ()))
    for g in range(group_size):
      st_CQ = lax.dot_general(
          k_CD, q2_GDQ[g], nn, preferred_element_type=jnp.float32
      )
      if masked:
        st_CQ = jnp.where(mask_CQ, st_CQ, _MASK_VALUE)
      # Online softmax over the C rows, per q column.
      m_prev = m_GQ[pl.ds(g, 1), :]
      m_new = jnp.maximum(m_prev, jnp.max(st_CQ, axis=0, keepdims=True))
      alpha = jnp.exp2(m_prev - m_new)
      pt_CQ = jnp.exp2(st_CQ - m_new)
      l_GQ[pl.ds(g, 1), :] = alpha * l_GQ[pl.ds(g, 1), :] + jnp.sum(
          pt_CQ, axis=0, keepdims=True
      )
      m_GQ[pl.ds(g, 1), :] = m_new
      acc_GDQ[g] = alpha * acc_GDQ[g] + lax.dot_general(
          vt_DC, pt_CQ.astype(vt_DC.dtype), nn,
          preferred_element_type=jnp.float32,
      )

  @pl.when(kind == 1)
  def _():
    tile(masked=False)

  @pl.when(kind == 2)
  def _():
    tile(masked=True)

  @pl.when(q_bounds_end_ref[step].astype(jnp.bool_))
  def _():
    for g in range(group_size):
      l = l_GQ[pl.ds(g, 1), :]
      o_GDQ_ref[g] = (acc_GDQ[g] / l).astype(o_GDQ_ref.dtype)
    lse2 = m_GQ[...] + jnp.log2(l_GQ[...])
    lse2_GQ_ref[...] = lse2[:group_size].astype(lse2_GQ_ref.dtype)


def splash_tfwd_call(
    sched,
    q_BHDS,
    k_BNSD,
    v_BNDS,
    qseg_B8S,
    kvseg_BSL,
    *,
    block: int,
    q_scale: float,
    q_ratio: float = 1.0,
    single_buffer_kv: bool = False,
):
  """The pallas_call on kernel layouts.

  `transposed_attention_fwd` is the model-layout entry point.

  Args:
    sched: `splash_tbwd.build_schedule` of the segment ids.
    q_BHDS: unscaled q, sequence-minor.
    k_BNSD, v_BNDS: K head-dim-minor, V sequence-minor.
    qseg_B8S, kvseg_BSL: segment ids broadcast to 8 rows / 128 lanes.
    block: tile size (q block = kv block).
    q_scale, q_ratio: see `splash_tbwd.splash_tbwd_call`.
    single_buffer_kv: one VMEM buffer for the per-(example, kv head) K / V^T.

  Returns:
    o^T [B, H, D, S] in q's dtype; lse2 [B, N, G, S] f32.
  """
  num_b, num_h, d, s = q_BHDS.shape
  num_kv = k_BNSD.shape[1]
  g = num_h // num_kv

  def q_seq_minor(h, t, b, qi, *_):
    return (b[t], h, 0, qi[t])

  def per_kv_head(h, t, b, *_):
    return (b[t], h, 0, 0)

  def q_segments(h, t, b, qi, *_):
    return (b[t], 0, qi[t])

  def kv_segments(h, t, b, qi, kj, *_):
    return (b[t], kj[t], 0)

  resident = dict(pipeline_mode=pl.Buffered(1)) if single_buffer_kv else {}
  in_specs = [
      pl.BlockSpec((None, g, d, block), q_seq_minor),
      pl.BlockSpec((None, None, s, d), per_kv_head, **resident),
      pl.BlockSpec((None, None, d, s), per_kv_head, **resident),
      pl.BlockSpec((None, 8, block), q_segments),
      pl.BlockSpec((None, block, NUM_LANES), kv_segments),
  ]
  out_specs = [
      pl.BlockSpec((None, g, d, block), q_seq_minor),
      pl.BlockSpec((None, None, g, block), q_seq_minor),
  ]
  out_shape = [
      jax.ShapeDtypeStruct((num_b, num_h, d, s), q_BHDS.dtype),
      jax.ShapeDtypeStruct((num_b, num_kv, g, s), jnp.float32),
  ]
  g8 = -(-g // 8) * 8
  scratch = [
      pltpu.VMEM((g, d, block), q_BHDS.dtype),
      pltpu.VMEM((g, d, block), jnp.float32),
      pltpu.VMEM((g8, block), jnp.float32),
      pltpu.VMEM((g8, block), jnp.float32),
  ]
  return pl.pallas_call(
      functools.partial(
          _kernel, group_size=g, block=block, q_scale=q_scale, q_ratio=q_ratio
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(splash_tbwd._SCHEDULE_KEYS),  # pylint: disable=protected-access
          in_specs=in_specs,
          out_specs=out_specs,
          grid=(num_kv, sched["num_active"]),
          scratch_shapes=scratch,
      ),
      out_shape=out_shape,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary"),
          vmem_limit_bytes=splash_tbwd._vmem_limit_bytes(),  # pylint: disable=protected-access
      ),
      name="splash_tfwd",
  )(*(sched[key] for key in splash_tbwd._SCHEDULE_KEYS), q_BHDS, k_BNSD, v_BNDS, qseg_B8S, kvseg_BSL)  # pylint: disable=protected-access


def transposed_attention_fwd(
    q_BSHD, k_BSND, v_BSND, segment_ids_BS, *, q_scale: float, q_ratio: float = 1.0
):
  """Causal attention forward in the model layout [B, S, heads, D].

  Args:
    q_BSHD, k_BSND, v_BSND: q (unscaled), k, v.
    segment_ids_BS: document id per token; None means one document per example.
    q_scale, q_ratio: see `splash_tbwd.splash_tbwd_call`.

  Raises:
    ValueError: for shapes outside splash_tbwd's "Supported shapes".

  Returns:
    out [B, S, H, D] in q's dtype; lse2 [B, H, S] f32, the log2-domain
    logsumexp of the base-2 logits (splash_tbwd's / Tokamax's residual).
  """
  reason = splash_tbwd.unsupported_reason(
      q_BSHD.shape, k_BSND.shape, v_BSND.shape, q_BSHD.dtype
  )
  if reason is not None:
    raise ValueError(f"splash_tfwd: splash_tbwd {reason}.")
  num_b, s, num_h, d = q_BSHD.shape
  num_kv = k_BSND.shape[2]
  if segment_ids_BS is None:
    segment_ids_BS = jnp.zeros((num_b, s), jnp.int32)
  block = splash_tbwd.block_size(s)
  sched = splash_tbwd.build_schedule(segment_ids_BS, block)
  seg = segment_ids_BS.astype(jnp.int32)
  ot_BHDS, lse2_BNGS = splash_tfwd_call(
      sched,
      jnp.transpose(q_BSHD, (0, 2, 3, 1)),
      jnp.swapaxes(k_BSND, 1, 2),
      jnp.transpose(v_BSND, (0, 2, 3, 1)),
      jnp.broadcast_to(seg[:, None, :], (num_b, 8, s)),
      jnp.broadcast_to(seg[:, :, None], (num_b, s, NUM_LANES)),
      block=block,
      q_scale=q_scale,
      q_ratio=q_ratio,
  )
  return (
      jnp.transpose(ot_BHDS, (0, 3, 1, 2)),
      lse2_BNGS.reshape(num_b, num_h, s),
  )
