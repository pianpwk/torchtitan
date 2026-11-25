"""Fused chunked Kimi Delta Attention (KDA) for TPU, in the model's layout.

Recurrence per head (per-channel log-decay ``g_t <= 0``, natural log)::

  S_t = (I - b_t k_t k_t^T) diag(exp(g_t)) S_{t-1} + b_t k_t v_t^T
  o_t = scale * q_t^T S_t

with ``q``, ``k`` L2-normalized and K3's gate ``g = lower * sigmoid(exp(a_log)
* (raw_gate + dt_bias))``, all evaluated in the kernels.

The forward kernel scans the chunks of a row in order (grid ``(B, H / M, S /
C)``, last axis sequential) with the ``[K, V]`` states of ``M`` heads in VMEM;
the backward kernel scans them in reverse with the state gradients in VMEM,
recomputing each chunk's intra-chunk terms from the inputs and the saved
per-chunk input state. Per chunk and head (``G`` = in-chunk inclusive cumsum of
``g`` in log2 units, ``G_C`` its last row)::

  A_kk[i, j] = b_i sum_c k_ic k_jc 2^(G_ic - G_jc)   (j < i, same document)
  A_qk[i, j] = scale sum_c q_ic k_jc 2^(G_ic - G_jc)  (j <= i, same document)
  T = (I + A_kk)^-1;  W = T (b k 2^G);  U = T (b v)
  v_new = U - W S0;  o = scale (q 2^G) S0 + A_qk v_new
  S = 2^G_C S0 + (k 2^(G_C - G))^T v_new

Inputs and outputs stay in ``[B, S, H * K]`` (a head is a 128-lane slice).
Documents are not padded to chunk boundaries: chunks may span documents, and
segment masks restrict ``A`` to one document, the ``S0`` terms to rows of the
document continuing from the previous chunk (``S0 = 0`` when a document starts
at the chunk), and the state update to rows of the chunk's last document.

Per-channel decays use 16-row sub-blocks so every ``exp2`` argument is
bounded: an off-diagonal sub-block factors ``2^(G_i - G_j)`` around the end of
its column block (both factors <= 1), a diagonal one around its middle row
(factors <= 2^(8 |g|max log2 e), which needs ``8 |lower| log2(e) < 126``).

Every matmul level is issued for all ``M`` heads before its consumers: the MXU
result path is in order, so a per-head program order pays each dependent
matmul's latency once per head.

Precision: fp32 accumulation everywhere. Everything that feeds the chunk
recurrence or the output is ~fp32: the A matrices, ``T [b k 2^G | b v]``, the
output products and the state update use bf16x3 (``hi*hi + hi*lo + lo*hi``),
``U`` stays fp32 and ``T`` is saved in fp32; ``W`` is rounded to bf16 once and
``W S0`` splits ``S0``. With ``beta ~ 1``, aligned keys and slow decay, ``v_new
= U - W S0`` and the state update cancel heavily, and single-pass bf16 there
gives ~3e-2 relative error (fp32 chunk algebra: ~1e-6). The gate cumsum and
its reverse are exact. In the backward, the products of the state-gradient
recurrence that have one exactly-bf16 operand split the fp32 one (``W^T
dv_new``, ``dv_new S0^T``), ``kg dS`` is bf16x3, and the gated products
(``dA -> dq, dk, dG``) round ``dA`` to bf16 once for both sides and split the
decay-factor operands: the gate gradient is a difference of nearly equal pair
terms. The other backward matmuls are single-pass bf16.

Example (chunk 4, one row, documents ``a`` (3 tokens) and ``b`` (6))::

  tokens   a0 a1 a2 b0 | b1 b2 b3 b4 | b5 . . .
  chunk 0  A within a and within b0; S0 = 0; state update from b0 only
  chunk 1  all rows continue b: S0 terms on every row
  chunk 2  ...

Shape suffixes: B = rows, S = tokens per row, H = heads, K = key head dim,
V = value head dim, C = chunk, N = chunks per row, M = heads per grid step.
"""

import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

_SUB = 16  # sub-block rows for the per-channel decay factorization
_BASE_BLOCK = 4  # diagonal blocks of the WY inverse's base case
_EPS = 1e-6  # L2-norm epsilon (FLA / Tokamax)
_LOG2E = 1.0 / math.log(2.0)


# --- matmul helpers ---------------------------------------------------------

_NN = (((1,), (0,)), ((), ()))
_NT = (((1,), (1,)), ((), ()))
_TN = (((0,), (0,)), ((), ()))


def _dot(a, b, dims=_NN):
  """bf16 operands, fp32 accumulation."""
  return jax.lax.dot_general(
      a.astype(jnp.bfloat16),
      b.astype(jnp.bfloat16),
      dims,
      preferred_element_type=jnp.float32,
  )


def _dot3(a, b, dims=_NN):
  """~fp32 matmul of two fp32 operands: hi*hi + hi*lo + lo*hi of bf16 parts."""
  ah, al = _split(a.astype(jnp.float32))
  bh, bl = _split(b.astype(jnp.float32))
  return _dot(ah, bh, dims) + (_dot(ah, bl, dims) + _dot(al, bh, dims))


def _split(x):
  hi = x.astype(jnp.bfloat16)
  return hi, (x - hi.astype(jnp.float32)).astype(jnp.bfloat16)


def _dot_s2(a, b, dims=_NN, *, split_a=False):
  """Matmul with one fp32 operand as two bf16 parts (``b``, or ``a`` if
  ``split_a``); the other operand must be exactly bf16."""
  if split_a:
    hi, lo = _split(a)
    return _dot(hi, b, dims) + _dot(lo, b, dims)
  hi, lo = _split(b)
  return _dot(a, hi, dims) + _dot(a, lo, dims)


def _tri(C, *, upper):
  r = jax.lax.broadcasted_iota(jnp.int32, (C, C), 0)
  c = jax.lax.broadcasted_iota(jnp.int32, (C, C), 1)
  return ((r <= c) if upper else (r >= c)).astype(jnp.bfloat16)


def _prefix_rows(x_list, *, reverse=False):
  """Inclusive (or reverse inclusive) prefix sums over rows, exact in fp32
  (three bf16 parts against a 0/1 triangle)."""
  C = x_list[0].shape[0]
  tri = _tri(C, upper=reverse)
  out = []
  for x in x_list:
    hi = x.astype(jnp.bfloat16)
    r = x - hi.astype(jnp.float32)
    mid = r.astype(jnp.bfloat16)
    lo = (r - mid.astype(jnp.float32)).astype(jnp.bfloat16)
    out.append((_dot(tri, hi) + _dot(tri, mid)) + _dot(tri, lo))
  return out


def _block_rows(row_1K_list):
  """``[C, K]`` with sub-block ``i``'s rows all equal to ``row_1K_list[i]``."""
  return jnp.concatenate(
      [jnp.broadcast_to(r, (_SUB, r.shape[-1])) for r in row_1K_list], axis=0
  )


def _neumann(X_list, n, eye):
  """``(I + X)^-1`` per head for ``X`` nilpotent of index ``n`` (a power of
  two): ``(I - X)(I + X^2)(I + X^4)...``, multiplied as a balanced tree."""
  facs = [[eye - x] for x in X_list]
  p = X_list
  for _ in range(int(math.log2(n)) - 1):
    p = [_dot(x, x) for x in p]
    for f, x in zip(facs, p):
      f.append(eye + x)
  while len(facs[0]) > 1:
    facs = [
        [_dot(f[i], f[i + 1]) for i in range(0, len(f) - 1, 2)]
        + ([f[-1]] if len(f) % 2 else [])
        for f in facs
    ]
  return [f[0] for f in facs]


def _wy(L_list, R_list):
  """``(I + L)^-1 R`` and ``(I + L)^-1`` per head for strictly lower ``L``.

  4x4 diagonal blocks are inverted with ``(I - D)(I + D^2)``; the block size
  then doubles with ``T <- T - T F T``, ``F`` the lower off-diagonal half-block
  of each doubled block (``(I + T F)^-1 = I - T F`` since ``(T F)^2 = 0``).
  Every intermediate is a block of the true inverse. (A Neumann product over
  16-row blocks instead forms powers ``D^k`` whose entries grow like binomial
  coefficients when ``A_kk`` entries approach +-1, e.g. beta ~ 1 with aligned
  keys and slow decay, and its bf16 roundings then cancel catastrophically.)
  """
  C = L_list[0].shape[0]
  r = jax.lax.broadcasted_iota(jnp.int32, (C, C), 0)
  c = jax.lax.broadcasted_iota(jnp.int32, (C, C), 1)
  eye = (r == c).astype(jnp.float32)
  b = _BASE_BLOCK
  T = _neumann([jnp.where(r // b == c // b, L, 0.0) for L in L_list], b, eye)
  while b < C:
    half = (r // (2 * b) == c // (2 * b)) & (r // b != c // b)
    F = [jnp.where(half, L, 0.0) for L in L_list]
    X = [_dot(f, t) for f, t in zip(F, T)]
    T = [t - _dot(t, x) for t, x in zip(T, X)]
    b *= 2
  return [_dot3(t, rr) for t, rr in zip(T, R_list)], T


# --- per-chunk shared parts -------------------------------------------------


def _chunk_masks(seg_col_ref, seg_row_ref, C):
  seg_C1 = seg_col_ref[0]
  seg_1C = seg_row_ref[0]
  ri = jax.lax.broadcasted_iota(jnp.int32, (C, C), 0)
  ci = jax.lax.broadcasted_iota(jnp.int32, (C, C), 1)
  same = seg_C1 == seg_1C
  return dict(
      qk_keep=same & (ci <= ri),
      kk_keep=same & (ci < ri),
      first=seg_C1 == seg_C1[0:1],  # rows continuing the carried doc
      last=seg_C1 == seg_C1[C - 1 : C],  # rows of the chunk's last doc
      carry=seg_C1[C - 1 : C] == seg_C1[0:1],  # [1, 1]: last doc == first doc
  )


def _load_heads(refs, M, K, V, lower, *, log_gate, valid):
  """Per-head normalized q, k (and rstd), v, beta, gate inputs and the log2
  per-token decays.

  ``log_gate``: ``g_ref`` holds natural-log decays, clamped to ``>= lower``;
  else K3's gate ``lower * sigmoid(exp(a_log) * (raw + dt_bias))``. ``valid``:
  ``[C, 1]`` mask of real (not tail-padding) rows, or None; padding rows get
  decay 1 so the state passes through them unchanged.
  """
  q_ref, k_ref, v_ref, g_ref, beta_ref, decay_ref, bias_ref = refs
  out = {n: [] for n in ("q", "k", "rq", "rk", "v", "beta", "z", "sig", "g2")}
  for h in range(M):
    kl = slice(h * K, (h + 1) * K)
    for n, ref in (("q", q_ref), ("k", k_ref)):
      x = ref[0, :, kl].astype(jnp.float32)
      r = jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + _EPS)
      out[n].append(x * r)
      out["r" + n].append(r)
    out["v"].append(v_ref[0, :, h * V : (h + 1) * V].astype(jnp.float32))
    out["beta"].append(beta_ref[0, 0, :, h : h + 1].astype(jnp.float32))
    if log_gate:
      z = g_ref[0, :, kl].astype(jnp.float32)
      out["z"].append(z)
      g2 = jnp.maximum(z, lower) * _LOG2E
    else:
      z = g_ref[0, :, kl].astype(jnp.float32) + bias_ref[:, kl]
      sig = jax.nn.sigmoid(decay_ref[:, kl] * z)
      out["z"].append(z)
      out["sig"].append(sig)
      g2 = (lower * _LOG2E) * sig
    if valid is not None:
      g2 = jnp.where(valid, g2, 0.0)
    out["g2"].append(g2)
  return out


def _valid_rows(chunk_index, C, valid_len, S):
  """``[C, 1]`` mask of rows before ``valid_len`` (None without padding)."""
  if valid_len >= S:
    return None
  rows = jax.lax.broadcasted_iota(jnp.int32, (C, 1), 0)
  return chunk_index * C + rows < valid_len


def _split_refs(refs, names):
  """Names refs positionally; ``names`` entries that are None are skipped."""
  out, i = {}, 0
  for n in names:
    if n is None:
      continue
    out[n] = refs[i]
    i += 1
  assert i == len(refs), (i, len(refs))
  return out


def _attention_terms(x, masks, scale, C, M):
  """A_qk, A_kk (masked) and the decay factors reused by the backward.

  For row block ``i``: ``lhs[i] = [q; k]_i * 2^(G_i - mid_i)`` and ``rhs[i]``
  = ``k * diag`` on block ``i``, ``k * off * vec[i][j]`` on blocks ``j < i``
  and 0 after, with ``diag = 2^(mid - G)`` and ``off = 2^(end - G)`` per row
  (``mid`` / ``end``: its block's middle / last row) and ``vec[i][j] =
  2^(mid_i - end_j)`` (a ``[1, K]`` row).
  """
  NB = C // _SUB
  out = {n: [] for n in ("e_row", "lhs", "rhs", "off", "diag", "vec")}
  zeros = jnp.zeros((_SUB, x["k"][0].shape[-1]), jnp.float32)
  for h in range(M):
    G, k = x["G"][h], x["k"][h]
    mid = [
        G[i * _SUB + _SUB // 2 : i * _SUB + _SUB // 2 + 1] for i in range(NB)
    ]
    end = [G[i * _SUB + _SUB - 1 : i * _SUB + _SUB] for i in range(NB)]
    mid_CK = _block_rows(mid)
    e = jnp.exp2(G - mid_CK)
    off = jnp.exp2(_block_rows(end) - G)
    diag = jnp.exp2(mid_CK - G)
    vec = [
        [jnp.exp2(jnp.minimum(mid[i] - end[j], 0.0)) for j in range(i)]
        for i in range(NB)
    ]
    k_off, k_diag = k * off, k * diag
    blk = lambda a, j: a[j * _SUB : (j + 1) * _SUB]
    out["rhs"].append(
        [
            jnp.concatenate(
                [blk(k_off, j) * vec[i][j] for j in range(i)]
                + [blk(k_diag, i)]
                + [zeros] * (NB - 1 - i),
                0,
            )
            for i in range(NB)
        ]
    )
    lq, lk = x["q"][h] * e, k * e
    out["lhs"].append(
        [jnp.concatenate([blk(lq, i), blk(lk, i)], 0) for i in range(NB)]
    )
    out["e_row"].append(e)
    out["off"].append(off)
    out["diag"].append(diag)
    out["vec"].append(vec)
  p = [
      [_dot3(out["lhs"][h][i], out["rhs"][h][i], _NT) for i in range(NB)]
      for h in range(M)
  ]
  out["a_qk"], out["a_kk"], out["a_kk_raw"] = [], [], []
  for h in range(M):
    raw_qk = jnp.concatenate([t[:_SUB] for t in p[h]], axis=0)
    raw_kk = jnp.concatenate([t[_SUB:] for t in p[h]], axis=0)
    out["a_qk"].append(jnp.where(masks["qk_keep"], raw_qk * scale, 0.0))
    out["a_kk_raw"].append(jnp.where(masks["kk_keep"], raw_kk, 0.0))
    out["a_kk"].append(out["a_kk_raw"][-1] * x["beta"][h])
  return out


# --- forward ----------------------------------------------------------------


def _fwd_kernel(
    reset_ref,
    *refs,
    M,
    K,
    V,
    scale,
    lower,
    store_state,
    log_gate,
    has_init,
    has_final,
    valid_len,
    N,
):
  r = _split_refs(
      refs,
      (
          "q",
          "k",
          "v",
          "g",
          "beta",
          "seg_col",
          "seg_row",
          None if log_gate else "decay",
          None if log_gate else "bias",
          "init" if has_init else None,
          "o",
          "h" if store_state else None,
          "t" if store_state else None,
          "final" if has_final else None,
          "state",
      ),
  )
  q_ref, o_ref, state_ref = r["q"], r["o"], r["state"]
  C = q_ref.shape[1]
  c = pl.program_id(2)
  masks = _chunk_masks(r["seg_col"], r["seg_row"], C)
  reset = reset_ref[pl.program_id(0), c] != 0
  x = _load_heads(
      (q_ref, r["k"], r["v"], r["g"], r["beta"], r.get("decay"), r.get("bias")),
      M,
      K,
      V,
      lower,
      log_gate=log_gate,
      valid=_valid_rows(c, C, valid_len, N * C),
  )
  x["G"] = _prefix_rows(x["g2"])  # [C, K], <= 0
  at = _attention_terms(x, masks, scale, C, M)
  heads = range(M)
  rhs = [
      jnp.concatenate(
          [
              x["k"][h] * jnp.exp2(x["G"][h]) * x["beta"][h],
              x["v"][h] * x["beta"][h],
          ],
          1,
      )
      for h in heads
  ]
  wu, T = _wy(at["a_kk"], rhs)
  if store_state:
    for h in heads:
      r["t"][0, 0, h] = T[h].astype(r["t"].dtype)

  w, u, qg, kgt, dec, s0 = [], [], [], [], [], []
  for h in heads:
    G = x["G"][h]
    g_last = G[C - 1 : C]
    w.append(jnp.where(masks["first"], wu[h][:, :K], 0.0).astype(jnp.bfloat16))
    u.append(wu[h][:, K:])
    qg.append(jnp.where(masks["first"], x["q"][h] * jnp.exp2(G) * scale, 0.0))
    kgt.append(
        jnp.where(masks["last"], x["k"][h] * jnp.exp2(g_last - G), 0.0).T
    )
    dec.append(jnp.where(masks["carry"], jnp.exp2(g_last), 0.0).T)
    s = jnp.where(reset, 0.0, state_ref[h])
    if has_init:
      # The row's first document continues from the initial state.
      s = jnp.where(c == 0, r["init"][0, h].astype(jnp.float32), s)
    if store_state:
      r["h"][0, 0, h] = s.astype(r["h"].dtype)
    s0.append(s)
  ws = [_dot_s2(w[h], s0[h]) for h in heads]
  qs = [_dot3(qg[h], s0[h]) for h in heads]
  v_new = [u[h] - ws[h] for h in heads]
  for h in heads:
    o = qs[h] + _dot3(at["a_qk"][h], v_new[h])
    o_ref[0, :, h * V : (h + 1) * V] = o.astype(o_ref.dtype)
    state_ref[h] = dec[h] * s0[h] + _dot3(kgt[h], v_new[h])
  if has_final:

    @pl.when(c == N - 1)
    def _():
      for h in heads:
        r["final"][0, h] = state_ref[h]


# --- backward ---------------------------------------------------------------


def _bwd_kernel(
    reset_ref,
    *refs,
    M,
    K,
    V,
    scale,
    lower,
    N,
    log_gate,
    has_init,
    has_final,
    valid_len,
):
  r = _split_refs(
      refs,
      (
          "q",
          "k",
          "v",
          "g",
          "beta",
          "seg_col",
          "seg_row",
          None if log_gate else "decay",
          None if log_gate else "bias",
          "h",
          "t",
          "do",
          "dfinal" if has_final else None,
          "dq",
          "dk",
          "dv",
          "dg",
          "dbeta",
          None if log_gate else "dbias",
          None if log_gate else "dalog",
          "dinit" if has_init else None,
          "ds",
      ),
  )
  q_ref, h_ref, t_ref, do_ref, ds_ref = r["q"], r["h"], r["t"], r["do"], r["ds"]
  dq_ref, dk_ref, dv_ref, dg_ref, dbeta_ref = (
      r["dq"],
      r["dk"],
      r["dv"],
      r["dg"],
      r["dbeta"],
  )
  decay_ref = r.get("decay")
  b = pl.program_id(0)
  c = pl.program_id(2)
  n = N - 1 - c
  C = q_ref.shape[1]
  NB = C // _SUB
  heads = range(M)
  masks = _chunk_masks(r["seg_col"], r["seg_row"], C)
  reset = reset_ref[b, n] != 0

  @pl.when(c == 0)
  def _():
    if has_final:
      ds_ref[...] = r["dfinal"][0].astype(jnp.float32)
    else:
      ds_ref[...] = jnp.zeros_like(ds_ref)
    if not log_gate:
      r["dbias"][...] = jnp.zeros_like(r["dbias"])
      r["dalog"][...] = jnp.zeros_like(r["dalog"])

  # --- recompute the chunk's forward terms ---
  valid = _valid_rows(n, C, valid_len, N * C)
  x = _load_heads(
      (q_ref, r["k"], r["v"], r["g"], r["beta"], decay_ref, r.get("bias")),
      M,
      K,
      V,
      lower,
      log_gate=log_gate,
      valid=valid,
  )
  x["G"] = _prefix_rows(x["g2"])
  at = _attention_terms(x, masks, scale, C, M)
  g_exp = [jnp.exp2(G) for G in x["G"]]
  r_w = [x["k"][h] * g_exp[h] * x["beta"][h] for h in heads]
  r_u = [x["v"][h] * x["beta"][h] for h in heads]
  # T = (I + A_kk)^-1 as saved by the forward (fp32).
  T = [t_ref[0, 0, h] for h in heads]
  wu = [_dot3(T[h], jnp.concatenate([r_w[h], r_u[h]], 1)) for h in heads]
  first, last = masks["first"], masks["last"]
  s0, w, u, qg, kg, dec, kdec = [], [], [], [], [], [], []
  for h in heads:
    G = x["G"][h]
    g_last = G[C - 1 : C]
    s0.append(h_ref[0, 0, h])  # bf16, exact
    w.append(jnp.where(first, wu[h][:, :K], 0.0).astype(jnp.bfloat16))
    u.append(wu[h][:, K:])
    qg.append(jnp.where(first, x["q"][h] * g_exp[h] * scale, 0.0))
    kd = jnp.exp2(g_last - G)
    kdec.append(kd)
    kg.append(jnp.where(last, x["k"][h] * kd, 0.0))
    dec.append(jnp.where(masks["carry"], jnp.exp2(g_last), 0.0))  # [1, K]
  v_new = [u[h] - _dot(w[h], s0[h]) for h in heads]  # both exactly bf16

  # --- inter-chunk backward ---
  ds1 = [ds_ref[h] for h in heads]  # dL/dS after chunk
  do = [do_ref[0, :, h * V : (h + 1) * V] for h in heads]  # bf16
  dvn = [_dot(at["a_qk"][h], do[h], _TN) + _dot3(kg[h], ds1[h]) for h in heads]
  ds0 = [
      _dot(qg[h], do[h], _TN) + dec[h].T * ds1[h] - _dot_s2(w[h], dvn[h], _TN)
      for h in heads
  ]
  for h in heads:
    ds_ref[h] = jnp.where(reset, 0.0, ds0[h])
  if has_init:

    @pl.when(c == N - 1)
    def _():
      for h in heads:
        r["dinit"][0, h] = ds0[h]

  da_qk = [
      jnp.where(masks["qk_keep"], _dot(do[h], v_new[h], _NT), 0.0)
      for h in heads
  ]
  dqg = [jnp.where(first, _dot(do[h], s0[h], _NT), 0.0) for h in heads]
  dkg = [jnp.where(last, _dot(v_new[h], ds1[h], _NT), 0.0) for h in heads]
  dw = [
      jnp.where(first, -_dot_s2(dvn[h], s0[h], _NT, split_a=True), 0.0)
      for h in heads
  ]
  # d decay(G_C) = sum_v dS1 * S0 (row-sum over V, as a [1, K] row).
  ddec = [
      jnp.sum(ds1[h] * s0[h].astype(jnp.float32), axis=1, keepdims=True).T
      for h in heads
  ]

  # --- WY backward: W = T R_w, U = T R_u, T = (I + A_kk)^-1 ---
  dwu = [jnp.concatenate([dw[h], dvn[h]], 1) for h in heads]
  dr = [_dot(T[h], dwu[h], _TN) for h in heads]  # T^T [dW | dU]
  dT = [_dot(dwu[h], jnp.concatenate([r_w[h], r_u[h]], 1), _NT) for h in heads]
  m1 = [_dot(T[h], dT[h], _TN) for h in heads]
  da_kk = [
      jnp.where(masks["kk_keep"], -_dot(m1[h], T[h], _NT), 0.0) for h in heads
  ]

  # --- gated products A_qk (x = q, y = k) and A_kk (x = k, y = k) ---
  # dL/dG gets +x*dx at row i and -y*dy at row j for every pair (i, j); after
  # the reverse cumsum these cancel for all t <= j, so both sides must round
  # alike: dB is rounded to bf16 once and shared, and the decay-factor operands
  # are split in two bf16 parts. Single-pass bf16 on both operands leaves gate
  # gradients with ~8% (default gates) relative error and d dt_bias with ~50%.
  dq_x, dk_x, dk_y = [], [], []
  dB = []
  for h in heads:
    dBq = da_qk[h] * scale
    dBk = da_kk[h] * x["beta"][h]
    dB.append(
        [
            jnp.concatenate(
                [
                    dBq[i * _SUB : (i + 1) * _SUB],
                    dBk[i * _SUB : (i + 1) * _SUB],
                ],
                0,
            ).astype(jnp.bfloat16)
            for i in range(NB)
        ]
    )
  dxx = [[_dot_s2(dB[h][i], at["rhs"][h][i]) for i in range(NB)] for h in heads]
  dyy = [
      [_dot_s2(dB[h][i], at["lhs"][h][i], _TN) for i in range(NB)]
      for h in heads
  ]
  blk = lambda a, j: a[j * _SUB : (j + 1) * _SUB]
  for h in heads:
    e = at["e_row"][h]
    dq_x.append(e * jnp.concatenate([t[:_SUB] for t in dxx[h]], 0))
    dk_x.append(e * jnp.concatenate([t[_SUB:] for t in dxx[h]], 0))
    # Column block j collects rhs[i]'s factor from every row block i >= j.
    rows = []
    for j in range(NB):
      acc = blk(at["diag"][h], j) * blk(dyy[h][j], j)
      if j + 1 < NB:
        off_sum = at["vec"][h][j + 1][j] * blk(dyy[h][j + 1], j)
        for i in range(j + 2, NB):
          off_sum = off_sum + at["vec"][h][i][j] * blk(dyy[h][i], j)
        acc = acc + blk(at["off"][h], j) * off_sum
      rows.append(acc)
    dk_y.append(jnp.concatenate(rows, 0))

  # --- assemble per-token gradients ---
  dG = []
  for h in heads:
    q_, k_, v_, beta = x["q"][h], x["k"][h], x["v"][h], x["beta"][h]
    drw, dru = dr[h][:, :K], dr[h][:, K:]
    dqn = dq_x[h] + scale * g_exp[h] * dqg[h]
    dkn = dk_x[h] + dk_y[h] + beta * g_exp[h] * drw + kdec[h] * dkg[h]
    dvv = beta * dru
    if K == V:
      d_r = jnp.sum(drw * k_ * g_exp[h] + dru * v_, axis=1, keepdims=True)
    else:
      d_r = jnp.sum(drw * k_ * g_exp[h], axis=1, keepdims=True) + jnp.sum(
          dru * v_, axis=1, keepdims=True
      )
    dbeta = jnp.sum(da_kk[h] * at["a_kk_raw"][h], axis=1, keepdims=True) + d_r
    # dL/dG (without the ln 2 that the gate's log2(e) cancels).
    kgf = kg[h]
    g_col = (
        q_ * dq_x[h]
        + k_ * (dk_x[h] - dk_y[h])
        + r_w[h] * drw
        + qg[h] * dqg[h]
        - kgf * dkg[h]
    )
    g_last_row = jnp.sum(kgf * dkg[h], axis=0, keepdims=True) + dec[h] * ddec[h]
    rows = jax.lax.broadcasted_iota(jnp.int32, (C, 1), 0)
    dG.append(g_col + jnp.where(rows == C - 1, g_last_row, 0.0))
    # L2-norm backward.
    dq = x["rq"][h] * (dqn - q_ * jnp.sum(q_ * dqn, axis=1, keepdims=True))
    dk = x["rk"][h] * (dkn - k_ * jnp.sum(k_ * dkn, axis=1, keepdims=True))
    dq_ref[0, :, h * K : (h + 1) * K] = dq.astype(dq_ref.dtype)
    dk_ref[0, :, h * K : (h + 1) * K] = dk.astype(dk_ref.dtype)
    dv_ref[0, :, h * V : (h + 1) * V] = dvv.astype(dv_ref.dtype)
    dbeta_ref[0, 0, :, h : h + 1] = dbeta.astype(dbeta_ref.dtype)
  dg2 = _prefix_rows(dG, reverse=True)
  for h in heads:
    kl = slice(h * K, (h + 1) * K)
    if log_gate:
      # Natural-log decay gradient; zero where the clamp to `lower` is active.
      dg_ref[0, :, kl] = jnp.where(x["z"][h] >= lower, dg2[h], 0.0).astype(
          dg_ref.dtype
      )
      continue
    sig = x["sig"][h]
    dz = dg2[h] * (lower * sig * (1.0 - sig))
    if valid is not None:
      dz = jnp.where(valid, dz, 0.0)
    dec_k = decay_ref[:, kl]
    draw = dz * dec_k
    dg_ref[0, :, kl] = draw.astype(dg_ref.dtype)
    r["dbias"][0, :, kl] += jnp.sum(draw, axis=0, keepdims=True)
    r["dalog"][0, :, kl] += jnp.sum(draw * x["z"][h], axis=0, keepdims=True)


# --- kernel wrappers (kernel layout: [B, S, H * K], S % chunk == 0) ---------

# Largest |log decay| per token: the diagonal sub-block factors reach
# 2^(_SUB / 2 * |g| * log2(e)) and the backward multiplies them by dA, so this
# keeps ~2^35 headroom below the fp32 range (|g| = 8: 2^92).
MAX_ABS_LOG_DECAY = 8.0
CHUNK_SIZES = (16, 32, 64, 128)


def _heads_per_step(H: int, cap: int) -> int:
  m = min(H, cap)
  while H % m:
    m -= 1
  return m


def _default_heads_per_step(H, K, V, C):
  # 32 heads of K = V = 128 at chunk 64 fit the 64 MiB VMEM limit (stage-major
  # temporaries scale with K * V and C).
  cap = max(1, (32 * 128 * 128 * 64) // (K * V * max(C, 64)))
  return _heads_per_step(H, cap)


def _prep(beta_BSH, segment_ids_BS, a_log_H, dt_bias_HK, *, C, M, K):
  B, S, H = beta_BSH.shape
  N = S // C
  seg = segment_ids_BS.astype(jnp.int32)
  starts = seg[:, ::C]
  prev = jnp.concatenate(
      [jnp.full((B, 1), -1, jnp.int32), seg[:, C - 1 :: C][:, :-1]], axis=1
  )
  p = dict(
      reset=(starts != prev).astype(jnp.int32),
      beta=beta_BSH.reshape(B, S, H // M, M).transpose(0, 2, 1, 3),
      seg_col=seg[:, :, None],
      seg_row=seg.reshape(B * N, 1, C),
  )
  if a_log_H is not None:
    p["decay"] = jnp.repeat(jnp.exp(a_log_H.astype(jnp.float32)), K)[None]
    p["bias"] = dt_bias_HK.astype(jnp.float32).reshape(1, H * K)
  return p


def _common_specs(C, M, K, V, N, *, reverse, log_gate):
  if reverse:
    ch = lambda c: N - 1 - c
  else:
    ch = lambda c: c
  tok = lambda d: pl.BlockSpec((1, C, M * d), lambda b, g, c, *_: (b, ch(c), g))
  per_head = lambda d: pl.BlockSpec((1, M * d), lambda b, g, c, *_: (0, g))
  specs = [
      tok(K),
      tok(K),
      tok(V),
      tok(K),
      pl.BlockSpec((1, 1, C, M), lambda b, g, c, *_: (b, g, ch(c), 0)),
      pl.BlockSpec((1, C, 1), lambda b, g, c, *_: (b, ch(c), 0)),
      pl.BlockSpec((1, 1, C), lambda b, g, c, *_: (b * N + ch(c), 0, 0)),
  ]
  if not log_gate:
    specs += [per_head(K), per_head(K)]
  state = pl.BlockSpec((1, 1, M, K, V), lambda b, g, c, *_: (b, ch(c), g, 0, 0))
  return specs, tok, state


def _row_state_spec(M, K, V):
  """``[B, H, K, V]`` state of one row and head group (resident over chunks)."""
  return pl.BlockSpec((1, M, K, V), lambda b, g, c, *_: (b, g, 0, 0))


def _t_spec(C, M, N=None, *, reverse=False):
  ch = (lambda c: N - 1 - c) if reverse else (lambda c: c)
  return pl.BlockSpec((1, 1, M, C, C), lambda b, g, c, *_: (b, ch(c), g, 0, 0))


def _check(K, V, S, C, lower_bound):
  if K % 128 or V % 128:
    raise ValueError(f"head dims must be multiples of 128, got {K=} {V=}")
  if S % C or C not in CHUNK_SIZES:
    raise ValueError(
        f"need S % chunk == 0 and chunk in {CHUNK_SIZES}: {S=} {C=}"
    )
  if not -MAX_ABS_LOG_DECAY <= lower_bound < 0:
    raise ValueError(
        f"lower_bound={lower_bound} must be in [-{MAX_ABS_LOG_DECAY}, 0)"
    )


def kda_fused_fwd(
    q_BSC,
    k_BSC,
    v_BSC,
    g_BSC,
    a_log_H,
    dt_bias_HK,
    beta_BSH,
    segment_ids_BS,
    *,
    lower_bound,
    chunk=64,
    heads_per_step=None,
    store_state=False,
    scale=None,
    initial_state_BHKV=None,
    output_final_state=False,
    valid_len=None,
):
  """KDA forward in the kernel layout ``[B, S, H * K]``.

  ``a_log_H`` / ``dt_bias_HK`` given: ``g_BSC`` is K3's raw gate; both None:
  ``g_BSC`` is the natural-log decay, clamped to ``>= lower_bound``. Tokens at
  positions ``>= valid_len`` (default ``S``) are tail padding: decay 1 (they
  must have zero q / k / v). Returns ``(o_BSC, residuals, final_state)``:
  ``residuals`` with ``store_state`` are the per-chunk input states ``[B, N, H,
  K, V]`` (bf16) and ``(I + A_kk)^-1`` ``[B, N, H, C, C]`` (fp32), else None;
  ``final_state`` ``[B, H, K, V]`` fp32 with ``output_final_state``, else None.
  """
  B, S, H = beta_BSH.shape
  K = q_BSC.shape[-1] // H
  V = v_BSC.shape[-1] // H
  C = chunk
  _check(K, V, S, C, lower_bound)
  N = S // C
  M = heads_per_step or _default_heads_per_step(H, K, V, C)
  if H % M:
    raise ValueError(f"heads_per_step={M} must divide H={H}")
  log_gate = a_log_H is None
  has_init = initial_state_BHKV is not None
  p = _prep(beta_BSH, segment_ids_BS, a_log_H, dt_bias_HK, C=C, M=M, K=K)
  in_specs, tok, state_spec = _common_specs(
      C, M, K, V, N, reverse=False, log_gate=log_gate
  )
  args = [
      p["reset"],
      q_BSC,
      k_BSC,
      v_BSC,
      g_BSC,
      p["beta"],
      p["seg_col"],
      p["seg_row"],
  ]
  if not log_gate:
    args += [p["decay"], p["bias"]]
  if has_init:
    in_specs.append(_row_state_spec(M, K, V))
    args.append(initial_state_BHKV.astype(jnp.float32))
  out_specs = [tok(V)]
  out_shape = [jax.ShapeDtypeStruct((B, S, H * V), q_BSC.dtype)]
  if store_state:
    out_specs += [state_spec, _t_spec(C, M)]
    out_shape += [
        jax.ShapeDtypeStruct((B, N, H, K, V), jnp.bfloat16),
        jax.ShapeDtypeStruct((B, N, H, C, C), jnp.float32),
    ]
  if output_final_state:
    out_specs.append(_row_state_spec(M, K, V))
    out_shape.append(jax.ShapeDtypeStruct((B, H, K, V), jnp.float32))
  outs = pl.pallas_call(
      functools.partial(
          _fwd_kernel,
          M=M,
          K=K,
          V=V,
          scale=K**-0.5 if scale is None else scale,
          lower=lower_bound,
          store_state=store_state,
          log_gate=log_gate,
          has_init=has_init,
          has_final=output_final_state,
          valid_len=S if valid_len is None else valid_len,
          N=N,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(B, H // M, N),
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=[pltpu.VMEM((M, K, V), jnp.float32)],
      ),
      out_shape=out_shape,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=64 * 1024 * 1024,
      ),
      name="kda_fused_fwd",
  )(*args)
  residuals = tuple(outs[1:3]) if store_state else None
  final = outs[-1] if output_final_state else None
  return outs[0], residuals, final


def kda_fused_bwd(
    q_BSC,
    k_BSC,
    v_BSC,
    g_BSC,
    a_log_H,
    dt_bias_HK,
    beta_BSH,
    segment_ids_BS,
    residuals,
    do_BSC,
    *,
    lower_bound,
    chunk=64,
    heads_per_step=None,
    scale=None,
    has_initial_state=False,
    dfinal_state_BHKV=None,
    valid_len=None,
):
  """Gradients ``(dq, dk, dv, dg, d a_log, d dt_bias, d beta, d init)``.

  ``d a_log`` / ``d dt_bias`` are None with a per-head log-decay ``g``; ``d init`` (the
  initial state's gradient) is None unless ``has_initial_state``. ``dfinal_state_BHKV``:
  the final state's cotangent, or None (zero).
  """
  B, S, H = beta_BSH.shape
  K = q_BSC.shape[-1] // H
  V = v_BSC.shape[-1] // H
  C = chunk
  _check(K, V, S, C, lower_bound)
  N = S // C
  M = heads_per_step or _default_heads_per_step(H, K, V, C)
  if H % M:
    raise ValueError(f"heads_per_step={M} must divide H={H}")
  log_gate = a_log_H is None
  has_final = dfinal_state_BHKV is not None
  p = _prep(beta_BSH, segment_ids_BS, a_log_H, dt_bias_HK, C=C, M=M, K=K)
  in_specs, tok, state_spec = _common_specs(
      C, M, K, V, N, reverse=True, log_gate=log_gate
  )
  in_specs += [state_spec, _t_spec(C, M, N, reverse=True), tok(V)]
  args = [
      p["reset"],
      q_BSC,
      k_BSC,
      v_BSC,
      g_BSC,
      p["beta"],
      p["seg_col"],
      p["seg_row"],
  ]
  if not log_gate:
    args += [p["decay"], p["bias"]]
  args += [*residuals, do_BSC]
  if has_final:
    in_specs.append(_row_state_spec(M, K, V))
    args.append(dfinal_state_BHKV.astype(jnp.float32))
  acc = pl.BlockSpec((1, 1, M * K), lambda b, g, c, *_: (b, 0, g))
  beta_spec = in_specs[4]
  out_specs = [tok(K), tok(K), tok(V), tok(K), beta_spec]
  out_shape = [
      jax.ShapeDtypeStruct((B, S, H * K), q_BSC.dtype),
      jax.ShapeDtypeStruct((B, S, H * K), k_BSC.dtype),
      jax.ShapeDtypeStruct((B, S, H * V), v_BSC.dtype),
      jax.ShapeDtypeStruct((B, S, H * K), g_BSC.dtype),
      jax.ShapeDtypeStruct(p["beta"].shape, jnp.float32),
  ]
  if not log_gate:
    out_specs += [acc, acc]
    out_shape += [jax.ShapeDtypeStruct((B, 1, H * K), jnp.float32)] * 2
  if has_initial_state:
    out_specs.append(_row_state_spec(M, K, V))
    out_shape.append(jax.ShapeDtypeStruct((B, H, K, V), jnp.float32))
  outs = pl.pallas_call(
      functools.partial(
          _bwd_kernel,
          M=M,
          K=K,
          V=V,
          scale=K**-0.5 if scale is None else scale,
          lower=lower_bound,
          N=N,
          log_gate=log_gate,
          has_init=has_initial_state,
          has_final=has_final,
          valid_len=S if valid_len is None else valid_len,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(B, H // M, N),
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=[pltpu.VMEM((M, K, V), jnp.float32)],
      ),
      out_shape=out_shape,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=64 * 1024 * 1024,
      ),
      name="kda_fused_bwd",
  )(*args)
  dq, dk, dv, dg, dbeta = outs[:5]
  dbeta_BSH = (
      dbeta.transpose(0, 2, 1, 3).reshape(B, S, H).astype(beta_BSH.dtype)
  )
  dalog_H = dbias_HK = None
  if not log_gate:
    dbias, dalog = outs[5:7]
    dbias_HK = (
        jnp.sum(dbias, axis=(0, 1)).reshape(H, K).astype(dt_bias_HK.dtype)
    )
    # d a_log = sum dz * z * exp(a_log) = sum draw * z (draw = dz * exp(a_log)).
    dalog_H = (
        jnp.sum(dalog, axis=(0, 1)).reshape(H, K).sum(-1).astype(a_log_H.dtype)
    )
  dinit = outs[-1] if has_initial_state else None
  return dq, dk, dv, dg, dalog_H, dbias_HK, dbeta_BSH, dinit


# --- public JAX API (model layout [B, S, H, K]) -------------------------------


def _round_up(x, m):
  return -(-x // m) * m


def _pad(x, widths, **kw):
  """``jnp.pad`` with trailing widths, skipped when there is nothing to pad."""
  widths = tuple(widths) + ((0, 0),) * (x.ndim - len(widths))
  if all(w == (0, 0) for w in widths):
    return x
  return jnp.pad(x, widths, **kw)


def _dtype_name(x):
  """``bfloat16`` / ``float32`` / ``int32`` ... for JAX and torch arrays alike
  (validation must not trace JAX inside ``torch.compile``)."""
  return str(x.dtype).rsplit(".", 1)[-1]


def _validate(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    segment_ids,
    initial_state,
    lower_bound,
    chunk,
):
  """Shape / dtype checks on JAX arrays, torch tensors or ShapeDtypeStructs."""
  q_shape, k_shape = tuple(q.shape), tuple(k.shape)
  if len(q_shape) != 4 or k_shape != q_shape:
    raise ValueError(
        f"q, k must be [B, S, Hk, K] and equal, got {q_shape} {k_shape}"
    )
  B, S, Hk, K = q_shape
  if len(v.shape) != 4 or tuple(v.shape[:2]) != (B, S) or v.shape[2] % Hk:
    raise ValueError(
        f"v must be [B, S, H, V] with H a multiple of q's {Hk} heads, got"
        f" {v.shape}"
    )
  H, V = v.shape[2], v.shape[3]
  if not (0 < K <= 256 and 0 < V <= 256):
    raise ValueError(f"head dims must be in [1, 256], got K={K} V={V}")
  if tuple(beta.shape) != (B, S, H):
    raise ValueError(f"beta must be [B, S, H] = {(B, S, H)}, got {beta.shape}")
  for n, x in (("q", q), ("k", k), ("v", v)):
    if _dtype_name(x) not in ("bfloat16", "float32"):
      raise ValueError(f"{n} must be bf16 or fp32, got {x.dtype}")
  if not (_dtype_name(q) == _dtype_name(k) == _dtype_name(v)):
    raise ValueError(f"q, k, v dtypes differ: {q.dtype} {k.dtype} {v.dtype}")
  for n, x in (("g", g), ("beta", beta)):
    if _dtype_name(x) not in ("bfloat16", "float16", "float32", "float64"):
      raise ValueError(f"{n} must be floating, got {x.dtype}")
  if (a_log is None) != (dt_bias is None):
    raise ValueError(
        "a_log and dt_bias must be given together (K3 gate) or not at all"
    )
  if a_log is None and tuple(g.shape) != (B, S, H):
    raise ValueError(
        f"g must be the per-head log decay [B, S, H] = {(B, S, H)} without"
        f" a_log / dt_bias, got {g.shape}"
    )
  if a_log is not None:
    if tuple(g.shape) != (B, S, H, K):
      raise ValueError(
          "the in-kernel gate needs a per-channel raw gate [B, S, H, K]"
      )
    if tuple(a_log.shape) != (H,) or tuple(dt_bias.shape) != (H, K):
      raise ValueError(
          f"a_log must be [H], dt_bias [H, K], got {a_log.shape}"
          f" {dt_bias.shape}"
      )
  if segment_ids is not None and (
      tuple(segment_ids.shape) != (B, S)
      or not _dtype_name(segment_ids).startswith(("int", "uint"))
  ):
    raise ValueError(
        f"segment_ids must be integer [B, S], got {segment_ids.shape}"
        f" {segment_ids.dtype}"
    )
  if initial_state is not None and tuple(initial_state.shape) != (B, H, K, V):
    raise ValueError(
        f"initial_state must be [B, H, K, V] = {(B, H, K, V)}, got"
        f" {initial_state.shape}"
    )
  if chunk not in CHUNK_SIZES:
    raise ValueError(f"chunk must be one of {CHUNK_SIZES}, got {chunk}")
  if not -MAX_ABS_LOG_DECAY <= lower_bound < 0:
    raise ValueError(
        f"lower_bound must be in [-{MAX_ABS_LOG_DECAY}, 0), got {lower_bound}"
    )


def _to_kernel_layout(
    q, k, v, g, beta, a_log, dt_bias, segment_ids, initial_state, *, H, K, chunk
):
  """Flat ``[B, S, heads * dim]`` inputs -> the kernel layout: repeats grouped
  q / k heads, broadcasts a per-head decay, pads head dims to 128 lanes and S
  to the chunk. Inputs that need none of this pass through untouched (a
  ``[B, S, H, K]`` <-> ``[B, S, H * K]`` reshape is a relayout copy on TPU)."""
  B, S = q.shape[:2]
  Hk, V = q.shape[-1] // K, v.shape[-1] // H
  Kp, Vp, Sp = _round_up(K, 128), _round_up(V, 128), _round_up(S, chunk)
  if segment_ids is None:
    segment_ids = jnp.zeros((B, S), jnp.int32)
  per_head_g = g.shape[-1] != H * K
  if (Hk, Kp, Vp, Sp) == (H, K, V, S) and not per_head_g:
    if initial_state is not None:
      initial_state = initial_state.astype(jnp.float32)
    return q, k, v, g, beta, a_log, dt_bias, segment_ids, initial_state
  q, k = q.reshape(B, S, Hk, K), k.reshape(B, S, Hk, K)
  v = v.reshape(B, S, H, V)
  if Hk != H:  # grouped q / k heads: value head h reads key head h // (H / Hk)
    q, k = jnp.repeat(q, H // Hk, axis=2), jnp.repeat(k, H // Hk, axis=2)
  if per_head_g:  # one decay per head (Gated DeltaNet): broadcast over K
    g = jnp.broadcast_to(g[..., None].astype(jnp.float32), (B, S, H, K))
  else:
    g = g.reshape(B, S, H, K)
  pad4 = lambda x, d, dp: _pad(x, ((0, 0), (0, Sp - S), (0, 0), (0, dp - d)))
  q, k, g = pad4(q, K, Kp), pad4(k, K, Kp), pad4(g, K, Kp)
  v = pad4(v, V, Vp)
  beta = _pad(beta, ((0, 0), (0, Sp - S)))
  # Padding tokens continue the row's last document (decay 1, zero k / v).
  segment_ids = _pad(
      segment_ids.astype(jnp.int32), ((0, 0), (0, Sp - S)), mode="edge"
  )
  if dt_bias is not None:
    dt_bias = _pad(dt_bias, ((0, 0), (0, Kp - K)))
  if initial_state is not None:
    initial_state = _pad(
        initial_state.astype(jnp.float32),
        ((0, 0), (0, 0), (0, Kp - K), (0, Vp - V)),
    )
  flat = lambda x: x.reshape(B, Sp, -1)
  return (
      flat(q),
      flat(k),
      flat(v),
      flat(g),
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
  )


def _kda_fwd_impl(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    segment_ids,
    initial_state,
    *,
    H,
    K,
    lower_bound,
    scale,
    chunk,
    output_final_state,
    store_state,
):
  """Flat ``[B, S, heads * dim]`` in and out (``H`` value heads, key dim K)."""
  B, S = q.shape[:2]
  V = v.shape[-1] // H
  kq, kk, kv, kg, kb, ka, kd, ks, ki = _to_kernel_layout(
      q,
      k,
      v,
      g,
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
      H=H,
      K=K,
      chunk=chunk,
  )
  o, res, final = kda_fused_fwd(
      kq,
      kk,
      kv,
      kg,
      ka,
      kd,
      kb,
      ks,
      lower_bound=lower_bound,
      chunk=chunk,
      store_state=store_state,
      scale=K**-0.5 if scale is None else scale,
      initial_state_BHKV=ki,
      output_final_state=output_final_state,
      valid_len=S,
  )
  Sp, Vp = o.shape[1], o.shape[-1] // H
  if (Sp, Vp) != (S, V):
    o = o.reshape(B, Sp, H, Vp)[:, :S, :, :V].reshape(B, S, H * V)
  if final is not None and final.shape[2:] != (K, V):
    final = final[:, :, :K, :V]
  return o, final, res


def _kda_bwd_impl(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    segment_ids,
    initial_state,
    res,
    do,
    dfinal,
    *,
    H,
    K,
    lower_bound,
    scale,
    chunk,
):
  """Gradients in the (flat) shapes of the inputs."""
  B, S = q.shape[:2]
  Hk, V = q.shape[-1] // K, v.shape[-1] // H
  kq, kk, kv, kg, kb, ka, kd, ks, ki = _to_kernel_layout(
      q,
      k,
      v,
      g,
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
      H=H,
      K=K,
      chunk=chunk,
  )
  Sp = kq.shape[1]
  Kp, Vp = kq.shape[-1] // H, kv.shape[-1] // H
  do = do.astype(q.dtype)
  if (Sp, Vp) != (S, V):
    do = _pad(
        do.reshape(B, S, H, V), ((0, 0), (0, Sp - S), (0, 0), (0, Vp - V))
    )
    do = do.reshape(B, Sp, H * Vp)
  if dfinal is not None:
    dfinal = _pad(
        dfinal.astype(jnp.float32), ((0, 0), (0, 0), (0, Kp - K), (0, Vp - V))
    )
  dq, dk, dv, dg, dalog, dbias, dbeta, dinit = kda_fused_bwd(
      kq,
      kk,
      kv,
      kg,
      ka,
      kd,
      kb,
      ks,
      res,
      do,
      lower_bound=lower_bound,
      chunk=chunk,
      scale=K**-0.5 if scale is None else scale,
      has_initial_state=initial_state is not None,
      dfinal_state_BHKV=dfinal,
      valid_len=S,
  )
  if kq is not q:  # undo _to_kernel_layout
    unk = lambda x, d: x.reshape(B, Sp, H, -1)[:, :S, :, :d]
    dq, dk, dv, dg = unk(dq, K), unk(dk, K), unk(dv, V), unk(dg, K)
    if Hk != H:  # sum the gradients of each key head's group
      group = lambda x: jnp.sum(
          x.astype(jnp.float32).reshape(B, S, Hk, H // Hk, K), axis=3
      ).astype(x.dtype)
      dq, dk = group(dq), group(dk)
    if g.shape[-1] != H * K:
      dg = jnp.sum(dg.astype(jnp.float32), axis=-1)
    flat = lambda x: x.reshape(B, S, -1)
    dq, dk, dv = flat(dq), flat(dk), flat(dv)
    dg = dg.reshape(g.shape).astype(g.dtype)
    dbeta = dbeta[:, :S]
    if dbias is not None:
      dbias = dbias[:, :K]
    if dinit is not None:
      dinit = dinit[:, :, :K, :V]
  if dinit is not None:
    dinit = dinit.astype(initial_state.dtype)
  return dq, dk, dv, dg, dbeta, dalog, dbias, dinit


@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14))
def _kda_core(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    segment_ids,
    initial_state,
    H,
    K,
    lower_bound,
    scale,
    chunk,
    output_final_state,
):
  o, final, _ = _kda_fwd_impl(
      q,
      k,
      v,
      g,
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
      H=H,
      K=K,
      lower_bound=lower_bound,
      scale=scale,
      chunk=chunk,
      output_final_state=output_final_state,
      store_state=False,
  )
  return o, final


def _core_fwd(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    segment_ids,
    initial_state,
    H,
    K,
    lower_bound,
    scale,
    chunk,
    output_final_state,
):
  o, final, res = _kda_fwd_impl(
      q,
      k,
      v,
      g,
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
      H=H,
      K=K,
      lower_bound=lower_bound,
      scale=scale,
      chunk=chunk,
      output_final_state=output_final_state,
      store_state=True,
  )
  return (o, final), (
      q,
      k,
      v,
      g,
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
      res,
  )


def _core_bwd(H, K, lower_bound, scale, chunk, output_final_state, saved, cot):
  q, k, v, g, beta, a_log, dt_bias, segment_ids, initial_state, res = saved
  do, dfinal = cot
  dq, dk, dv, dg, dbeta, dalog, dbias, dinit = _kda_bwd_impl(
      q,
      k,
      v,
      g,
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
      res,
      do,
      dfinal if output_final_state else None,
      H=H,
      K=K,
      lower_bound=lower_bound,
      scale=scale,
      chunk=chunk,
  )
  return dq, dk, dv, dg, dbeta, dalog, dbias, None, dinit


_kda_core.defvjp(_core_fwd, _core_bwd)


def kda(
    q,
    k,
    v,
    g,
    beta,
    *,
    lower_bound,
    a_log=None,
    dt_bias=None,
    segment_ids=None,
    initial_state=None,
    output_final_state=False,
    scale=None,
    chunk=64,
):
  """Chunked (Kimi / Gated) delta attention with fused Pallas fwd and bwd.

  Per head, with ``q``, ``k`` L2-normalized in the kernel::

    S_t = (I - beta_t k_t k_t^T) diag(exp(g_t)) S_{t-1} + beta_t k_t v_t^T
    o_t = scale * S_t^T q_t

  Args:
    q, k: ``[B, S, Hk, K]``, bf16 or fp32 (matmuls run on bf16 operands);
      ``Hk`` divides ``H``: value head ``h`` uses q / k head ``h // (H / Hk)``.
    v: ``[B, S, H, V]``, same dtype.
    g: without ``a_log`` / ``dt_bias``, the per-token log decay of each head
      (natural log, ``<= 0``, Gated DeltaNet) ``[B, S, H]``, clamped to
      ``>= lower_bound``. With them (KDA), the raw gate ``[B, S, H, K]`` of
      ``g = lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``.
    beta: ``[B, S, H]`` delta-rule rate, typically in ``[0, 1]``.
    lower_bound: ``[-8, 0)``; the gate's lower bound or the log-decay clamp.
    a_log, dt_bias: ``[H]``, ``[H, K]`` for the in-kernel gate, or None.
    segment_ids: ``[B, S]`` int; contiguous runs of equal ids are documents
      (the state restarts at each). None: one sequence per row.
    initial_state: ``[B, H, K, V]`` state entering the first document of each
      row (fp32), or None (zero).
    output_final_state: also return the state after each row's last token.
    scale: query scale, default ``K ** -0.5``.
    chunk: chunk size, one of ``CHUNK_SIZES``.

  Returns:
    ``(o [B, S, H, V] in q's dtype, final_state [B, H, K, V] fp32 or None)``.
  """
  _validate(
      q,
      k,
      v,
      g,
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
      lower_bound,
      chunk,
  )
  B, S, _, K = q.shape
  H, V = v.shape[2:]
  flat = lambda x: x.reshape(B, S, -1)
  o, final = _kda_core(
      flat(q),
      flat(k),
      flat(v),
      g if g.ndim == 3 else flat(g),
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
      H,
      K,
      lower_bound,
      scale,
      chunk,
      output_final_state,
  )
  return o.reshape(B, S, H, V), final


# --- PyTorch entry point (torch_tpu jax_op) -----------------------------------
#
# Optional tensors are passed as None; ``jax_op`` reads operands by shape, so
# every tensor is made contiguous first. q / k / v / g / out cross the op
# boundary flat (``[B, S, heads * dim]``), the kernels' layout: reshaping a
# ``[B, S, H, K]`` JAX array is a relayout copy on TPU.


@pallas.jax_op("tt_tpu_kernels::kda_fwd")
def _kda_fwd_op(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    a_log: jax.Array | None,
    dt_bias: jax.Array | None,
    segment_ids: jax.Array,
    initial_state: jax.Array | None,
    lower_bound: float,
    scale: float,
    chunk: int,
    output_final_state: bool,
    num_heads: int,
    key_dim: int,
) -> list[jax.Array]:
  """``[out]`` or ``[out, final_state]``; saves no residuals."""
  o, final, _ = _kda_fwd_impl(
      q,
      k,
      v,
      g,
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
      H=num_heads,
      K=key_dim,
      lower_bound=lower_bound,
      scale=scale,
      chunk=chunk,
      output_final_state=output_final_state,
      store_state=False,
  )
  return [o] + ([final] if output_final_state else [])


@pallas.jax_op("tt_tpu_kernels::kda_fwd_res")
def _kda_fwd_res_op(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    a_log: jax.Array | None,
    dt_bias: jax.Array | None,
    segment_ids: jax.Array,
    initial_state: jax.Array | None,
    lower_bound: float,
    scale: float,
    chunk: int,
    output_final_state: bool,
    num_heads: int,
    key_dim: int,
) -> list[jax.Array]:
  """``[out, (final_state,) chunk states, (I + A_kk)^-1]``."""
  o, final, res = _kda_fwd_impl(
      q,
      k,
      v,
      g,
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
      H=num_heads,
      K=key_dim,
      lower_bound=lower_bound,
      scale=scale,
      chunk=chunk,
      output_final_state=output_final_state,
      store_state=True,
  )
  return [o] + ([final] if output_final_state else []) + list(res)


@pallas.jax_op("tt_tpu_kernels::kda_bwd")
def _kda_bwd_op(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    a_log: jax.Array | None,
    dt_bias: jax.Array | None,
    segment_ids: jax.Array,
    initial_state: jax.Array | None,
    chunk_states: jax.Array,
    wy_inverse: jax.Array,
    grad_out: jax.Array,
    grad_final_state: jax.Array | None,
    lower_bound: float,
    scale: float,
    chunk: int,
    num_heads: int,
    key_dim: int,
) -> list[jax.Array]:
  """``[dq, dk, dv, dg, dbeta, (d a_log, d dt_bias,) (d initial_state)]``."""
  dq, dk, dv, dg, dbeta, dalog, dbias, dinit = _kda_bwd_impl(
      q,
      k,
      v,
      g,
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
      (chunk_states, wy_inverse),
      grad_out,
      grad_final_state,
      H=num_heads,
      K=key_dim,
      lower_bound=lower_bound,
      scale=scale,
      chunk=chunk,
  )
  out = [dq, dk, dv, dg, dbeta]
  if dalog is not None:
    out += [dalog, dbias]
  if dinit is not None:
    out.append(dinit)
  return out


_NUM_TENSOR_INPUTS = (
    9  # q, k, v, g, beta, a_log, dt_bias, segment_ids, initial_state
)


def _setup_context(ctx, inputs, output):
  tensors = inputs[:_NUM_TENSOR_INPUTS]
  (
      ctx.lower_bound,
      ctx.scale,
      ctx.chunk,
      ctx.output_final_state,
      ctx.num_heads,
      ctx.key_dim,
  ) = inputs[_NUM_TENSOR_INPUTS:]
  n_out = 2 if ctx.output_final_state else 1
  ctx.mark_non_differentiable(*output[n_out:])
  ctx.set_materialize_grads(False)
  ctx.has = [t is not None for t in tensors]
  ctx.save_for_backward(*[t for t in tensors if t is not None], *output[n_out:])


def _backward(ctx, grad_outputs):
  saved = list(ctx.saved_tensors)
  tensors = [saved.pop(0) if h else None for h in ctx.has]
  chunk_states, wy_inverse = saved
  grad_out = grad_outputs[0]
  grad_final = grad_outputs[1] if ctx.output_final_state else None
  if grad_out is None and grad_final is None:
    return (None,) * (_NUM_TENSOR_INPUTS + 6)
  if grad_out is None:
    grad_out = torch.zeros(
        tensors[2].shape, dtype=tensors[0].dtype, device=tensors[0].device
    )
  grads = list(
      _kda_bwd_op(
          *tensors,
          chunk_states,
          wy_inverse,
          grad_out.contiguous(),
          None if grad_final is None else grad_final.contiguous(),
          ctx.lower_bound,
          ctx.scale,
          ctx.chunk,
          ctx.num_heads,
          ctx.key_dim,
      )
  )
  dq, dk, dv, dg, dbeta = grads[:5]
  rest = grads[5:]
  dalog = dbias = dinit = None
  if tensors[5] is not None:
    dalog, dbias = rest[:2]
    rest = rest[2:]
  if tensors[8] is not None:
    dinit = rest[0]
  return (dq, dk, dv, dg, dbeta, dalog, dbias, None, dinit) + (None,) * 6


_kda_fwd_res_op.register_autograd(_backward, setup_context=_setup_context)


def kda_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    lower_bound: float,
    a_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
  """PyTorch (TPU) version of ``kda``: same arguments, shapes and returns.

  ``a_log``, ``dt_bias``, ``beta`` and ``initial_state`` are used in fp32.
  Differentiable in every float input; without a gradient to compute
  (``no_grad`` or no input requiring grad) the forward saves no residuals.
  """
  if q.device.type != "tpu":
    raise ValueError(f"kda_torch runs on TPU only, got device {q.device.type}")
  _validate(
      q,
      k,
      v,
      g,
      beta,
      a_log,
      dt_bias,
      segment_ids,
      initial_state,
      lower_bound,
      chunk,
  )
  if segment_ids is None:
    segment_ids = torch.zeros(q.shape[:2], dtype=torch.int32, device=q.device)
  B, S, _, K = q.shape
  H, V = v.shape[2], v.shape[3]
  f32 = lambda t: None if t is None else t.float().contiguous()
  flat = lambda t: t.flatten(2).contiguous()
  inputs = [
      flat(q),
      flat(k),
      flat(v),
      flat(g) if g.ndim == 4 else g.contiguous(),
      f32(beta),
      f32(a_log),
      f32(dt_bias),
      segment_ids.to(torch.int32).contiguous(),
      f32(initial_state),
  ]
  statics = (
      float(lower_bound),
      float(K**-0.5 if scale is None else scale),
      int(chunk),
      bool(output_final_state),
      int(H),
      int(K),
  )
  needs_grad = torch.is_grad_enabled() and any(
      t is not None and t.requires_grad for t in inputs
  )
  op = _kda_fwd_res_op if needs_grad else _kda_fwd_op
  outs = op(*inputs, *statics)
  return outs[0].unflatten(-1, (H, V)), (
      outs[1] if output_final_state else None
  )
