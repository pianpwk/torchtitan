# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek MLA attention core on MaxText Lineage's fused FP8 q/kv up kernels.

``maxtext_mla_attention(q_down, kv_down, wq_b, wkv_b, q_norm_w, kv_norm_w,
cos, sin, segment_ids, ...)`` covers MLA from the compressed projections to
the Splash output:

  cq = rms_norm(q_down);  ckv, k_pe = split(kv_down)
  q = [cq @ wq_b] with q_pe roped, * softmax_scale
  k = [rms_norm(ckv) @ wk_b | rope(k_pe) broadcast to every head]
  v = rms_norm(ckv) @ wv_b
  o, lse = splash(q, k, v)  (causal, packed documents by segment_ids)

The q / kv up projections run in Lineage's fused kernel
(``maxtext_lineage_mla/mla_qkv_up_kernel.py``): the norms, the static FP8
quantization (``fp8_full``), the rope, the scale and the k_pe broadcast all
happen in VMEM, and q / k come out seq-minor ``[B, N, D, T]`` and v
``[B, N, T, V]``, the layouts Tokamax Splash takes (``SEQ_MINOR`` q / k), so
the concat, broadcast and transposes of the upstream path are gone. The
backward recomputes q, k and v with the same kernel (only q_down, kv_down,
o and lse are saved), runs Splash's backward, and then Lineage's fused
dgrad / wgrad kernels (``mla_qkv_up_bwd_kernels.py``).

Mirrors MaxText's Lineage DeepSeek-V3 ``qkv_up_from_down_fused`` /
``qkv_up_from_qkv_down_bwd`` (``maxtext/experimental/lineage/dsv3_mla.py``)
per device, without ``shard_map``: torch_tpu runs each device's part, and
heads are not sharded. Two differences:

* Splash is stock Tokamax (``kernels/tokamax_splash_attention.py``'s
  kernels, without its 1 / sqrt(D) q scaling, since the scale is folded into
  q here), not Lineage's Splash fork.
* RoPE: torchtitan's DeepSeek rope rotates interleaved pairs
  ``(x[2i], x[2i+1])``; Lineage's ``yarn`` rotates ``(x[i], x[i + R/2])``.
  The rope dims of ``wq_b``'s q_pe rows and of k_pe are de-interleaved before
  the kernel (and their gradients re-interleaved after). q_pe . k_pe is
  invariant under the same permutation of both, so this is exact.

Shape legend: B batch, T sequence, N heads, Cq q_lora_rank, Ckv
kv_lora_rank, R rope head dim, QK nope head dim, V v head dim.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal import pallas

from torchtitan.experiments.tpu.kernels import tokamax_splash_attention as tsa
from torchtitan.experiments.tpu.kernels.fp8_full import (
    FWD_DTYPE,
    STATIC_SCALE,
    static_quantize,
)
from torchtitan.experiments.tpu.kernels.maxtext_lineage_mla import (
    mla_qkv_up_bwd_kernels,
    mla_qkv_up_kernel,
)

_BWD_BLOCKS = (512, 8)  # Lineage's _QKV_UP_BWD_BLOCKS (token, head) tile.


# numpy, not jnp: index constants stay literals instead of captured jax.Arrays.
def _deinterleave(r: int) -> np.ndarray:
    return np.concatenate([np.arange(0, r, 2), np.arange(1, r, 2)])


def _interleave(r: int) -> np.ndarray:
    return np.argsort(_deinterleave(r))


def _rms_norm(x: jax.Array, w: jax.Array, eps: float) -> jax.Array:
    """bf16 RMSNorm computed in f32 (``mla_qkv_up_kernel._rms_norm``)."""
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps)
    return (y * w.astype(jnp.float32)).astype(jnp.bfloat16)


def _yarn(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Lineage ``dsv3_mla.yarn``: ``x * cos + rotate_half(x) * sin``."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return x * cos + jnp.concatenate([-x2, x1], axis=-1) * sin


def _kernel_weights(wq_b, wkv_b, *, n, qk, r, v):
    """``wq_b [N*(QK+R), Cq]`` / ``wkv_b [N*(QK+V), Ckv]`` -> e4m3, q_pe rows
    de-interleaved, as ``[N*D, C]`` (``heads_minor``) for the backward."""
    cq, ckv = wq_b.shape[1], wkv_b.shape[1]
    wq = wq_b.reshape(n, qk + r, cq)
    wq = jnp.concatenate([wq[:, :qk], wq[:, qk:][:, _deinterleave(r)]], axis=1)
    wkv = wkv_b.reshape(n, qk + v, ckv)
    return (
        static_quantize(wq.reshape(n * (qk + r), cq)),
        static_quantize(wkv[:, :qk].reshape(n * qk, ckv)),
        static_quantize(wkv[:, qk:].reshape(n * v, ckv)),
    )


def _heads_major(w_nd_c: jax.Array, n: int) -> jax.Array:
    """``[N*D, C] -> [C, N, D]`` (Lineage's ``heads_major``)."""
    return jnp.transpose(w_nd_c.reshape(n, -1, w_nd_c.shape[1]), (2, 0, 1))


def _qkv(q_down, kv_down, wq_q, wk_q, wv_q, q_norm_w, kv_norm_w, cos, sin, *,
         n, qk, r, ckv, softmax_scale, eps):
    """Fused q / kv up projections: q, k ``[B, N, QK+R, T]``, v ``[B, N, T, V]``."""
    kv = jnp.concatenate(
        [kv_down[..., :ckv], kv_down[..., ckv:][..., _deinterleave(r)]], -1)
    return mla_qkv_up_kernel.qkv_up_fwd(
        q_down,
        kv,
        _heads_major(wq_q, n),
        _heads_major(wk_q, n),
        _heads_major(wv_q, n),
        q_norm_w,
        kv_norm_w,
        (cos[:, :, None, :], sin[:, :, None, :]),
        qk_head_dim=qk,
        kv_lora_rank=ckv,
        softmax_scale=softmax_scale,
        norm_epsilon=eps,
        act_qtype=FWD_DTYPE,
        act_scale=STATIC_SCALE,
        weight_scale=STATIC_SCALE,
    )


def _splash(seq: int, block_q: int, block_kv: int, block_q_dkv: int,
            block_kv_dkv: int):
    """Stock Tokamax Splash with SEQ_MINOR q / k, as ``tsa`` builds it.

    Built inside each traced function (not cached): its mask-info arrays must
    not leak from one trace into another as captured constants."""
    kernel_lib = tsa.splash_attention_kernel
    config = kernel_lib.SplashConfig(
        block_q=min(block_q, seq),
        block_kv=min(block_kv, seq),
        block_kv_compute=min(block_kv, seq),
        block_q_dkv=min(block_q_dkv, seq),
        block_kv_dkv=min(block_kv_dkv, seq),
        block_kv_dkv_compute=min(block_kv_dkv, seq),
        use_fused_bwd_kernel=True,
        q_layout=kernel_lib.QKVLayout.SEQ_MINOR,
        k_layout=kernel_lib.QKVLayout.SEQ_MINOR,
        v_layout=kernel_lib.QKVLayout.HEAD_DIM_MINOR,
    )
    kernel = kernel_lib.make_splash_mha_single_device(
        mask=tsa._make_mask((seq, seq), True, None, 1),  # pylint: disable=protected-access
        config=config,
        q_seq_shards=1,
    )
    return kernel_lib, config, kernel


def _segment_ids(kernel_lib, seg_b, seq):
    return tsa.make_cp_segment_ids(kernel_lib.SegmentIds, seg_b, 1, seq, None)


def _splash_fwd(q, k, v, seg, blocks):
    """Per-example (vmapped) Splash forward on pre-scaled q.

    Tokamax takes logical ``[N, T, D]`` q / k and applies ``.mT`` itself for
    SEQ_MINOR, so the kernel's ``[N, D, T]`` are swapped here; XLA folds the
    two transposes."""
    seq = q.shape[-1]
    q, k = jnp.swapaxes(q, -1, -2), jnp.swapaxes(k, -1, -2)
    kernel_lib, config, kernel = _splash(seq, *blocks)
    kw = kernel.kwargs

    def one(q_b, k_b, v_b, seg_b):
        # pylint: disable=protected-access
        out_b, stats = kernel_lib._splash_attention_forward(
            kernel.fwd_mask_info, q_b, k_b, v_b,
            segment_ids=_segment_ids(kernel_lib, seg_b, seq),
            sinks=None,
            mask_value=kw.get("mask_value", tsa._DEFAULT_MASK_VALUE),
            is_mqa=False,
            config=config,
            save_residuals=True,
            mask_function=kw.get("mask_function", None),
            fwd_mask_sparsity=kw.get("fwd_mask_sparsity", 1.0),
        )
        return out_b, stats["logsumexp"]

    return jax.vmap(one)(q, k, v, seg)


def _splash_bwd(q, k, v, o, lse, do, seg, blocks):
    """Per-example (vmapped, not scanned) Splash backward (see _splash_fwd)."""
    seq = q.shape[-1]
    q, k = jnp.swapaxes(q, -1, -2), jnp.swapaxes(k, -1, -2)
    kernel_lib, config, kernel = _splash(seq, *blocks)
    kw = kernel.kwargs

    def one(q_b, k_b, v_b, o_b, lse_b, do_b, seg_b):
        res = (q_b, k_b, v_b, _segment_ids(kernel_lib, seg_b, seq), None, o_b,
               lse_b, kernel.dkv_mask_info, None)
        # pylint: disable=protected-access
        grads = kernel_lib._splash_attention_bwd(
            False,
            kw.get("mask_value", tsa._DEFAULT_MASK_VALUE),
            False,
            config,
            kw.get("mask_function", None),
            kw.get("fwd_mask_sparsity", 1.0),
            kw.get("dkv_mask_sparsity", 1.0),
            res,
            do_b,
        )
        return grads[2], grads[3], grads[4]

    return jax.vmap(one)(q, k, v, o, lse, do, seg)


def _head_dim_minor(x: jax.Array, d: int) -> jax.Array:
    """``[B, N, T, D]`` from a Splash gradient in either layout."""
    return x if x.shape[-1] == d else jnp.swapaxes(x, -1, -2)


def _jax_fwd(q_down: jax.Array, kv_down: jax.Array, wq_b: jax.Array,
             wkv_b: jax.Array, q_norm_w: jax.Array, kv_norm_w: jax.Array,
             cos: jax.Array, sin: jax.Array, seg: jax.Array, n: int, qk: int,
             r: int, v_dim: int, softmax_scale: float, eps: float,
             block_q: int, block_kv: int, block_q_dkv: int,
             block_kv_dkv: int) -> tuple[jax.Array, jax.Array]:
    ckv = wkv_b.shape[1]
    wq_q, wk_q, wv_q = _kernel_weights(wq_b, wkv_b, n=n, qk=qk, r=r, v=v_dim)
    q, k, v = _qkv(q_down, kv_down, wq_q, wk_q, wv_q, q_norm_w, kv_norm_w, cos,
                   sin, n=n, qk=qk, r=r, ckv=ckv, softmax_scale=softmax_scale,
                   eps=eps)
    o, lse = _splash_fwd(q, k, v, seg,
                         (block_q, block_kv, block_q_dkv, block_kv_dkv))
    return jnp.swapaxes(o, 1, 2), lse  # [B, T, N, V], [B, N, T]


def _jax_bwd(do: jax.Array, q_down: jax.Array, kv_down: jax.Array,
             wq_b: jax.Array, wkv_b: jax.Array, q_norm_w: jax.Array,
             kv_norm_w: jax.Array, cos: jax.Array, sin: jax.Array,
             seg: jax.Array, o: jax.Array, lse: jax.Array, n: int, qk: int,
             r: int, v_dim: int, softmax_scale: float, eps: float,
             block_q: int, block_kv: int, block_q_dkv: int, block_kv_dkv: int
             ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
                        jax.Array]:
    ckv = wkv_b.shape[1]
    d = qk + r
    wq_q, wk_q, wv_q = _kernel_weights(wq_b, wkv_b, n=n, qk=qk, r=r, v=v_dim)
    q, k, v = _qkv(q_down, kv_down, wq_q, wk_q, wv_q, q_norm_w, kv_norm_w, cos,
                   sin, n=n, qk=qk, r=r, ckv=ckv, softmax_scale=softmax_scale,
                   eps=eps)
    dq, dk, dv = _splash_bwd(
        q, k, v, jnp.swapaxes(o, 1, 2), lse, jnp.swapaxes(do, 1, 2), seg,
        (block_q, block_kv, block_q_dkv, block_kv_dkv))
    dq, dk, dv = (_head_dim_minor(dq, d), _head_dim_minor(dk, d),
                  _head_dim_minor(dv, v_dim))

    kv_perm = jnp.concatenate(
        [kv_down[..., :ckv], kv_down[..., ckv:][..., _deinterleave(r)]], -1)
    dims = mla_qkv_up_bwd_kernels.QkvUpBwdDims(
        qk_head_dim=qk, rope_head_dim=r, v_head_dim=v_dim,
        softmax_scale=softmax_scale, eps=eps, act_scale=STATIC_SCALE,
        act_qtype=FWD_DTYPE, weight_scale=STATIC_SCALE)
    t = dq.shape[2]
    kw = dict(dims=dims, block_t=min(_BWD_BLOCKS[0], t),
              block_n=min(_BWD_BLOCKS[1], n))
    cos_sin = jnp.concatenate([cos, sin], axis=-1)
    grad_cq, grad_kv, gq, dk_q, dv_q, scales = (
        mla_qkv_up_bwd_kernels.qkv_up_dgrad(
            dq, dk, dv, kv_perm, kv_norm_w.reshape(1, -1), cos_sin, wq_q, wk_q,
            wv_q, **kw))
    ckv_normed = _rms_norm(kv_perm[..., :ckv], kv_norm_w, eps)
    g_wq, g_wk, g_wv = mla_qkv_up_bwd_kernels.qkv_up_wgrad(
        q_down, ckv_normed, q_norm_w.reshape(1, -1), gq, dk_q, dv_q, scales,
        grad_dtype=wq_b.dtype, **kw)
    grad_cq = jnp.swapaxes(grad_cq, 1, 2)            # [B, T, Cq]
    grad_kv = jnp.swapaxes(grad_kv, 1, 2)            # [B, T, Ckv + R]
    grad_ckv, grad_k_pe = grad_kv[..., :ckv], grad_kv[..., ckv:]

    # q norm: the static act calibration clips cq; zero the clipped gradient
    # (Lineage's _ste), then the norm pullback.
    norm = functools.partial(_rms_norm, eps=eps)
    cq, q_norm_vjp = jax.vjp(norm, q_down, q_norm_w)
    act_max = STATIC_SCALE * float(jnp.finfo(FWD_DTYPE).max)
    grad_cq = jnp.where(
        jnp.abs(cq.astype(jnp.float32)) <= act_max, grad_cq, 0
    ).astype(grad_cq.dtype)
    grad_q_down, grad_q_norm_w = q_norm_vjp(grad_cq)
    # kv norm (the ckv STE is applied in qkv_up_dgrad) and the k_pe rope.
    _, kv_norm_vjp = jax.vjp(norm, kv_perm[..., :ckv], kv_norm_w)
    grad_ckv, grad_kv_norm_w = kv_norm_vjp(grad_ckv)
    _, rope_vjp = jax.vjp(lambda x: _yarn(x, cos, sin), kv_perm[..., ckv:])
    (grad_k_pe,) = rope_vjp(grad_k_pe.astype(jnp.float32))  # rope runs in f32
    grad_kv_down = jnp.concatenate(
        [grad_ckv, grad_k_pe[..., _interleave(r)]], -1).astype(kv_down.dtype)

    cq_dim = wq_b.shape[1]
    g_wq = g_wq.reshape(n, d, cq_dim)
    g_wq = jnp.concatenate([g_wq[:, :qk], g_wq[:, qk:][:, _interleave(r)]], 1)
    g_wkv = jnp.concatenate(
        [g_wk.reshape(n, qk, ckv), g_wv.reshape(n, v_dim, ckv)], axis=1)
    return (
        grad_q_down.astype(q_down.dtype),
        grad_kv_down,
        g_wq.reshape(wq_b.shape).astype(wq_b.dtype),
        g_wkv.reshape(wkv_b.shape).astype(wkv_b.dtype),
        grad_q_norm_w.astype(q_norm_w.dtype),
        grad_kv_norm_w.astype(kv_norm_w.dtype),
    )


maxtext_mla_attn_fwd = pallas.jax_op("pallas::maxtext_mla_attn_fwd", _jax_fwd)
maxtext_mla_attn_bwd = pallas.jax_op("pallas::maxtext_mla_attn_bwd", _jax_bwd)

_NUM_TENSOR_INPUTS = 9


def _setup_context(ctx, inputs, output, **kwargs):
    del kwargs
    ctx.save_for_backward(*inputs[:_NUM_TENSOR_INPUTS], *output)
    ctx.static = inputs[_NUM_TENSOR_INPUTS:]


def _backward(ctx, grad_o, grad_lse):
    del grad_lse
    saved = ctx.saved_tensors
    tensors, (o, lse) = saved[:_NUM_TENSOR_INPUTS], saved[_NUM_TENSOR_INPUTS:]
    grads = maxtext_mla_attn_bwd(grad_o.contiguous(), *tensors, o, lse,
                                 *ctx.static)
    return tuple(grads) + (None,) * (3 + len(ctx.static))


maxtext_mla_attn_fwd.register_autograd(_backward, setup_context=_setup_context)


def rope_half_tables(cis: torch.Tensor, batch: int, seq: int):
    """``(cos, sin) [B, T, R]`` f32 in ``rotate_half`` order from complex
    ``cis [B*T, R/2]``: ``cos = [c, c]``, ``sin = [s, s]``."""
    c, s = cis.real.float(), cis.imag.float()
    return (torch.cat([c, c], -1).view(batch, seq, -1),
            torch.cat([s, s], -1).view(batch, seq, -1))


def maxtext_mla_attention(
    q_down: torch.Tensor,
    kv_down: torch.Tensor,
    wq_b: torch.Tensor,
    wkv_b: torch.Tensor,
    q_norm_w: torch.Tensor,
    kv_norm_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    softmax_scale: float,
    eps: float,
    block_q: int = 1024,
    block_kv: int = 1024,
    block_q_dkv: int = 1024,
    block_kv_dkv: int = 1024,
) -> torch.Tensor:
    """Splash output ``[B, T, N, V]``; see the module docstring.

    Splash blocks: 2048 x 2048 forward blocks (Lineage's default) overflow the
    32 MiB scoped VMEM by 0.9 MB at q / k head dim 192 on stock Tokamax, so
    the defaults are 1024.
    """
    o, _ = maxtext_mla_attn_fwd(
        q_down, kv_down, wq_b, wkv_b, q_norm_w, kv_norm_w, cos, sin,
        segment_ids, num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
        float(softmax_scale), float(eps), block_q, block_kv, block_q_dkv,
        block_kv_dkv)
    return o


__all__ = ["maxtext_mla_attention", "rope_half_tables"]
