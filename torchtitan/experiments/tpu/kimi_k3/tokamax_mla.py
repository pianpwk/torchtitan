"""Kimi K3's gated MLA on TPU with Tokamax's Splash attention kernels.

Upstream ``torchtitan.models.kimi_k3.model.KimiMLAAttention`` runs its
attention with FlexAttention and a document-causal ``BlockMask``. The
``tokamax_mla_attention`` override swaps every ``KimiMLAAttention`` for
``TpuKimiMLAAttention``, which keeps upstream's projections, norms, output gate
and parameters and computes the attention with Tokamax Splash:

* ``TpuKimiMLAAttention``: ``KimiMLAAttention`` that turns ``positions`` into
  per-token document ids and hands them to its inner attention in place of the
  FlexAttention ``BlockMask`` (upstream's forward passes ``attention_masks``
  through to ``inner_attention`` untouched).
* ``TokamaxSplashInnerAttention``: unflattens ``[T]`` to ``[B, S]`` rows
  (``unflatten_shim``) and runs causal attention within each row and document
  (Splash ``segment_ids``) in the expanded (MHA) form: every head has its own
  K (``qk_nope_head_dim + qk_rope_head_dim``) and V (``v_head_dim``); Splash
  takes the two head dims directly.

K3 sets ``mla_use_nope=True``: the RoPE-sized query/key slices get no rotary
transform, so there is no RoPE here either. The softmax scale is upstream's
``(qk_nope_head_dim + qk_rope_head_dim) ** -0.5``; the Splash wrapper
multiplies q by it rounded to q's dtype (in bf16, ``192 ** -0.5`` rounds 0.13%
up), where FlexAttention scales the fp32 scores.
"""

from dataclasses import dataclass

import torch
from torchtitan.config.override import derive, override

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.components import unflatten_shim
from torchtitan.experiments.tpu.kernels import tokamax_splash_attention
from torchtitan.experiments.tpu.kimi_k3.sharding import enter_tp_region, reduce_tp_partial, tp_group
from torchtitan.models.common.attention import segment_ids_from_positions
from torchtitan.models.kimi_k3.model import KimiMLAAttention
from torchtitan.protocols.module import Module

# Shape suffixes:
# T = packed tokens, B = rows, S = tokens per row, D = model dimension,
# H = heads, K = query/key head dimension, V = value head dimension.

# Splash tiling for K3's head dims (K = 192, V = 128; debug flavor 96 / 64),
# from a TPU7x sweep at 4 x 4096 tokens, 48 heads (one TP2 shard): 1024-token
# blocks and a vmapped backward take fwd+bwd from 30.1 to 19.4 ms without
# documents and from 19.3 to 17.4 ms with 5 per row (512 blocks, scanned
# backward; see NOTES). Tiles cover whole heads; the blocks are capped at the
# row length, which they must divide. Every `TokamaxConfig` field is set, so a
# recipe's global Splash config for other attention layers does not leak here.
_BLOCK = 1024
_SPLASH_KERNEL_CONFIG = dict(
    block_q=_BLOCK,
    block_kv=_BLOCK,
    block_kv_compute=512,
    block_q_dkv=_BLOCK,
    block_kv_dkv=_BLOCK,
    block_kv_dkv_compute=_BLOCK,
    block_q_dq=None,
    block_kv_dq=None,
    use_fused_bwd_kernel=True,
    # One batched backward call instead of a scan over rows; its
    # intermediates grow with the rows per call.
    use_vmap_bwd=True,
    q_layout="HEAD_DIM_MINOR",
    k_layout="HEAD_DIM_MINOR",
    v_layout="HEAD_DIM_MINOR",
    num_stacked_q_heads=1,
    # A Python loop over rows in the forward: with more than one row per
    # rank, the vmapped segmented forward fails to compile inside the
    # GraphTrainer step (XLA: "reshape should have supported layout before
    # reaching the emitter" on the segment ids).
    use_batch_loop_fwd=True,
    use_transposed_bwd=False,
)


class TokamaxSplashInnerAttention(Module):
  """Causal attention within packed documents on ``[T, H, *]`` tokens.

  Rows are ``[B, S]`` (``unflatten_shim``; one row without it). A document
  that crosses a row boundary is split there, as every row is attended on its
  own.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Module.Config):
    pass

  def __init__(self, config: Config):
    del config  # No options.
    super().__init__()

  def forward(
      self,
      q_THK: torch.Tensor,
      k_THK: torch.Tensor,
      v_THV: torch.Tensor,
      *,
      attention_masks: torch.Tensor | None,
      scale: float,
  ) -> torch.Tensor:
    """``attention_masks``: optional ``[T]`` document ids, non-decreasing per row."""
    if q_THK.device.type != "tpu":
      raise ValueError(
          "TokamaxSplashInnerAttention runs on TPU only, got device"
          f" {q_THK.device.type}."
      )
    num_tokens = q_THK.shape[0]
    # TT-UNFLATTEN-SHIM: unflatten [T, ...] to [B, S, ...] for Splash.
    rows = [
        unflatten_shim.unflatten_tensor(x, expected_flat_ndim=3)
        for x in (q_THK, k_THK, v_THV)
    ]
    if rows[0].ndim == 3:
      # Without set_unflatten_seq_len, the [T] token stream is one row.
      rows = [x.unsqueeze(0) for x in rows]
    # `jax_op` reads its operands by shape alone, so they must be dense.
    q_BSHK, k_BSHK, v_BSHV = (x.contiguous() for x in rows)
    seq_len = q_BSHK.shape[1]
    if seq_len % min(_BLOCK, seq_len) or seq_len % 128:
      raise ValueError(
          "TokamaxSplashInnerAttention needs rows of a multiple of"
          f" {_BLOCK} tokens (or of 128 tokens up to {_BLOCK}), got {seq_len}."
      )
    segment_ids_BS = (
        None
        if attention_masks is None
        else attention_masks.reshape(q_BSHK.shape[:2]).contiguous()
    )
    out_BSHV, _ = tokamax_splash_attention.splash_op(
        q_BSHK,
        k_BSHK,
        v_BSHV,
        segment_ids_BS,
        None,  # cp_rank
        True,  # is_causal
        -1,  # no local window
        scale,
        *tokamax_splash_attention.kernel_config_args(**_SPLASH_KERNEL_CONFIG),
    )
    # TT-UNFLATTEN-SHIM: fold [B, S, ...] back to [T, ...].
    return out_BSHV.reshape(num_tokens, *out_BSHV.shape[2:])


class TpuKimiMLAAttention(KimiMLAAttention):
  """``KimiMLAAttention`` whose packed documents come from ``positions``.

  Upstream's forward drops ``positions`` and hands ``attention_masks`` (a
  FlexAttention ``BlockMask``) to ``inner_attention``; here that slot carries
  the ``[T]`` document ids of ``positions`` (static shape) instead. The
  projections are upstream's, with optional TP over heads (layouts in
  ``kimi_k3/sharding.py``): the replicated input enters the TP region, the
  LoRA down projections run whole, ``wq_b`` / ``wkv_b`` / ``gate`` and Splash
  run on this rank's heads, and the rowwise ``wo``'s partial sum is
  all-reduced.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(KimiMLAAttention.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks=None,
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    del attention_masks  # Documents come from `positions`.
    document_ids_T = None
    if positions is not None:
      if positions.numel() != x_TD.shape[0]:
        raise ValueError(
            "TpuKimiMLAAttention needs one position per token, got positions"
            f" of shape {tuple(positions.shape)} for {x_TD.shape[0]} tokens."
        )
      document_ids_T = segment_ids_from_positions(positions.reshape(-1))
    tp = tp_group()
    if tp is not None:
      x_TD = enter_tp_region(x_TD, tp)

    # Upstream's KimiMLAAttention.forward with -1 (this rank's heads) for the
    # head count.
    num_tokens = x_TD.shape[0]
    q_THK = self.wq_b(self.q_norm(self.wq_a(x_TD))).view(
        num_tokens, -1, self.q_head_dim
    )
    compressed_kv_TC = self.wkv_a(x_TD)
    kv_latent_TC, k_rope_TK = torch.split(
        compressed_kv_TC, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    kv_THC = self.wkv_b(self.kv_norm(kv_latent_TC)).view(
        num_tokens, -1, self.qk_nope_head_dim + self.v_head_dim
    )
    k_nope_THK, v_THV = torch.split(
        kv_THC, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    num_local_heads = q_THK.shape[1]
    k_rope_THK = k_rope_TK.view(num_tokens, 1, self.qk_rope_head_dim).expand(
        -1, num_local_heads, -1
    )
    k_THK = torch.cat((k_nope_THK, k_rope_THK), dim=-1)
    out_THV = self.inner_attention(
        q_THK,
        k_THK,
        v_THV,
        attention_masks=document_ids_T,
        scale=self.scale,
    )
    out_TD = out_THV.reshape(num_tokens, num_local_heads * self.v_head_dim)
    out_TD = self.wo(out_TD * torch.sigmoid(self.gate(x_TD)))
    if tp is not None:
      out_TD = reduce_tp_partial(out_TD, tp)
    return out_TD


@override(
    target=KimiMLAAttention.Config,
    exact=True,
    description="Run Kimi K3's gated MLA layers with Tokamax Splash on TPU.",
)
def tokamax_mla_attention(
    cfg: KimiMLAAttention.Config,
) -> TpuKimiMLAAttention.Config:
  """``TpuKimiMLAAttention.Config`` for an upstream ``KimiMLAAttention.Config``."""
  return derive(
      cfg,
      TpuKimiMLAAttention.Config,
      inner_attention=TokamaxSplashInnerAttention.Config(),
  )
