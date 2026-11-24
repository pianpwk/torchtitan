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

Why ``KimiMLAAttention`` is forked (subclassed with its own forward) rather
than only ``inner_attention`` swapped: upstream ``KimiMLAAttention.forward``
deletes ``positions`` and gives ``inner_attention`` a FlexAttention
``BlockMask``. ``TpuKimiMLAAttention`` overrides the forward so the mask slot
carries static per-token document ids from ``positions``; the projections,
norms, gate and parameters stay upstream's.

TODO: drop this fork once upstream passes documents to the kernel / inner
attention as a static segment-id mask derived from positions; then only the
kernel (KDAKernel) / inner_attention config needs swapping.

K3 sets ``mla_use_nope=True``: the RoPE-sized query/key slices get no rotary
transform, so there is no RoPE here either. The softmax scale is upstream's
``(qk_nope_head_dim + qk_rope_head_dim) ** -0.5``. The Splash wrapper
multiplies q by that scale rounded to q's dtype (bf16(192 ** -0.5) is 0.13%
high), where FlexAttention scales the fp32 scores.
"""

import dataclasses
from dataclasses import dataclass

import torch
from torchtitan.config.override import derive, override

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.components import unflatten_shim
from torchtitan.experiments.tpu.kernels import tokamax_splash_attention
from torchtitan.models.common.attention import segment_ids_from_positions
from torchtitan.models.kimi_k3.model import KimiMLAAttention
from torchtitan.protocols.module import Module

# Shape suffixes:
# T = packed tokens, B = rows, S = tokens per row, D = model dimension,
# H = heads, K = query/key head dimension, V = value head dimension.

# Splash fields tied to this wrapper, not tuning: head-dim-minor layouts and
# one query head per tile (the expanded MHA form has one KV head per query
# head), the fused backward kernel (so no separate dq blocks) and Tokamax's
# backward. Together with `SplashTilingConfig` every `TokamaxConfig` field is
# set, so a recipe's global Splash config for other attention layers does not
# leak here.
_SPLASH_WRAPPER_CONFIG = dict(
    block_q_dq=None,
    block_kv_dq=None,
    use_fused_bwd_kernel=True,
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


@dataclass(kw_only=True, slots=True)
class SplashTilingConfig:
  """Splash tiling (``TokamaxConfig`` fields of the same names) for MLA.

  Tiles cover whole heads. A block is capped at the row length, which it must
  divide; ``block_q`` and ``block_kv`` are checked per row.
  """

  block_q: int
  block_kv: int
  block_kv_compute: int
  block_q_dkv: int
  block_kv_dkv: int
  block_kv_dkv_compute: int
  # One batched backward call instead of a scan over rows; its intermediates
  # grow with the rows per call.
  use_vmap_bwd: bool

  def __post_init__(self):
    for name in (
        "block_q",
        "block_kv",
        "block_kv_compute",
        "block_q_dkv",
        "block_kv_dkv",
        "block_kv_dkv_compute",
    ):
      value = getattr(self, name)
      if value <= 0 or value % 128:
        raise ValueError(
            f"SplashTilingConfig.{name} must be a positive multiple of 128,"
            f" got {value}"
        )


class TokamaxSplashInnerAttention(Module):
  """Causal attention within packed documents on ``[T, H, *]`` tokens.

  Rows are ``[B, S]`` (``unflatten_shim``; one row without it). A document
  that crosses a row boundary is split there, as every row is attended on its
  own.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Module.Config):
    splash_tiling: SplashTilingConfig

  def __init__(self, config: Config):
    super().__init__()
    self.splash_tiling = config.splash_tiling

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
    tiling = self.splash_tiling
    for block in (tiling.block_q, tiling.block_kv):
      if seq_len % min(block, seq_len) or seq_len % 128:
        raise ValueError(
            "TokamaxSplashInnerAttention needs rows of a multiple of"
            f" {block} tokens (or of 128 tokens up to {block}), got"
            f" {seq_len}."
        )
    segment_ids_BS = None
    if attention_masks is not None:
      if attention_masks.shape != (num_tokens,):
        raise ValueError(
            "TokamaxSplashInnerAttention takes one document id per token, got"
            f" shape {tuple(attention_masks.shape)} for {num_tokens} tokens."
        )
      segment_ids_BS = attention_masks.reshape(q_BSHK.shape[:2]).contiguous()
    out_BSHV, _ = tokamax_splash_attention.splash_op(
        q_BSHK,
        k_BSHK,
        v_BSHV,
        segment_ids_BS,
        None,  # cp_rank
        True,  # is_causal
        -1,  # no local window
        scale,
        *tokamax_splash_attention.kernel_config_args(
            **dataclasses.asdict(tiling), **_SPLASH_WRAPPER_CONFIG
        ),
    )
    # TT-UNFLATTEN-SHIM: fold [B, S, ...] back to [T, ...].
    return out_BSHV.reshape(num_tokens, *out_BSHV.shape[2:])


class TpuKimiMLAAttention(KimiMLAAttention):
  """``KimiMLAAttention`` whose packed documents come from ``positions``.

  Upstream's forward drops ``positions`` and hands ``attention_masks`` (a
  FlexAttention ``BlockMask``) to ``inner_attention``; here that slot carries
  the ``[T]`` document ids of ``positions`` (static shape) instead. The
  projections, norms and output gate are upstream's.
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
    return super().forward(x_TD, attention_masks=document_ids_T)


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
      inner_attention=TokamaxSplashInnerAttention.Config(
          # Tuned on TPU7x at 4 x 4096 tokens, 48 heads: 1024-token blocks
          # and a vmapped backward take fwd+bwd from 30.1 to 19.4 ms without
          # documents and from 19.3 to 17.4 ms with 5 per row (vs 512 blocks
          # and a scanned backward). At F3's per-core shape (one row of
          # 16,384 tokens, 96 heads) no block_q / block_kv in {512, 1024,
          # 2048} or block_kv_compute in {256, 512, 1024} is >5% faster.
          splash_tiling=SplashTilingConfig(
              block_q=1024,
              block_kv=1024,
              block_kv_compute=512,
              block_q_dkv=1024,
              block_kv_dkv=1024,
              block_kv_dkv_compute=1024,
              use_vmap_bwd=True,
          ),
      ),
  )
