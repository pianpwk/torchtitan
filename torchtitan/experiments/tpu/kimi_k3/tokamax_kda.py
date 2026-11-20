"""Kimi K3's Kimi Delta Attention (KDA) on TPU with Tokamax's Mosaic kernels.

Upstream ``torchtitan.models.kimi_k3.kda`` runs the gate, the Q/K L2 norm, the
varlen causal ``conv1d`` and the chunked delta rule with CUDA-only
``attn_gym`` kernels. The ``tokamax_kda_attention`` override swaps in three
TPU modules, one per level of upstream's config tree, that compute the same
function:

* ``TpuKDA``: ``KDA.forward`` with packed documents from ``positions``
  (static ``[B, S]`` segment ids) instead of the varlen ``cu_seqlens``, whose
  shape changes with the number of documents.
* ``TpuInnerKDA``: unflattens ``[T]`` to ``[B, S]`` rows (``unflatten_shim``)
  and runs the depthwise causal ``conv1d`` + SiLU per row
  (``qwen3_5.gated_delta.causal_conv_silu``), restarting at every document.
* ``TokamaxKDAKernel``: K3's gate and beta in fp32 (the formula of
  ``attn_gym``'s ``bound_gate``), then Tokamax's Mosaic KDA forward and
  backward kernels through Qwen3.5's bridge
  (``qwen3_5.tokamax_gated_delta.tokamax_gated_delta`` with a per-channel
  decay), which also L2-normalize Q/K in-kernel as ``attn_gym``'s ``l2norm``
  does (fp32, ``eps=1e-6``, rounded to the input dtype).

No decay clamp: K3's gate is ``lower_bound * sigmoid(...)`` with
``lower_bound`` in ``[-5, 0)`` (``KDAKernel.Config``), so every log decay is in
``[-5, 0]`` (fp32 rounding keeps ``lower_bound * s`` >= ``lower_bound`` for
``s <= 1``), inside the Mosaic backward kernel's range (it overflows below
about -5.9; see ``qwen3_5.tokamax_gated_delta.MIN_SUPPORTED_LOG_DECAY``).

Packed documents use Tokamax's ``segment_ids`` / ``max_num_segments``: the
kernels lay every document out from a 64-token chunk boundary, so each row
always costs ``seq_len + 63 * max_num_segments`` kernel tokens, and documents
beyond ``max_num_segments`` share one sequence (in the convolution too).
"""

from dataclasses import dataclass

import torch
from torchtitan.config.override import derive, override

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.components import unflatten_shim
from torchtitan.experiments.tpu.qwen3_5.gated_delta import causal_conv_silu, per_row_segment_ids
from torchtitan.experiments.tpu.qwen3_5.tokamax_gated_delta import (
    MIN_NUM_SEGMENTS,
    MIN_TOKAMAX_VERSION,
    TOKENS_PER_SEGMENT,
    has_tokamax_kda,
    supports_current_tpu,
    tokamax_gated_delta,
    tokamax_kda_import_error,
)
from torchtitan.models.common.attention import segment_ids_from_positions
from torchtitan.models.kimi_k3.kda import InnerKDA, KDA, KDAKernel

# Shape suffixes:
# T = packed tokens, B = rows, S = tokens per row, D = model dimension,
# C = projection channels, H = heads, K = query/key head dimension,
# V = value head dimension, W = convolution kernel width.


class TokamaxKDAKernel(KDAKernel):
  """``KDAKernel`` on TPU: K3's gate in PyTorch, then the Tokamax kernels."""

  @dataclass(kw_only=True, slots=True)
  class Config(KDAKernel.Config):
    # Packed documents per row kept apart (Tokamax's `max_num_segments`);
    # later documents of a row share one sequence in the convolution and the
    # recurrence. Each row costs `seq_len + 63 * max_num_segments` kernel
    # tokens. None: one document per `TOKENS_PER_SEGMENT` tokens of the row,
    # at least `MIN_NUM_SEGMENTS` (8 per 4K row: +12% kernel tokens).
    max_num_segments: int | None = None

    def __post_init__(self):
      KDAKernel.Config.__post_init__(self)
      if self.max_num_segments is not None and self.max_num_segments < 1:
        raise ValueError(
            f"max_num_segments must be positive, got {self.max_num_segments}"
        )

  def __init__(self, config: Config):
    super().__init__(config)
    # Fail at model construction rather than at the first traced forward.
    if supports_current_tpu() and not has_tokamax_kda():
      raise ImportError(
          f"TokamaxKDAKernel needs tokamax>={MIN_TOKAMAX_VERSION} (the Kimi"
          " Delta Attention op with the Mosaic TPU backend): pip install -r"
          " torchtitan/experiments/tpu/requirements.txt"
          f" ({tokamax_kda_import_error()})"
      )
    self.max_num_segments = config.max_num_segments

  def num_segments(self, seq_len: int) -> int:
    """Documents per ``seq_len``-token row kept apart."""
    if self.max_num_segments is not None:
      return min(self.max_num_segments, max(1, seq_len))
    return min(
        max(MIN_NUM_SEGMENTS, seq_len // TOKENS_PER_SEGMENT), max(1, seq_len)
    )

  def document_ids(self, segment_ids_BS: torch.Tensor) -> torch.Tensor:
    """Per-row ids for the conv and the kernel; extra documents share the last."""
    return per_row_segment_ids(
        segment_ids_BS, self.num_segments(segment_ids_BS.shape[1])
    )

  def forward(  # pyrefly: ignore [bad-override]
      self,
      q_BSHK: torch.Tensor,
      k_BSHK: torch.Tensor,
      v_BSHV: torch.Tensor,
      raw_gate_BSHK: torch.Tensor,
      raw_beta_BSH: torch.Tensor,
      A_log_H: torch.Tensor,
      dt_bias_HK: torch.Tensor,
      *,
      segment_ids_BS: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """K3's KDA on ``[B, S]`` rows; ``segment_ids_BS`` from ``document_ids``.

    Without ``segment_ids_BS``, ``S`` must be a multiple of 64.
    """
    if q_BSHK.device.type != "tpu":
      raise ValueError(
          f"TokamaxKDAKernel runs on TPU only, got device {q_BSHK.device.type}."
      )
    # `attn_gym.linear.kda.bound_gate`:
    # lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias)), in fp32.
    gate_BSHK = self.lower_bound * torch.sigmoid(
        A_log_H.float().exp().unsqueeze(-1)
        * (raw_gate_BSHK.float() + dt_bias_HK.float())
    )
    return tokamax_gated_delta(
        q_BSHK,
        k_BSHK,
        v_BSHV,
        gate_BSHK,
        raw_beta_BSH.float().sigmoid(),
        segment_ids=segment_ids_BS,
        max_num_segments=(
            None
            if segment_ids_BS is None
            else self.num_segments(q_BSHK.shape[1])
        ),
    )


class TpuInnerKDA(InnerKDA):
  """``InnerKDA`` that convolves each ``[B, S]`` row on its own.

  Given ``positions``, the convolution and the recurrence also restart at
  every document packed into a row (``positions == 0``), as upstream's varlen
  kernels do at ``cu_seqlens``.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(InnerKDA.Config):
    pass

  def forward(  # pyrefly: ignore [bad-override]
      self,
      query_TC: torch.Tensor,
      key_TC: torch.Tensor,
      value_TC: torch.Tensor,
      raw_gate_THK: torch.Tensor,
      raw_beta_TH: torch.Tensor,
      conv_q_weight_C1W: torch.Tensor,
      conv_k_weight_C1W: torch.Tensor,
      conv_v_weight_C1W: torch.Tensor,
      A_log_H: torch.Tensor,
      dt_bias_HK: torch.Tensor,
      *,
      positions: torch.Tensor | None,
  ) -> torch.Tensor:
    num_tokens = query_TC.shape[0]
    # TT-UNFLATTEN-SHIM: unflatten [T, ...] to [B, S, ...] for conv + kernel.
    rows = [
        unflatten_shim.unflatten_tensor(x, expected_flat_ndim=x.ndim)
        for x in (query_TC, key_TC, value_TC, raw_gate_THK, raw_beta_TH)
    ]
    if rows[0].ndim == query_TC.ndim:
      # Without set_unflatten_seq_len, the [T] token stream is one row.
      rows = [x.unsqueeze(0) for x in rows]
    query_BSC, key_BSC, value_BSC, raw_gate_BSHK, raw_beta_BSH = rows
    segment_ids_BS = None
    if positions is not None:
      if positions.numel() != num_tokens:
        raise ValueError(
            "TpuInnerKDA needs one position per token, got positions of shape"
            f" {tuple(positions.shape)} for {num_tokens} tokens."
        )
      segment_ids_BS = self.kernel.document_ids(
          segment_ids_from_positions(positions.reshape(-1)).reshape(
              query_BSC.shape[:2]
          )
      )
    q_BSHK, k_BSHK, v_BSHV = (
        causal_conv_silu(x_BSC, weight_C1W, segment_ids_BS).unflatten(
            -1, (-1, self.head_dim)
        )
        for x_BSC, weight_C1W in (
            (query_BSC, conv_q_weight_C1W),
            (key_BSC, conv_k_weight_C1W),
            (value_BSC, conv_v_weight_C1W),
        )
    )
    out_BSHV = self.kernel(
        q_BSHK,
        k_BSHK,
        v_BSHV,
        raw_gate_BSHK,
        raw_beta_BSH,
        A_log_H,
        dt_bias_HK,
        segment_ids_BS=segment_ids_BS,
    )
    # TT-UNFLATTEN-SHIM: fold [B, S, ...] back to [T, ...].
    return out_BSHV.reshape(num_tokens, *out_BSHV.shape[2:])


class TpuKDA(KDA):
  """``KDA`` that passes ``positions`` to ``TpuInnerKDA``.

  Identical to upstream ``KDA.forward`` except that packed documents come from
  ``positions`` instead of the varlen ``attention_masks``' ``cu_seqlens``.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(KDA.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks=None,
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    del attention_masks  # Documents come from `positions`.
    if x_TD.ndim != 2:
      raise ValueError(
          f"KDA input must have shape [T, D], got {tuple(x_TD.shape)}."
      )
    num_tokens = x_TD.shape[0]
    raw_gate_THK = self.forget_b(self.forget_a(x_TD)).reshape(
        num_tokens, self.num_heads, self.head_dim
    )
    raw_beta_TH = self.beta(x_TD).reshape(num_tokens, self.num_heads)
    out_THV = self.inner_kda(
        self.q_proj(x_TD),
        self.k_proj(x_TD),
        self.v_proj(x_TD),
        raw_gate_THK,
        raw_beta_TH,
        self.q_conv.weight,
        self.k_conv.weight,
        self.v_conv.weight,
        self.A_log,
        self.dt_bias,
        positions=positions,
    )
    output_gate_THV = self.output_gate(x_TD).view_as(out_THV)
    return self.output_proj(
        self.output_norm(out_THV, output_gate_THV).flatten(-2)
    )


@override(
    target=KDA.Config,
    exact=True,
    description="Run Kimi K3's KDA layers with Tokamax's TPU kernels.",
)
def tokamax_kda_attention(
    cfg: KDA.Config, *, max_num_segments: int | None = None
) -> TpuKDA.Config:
  """``TpuKDA.Config`` with ``TokamaxKDAKernel`` for an upstream ``KDA.Config``.

  For example ``(TOKAMAX_KDA_OVERRIDE, {"max_num_segments": 16})`` in
  ``override.imports`` keeps up to 16 packed documents per row apart; see
  ``TokamaxKDAKernel.Config``.
  """
  inner = cfg.inner_kda
  return derive(
      cfg,
      TpuKDA.Config,
      inner_kda=derive(
          inner,
          TpuInnerKDA.Config,
          kernel=derive(
              inner.kernel,
              TokamaxKDAKernel.Config,
              max_num_segments=max_num_segments,
          ),
      ),
  )
