"""Qwen3.5 GatedDeltaNet recurrence on the fused Pallas KDA kernels.

``FusedGatedDeltaKernel`` runs ``kernels.kda.kda_torch`` with one log decay per
head (``g [B, S, H]``) and the model's grouped key heads (``Hk`` q / k heads
shared by ``H`` value heads), so neither the ``[B, S, H, K]`` decay broadcast
nor the q / k repeat leaves the kernel op. Packed documents are masked inside
the kernel's chunks: every document of a row stays apart (as in the XLA
``chunked_gated_delta``), with no per-row document budget and no padding to
chunk boundaries, unlike ``tokamax_gated_delta.TokamaxGatedDeltaKernel``.

Decay clamp: log decays below ``min_log_decay`` (default -8, the kernels'
limit) are computed as ``min_log_decay`` and pass no gradient, as in the
Tokamax kernel (whose limit is -5). Such a token keeps ``exp(-8)`` = 0.03% of
the state instead of less.

Select it with the ``fused_gated_delta_kernel`` override, e.g.
``--override.imports
torchtitan.experiments.tpu.qwen3_5.fused_gated_delta.fused_gated_delta_kernel``
in place of ``TOKAMAX_GATED_DELTA_OVERRIDE``.
"""

from dataclasses import dataclass

import torch
from torchtitan.config.override import derive, override
from torchtitan.experiments.tpu.kernels import kda
from torchtitan.experiments.tpu.qwen3_5.gated_delta import TpuGatedDeltaKernel

DEFAULT_MIN_LOG_DECAY = -kda.MAX_ABS_LOG_DECAY


class FusedGatedDeltaKernel(TpuGatedDeltaKernel):
  """``TpuGatedDeltaKernel`` with the fused Pallas KDA kernels on TPU.

  ``chunk_size`` is the kernels' chunk (one of ``kda.CHUNK_SIZES``). Inputs
  off TPU (CPU tests) run the parent's XLA recurrence on the same clamped
  decays.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TpuGatedDeltaKernel.Config):
    # Per-token log decays below this are clamped to it, in
    # [-kda.MAX_ABS_LOG_DECAY, 0).
    min_log_decay: float = DEFAULT_MIN_LOG_DECAY

    def __post_init__(self):
      if self.chunk_size not in kda.CHUNK_SIZES:
        raise ValueError(
            f"chunk_size must be one of {kda.CHUNK_SIZES}, got"
            f" {self.chunk_size}"
        )
      if not -kda.MAX_ABS_LOG_DECAY <= self.min_log_decay < 0:
        raise ValueError(
            f"min_log_decay must be in [-{kda.MAX_ABS_LOG_DECAY}, 0), got"
            f" {self.min_log_decay}"
        )

  def __init__(self, config: Config):
    super().__init__(config)
    self.min_log_decay = config.min_log_decay

  def forward(
      self,
      xq_BSHK: torch.Tensor,
      xk_BSHK: torch.Tensor,
      xv_BSHV: torch.Tensor,
      g_BSH: torch.Tensor,
      beta_BSH: torch.Tensor,
      *,
      cu_seqlens: torch.Tensor | None = None,
      segment_ids_BS: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """The gated delta rule; q / k keep their ``Hk`` key heads."""
    if xq_BSHK.device.type != "tpu":
      g_BSH = g_BSH.clamp(min=self.min_log_decay)
      return super().forward(
          xq_BSHK,
          xk_BSHK,
          xv_BSHV,
          g_BSH,
          beta_BSH,
          cu_seqlens=cu_seqlens,
          segment_ids_BS=segment_ids_BS,
      )
    if cu_seqlens is not None:
      raise NotImplementedError(
          "FusedGatedDeltaKernel does not support varlen cu_seqlens; pass"
          " segment_ids_BS."
      )
    out_BSHV, _ = kda.kda_torch(
        xq_BSHK,
        xk_BSHK,
        xv_BSHV,
        g_BSH,
        beta_BSH,
        lower_bound=self.min_log_decay,
        segment_ids=segment_ids_BS,
        chunk=self.chunk_size,
    )
    return out_BSHV


@override(
    target=TpuGatedDeltaKernel.Config,
    exact=True,
    description="Run the Qwen3.5 GatedDeltaNet with the fused KDA kernels.",
)
def fused_gated_delta_kernel(
    cfg: TpuGatedDeltaKernel.Config,
    *,
    min_log_decay: float = DEFAULT_MIN_LOG_DECAY,
) -> FusedGatedDeltaKernel.Config:
  """``FusedGatedDeltaKernel.Config`` with the override's arguments.

  Args:
    cfg: The ``TpuGatedDeltaKernel.Config`` to replace.
    min_log_decay: See ``FusedGatedDeltaKernel.Config``.

  Returns:
    The fused kernel's config.
  """
  return derive(cfg, FusedGatedDeltaKernel.Config, min_log_decay=min_log_decay)
