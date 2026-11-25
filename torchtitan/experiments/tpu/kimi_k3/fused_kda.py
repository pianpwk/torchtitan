"""Kimi K3's KDA layers on TPU with the fused Pallas kernels (``kernels/kda.py``).

``fused_kda_attention`` is the alternative to ``tokamax_kda_attention``: the
same ``TpuKDA`` / ``TpuInnerKDA`` layers (per-row convolution restarting at
every document; the Pallas conv and output norm) with ``FusedKDAKernel``, which runs K3's beta in PyTorch and
then one fused forward and one fused backward Pallas kernel. Those take the
``[B, S, H * K]`` projections as they are (no layout passes), evaluate the Q/K
L2 norm and K3's gate in-kernel, and keep every packed document of a row apart
(no ``max_num_segments`` bound, no padding to chunk boundaries).
"""

from dataclasses import dataclass

import torch
from torchtitan.config.override import override

from torchtitan.experiments.tpu.kernels import kda
from torchtitan.experiments.tpu.kimi_k3.tokamax_kda import (
    PallasCausalConvSilu,
    PallasGatedHeadRMSNorm,
    tpu_kda_config,
    TpuKDA,
)
from torchtitan.experiments.tpu.qwen3_5.gated_delta import per_row_segment_ids
from torchtitan.models.kimi_k3.kda import KDA, KDAKernel

# Shape suffixes: B = rows, S = tokens per row, H = heads, K = query/key head
# dimension, V = value head dimension.


class FusedKDAKernel(KDAKernel):
  """``KDAKernel`` on TPU: K3's beta in PyTorch, then the fused kernels."""

  @dataclass(kw_only=True, slots=True)
  class Config(KDAKernel.Config):
    pass

  def document_ids(self, segment_ids_BS: torch.Tensor) -> torch.Tensor:
    """Per-row ids for the conv and the kernels; every document kept apart."""
    return per_row_segment_ids(segment_ids_BS, segment_ids_BS.shape[1])

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
    """K3's KDA on ``[B, S]`` rows; ``segment_ids_BS`` from ``document_ids``."""
    if q_BSHK.device.type != "tpu":
      raise ValueError(
          f"FusedKDAKernel runs on TPU only, got device {q_BSHK.device.type}."
      )
    if segment_ids_BS is None:
      segment_ids_BS = torch.ones(
          q_BSHK.shape[:2], dtype=torch.int32, device=q_BSHK.device
      )
    out_BSHV, _ = kda.kda_torch(
        q_BSHK, k_BSHK, v_BSHV, raw_gate_BSHK, raw_beta_BSH.float().sigmoid(),
        lower_bound=self.lower_bound, a_log=A_log_H, dt_bias=dt_bias_HK,
        segment_ids=segment_ids_BS,
    )
    return out_BSHV


@override(
    target=KDA.Config,
    exact=True,
    description="Run Kimi K3's KDA layers with the fused Pallas TPU kernels.",
)
def fused_kda_attention(cfg: KDA.Config) -> TpuKDA.Config:
  """``TpuKDA.Config`` with ``FusedKDAKernel`` for an upstream ``KDA.Config``.

  With the Pallas short conv and output norm (``PallasCausalConvSilu``,
  ``PallasGatedHeadRMSNorm``), as the fused path has always run.
  """
  return tpu_kda_config(
      cfg,
      FusedKDAKernel.Config,
      conv=PallasCausalConvSilu.Config(),
      output_norm=PallasGatedHeadRMSNorm.Config,
  )
