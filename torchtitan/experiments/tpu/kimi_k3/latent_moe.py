"""Kimi K3's LatentMoE on TPU with MegaBlox grouped matmuls.

Upstream ``torchtitan.models.kimi_k3.moe.KimiLatentMoE`` routes on the model
dim (sigmoid scores from an FP32 router gate, top-k over scores plus the
load-balancing bias, gating weights from the unbiased scores, normalized to sum
to 1), projects tokens down to the latent dim, runs the routed experts there,
weights each expert output by its score after W2, sums the routes per token,
then applies ``routed_norm`` and the up projection and adds the shared experts
(a SiTU-GLU FFN on the model dim). Only the routed experts change on TPU, so
the override keeps ``KimiLatentMoE`` and its ``RoutedExperts`` and replaces
two children:

* ``TpuKimiGroupedExperts``: ``TpuGroupedExperts`` (Pallas MegaBlox GMM,
  kernel-native weight layout) with K3's SiTU-GLU activation. Upstream's
  ``KimiGroupedExperts`` subclasses the common ``GroupedExperts`` on
  ``torch._grouped_mm``; TPU runs the existing ``TpuGroupedExperts``
  component instead, so only the activation hook (``_glu``) is K3-specific.
* ``TpuAllToAllTokenDispatcher``: TPU local-reorder VJPs. The scores are
  applied after W2, as upstream does.
"""

from dataclasses import dataclass

import torch
from torchtitan.config.override import derive, override

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.components.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
)
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.kimi_k3.moe import KimiLatentMoE

# Shape suffixes: R = routed rows, F = expert hidden dimension.

# MegaBlox tiling for K3's experts (latent dim 3584, hidden dim 3072): every
# full-K or full-N tile runs out of VMEM. TPU7x sweep at 262,144 rows over 56
# experts: fwd 55.6 ms, fwd+bwd 149.5 ms. Zero selects the full
# dimension, which smaller experts (the debug flavor) use.
_K3_EXPERT_TILING = dict(
    pallas_grouped_mm_tile_m=1024,
    pallas_grouped_mm_tile_k=1792,
    pallas_grouped_mm_tile_n=768,
    pallas_grouped_mm_tgmm_tiling=(512, 1792, 1536),
    pallas_grouped_mm_w2_tgmm_tiling=(512, 1536, 1792),
)
_FULL_TILE_EXPERT_TILING = dict(
    pallas_grouped_mm_tile_m=512,
    pallas_grouped_mm_tile_k=0,
    pallas_grouped_mm_tile_n=0,
    pallas_grouped_mm_tgmm_tiling=(512, 0, 0),
)


def _expert_kernel_config(dim: int, hidden_dim: int) -> dict:
  # The K3 tiles are smaller than K3's expert dims (K tile 1792 of 3584,
  # dWeight tiles 1536 of 3072); smaller experts fit whole in VMEM.
  tiling = (
      _K3_EXPERT_TILING
      if dim > 1792 and hidden_dim > 1536
      else _FULL_TILE_EXPERT_TILING
  )
  return dict(
      use_pallas_grouped_mm=True, use_kernel_native_weight_layout=True, **tiling
  )


def _situ_glu(
    gate_RF: torch.Tensor,
    up_RF: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
  """Kimi's SiTU-GLU in FP32.

  A copy of upstream ``kimi_k3.moe._situ_glu``, which is private to that
  module. TODO: import it instead if upstream makes it public.
  """
  gate_RF32, up_RF32 = gate_RF.float(), up_RF.float()
  gate_RF32 = beta * torch.tanh(gate_RF32 / beta) * torch.sigmoid(gate_RF32)
  if linear_beta is not None:
    up_RF32 = linear_beta * torch.tanh(up_RF32 / linear_beta)
  return (gate_RF32 * up_RF32).to(gate_RF.dtype)


class TpuKimiGroupedExperts(TpuGroupedExperts):
  """``TpuGroupedExperts`` with Kimi's SiTU-GLU activation."""

  @dataclass(kw_only=True, slots=True)
  class Config(TpuGroupedExperts.Config):
    beta: float
    linear_beta: float | None

  def __init__(self, config: Config):
    if config.use_fp8_grouped_mm:
      # The fused FP8 SwiGLU + W2 op always computes silu(gate) * up.
      raise ValueError("TpuKimiGroupedExperts does not support FP8 experts")
    super().__init__(config)
    self.beta = config.beta
    self.linear_beta = config.linear_beta

  def _glu(self, gate_RF: torch.Tensor, up_RF: torch.Tensor) -> torch.Tensor:
    return _situ_glu(gate_RF, up_RF, self.beta, self.linear_beta)


@override(
    target=KimiLatentMoE.Config,
    exact=True,
    description="Kimi K3 LatentMoE on TPU: MegaBlox experts.",
)
def tpu_latent_moe(cfg: KimiLatentMoE.Config) -> KimiLatentMoE.Config:
  """K3 LatentMoE with TPU experts and the TPU dispatcher."""
  routed = cfg.routed_experts
  if type(routed.token_dispatcher) is not AllToAllTokenDispatcher.Config:
    raise ValueError(
        "tpu_latent_moe replaces the standard dispatcher, got"
        f" {type(routed.token_dispatcher).__qualname__}"
    )
  return derive(
      cfg,
      KimiLatentMoE.Config,
      routed_experts=derive(
          routed,
          type(routed),
          inner_experts=derive(
              routed.inner_experts,
              TpuKimiGroupedExperts.Config,
              **_expert_kernel_config(
                  routed.inner_experts.dim, routed.inner_experts.hidden_dim
              ),
          ),
          token_dispatcher=derive(
              routed.token_dispatcher,
              TpuAllToAllTokenDispatcher.Config,
              fuse_route_score_backward=True,
          ),
      ),
  )
