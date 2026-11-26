"""Kimi K3's LatentMoE on TPU with MegaBlox grouped matmuls.

Upstream ``torchtitan.models.kimi_k3.moe.KimiLatentMoE`` routes on the model
dim (sigmoid scores from an FP32 router gate, top-k over scores plus the
load-balancing bias, gating weights from the unbiased scores, normalized to sum
to 1), projects tokens down to the latent dim, runs the routed experts there,
weights each expert output by its score after W2, sums the routes per token,
then applies ``routed_norm`` and the up projection and adds the shared experts
(a SiTU-GLU FFN on the model dim). Only the routed experts change on TPU, so
the overrides keep ``KimiLatentMoE`` and replace its ``routed_experts``
children:

* ``TpuKimiGroupedExperts``: ``TpuGroupedExperts`` (Pallas MegaBlox GMM,
  kernel-native weight layout) with K3's SiTU-GLU activation. Upstream's
  ``KimiGroupedExperts`` subclasses the common ``GroupedExperts`` on
  ``torch._grouped_mm``; TPU runs the existing ``TpuGroupedExperts``
  component instead, so only the activation hooks (``_glu``,
  ``_gmm_glu_act``) are K3-specific. ``tpu_latent_moe`` runs it on MegaBlox
  (the reference path); ``tpu_latent_moe_gmm_glu`` (an option, used in its
  place) selects ``pallas_grouped_mm_kernel="gmm_glu"``: W1 / W3 and the
  SiTU-GLU as one fused grouped matmul (``kernels/gmm_glu.py``), W2 on
  Tokamax's ``gmm_v2`` / ``tgmm_v2``.
* ``TpuAllToAllTokenDispatcher``: TPU local-reorder VJPs without EP; with EP,
  the Ring-of-Experts transport (``ep_transport="ring_of_experts"``, set
  explicitly rather than ``"auto"``, which would switch to ragged all-to-all
  above EP = top_k): all-gather of the latent tokens, scores and expert ids,
  the local experts on a static buffer, and a reduce-scatter of the per-token
  sums. The scores are applied after W2, as upstream does.
* ``TpuRoutedExperts``: upstream's ``RoutedExperts`` over the TPU dispatcher
  (one token chunk).
"""

from dataclasses import dataclass

import torch
from torchtitan.config.override import derive, override

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.components.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExperts,
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

  def _gmm_glu_act(self):
    # Imported lazily, like TpuGroupedExperts' kernels: it pulls in tokamax.
    from torchtitan.experiments.tpu.kernels import gmm_glu

    return gmm_glu.SiTUGLU(self.beta, self.linear_beta)


def _tpu_latent_moe(
    cfg: KimiLatentMoE.Config,
    *,
    ep_capacity_factor: float | None,
    expert_kernel_config: dict,
    num_token_chunks: int,
) -> KimiLatentMoE.Config:
  routed = cfg.routed_experts
  if type(routed.token_dispatcher) is not AllToAllTokenDispatcher.Config:
    raise ValueError(
        "tpu_latent_moe replaces the standard dispatcher, got"
        f" {type(routed.token_dispatcher).__qualname__}"
    )
  return derive(
      cfg,
      type(cfg),
      routed_experts=derive(
          routed,
          TpuRoutedExperts.Config,
          num_token_chunks=num_token_chunks,
          inner_experts=derive(
              routed.inner_experts,
              TpuKimiGroupedExperts.Config,
              **expert_kernel_config,
          ),
          token_dispatcher=derive(
              routed.token_dispatcher,
              TpuAllToAllTokenDispatcher.Config,
              fuse_route_score_backward=True,
              ep_transport="ring_of_experts",
              ep_capacity_factor=ep_capacity_factor,
          ),
      ),
  )


@override(
    target=KimiLatentMoE.Config,
    description="Kimi K3 LatentMoE on TPU: MegaBlox experts, Ring-of-Experts EP.",
)
def tpu_latent_moe(
    cfg: KimiLatentMoE.Config,
    *,
    ep_capacity_factor: float | None = None,
    num_token_chunks: int = 1,
) -> KimiLatentMoE.Config:
  """K3 LatentMoE with TPU MegaBlox experts and the TPU dispatcher.

  The MoE keeps its config type: ``KimiLatentMoE`` or
  ``QuantileBalancedKimiLatentMoE`` (``quantile_balance.py``).

  ``ep_capacity_factor`` (Ring-of-Experts expert rows per EP rank, in units of
  the local routes; None is drop-free) applies only under EP; see
  ``TpuAllToAllTokenDispatcher.Config``. ``num_token_chunks`` > 1 pipelines
  the EP dispatch, experts and combine over that many token chunks per layer
  (``TpuRoutedExperts``; needs SimpleFSDP).
  """
  inner = cfg.routed_experts.inner_experts
  return _tpu_latent_moe(
      cfg,
      ep_capacity_factor=ep_capacity_factor,
      num_token_chunks=num_token_chunks,
      expert_kernel_config=_expert_kernel_config(inner.dim, inner.hidden_dim),
  )


@override(
    target=KimiLatentMoE.Config,
    description=(
        "Kimi K3 LatentMoE on TPU: fused gmm_glu / gmm_v2 experts,"
        " Ring-of-Experts EP."
    ),
)
def tpu_latent_moe_gmm_glu(
    cfg: KimiLatentMoE.Config,
    *,
    ep_capacity_factor: float | None = None,
    num_token_chunks: int = 1,
) -> KimiLatentMoE.Config:
  """``tpu_latent_moe`` with the fused gmm_glu experts (an option).

  W1 / W3 and the SiTU-GLU run as one gmm_glu call, W2 on gmm_v2 / tgmm_v2.
  Use it in place of ``tpu_latent_moe`` in ``override.imports``. Both kernels
  pick their own tiles, so the MegaBlox tiling does not apply.
  """
  return _tpu_latent_moe(
      cfg,
      ep_capacity_factor=ep_capacity_factor,
      num_token_chunks=num_token_chunks,
      expert_kernel_config=dict(
          use_pallas_grouped_mm=True,
          use_kernel_native_weight_layout=True,
          pallas_grouped_mm_kernel="gmm_glu",
      ),
  )
