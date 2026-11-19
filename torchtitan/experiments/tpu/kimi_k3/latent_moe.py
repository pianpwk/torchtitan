"""Kimi K3's LatentMoE on TPU with MegaBlox grouped matmuls.

Upstream ``torchtitan.models.kimi_k3.moe.KimiLatentMoE`` routes on the model
dim (sigmoid scores from an FP32 router gate, top-k over scores plus the
load-balancing bias, gating weights from the unbiased scores, normalized to sum
to 1), projects tokens down to the latent dim, runs the routed experts there,
weights each expert output by its score after W2, sums the routes per token,
then applies ``routed_norm`` and the up projection and adds the shared experts
(a SiTU-GLU FFN on the model dim). Only the routed experts change on TPU, so
these overrides keep ``KimiLatentMoE`` and replace its ``routed_experts``
children:

* ``TpuKimiGroupedExperts``: ``TpuGroupedExperts`` (Pallas MegaBlox GMM,
  kernel-native weight layout) with K3's SiTU-GLU activation.
* ``TpuAllToAllTokenDispatcher``: TPU local-reorder VJPs without EP; with EP,
  its default Ring-of-Experts transport (``components/ring_of_experts.py``):
  all-gather of the latent tokens, scores and expert ids, the local experts on
  a static ``max_R``-row buffer, and a reduce-scatter of the per-token sums.
  The scores are applied after W2, as upstream does.
* ``TpuRoutedExperts``: optional Ring-of-Experts token chunking.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch.fx.traceback import annotate
from torchtitan.config.override import derive, override

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.components.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExperts,
)
from torchtitan.experiments.tpu.kimi_k3.quantile_balance import (
    QuantileBalancedKimiLatentMoE,
)
from torchtitan.experiments.tpu.kimi_k3.sharding import reduce_tp_partial, tp_group
from torchtitan.models.common.moe import _tokens_per_expert  # pylint: disable=protected-access
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.kimi_k3.moe import KimiLatentMoE

# Shape suffixes: R = routed rows, F = expert hidden dimension.

# MegaBlox tiling for K3's experts (latent dim 3584, hidden dim 3072): every
# full-K or full-N tile runs out of VMEM. TPU7x sweep at 262,144 rows over 56
# experts (see NOTES): fwd 55.6 ms, fwd+bwd 149.5 ms. Zero selects the full
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
  """Kimi's SiTU-GLU in FP32 (upstream ``kimi_k3.moe._situ_glu``)."""
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
    super().__init__(config)
    self.beta = config.beta
    self.linear_beta = config.linear_beta

  def _glu(self, gate_RF: torch.Tensor, up_RF: torch.Tensor) -> torch.Tensor:
    return _situ_glu(gate_RF, up_RF, self.beta, self.linear_beta)


class TpuKimiLatentMoE(KimiLatentMoE):
  """``KimiLatentMoE`` with the routed branch on a token shard under TP.

  Without TP this is upstream's forward. With TP (layouts in
  ``kimi_k3/sharding.py``) the two ranks of a TP group hold the same tokens,
  so each routes and dispatches only its half (``I -> S(0)``; the backward
  all-gathers the input gradient): the router, ``routed_down``, the routed
  experts (Ring-of-Experts under EP), ``routed_norm`` and ``routed_up`` see
  this rank's tokens only. The routed output is zero-filled to a TP partial
  sum (``S(0) -> P``), added to the shared experts' rowwise partial sum, and
  both are reduced with one all-reduce, as upstream torchtitan's MoE does
  for EP x TP.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(KimiLatentMoE.Config):
    pass

  def forward(self, x_TD: torch.Tensor, **router_kwargs) -> torch.Tensor:
    tp = tp_group()
    routed_x_TD = x_TD
    if tp is not None:
      # "TP" marks the backward token all-gather as activation traffic for
      # GraphSplitter's fsdp_pipeline (graph_trainer/graph_split.py).
      with annotate({"TP": "token_shard"}):
        routed_x_TD = spmd.redistribute(
            x_TD,
            tp,
            src=spmd.I,
            dst=spmd.S(0),
            backward_options={"op_dtype": x_TD.dtype},
        )
    weights_TK, expert_ids_TK, routing_map_TE = self.router(
        routed_x_TD, self.expert_bias_E, **router_kwargs
    )
    num_tokens_per_expert_E = (
        routing_map_TE.sum(dim=0)
        if routing_map_TE is not None
        else _tokens_per_expert(expert_ids_TK, self.tokens_per_expert_E.numel())
    )
    if self.training:
      with torch.no_grad():
        self.tokens_per_expert_E.add_(num_tokens_per_expert_E)

    routed_TD = self.routed_experts(
        self.routed_down(routed_x_TD),
        weights_TK,
        expert_ids_TK,
        num_tokens_per_expert_E,
    )
    out_TD = self.routed_up(self.routed_norm(routed_TD))
    if tp is not None:
      # Zero-fill the other ranks' tokens: no communication.
      out_TD = spmd.convert(
          out_TD, tp, src=spmd.S(0), dst=spmd.P, expert_mode=True
      )
    if self.shared_experts is not None:
      out_TD = out_TD + self.shared_experts(x_TD)
    if tp is not None:
      out_TD = reduce_tp_partial(out_TD, tp)
    return out_TD


class TpuQuantileBalancedKimiLatentMoE(
    TpuKimiLatentMoE, QuantileBalancedKimiLatentMoE
):
  """``TpuKimiLatentMoE`` with ``QuantileBalancedKimiLatentMoE``'s expert bias."""

  @dataclass(kw_only=True, slots=True)
  class Config(TpuKimiLatentMoE.Config, QuantileBalancedKimiLatentMoE.Config):
    pass


# Source config type -> TPU config type.
_TPU_LATENT_MOE = {
    KimiLatentMoE.Config: TpuKimiLatentMoE.Config,
    QuantileBalancedKimiLatentMoE.Config: (
        TpuQuantileBalancedKimiLatentMoE.Config
    ),
}


@override(
    target=KimiLatentMoE.Config,
    description="Kimi K3 LatentMoE on TPU: MegaBlox experts, Ring-of-Experts EP.",
)
def tpu_latent_moe(
    cfg: KimiLatentMoE.Config,
    *,
    capacity_factor: float | None = None,
    num_moe_token_chunks: int = 1,
) -> TpuKimiLatentMoE.Config:
  """K3 LatentMoE with TPU experts and the TPU dispatcher (TP-aware).

  ``KimiLatentMoE`` becomes ``TpuKimiLatentMoE`` and
  ``QuantileBalancedKimiLatentMoE`` becomes ``TpuQuantileBalancedKimiLatentMoE``.

  ``capacity_factor`` (Ring-of-Experts rows per EP rank, in units of the local
  routes; None is drop-free) and ``num_moe_token_chunks`` (Ring-of-Experts
  token chunks) apply only under EP; see ``TpuAllToAllTokenDispatcher`` and
  ``TpuRoutedExperts``.
  """
  tpu_config_cls = _TPU_LATENT_MOE.get(type(cfg))
  if tpu_config_cls is None:
    raise ValueError(
        f"tpu_latent_moe does not support {type(cfg).__qualname__}"
    )
  routed = cfg.routed_experts
  if type(routed.token_dispatcher) is not AllToAllTokenDispatcher.Config:
    raise ValueError(
        "tpu_latent_moe replaces the standard dispatcher, got"
        f" {type(routed.token_dispatcher).__qualname__}"
    )
  return derive(
      cfg,
      tpu_config_cls,
      routed_experts=derive(
          routed,
          TpuRoutedExperts.Config,
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
              capacity_factor=capacity_factor,
          ),
          num_moe_token_chunks=num_moe_token_chunks,
      ),
  )
