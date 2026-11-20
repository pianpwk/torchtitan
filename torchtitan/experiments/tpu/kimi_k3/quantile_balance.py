"""Quantile balancing for Kimi K3's MoE router (K3 report sec. 2.3.3, App. D).

K3 balances its experts with this alone: auxiliary-loss-free routing, as in the
Kimi K3 report.

A copy of upstream torchtitan's implementation (pytorch/torchtitan#4577; the
code as of commit ff36f8471), which this fork predates:

* ``QuantileBalancedTopKRouter``, ``QuantileBalancer`` and
  ``register_moe_quantile_balancing_hook`` from
  ``torchtitan/models/common/moe/moe.py``.
* ``QuantileBalancedKimiLatentMoE``: the expert-bias handling of upstream's
  ``torchtitan/models/kimi_k3/moe.py::KimiLatentMoE`` (an always-present
  persistent ``expert_bias_E``, and no sign-based ``load_balance_coeff``).

The router takes the biased top-(k+1) scores, routes on the first k and keeps
the (k+1)-th as the cutoff. Every token and expert adds ``cutoff - score``
(the bias the expert would need to be selected) to a per-expert histogram of
``num_bins`` uniform bins over ``[min(bias) - 1, max(bias) + 1]``. Before each
optimizer step the hook sums the histograms over the token-sharding groups and
sets each expert's bias to the ``top_k / num_experts`` quantile of its
histogram, mean-centered.

Adaptations to this fork (TODO: drop this file once the fork rebases onto
upstream's ``common/moe``):

* The fork's MoE, not its router, owns ``tokens_per_expert_E``; the hook zeroes
  it there.
* The fork's router has no padding mask, so ``observe`` always gets None.
* The fork's router names its score function by string, and its node-limited
  routing and ``_debug_force_load_balance`` bypass the quantile cutoff; the
  router rejects both.
* The fork's MoE has no ``freeze_expert_bias``.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch import nn
from torch.optim import Optimizer
import torch_remat as remat

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.kimi_k3.moe import KimiLatentMoE
from torchtitan.protocols.module import Module

# Shape suffixes: T = tokens, E = experts, B = histogram bins, L = MoE layers.


class QuantileBalancedTopKRouter(TokenChoiceTopKRouter):
  """Top-k router that uses a biased Top-(k+1) cutoff during training."""

  @dataclass(kw_only=True, slots=True)
  class Config(TokenChoiceTopKRouter.Config):
    num_bins: int

  def __init__(self, config: Config):
    super().__init__(config)
    if self.score_func != "sigmoid":
      raise ValueError("Quantile balancing requires sigmoid router scores.")
    if self.num_expert_groups is not None or self._debug_force_load_balance:
      raise ValueError(
          "Quantile balancing does not support node-limited routing or"
          " _debug_force_load_balance, which bypass the top-(k+1) cutoff."
      )
    self.quantile_balancer = QuantileBalancer.Config(
        num_experts=self.num_experts,
        top_k=self.top_k,
        num_bins=config.num_bins,
    ).build()

  def _select_experts(
      self,
      scores_TE: torch.Tensor,
      expert_bias_E: torch.Tensor | None = None,
      **router_kwargs,
  ) -> torch.Tensor:
    del router_kwargs
    if expert_bias_E is None:
      raise ValueError("Quantile balancing requires an expert bias.")
    biased_scores_TE = scores_TE + expert_bias_E
    topk_plus_one_scores, topk_plus_one_expert_ids = torch.topk(
        biased_scores_TE,
        k=self.top_k + 1,
        dim=-1,
        sorted=True,
    )
    if self.training and not remat.is_recomputing():
      self.quantile_balancer.observe(
          scores_TE,
          topk_plus_one_scores[:, self.top_k :],
          expert_bias_E,
          None,
      )
    return topk_plus_one_expert_ids[:, : self.top_k].contiguous()


class QuantileBalancer(Module):
  """Accumulate and recover histogram-based quantile bias updates.

  For scores bounded between zero and one, required expert biases lie between
  the current minimum bias minus one and maximum bias plus one. Each training
  micro-batch is accumulated into uniform bins over that interval.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Module.Config):
    num_experts: int
    top_k: int
    num_bins: int

  def __init__(self, config: Config):
    super().__init__()
    if not 0 < config.top_k < config.num_experts:
      raise ValueError("top_k must be between zero and num_experts.")
    self.num_experts = config.num_experts
    self.top_k = config.top_k
    self.num_bins = config.num_bins
    self.register_buffer(
        "required_bias_histogram_EB",
        torch.zeros(config.num_experts, config.num_bins, dtype=torch.int32),
        persistent=False,
    )

  def observe(
      self,
      scores_TE: torch.Tensor,
      cutoff_T1: torch.Tensor,
      expert_bias_E: torch.Tensor,
      padding_mask_T: torch.Tensor | None,
  ) -> None:
    """Accumulate required-bias histograms for one local micro-batch."""
    if not self.training:
      return

    with spmd.no_typecheck(), torch.no_grad():
      lower_bound = expert_bias_E.min() - 1.0
      bin_width = (
          expert_bias_E.max() - expert_bias_E.min() + 2.0
      ) / self.num_bins
      required_bias_TE = cutoff_T1 - scores_TE
      bin_indices_TE = torch.floor(
          (required_bias_TE - lower_bound) / bin_width
      ).to(torch.int64)
      bin_indices_ET = bin_indices_TE.clamp_(0, self.num_bins - 1).transpose(
          0, 1
      )
      histogram_updates_ET = torch.ones_like(
          bin_indices_ET,
          dtype=self.required_bias_histogram_EB.dtype,
      )
      if padding_mask_T is not None:
        histogram_updates_ET = histogram_updates_ET * (
            ~padding_mask_T
        ).unsqueeze(0)
      self.required_bias_histogram_EB.scatter_add_(
          1,
          bin_indices_ET,
          histogram_updates_ET,
      )

  def estimate_expert_bias(
      self,
      histogram_EB: torch.Tensor,
      expert_bias_E: torch.Tensor,
  ) -> torch.Tensor:
    """Estimate the next mean-centered expert bias from the histogram."""
    counts_E = histogram_EB.sum(dim=-1, dtype=torch.int64)
    target_count_E = counts_E.float() * (self.top_k / self.num_experts)
    cumulative_counts_EB = histogram_EB.cumsum(dim=-1, dtype=torch.int64)
    target_rank_E = target_count_E.ceil().to(torch.int64)
    target_bin_E = (cumulative_counts_EB < target_rank_E.unsqueeze(-1)).sum(
        dim=-1
    )

    target_bin_E1 = target_bin_E.unsqueeze(-1)
    counts_in_bin_E = histogram_EB.gather(-1, target_bin_E1).squeeze(-1)
    counts_before_E = (
        cumulative_counts_EB.gather(-1, target_bin_E1).squeeze(-1)
        - counts_in_bin_E
    )
    fraction_E = (
        target_count_E - counts_before_E.float()
    ) / counts_in_bin_E.float()

    bin_width = (
        expert_bias_E.max() - expert_bias_E.min() + 2.0
    ) / self.num_bins
    quantile_position_E = target_bin_E.float() + fraction_E
    return (quantile_position_E - quantile_position_E.mean()) * bin_width

  def _init_self_buffers(
      self, *, buffer_device: torch.device | None = None
  ) -> None:
    if buffer_device is None:
      buffer_device = self.required_bias_histogram_EB.device
    with torch.device(buffer_device):
      self.required_bias_histogram_EB = torch.zeros(
          self.num_experts,
          self.num_bins,
          dtype=torch.int32,
      )


class QuantileBalancedKimiLatentMoE(KimiLatentMoE):
  """``KimiLatentMoE`` whose expert bias is set by quantile balancing."""

  @dataclass(kw_only=True, slots=True)
  class Config(KimiLatentMoE.Config):
    pass

  def __init__(self, config: Config):
    if config.load_balance_coeff is not None:
      raise ValueError(
          "KimiLatentMoE cannot combine sign-based and quantile balancing."
      )
    if not isinstance(config.router, QuantileBalancedTopKRouter.Config):
      raise ValueError(
          "QuantileBalancedKimiLatentMoE needs a QuantileBalancedTopKRouter,"
          f" got {type(config.router).__qualname__}"
      )
    super().__init__(config)
    del self.expert_bias_E
    self.register_buffer(
        "expert_bias_E",
        torch.zeros(config.num_experts, dtype=torch.float32),
        persistent=True,
    )

  def _init_self_buffers(
      self, *, buffer_device: torch.device | None = None
  ) -> None:
    if buffer_device is None:
      buffer_device = self.tokens_per_expert_E.device
    super()._init_self_buffers(buffer_device=buffer_device)
    with torch.device(buffer_device):
      self.expert_bias_E = torch.zeros(
          self.router.num_experts,
          dtype=torch.float32,
      )


def register_moe_quantile_balancing_hook(
    optimizers: Optimizer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
  """Update quantile-balanced expert biases before each optimizer step."""
  moe_layers: list[tuple[KimiLatentMoE, QuantileBalancedTopKRouter]] = []
  for model_part in model_parts:
    for module in model_part.modules():
      if isinstance(module, KimiLatentMoE) and isinstance(
          module.router, QuantileBalancedTopKRouter
      ):
        moe_layers.append((module, module.router))
  if not moe_layers:
    return

  @torch.no_grad()
  def _update_expert_bias() -> None:
    reduction_groups = []
    loss_mesh = parallel_dims.get_optional_mesh("loss")
    if loss_mesh is not None:
      reduction_groups.append(loss_mesh.get_group())

    histograms = [
        router.quantile_balancer.required_bias_histogram_EB
        for _moe, router in moe_layers
    ]
    if reduction_groups:
      reduced_histograms_LEB = torch.stack(histograms)
      for group in reduction_groups:
        torch.distributed.all_reduce(
            reduced_histograms_LEB,
            group=group,
            op=torch.distributed.ReduceOp.SUM,
        )
      histograms = list(reduced_histograms_LEB.unbind())

    for histogram_EB, (moe, router) in zip(
        histograms,
        moe_layers,
        strict=True,
    ):
      expert_bias_E = moe.expert_bias_E
      assert expert_bias_E is not None
      quantile_balancer = router.quantile_balancer
      next_expert_bias_E = quantile_balancer.estimate_expert_bias(
          histogram_EB,
          expert_bias_E,
      )
      expert_bias_E.copy_(next_expert_bias_E)
      quantile_balancer.required_bias_histogram_EB.zero_()
      moe.tokens_per_expert_E.zero_()

  optimizers.register_step_pre_hook(
      lambda *args, **kwargs: _update_expert_bias()
  )
