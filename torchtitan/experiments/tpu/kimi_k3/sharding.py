"""Tensor-parallel (TP) parameter layouts for Kimi K3 on TPU.

Upstream's K3 declares only its data-parallel and expert layouts
(``models/kimi_k3/sharding.py``) and rejects TP. With TP > 1 the TPU K3
(``GraphTrainerKimiK3Model``) declares the TP layout of every parameter here;
the TP collectives on activations run in the TPU layers' forwards
(``TpuKDA``, ``TpuKimiMLAAttention``, ``TpuKimiLatentMoE``), except for the
dense FFN of layer 0, which keeps upstream's ``FeedForward`` and declares its
redistributions here. No sequence parallelism: the residual stream is
invariant (``I``) over TP and every rank computes it whole.

Per TP axis type (the gradient semantics follow from it, see
``graph_trainer/simple_fsdp.py``):

* ``S(0)`` / ``S(1)``: colwise / rowwise shards. Attention and FFN shards are
  whole heads (KDA, MLA) or whole hidden units.
* ``R``: a replicated weight whose input is replicated (``R``) or a token
  shard, so its gradient is partial over TP; SimpleFSDP all-reduces it.
* ``I`` (default for every parameter not listed): used only on the invariant
  residual stream (norms, attention residuals), so every rank already holds
  the whole gradient.

The embedding and LM head are vocab-sharded as upstream's decoders declare
them (``set_decoder_sharding_config``): the loss under TP is loss-parallel
cross-entropy.

Layer by layer:

* KDA: q/k/v, ``forget_b``, ``output_gate`` and the short-conv weights shard
  heads; ``forget_a`` (model dim -> head dim), the per-head ``output_norm``
  weight and the small per-head ``beta``, ``A_log`` and ``dt_bias`` (sliced
  to this rank's heads in the forward) are ``R``; ``output_proj`` is rowwise.
* MLA: ``wq_b``, ``wkv_b`` and ``gate`` shard heads; the LoRA down projections
  and their norms are ``R`` (the shared ``k_rope`` slice of ``wkv_a`` feeds
  every head); ``wo`` is rowwise.
* LatentMoE: the routed branch (router, ``routed_down`` / ``routed_norm`` /
  ``routed_up``, routed experts) runs on a token shard per TP rank, so its
  weights are ``R`` without EP. With EP the routed experts keep upstream's
  sparse-mesh layout (the EP axis contains the TP pair). Shared experts are
  colwise / rowwise.
"""

import spmd_types as spmd
import torch

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
)
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.moe_sharding import expert_param_placement_sparse
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.protocols.sharding import ShardingConfig


def _token_shard_counts() -> spmd.SpmdType:
  """Expert counts of a token shard: partial over every token-sharding axis."""
  return spmd.SpmdType({
      MeshAxisName.DP: spmd.P,
      MeshAxisName.CP: spmd.P,
      MeshAxisName.TP: spmd.P,
  })


def _state(**placements: spmd.PerMeshAxisSpmdType) -> ShardingConfig:
  return ShardingConfig(
      state_shardings={
          name: dense_param_placement(tp=placement)
          for name, placement in placements.items()
      }
  )


def _colwise() -> ShardingConfig:
  return _state(weight=spmd.S(0))


def _rowwise() -> ShardingConfig:
  return _state(weight=spmd.S(1))


def _replicated() -> ShardingConfig:
  return _state(weight=spmd.R)


def _set_kda_sharding(kda) -> None:
  # The per-head A_log, dt_bias and `beta` (model dim -> heads) are small and
  # replicated; `TpuKDA` slices this rank's heads. As TP shards, their dim 0
  # (heads) can be smaller than the FSDP degree, which SimpleFSDP's
  # replicate-if-small check (on the global shape) does not handle.
  kda.sharding_config = _state(A_log=spmd.R, dt_bias=spmd.R)
  for name in ("q_proj", "k_proj", "v_proj", "forget_b", "output_gate",
               "q_conv", "k_conv", "v_conv"):
    getattr(kda, name).sharding_config = _colwise()
  kda.beta.sharding_config = _replicated()
  kda.forget_a.sharding_config = _replicated()
  kda.output_norm.sharding_config = _replicated()
  kda.output_proj.sharding_config = _rowwise()
  # Upstream's DP-only local region around Attention Gym; the TPU inner KDA
  # takes local head shards.
  kda.inner_kda.sharding_config = None


def _set_mla_sharding(mla) -> None:
  for name in ("wq_b", "wkv_b", "gate"):
    getattr(mla, name).sharding_config = _colwise()
  for name in ("wq_a", "q_norm", "wkv_a", "kv_norm"):
    getattr(mla, name).sharding_config = _replicated()
  mla.wo.sharding_config = _rowwise()


def _set_latent_moe_sharding(moe, *, enable_ep: bool) -> None:
  moe.sharding_config = ShardingConfig(
      state_shardings={
          "expert_bias_E": dense_param_placement(tp=spmd.R),
          "tokens_per_expert_E": _token_shard_counts(),
      }
  )
  moe.router.gate.sharding_config = _replicated()
  moe.routed_down.sharding_config = _replicated()
  moe.routed_norm.sharding_config = _replicated()
  moe.routed_up.sharding_config = _replicated()

  token_shard = dense_sequence_parallel_placement()
  counts = _token_shard_counts()
  in_types = {
      "x_TD": token_shard,
      "topk_scores_TK": token_shard,
      "topk_expert_ids_TK": token_shard,
      "num_local_tokens_per_expert_E": counts,
  }
  routed = moe.routed_experts
  routed.sharding_config = ShardingConfig(
      in_src_shardings=in_types,
      in_dst_shardings=in_types,
      out_src_shardings=token_shard,
      out_dst_shardings=token_shard,
      local_spmd=True,
  )
  params = ("w1_EFD", "w2_EDF", "w3_EFD")
  routed.inner_experts.sharding_config = ShardingConfig(
      state_shardings={
          name: (
              expert_param_placement_sparse()
              if enable_ep
              else dense_param_placement(tp=spmd.R)
          )
          for name in params
      }
  )

  shared = moe.shared_experts
  if shared is not None:
    shared.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        },
    )
    shared.w1.sharding_config = _colwise()
    shared.w3.sharding_config = _colwise()
    # Partial output: `TpuKimiLatentMoE` adds the zero-filled routed output
    # and reduces both with one all-reduce.
    shared.w2.sharding_config = _rowwise()


def set_kimi_k3_tp_sharding_config(
    config: KimiK3Model.Config, *, enable_ep: bool
) -> None:
  """Declares the TP layout of every TP-sharded or TP-reduced K3 parameter.

  Runs after upstream's ``set_kimi_k3_sharding_config`` and replaces its
  per-layer declarations. Parameters left undeclared are invariant over TP
  (``annotate_tp_invariant_parameters``).
  """
  if config.vision_encoder is not None:
    raise ValueError("TPU K3 TP supports the text-only model")
  # Vocab-sharded embedding and LM head (the loss is TP loss-parallel).
  set_decoder_sharding_config(config, enable_sp=False)
  for layer in config.layers:
    if layer.delta_attention is not None:
      _set_kda_sharding(layer.delta_attention)
    if layer.attention is not None:
      _set_mla_sharding(layer.attention)
    if layer.moe is not None:
      _set_latent_moe_sharding(layer.moe, enable_ep=enable_ep)
    if layer.feed_forward is not None:
      set_dense_ffn_sharding(
          layer.feed_forward,
          attn_x_layout=dense_activation_placement(tp=spmd.I, cp=spmd.S(0)),
          enable_sp=False,
      )


def tp_group() -> torch.distributed.ProcessGroup | None:
  """The dense TP group of the current SPMD mesh, or None without TP."""
  return spmd_mesh_group(MeshAxisName.TP)


def enter_tp_region(x: torch.Tensor, group) -> torch.Tensor:
  """``I -> R``: the identity forward, the TP all-reduce of the grad backward."""
  return spmd.convert(
      x, group, src=spmd.I, dst=spmd.R, backward_options={"op_dtype": x.dtype}
  )


def reduce_tp_partial(x: torch.Tensor, group) -> torch.Tensor:
  """``P -> I``: the TP all-reduce forward, the identity backward."""
  return spmd.redistribute(
      x, group, src=spmd.P, dst=spmd.I, backward_options={"op_dtype": x.dtype}
  )
