"""TPU recipes for Kimi K3 (text only) on the TPU GraphTrainer."""

import dataclasses
import os
from typing import Literal

import torch
from torch.distributed.tensor import Shard
from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import (
    default_adamw,
    OptimizersContainer,
    ParamGroupConfig,
)
from torchtitan.components.validate import Validator
from torchtitan.config.configs import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.experiments.tpu.components.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.graph_trainer.config import (
    TpuGraphTrainerCompileConfig,
)
from torchtitan.experiments.tpu.graph_trainer.kimi_k3 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.kimi_k3 import model_registry
from torchtitan.experiments.tpu.kimi_k3.attention_residual import (
    use_tpu_attention_residual,
)
from torchtitan.experiments.tpu.spmd_dist_muon import TpuOptimizersContainer
from torchtitan.experiments.tpu.tpu_job_config import (
    TPUConfig,
    TPUTrainerConfig,
)

_OVERRIDES = "torchtitan.experiments.tpu.kimi_k3"


def graph_trainer_kimi_k3_tiny() -> TpuGraphTrainer.Config:
  """The ``tiny`` K3 on one TPU core with the three TPU layer overrides.

  One packed 4096-token row per step (``c4_test``, documents kept apart in
  KDA and MLA), AdamW with FP32 master weights and BF16 compute, full
  rematerialization, the whole step (forward, loss, backward) as one compiled
  executable. With ``--parallelism.expert_parallel_degree`` > 1 the routed
  experts use Ring-of-Experts at capacity factor 1.0.
  """
  seq_len = 4096
  base = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("tiny", seq_len=seq_len),
      dataloader=c4_test_dataloader(),
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=5,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.1,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=seq_len,
          max_context_length=seq_len,
          steps=40,
          mixed_precision_param="bfloat16",
          mixed_precision_reduce="float32",
      ),
      parallelism=ParallelismConfig(use_simple_fsdp=True),
      checkpoint=CheckpointManager.Config(interval=1000),
      activation_checkpoint=FullAC.Config(preserve_rng_state=False),
      validator=Validator.Config(enable=False),
      tpu_config=TPUConfig(enable_amp=False),
  )
  base.override.imports = [
      # Tokamax lays every document out from a 64-token chunk boundary, so a
      # 4096-token row costs 4096 + 63 * max_num_segments kernel tokens: 8
      # gives 4600 (32 gave 6112, ~8% slower K3-width KDA layers). The first
      # 40 c4_test rows hold at most 7 documents; documents past the 8th in a
      # row share the 8th's sequence (rare in c4: one 9-document row in the
      # k3_slice benchmark's first 96 rows).
      (f"{_OVERRIDES}.tokamax_kda.tokamax_kda_attention",
       {"max_num_segments": 8}),
      f"{_OVERRIDES}.tokamax_mla.tokamax_mla_attention",
      (f"{_OVERRIDES}.latent_moe.tpu_latent_moe", {"ep_capacity_factor": 1.0}),
  ]
  cfg = TpuGraphTrainer.Config.from_tpu_trainer_config(
      base, graph_trainer_model_registry
  )
  cfg.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      mode="aot_fx_trace",
      memory_policy="full",
  )
  return cfg


# Benchmark recipes use every K3 kernel option (see the kimi_k3 module
# docstring): Pallas short conv and output norm (with the Tokamax or the fused
# KDA kernel), gmm_glu experts.
_PALLAS_TOKAMAX_KDA = (
    f"{_OVERRIDES}.tokamax_kda.tokamax_kda_attention_pallas_conv_norm",
    {"max_num_segments": 8},
)
_FUSED_KDA = f"{_OVERRIDES}.fused_kda.fused_kda_attention"


def _graph_trainer_kimi_k3_slice(
    flavor: str, *, kda_override: str | tuple = _PALLAS_TOKAMAX_KDA
) -> TpuGraphTrainer.Config:
  """A K3-width ``k3_slice_*`` flavor for comm / memory benchmarks.

  Eight packed 4096-token rows (32,768 tokens) per data-parallel rank and
  microbatch; pass ``--training.num_tokens_per_microbatch_per_dp_rank
  16384`` for half of that. Balanced forced routing (see the flavor)
  and Ring-of-Experts at capacity factor 1.0; full rematerialization, the
  whole step as one compiled executable, AdamW eager. The K3 kernel options
  are on: ``kda_override`` (Tokamax KDA with the Pallas conv / norm, or the
  fused KDA), gmm_glu experts and the TPU attention residual.
  """
  cfg = graph_trainer_kimi_k3_tiny()
  seq_len = 4096
  tokens = 8 * seq_len
  cfg.model_spec = dataclasses.replace(
      graph_trainer_model_registry(flavor), max_context_length=seq_len
  )
  use_tpu_attention_residual(cfg.model_spec.model)
  cfg.override.imports = [
      kda_override,
      f"{_OVERRIDES}.tokamax_mla.tokamax_mla_attention",
      (f"{_OVERRIDES}.latent_moe.tpu_latent_moe_gmm_glu",
       {"ep_capacity_factor": 1.0}),
  ]
  cfg.training.num_tokens_per_microbatch_per_dp_rank = tokens
  cfg.training.steps = 8
  # Balanced round-robin routing. The trainer sets every router's
  # ``_debug_force_load_balance`` from this field when it finalizes the model
  # config, so a flag set in the flavor alone is overwritten with False.
  cfg.debug.moe_force_load_balance = True
  return cfg


def graph_trainer_kimi_k3_slice_e448() -> TpuGraphTrainer.Config:
  """``k3_slice_e448``: 56 experts per core on EP8."""
  return _graph_trainer_kimi_k3_slice("k3_slice_e448")


def graph_trainer_kimi_k3_slice_e896() -> TpuGraphTrainer.Config:
  """``k3_slice_e896``: 56 experts per core on EP16."""
  return _graph_trainer_kimi_k3_slice("k3_slice_e896")


def graph_trainer_kimi_k3_deep8_e448() -> TpuGraphTrainer.Config:
  """``k3_deep8_e448``: eight layers in K3's KDA:MLA mix, EP8."""
  return _graph_trainer_kimi_k3_slice("k3_deep8_e448")


def graph_trainer_kimi_k3_deep8_e896() -> TpuGraphTrainer.Config:
  """``k3_deep8_e896``: eight layers in K3's KDA:MLA mix, EP16."""
  return _graph_trainer_kimi_k3_slice("k3_deep8_e896")


def graph_trainer_kimi_k3_slice_e64() -> TpuGraphTrainer.Config:
  """``k3_slice_e64``: small expert count for debugging on 2x2x1."""
  return _graph_trainer_kimi_k3_slice("k3_slice_e64")


def graph_trainer_kimi_k3_deep8_e448_fused_kda() -> TpuGraphTrainer.Config:
  """``k3_deep8_e448`` with the fused Pallas KDA kernels (``fused_kda.py``)."""
  return _graph_trainer_kimi_k3_slice("k3_deep8_e448", kda_override=_FUSED_KDA)


def graph_trainer_kimi_k3_half47_e896_fused_kda() -> TpuGraphTrainer.Config:
  """``k3_half47_e896`` (half-depth K3, full widths, 56 experts per core on
  EP16) with the fused Pallas KDA kernels and host offload of the saved layer
  boundaries (full recompute + ``full_and_offload``)."""
  cfg = _graph_trainer_kimi_k3_slice("k3_half47_e896", kda_override=_FUSED_KDA)
  cfg.compile.memory_policy = "full_and_offload"
  return cfg


def graph_trainer_kimi_k3_full93_e896_fused_kda() -> TpuGraphTrainer.Config:
  """``k3_full93_e896`` (full-depth K3, 56 experts per core on EP16) with the
  fused Pallas KDA kernels and host offload of the saved layer boundaries."""
  cfg = _graph_trainer_kimi_k3_slice("k3_full93_e896", kda_override=_FUSED_KDA)
  cfg.compile.memory_policy = "full_and_offload"
# Mesh axes a dense K3 parameter can be stored on: SimpleFSDP's ``fsdp`` (FSDP2
# names it ``dp_shard``) and ``tp``. Layout axes absent from a parameter's mesh
# are ignored, so one layout covers FSDP and FSDP x TP.
_DENSE_AXES = ("fsdp", "dp_shard", "tp")

# Muon matrices by raw parameter name; every other parameter goes to AdamW:
# embedding, LM head, norms, KDA's A_log / dt_bias / short convolutions, the
# per-head scalar matrices of KDA's ``beta`` and the single-row attention
# residual projections (Newton-Schulz would only rescale those).
_K3_MUON_PATTERN = (
    r"(?:^|\.)(?:"
    r"delta_attention\.(?:q_proj|k_proj|v_proj|forget_a|forget_b|output_gate"
    r"|output_proj)\.weight"
    r"|attention\.(?:wq_a|wq_b|wkv_a|wkv_b|gate|wo)\.weight"
    r"|feed_forward\.w[123]\.weight"
    r"|moe\.(?:router\.gate|shared_experts\.w[123]|routed_down|routed_up)"
    r"\.weight"
    r"|moe\.routed_experts\.inner_experts\.w[123]_E[DF][DF]"
    r")$"
)
_K3_ROUTED_EXPERT_PATTERN = (
    r"(?:^|\.)moe\.routed_experts\.inner_experts\.w[123]_E[DF][DF]$"
)


# ``bucketing="by_shape"``: Muon matrix families that share one bucket across
# all layers (same matrix shape or same per-head block shape). Parameters not
# listed get a bucket of their own name (e.g. ``feed_forward.w1`` of layer 0).
_K3_SHAPE_FAMILIES = {
    "delta_attention.output_proj": "o_proj",
    "attention.wo": "o_proj",
    "moe.shared_experts.w1": "shared_w13",
    "moe.shared_experts.w3": "shared_w13",
    "moe.shared_experts.w2": "shared_w2",
    "moe.routed_down": "latent",
    "moe.routed_up": "latent",
    "moe.router.gate": "router",
    "delta_attention.forget_a": "lora_down",
    "attention.wq_a": "lora_down",
    "attention.wkv_a": "lora_down",
    # Per-head [head_dim, dim] blocks.
    "delta_attention.q_proj": "head_q",
    "delta_attention.k_proj": "head_k",
    "delta_attention.v_proj": "head_v",
    "delta_attention.output_gate": "head_gate",
    "attention.gate": "head_gate",
    # Small per-head blocks.
    "delta_attention.forget_b": "head_small",
    "attention.wq_b": "head_small",
    "attention.wkv_b": "head_small",
    "feed_forward.w1": "dense_ffn",
    "feed_forward.w2": "dense_ffn",
    "feed_forward.w3": "dense_ffn",
}


def _k3_dist_muon_optimizer(
    model_config,
    *,
    lr: float = 8e-4,
    bucketing: Literal["by_layer", "by_shape"] = "by_layer",
    spmd_expert_bucket_size: int | None = None,
) -> OptimizersContainer.Config:
  """DistMuon on every K3 matrix (upstream K3's split), AdamW on the rest.

  Layouts follow upstream's ``kimi_k3`` test recipe in this fork's parameter
  names, extended to FSDP x TP: each layout lists the dense TP axis next to
  the FSDP axis, so one transport group spans ``fsdp x tp``.

  * Per-head Newton-Schulz (``BlockShard`` rows) for the head-sharded
    projections: KDA q / k / v, ``forget_b``, ``output_gate``; MLA ``wq_b``
    ([q_nope; q_rope] per head), ``wkv_a`` ([kv_latent; k_rope]), ``wkv_b``
    ([k_nope; v] per head) and ``gate``.
  * Whole-matrix (``Owned``): KDA ``forget_a`` / ``output_proj``, MLA
    ``wq_a`` / ``wo``, the dense FFN and shared experts (w1, w2, w3), the
    router gate and the latent ``routed_down`` / ``routed_up``.
  * Routed experts ``[E, ., .]``: one matrix per expert, ``Shard(0)`` over the
    expert dim on the FSDP / expert-FSDP / EP axes (EP first, as stored), TP
    replicated as stored. Every rank then updates its own experts in place.

  Buckets (``bucketing``):

  * ``by_layer``: layer 0 alone, then pairs of layers, routed experts in their
    own bucket (upstream's grouping).
  * ``by_shape``: the non-expert matrices bucket by family across all layers
    (``_K3_SHAPE_FAMILIES``), routed experts as in ``by_layer``. On TPU every
    all-to-all pads each (source, destination) segment to the largest pair
    (equal splits), so a bucket costs every rank about the largest volume one
    owner receives. A bucket of many same-shape matrices keeps that close to the
    packed volume; a layer bucket mixing one large and many small matrices does
    not.

  With ``spmd_expert_bucket_size`` set, the routed experts move to a second
  optimizer, ``SpmdDistMuon`` (one captured ``spmd_safe`` TPU program; see
  ``experiments/tpu/spmd_dist_muon.py``), in buckets of that many same-shape
  expert tensors (w1 / w3 of consecutive MoE layers, or w2); the dense
  matrices stay in the eager DistMuon. A bucket is captured when its per-rank
  program is identical: ``P`` tensors of ``E`` local experts over an
  ``N``-rank transport group split into equal runs of ``k = P * E / N``
  experts when ``k`` divides ``E`` (``E`` = 56: ``N`` = 16 needs ``P`` = 2,
  ``N`` = 64 needs ``P`` = 8); other buckets run eagerly. Recipes using it also
  need ``_with_spmd_dist_muon_flags``.
  """
  bf16 = torch.bfloat16

  def dense(sharding) -> ComputeLayout:
    return ComputeLayout(
        shardings_by_mesh_axis={axis: sharding for axis in _DENSE_AXES},
        compute_dtype=bf16,
    )

  owned = dense(Owned())

  def blocks_of(*num_rows: int) -> ComputeLayout:
    return dense(BlockShard(dim=0, block_sizes=num_rows))

  per_expert = ComputeLayout(
      shardings_by_mesh_axis={
          axis: Shard(0) for axis in ("fsdp", "dp_shard", "efsdp", "ep")
      },
      shard_order_by_tensor_dim={0: ("ep", "efsdp")},
      compute_dtype=bf16,
  )

  layer_fqns: list[dict[str, ComputeLayout]] = []
  for layer_id, layer in enumerate(model_config.layers):
    prefix = f"layers.{layer_id}"
    shardings: dict[str, ComputeLayout] = {}
    if layer.delta_attention is not None:
      per_head = blocks_of(layer.delta_attention.head_dim)
      for name, layout in (
          ("q_proj", per_head), ("k_proj", per_head), ("v_proj", per_head),
          ("forget_a", owned), ("forget_b", per_head),
          ("output_gate", per_head), ("output_proj", owned),
      ):
        shardings[f"{prefix}.delta_attention.{name}.weight"] = layout
    if layer.attention is not None:
      mla = layer.attention
      for name, layout in (
          ("wq_a", owned),
          ("wq_b", blocks_of(mla.qk_nope_head_dim, mla.qk_rope_head_dim)),
          ("wkv_a", blocks_of(mla.kv_lora_rank, mla.qk_rope_head_dim)),
          ("wkv_b", blocks_of(mla.qk_nope_head_dim, mla.v_head_dim)),
          ("gate", blocks_of(mla.v_head_dim)),
          ("wo", owned),
      ):
        shardings[f"{prefix}.attention.{name}.weight"] = layout
    if layer.feed_forward is not None:
      for name in ("w1", "w2", "w3"):
        shardings[f"{prefix}.feed_forward.{name}.weight"] = owned
    if layer.moe is not None:
      for name in ("router.gate", "routed_down", "routed_up") + tuple(
          f"shared_experts.{w}" for w in ("w1", "w2", "w3")
      ):
        shardings[f"{prefix}.moe.{name}.weight"] = owned
      # TpuGroupedExperts' kernel-native (transposed) names.
      for name in ("w1_EDF", "w2_EFD", "w3_EDF"):
        shardings[f"{prefix}.moe.routed_experts.inner_experts.{name}"] = (
            per_expert
        )
    layer_fqns.append(shardings)

  num_layers = len(layer_fqns)
  bucket_layer_ids = [(0,)] + [
      tuple(range(first, min(first + 2, num_layers)))
      for first in range(1, num_layers, 2)
  ]
  bucket_configs = []
  for layer_ids in bucket_layer_ids:
    name = "layers." + "-".join(map(str, layer_ids))
    fqns = [fqn for i in layer_ids for fqn in layer_fqns[i]]
    routed = tuple(f for f in fqns if ".routed_experts." in f)
    dense = tuple(f for f in fqns if f not in routed)
    if bucketing == "by_layer":
      bucket_configs.append(BucketConfig(name=name, patterns=dense))
    if routed and spmd_expert_bucket_size is None:
      bucket_configs.append(
          BucketConfig(name=f"{name}.routed-experts", patterns=routed))
  if bucketing == "by_shape":
    # One bucket per matrix family across all layers (see the docstring).
    families: dict[str, list[str]] = {}
    for shardings in layer_fqns:
      for fqn in shardings:
        if ".routed_experts." in fqn:
          continue
        suffix = fqn.split(".", 2)[2].removesuffix(".weight")
        family = _K3_SHAPE_FAMILIES.get(suffix, suffix)
        families.setdefault(family, []).append(fqn)
    bucket_configs = [
        BucketConfig(name=family, patterns=tuple(fqns))
        for family, fqns in families.items()
    ] + bucket_configs
  layouts = {fqn: layout for d in layer_fqns for fqn, layout in d.items()}
  muon_kwargs = {
      "lr": lr,
      "weight_decay": 0.1,
      "adjust_lr_fn": "match_rms_adamw",
  }
  if spmd_expert_bucket_size is not None:
    # Same-shape classes: w1 / w3 are [E, D, F], w2 is [E, F, D].
    routed_by_class = {
        cls: [
            f for d in layer_fqns for f in d
            if f.rsplit(".", 1)[-1] in names
        ]
        for cls, names in (("w13", ("w1_EDF", "w3_EDF")), ("w2", ("w2_EFD",)))
    }
    expert_buckets = tuple(
        BucketConfig(
            name=f"routed-experts.{cls}.{i // spmd_expert_bucket_size}",
            patterns=tuple(fqns[i:i + spmd_expert_bucket_size]),
        )
        for cls, fqns in routed_by_class.items()
        for i in range(0, len(fqns), spmd_expert_bucket_size)
    )
    return TpuOptimizersContainer.Config(
        implementation="for-loop",
        param_groups=[
            ParamGroupConfig(
                pattern=_K3_ROUTED_EXPERT_PATTERN,
                optimizer_name="SpmdDistMuon",
                optimizer_kwargs=muon_kwargs,
            ),
            ParamGroupConfig(
                pattern=_K3_MUON_PATTERN,
                optimizer_name="DistMuon",
                optimizer_kwargs=muon_kwargs,
            ),
            _k3_adamw_rest(lr),
        ],
        optimizer_factory_kwargs_by_name={
            "SpmdDistMuon": {
                "compute_sharding_by_fqn": {
                    f: layouts[f] for b in expert_buckets for f in b.patterns
                },
                "bucket_configs": expert_buckets,
            },
            "DistMuon": {
                "compute_sharding_by_fqn": {
                    f: v for f, v in layouts.items()
                    if ".routed_experts." not in f
                },
                "bucket_configs": tuple(bucket_configs),
            },
        },
    )
  return OptimizersContainer.Config(
      implementation="for-loop",
      param_groups=[
          ParamGroupConfig(
              pattern=_K3_MUON_PATTERN,
              optimizer_name="DistMuon",
              optimizer_kwargs=muon_kwargs,
          ),
          _k3_adamw_rest(lr),
      ],
      optimizer_factory_kwargs_by_name={
          "DistMuon": {
              "compute_sharding_by_fqn": layouts,
              "bucket_configs": tuple(bucket_configs),
          }
      },
  )


def _k3_adamw_rest(lr: float) -> ParamGroupConfig:
  return ParamGroupConfig(
      pattern=r".*",
      optimizer_name="AdamW",
      optimizer_kwargs={
          "lr": lr,
          "betas": (0.9, 0.95),
          "eps": 1e-8,
          "weight_decay": 0.1,
      },
  )


def graph_trainer_kimi_k3_tiny_dist_muon() -> TpuGraphTrainer.Config:
  """``graph_trainer_kimi_k3_tiny`` with DistMuon (eager) on every matrix.

  The forward / backward stays one compiled executable; the optimizer runs
  eager. Works with FSDP, FSDP x TP2 and EP (``--parallelism.*``).
  """
  cfg = graph_trainer_kimi_k3_tiny()
  cfg.optimizer = _k3_dist_muon_optimizer(cfg.model_spec.model)
  return cfg


def graph_trainer_kimi_k3_tiny_dist_muon_by_shape() -> TpuGraphTrainer.Config:
  """``graph_trainer_kimi_k3_tiny_dist_muon`` with ``by_shape`` buckets."""
  cfg = graph_trainer_kimi_k3_tiny()
  cfg.optimizer = _k3_dist_muon_optimizer(
      cfg.model_spec.model, bucketing="by_shape"
  )
  return cfg


def graph_trainer_kimi_k3_slice_e448_dist_muon_by_shape(
) -> TpuGraphTrainer.Config:
  """``graph_trainer_kimi_k3_slice_e448_dist_muon``, ``by_shape`` buckets."""
  cfg = graph_trainer_kimi_k3_slice_e448()
  cfg.optimizer = _k3_dist_muon_optimizer(
      cfg.model_spec.model, bucketing="by_shape"
  )
  return cfg


def graph_trainer_kimi_k3_slice_l12_e64() -> TpuGraphTrainer.Config:
  """``k3_slice_l12_e64``: 12 K3-width layers, 64 experts (AdamW)."""
  return _graph_trainer_kimi_k3_slice("k3_slice_l12_e64")


def graph_trainer_kimi_k3_slice_l12_e64_dist_muon(
) -> TpuGraphTrainer.Config:
  """``k3_slice_l12_e64`` with DistMuon, ``by_layer`` buckets."""
  cfg = graph_trainer_kimi_k3_slice_l12_e64()
  cfg.optimizer = _k3_dist_muon_optimizer(cfg.model_spec.model)
  return cfg


def graph_trainer_kimi_k3_slice_l12_e64_dist_muon_by_shape(
) -> TpuGraphTrainer.Config:
  """``k3_slice_l12_e64`` with DistMuon, ``by_shape`` buckets."""
  cfg = graph_trainer_kimi_k3_slice_l12_e64()
  cfg.optimizer = _k3_dist_muon_optimizer(
      cfg.model_spec.model, bucketing="by_shape"
  )
  return cfg


def graph_trainer_kimi_k3_slice_e448_dist_muon() -> TpuGraphTrainer.Config:
  """``graph_trainer_kimi_k3_slice_e448`` with DistMuon (eager)."""
  cfg = graph_trainer_kimi_k3_slice_e448()
  cfg.optimizer = _k3_dist_muon_optimizer(cfg.model_spec.model)
  return cfg


def graph_trainer_kimi_k3_slice_e64_dist_muon() -> TpuGraphTrainer.Config:
  """``graph_trainer_kimi_k3_slice_e64`` with DistMuon (eager)."""
  cfg = graph_trainer_kimi_k3_slice_e64()
  cfg.optimizer = _k3_dist_muon_optimizer(cfg.model_spec.model)
  return cfg


def graph_trainer_kimi_k3_deep8_e448_dist_muon() -> TpuGraphTrainer.Config:
  """``graph_trainer_kimi_k3_deep8_e448`` with DistMuon (eager, experts too)."""
  cfg = graph_trainer_kimi_k3_deep8_e448()
  cfg.optimizer = _k3_dist_muon_optimizer(cfg.model_spec.model)
  return cfg


def graph_trainer_kimi_k3_deep8_e896_dist_muon() -> TpuGraphTrainer.Config:
  """``graph_trainer_kimi_k3_deep8_e896`` with DistMuon (eager, experts too)."""
  cfg = graph_trainer_kimi_k3_deep8_e896()
  cfg.optimizer = _k3_dist_muon_optimizer(cfg.model_spec.model)
  return cfg


def _with_spmd_dist_muon_flags(cfg: TpuGraphTrainer.Config) -> None:
  """TPU flags ``SpmdDistMuon`` needs (see ``_k3_dist_muon_optimizer``).

  * Async all-to-all: XLA's TPU all-to-all is synchronous by default, so
    nothing could overlap a bucket's exchange.
  * A per-rank XLA dump (post-optimization HLO as text) of the DistMuon
    programs: the startup check compares the captured program's collective
    order across ranks. This process's rank is read from torchrun's ``RANK``
    because XLA latches the dump flags before the trainer exists.
  """
  rank = os.environ.get("RANK", "0")
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags, xla_tpu_enable_async_all_to_all=True
  )
  cfg.xla_flags = tpu_flags.merge_flags(
      cfg.xla_flags,
      xla_dump_to=f"/tmp/spmd_dist_muon_xla_dump/r{rank}",
      xla_dump_hlo_as_text=True,
      # TorchTPU names a program after the source of its last op: the
      # captured optimizer ends in dist_muon.py's update. Dumping only those
      # modules keeps the rest of the step's compiles fast.
      xla_dump_hlo_module_re=".*dist_muon.*",
  )


def graph_trainer_kimi_k3_slice_e64_spmd_dist_muon() -> TpuGraphTrainer.Config:
  """``graph_trainer_kimi_k3_slice_e64_dist_muon`` with captured experts.

  Routed experts in ``SpmdDistMuon`` (one bucket per expert tensor, async
  all-to-all), dense matrices in the eager DistMuon (``by_layer``).
  """
  cfg = graph_trainer_kimi_k3_slice_e64()
  cfg.optimizer = _k3_dist_muon_optimizer(
      cfg.model_spec.model, spmd_expert_bucket_size=1
  )
  _with_spmd_dist_muon_flags(cfg)
  return cfg


def graph_trainer_kimi_k3_slice_e448_spmd_dist_muon() -> TpuGraphTrainer.Config:
  """``k3_slice_e448`` with captured experts in buckets of two tensors.

  Two same-shape expert tensors per bucket split evenly at 56 local experts
  over eFSDP 16 (EP8 on 4x4x4) or 28 over eFSDP 8 (EP16).
  """
  cfg = graph_trainer_kimi_k3_slice_e448()
  cfg.optimizer = _k3_dist_muon_optimizer(
      cfg.model_spec.model, spmd_expert_bucket_size=2
  )
  _with_spmd_dist_muon_flags(cfg)
  return cfg


def graph_trainer_kimi_k3_deep8_e448_spmd_dist_muon() -> TpuGraphTrainer.Config:
  """``k3_deep8_e448`` with captured experts (see the e448 slice recipe)."""
  cfg = graph_trainer_kimi_k3_deep8_e448()
  cfg.optimizer = _k3_dist_muon_optimizer(
      cfg.model_spec.model, spmd_expert_bucket_size=2
  )
  _with_spmd_dist_muon_flags(cfg)
  return cfg


def graph_trainer_kimi_k3_deep8_e896_spmd_dist_muon() -> TpuGraphTrainer.Config:
  """``k3_deep8_e896`` with captured experts in buckets of two tensors.

  56 local experts on EP16; at eFSDP 8 (4x4x4, TP1) each rank already holds
  seven whole experts, so the expert buckets may need no exchange.
  """
  cfg = graph_trainer_kimi_k3_deep8_e896()
  cfg.optimizer = _k3_dist_muon_optimizer(
      cfg.model_spec.model, spmd_expert_bucket_size=2
  )
  _with_spmd_dist_muon_flags(cfg)
  return cfg
