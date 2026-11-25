"""TPU recipes for Kimi K3 (text only) on the TPU GraphTrainer."""

import dataclasses
import os

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
from torchtitan.experiments.tpu import tpu_flags
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
# docstring): Pallas output norm (with the Tokamax or the fused KDA kernel),
# gmm_v2 experts.
_PALLAS_TOKAMAX_KDA = (
    f"{_OVERRIDES}.tokamax_kda.tokamax_kda_attention_pallas_norm",
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
  are on: ``kda_override`` (Tokamax KDA with the Pallas output norm, or the
  fused KDA), gmm_v2 experts and the TPU attention residual.
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
      (f"{_OVERRIDES}.latent_moe.tpu_latent_moe_gmm_v2",
       {"ep_capacity_factor": 1.0}),
  ]
  cfg.training.num_tokens_per_microbatch_per_dp_rank = tokens
  cfg.training.steps = 8
  # Compile the graph_split executables concurrently (deep flavors: tens of
  # executables per rank).
  cfg.compile.async_compile_workers = 8
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
  return cfg


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


def _k3_muon_layouts(model_config) -> list[dict[str, ComputeLayout]]:
  """DistMuon compute layout of every K3 Muon matrix, per layer.

  Upstream K3's split in this fork's parameter names, extended to FSDP x TP
  (each dense layout lists the TP axis next to the FSDP axis, so one
  transport group spans ``fsdp x tp``):

  * Per head (``BlockShard`` rows): KDA q / k / v, ``forget_b``,
    ``output_gate``; MLA ``wq_b`` ([q_nope; q_rope] per head), ``wkv_a``
    ([kv_latent; k_rope]), ``wkv_b`` ([k_nope; v] per head) and ``gate``.
  * Whole matrix (``Owned``): KDA ``forget_a`` / ``output_proj``, MLA
    ``wq_a`` / ``wo``, the dense FFN and shared experts, the router gate and
    the latent ``routed_down`` / ``routed_up``.
  * Routed experts ``[E, ., .]``: one matrix per expert, ``Shard(0)`` over
    the expert dim on the FSDP / expert-FSDP / EP axes (EP first, as
    stored); TP as stored.
  """

  def dense(sharding) -> ComputeLayout:
    return ComputeLayout(
        shardings_by_mesh_axis={axis: sharding for axis in _DENSE_AXES},
        compute_dtype=torch.bfloat16,
    )

  owned = dense(Owned())

  def blocks_of(*num_rows: int) -> ComputeLayout:
    return dense(BlockShard(dim=0, block_sizes=num_rows))

  per_expert = ComputeLayout(
      shardings_by_mesh_axis={
          axis: Shard(0) for axis in ("fsdp", "dp_shard", "efsdp", "ep")
      },
      shard_order_by_tensor_dim={0: ("ep", "efsdp")},
      compute_dtype=torch.bfloat16,
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
  return layer_fqns


def _k3_layer_buckets(
    layer_fqns: list[dict[str, ComputeLayout]], *, routed_experts: bool
) -> list[BucketConfig]:
  """Upstream's grouping: layer 0 alone, then pairs of layers; each group's
  routed experts (``routed_experts=True``) in a bucket of their own."""
  num_layers = len(layer_fqns)
  buckets = []
  for first in (0, *range(1, num_layers, 2)):
    layer_ids = range(first, 1 if first == 0 else min(first + 2, num_layers))
    name = "layers." + "-".join(map(str, layer_ids))
    fqns = [fqn for i in layer_ids for fqn in layer_fqns[i]]
    routed = tuple(f for f in fqns if ".routed_experts." in f)
    buckets.append(BucketConfig(
        name=name, patterns=tuple(f for f in fqns if f not in routed)))
    if routed and routed_experts:
      buckets.append(BucketConfig(name=f"{name}.routed-experts", patterns=routed))
  return buckets


def _k3_muon_kwargs(lr: float) -> dict[str, object]:
  return {"lr": lr, "weight_decay": 0.1, "adjust_lr_fn": "match_rms_adamw"}


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


def _k3_dist_muon_optimizer(
    model_config, *, lr: float = 8e-4
) -> OptimizersContainer.Config:
  """DistMuon (eager) on every K3 matrix (``_k3_muon_layouts``), AdamW on the
  rest; ``_k3_layer_buckets`` buckets. lr 8e-4, weight decay 0.1, AdamW-RMS
  matched Muon LR."""
  layer_fqns = _k3_muon_layouts(model_config)
  return OptimizersContainer.Config(
      implementation="for-loop",
      param_groups=[
          ParamGroupConfig(
              pattern=_K3_MUON_PATTERN,
              optimizer_name="DistMuon",
              optimizer_kwargs=_k3_muon_kwargs(lr),
          ),
          _k3_adamw_rest(lr),
      ],
      optimizer_factory_kwargs_by_name={
          "DistMuon": {
              "compute_sharding_by_fqn": {
                  fqn: layout for d in layer_fqns for fqn, layout in d.items()
              },
              "bucket_configs": tuple(
                  _k3_layer_buckets(layer_fqns, routed_experts=True)
              ),
          }
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


def _k3_spmd_dist_muon_optimizer(
    model_config, *, expert_bucket_size: int, lr: float = 8e-4
) -> TpuOptimizersContainer.Config:
  """``_k3_dist_muon_optimizer`` with the routed experts in ``SpmdDistMuon``.

  ``SpmdDistMuon`` runs its buckets as one captured ``spmd_safe`` TPU program
  (``experiments/tpu/spmd_dist_muon.py``); the dense matrices stay in the
  eager DistMuon (their owner-based exchange differs per rank). Expert
  buckets hold ``expert_bucket_size`` (``P``) same-shape tensors: w1 / w3 of
  consecutive MoE layers, or w2. A bucket is captured when every rank runs
  the same program: ``P`` tensors of ``E`` local experts over an ``N``-rank
  transport group split into equal runs of ``k = P * E / N`` experts when
  ``k`` divides ``E`` (``E`` = 56: ``N`` = 16 needs ``P`` = 2); other
  buckets (e.g. a remainder) run eagerly after the program. Recipes using it
  also need ``_with_spmd_dist_muon_flags``.
  """
  layer_fqns = _k3_muon_layouts(model_config)
  layouts = {fqn: layout for d in layer_fqns for fqn, layout in d.items()}
  routed_by_class = {
      cls: [f for d in layer_fqns for f in d if f.rsplit(".", 1)[-1] in names]
      for cls, names in (("w13", ("w1_EDF", "w3_EDF")), ("w2", ("w2_EFD",)))
  }
  expert_buckets = tuple(
      BucketConfig(
          name=f"routed-experts.{cls}.{i // expert_bucket_size}",
          patterns=tuple(fqns[i:i + expert_bucket_size]),
      )
      for cls, fqns in routed_by_class.items()
      for i in range(0, len(fqns), expert_bucket_size)
  )
  return TpuOptimizersContainer.Config(
      implementation="for-loop",
      param_groups=[
          ParamGroupConfig(
              pattern=_K3_ROUTED_EXPERT_PATTERN,
              optimizer_name="SpmdDistMuon",
              optimizer_kwargs=_k3_muon_kwargs(lr),
          ),
          ParamGroupConfig(
              pattern=_K3_MUON_PATTERN,
              optimizer_name="DistMuon",
              optimizer_kwargs=_k3_muon_kwargs(lr),
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
                  f: v for f, v in layouts.items() if ".routed_experts." not in f
              },
              "bucket_configs": tuple(
                  _k3_layer_buckets(layer_fqns, routed_experts=False)
              ),
          },
      },
  )


def _with_spmd_dist_muon_flags(cfg: TpuGraphTrainer.Config) -> None:
  """TPU flags ``SpmdDistMuon`` needs.

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


def graph_trainer_kimi_k3_deep8_e448_spmd_dist_muon() -> TpuGraphTrainer.Config:
  """``graph_trainer_kimi_k3_deep8_e448_dist_muon`` with captured experts.

  Buckets of two expert tensors split evenly at 56 local experts over eFSDP 16
  (EP8 on 4x4x4) or 28 over eFSDP 8 (EP16).
  """
  cfg = graph_trainer_kimi_k3_deep8_e448()
  cfg.optimizer = _k3_spmd_dist_muon_optimizer(
      cfg.model_spec.model, expert_bucket_size=2
  )
  _with_spmd_dist_muon_flags(cfg)
  return cfg


def graph_trainer_kimi_k3_deep8_e896_spmd_dist_muon() -> TpuGraphTrainer.Config:
  """``graph_trainer_kimi_k3_deep8_e896_dist_muon`` with captured experts.

  56 local experts on EP16; at eFSDP 8 (4x4x4, TP1) each rank already holds
  seven whole experts, so the expert buckets need no exchange.
  """
  cfg = graph_trainer_kimi_k3_deep8_e896()
  cfg.optimizer = _k3_spmd_dist_muon_optimizer(
      cfg.model_spec.model, expert_bucket_size=2
  )
  _with_spmd_dist_muon_flags(cfg)
  return cfg
