"""TPU recipes for Kimi K3 (text only) on the TPU GraphTrainer."""

import dataclasses

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
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
