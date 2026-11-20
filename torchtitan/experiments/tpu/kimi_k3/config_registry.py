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
from torchtitan.experiments.tpu.tpu_job_config import (
    TPUConfig,
    TPUTrainerConfig,
)

_OVERRIDES = "torchtitan.experiments.tpu.kimi_k3"


def graph_trainer_kimi_k3_tiny() -> TpuGraphTrainer.Config:
  """The ``tiny`` K3 with the three TPU layer overrides.

  Model (``kimi_k3.tiny``): 67.6M parameters, 29.1M activated; 8 layers
  (0 KDA + dense FFN, 3 and 7 gated MLA + LatentMoE, the rest KDA +
  LatentMoE), dim 512, vocab 2048; KDA 4 heads x 128, MLA 4 heads (qk 128 +
  64, v 128); 32 routed experts top-4 (hidden 256, latent 256) + 2 shared;
  quantile-balanced router.
  Run: one packed 4096-token row (``c4_test``, documents kept apart in KDA
  and MLA) per data-parallel rank and step, 40 steps; AdamW lr 8e-4 (5
  warmup steps, linear decay over the last 80% to 0.1x), FP32 master weights
  and BF16 compute; full rematerialization; the whole step (forward, loss,
  backward) as one compiled executable. Parallelism: SimpleFSDP over every
  rank (one rank: a single core); with
  ``--parallelism.expert_parallel_degree`` > 1 the routed experts use
  Ring-of-Experts at capacity factor 1.0.
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
      # Keeps every packed document apart: the first 40 c4_test rows of 4096
      # tokens hold at most 7 (Tokamax lays each out from a 64-token chunk
      # boundary, so a smaller budget is cheaper).
      (f"{_OVERRIDES}.tokamax_kda.tokamax_kda_attention",
       {"max_num_segments": 32}),
      f"{_OVERRIDES}.tokamax_mla.tokamax_mla_attention",
      (f"{_OVERRIDES}.latent_moe.tpu_latent_moe", {"capacity_factor": 1.0}),
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


def _graph_trainer_kimi_k3_slice(flavor: str) -> TpuGraphTrainer.Config:
  """A K3-width ``k3_slice_*`` flavor for comm / memory benchmarks.

  Model: see ``kimi_k3._k3_slice`` (3 layers at K3's widths; 34.2B / 8.8B
  parameters, 5.6B activated, for 448 / 64 routed experts).
  Run: eight packed 4096-token rows (32,768 tokens) per data-parallel rank
  and microbatch, 8 steps; otherwise ``graph_trainer_kimi_k3_tiny``'s
  optimizer, schedule, rematerialization and compilation. Balanced forced
  routing (see the flavor) and Ring-of-Experts at capacity factor 1.0 under
  ``--parallelism.expert_parallel_degree`` > 1. Parallelism: SimpleFSDP over
  every rank by default; set EP by flag.
  """
  cfg = graph_trainer_kimi_k3_tiny()
  seq_len = 4096
  tokens = 8 * seq_len
  cfg.model_spec = dataclasses.replace(
      graph_trainer_model_registry(flavor), max_context_length=seq_len
  )
  cfg.training.num_tokens_per_microbatch_per_dp_rank = tokens
  cfg.training.steps = 8
  return cfg


def graph_trainer_kimi_k3_slice_e448() -> TpuGraphTrainer.Config:
  """``k3_slice_e448``: 56 experts per core on EP8."""
  return _graph_trainer_kimi_k3_slice("k3_slice_e448")


def graph_trainer_kimi_k3_slice_e64() -> TpuGraphTrainer.Config:
  """``k3_slice_e64``: small expert count for debugging on 2x2x1."""
  return _graph_trainer_kimi_k3_slice("k3_slice_e64")
