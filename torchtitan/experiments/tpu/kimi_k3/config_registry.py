"""TPU recipes for Kimi K3 (text only) on the TPU GraphTrainer."""

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
