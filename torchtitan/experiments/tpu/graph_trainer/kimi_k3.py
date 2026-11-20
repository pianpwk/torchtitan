"""Kimi K3 (text-only) on the TPU GraphTrainer path.

The model config comes from ``experiments.tpu.kimi_k3.model_registry``; its
``override.imports`` entries swap in the TPU KDA, MLA and LatentMoE layers.
``parallelize_kimi_k3`` follows upstream GraphTrainer's
``parallelize_deepseekv3`` (``model.parallelize``, SimpleFSDP, compile) plus
the TPU SimpleFSDP policy of ``graph_trainer/deepseek_v3.py``.
"""

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    disable_active_parametrization,
)
from torchtitan.experiments.tpu import kimi_k3 as tpu_kimi_k3
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    configure_simple_fsdp_shard_dims,
)
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import (
    install_tpu_replicate_computation,
)
from torchtitan.models.kimi_k3 import KimiK3Model
from torchtitan.protocols.model_spec import ModelSpec


class GraphTrainerKimiK3Model(KimiK3Model):
  """``KimiK3Model`` that initializes the sharded, not the gathered, states."""

  @dataclass(kw_only=True, slots=True)
  class Config(KimiK3Model.Config):
    pass

  def init_states(self, *, buffer_device: torch.device | None = None) -> None:
    with disable_active_parametrization():
      super().init_states(buffer_device=buffer_device)


def parallelize_kimi_k3(
    model: GraphTrainerKimiK3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
  """Applies SimpleFSDP and GraphTrainer compile to text-only K3."""
  # GraphTrainer recomputes per `compile.memory_policy`.
  del ac_config
  # The TPU registry has no pipelining_fn; CP is not wired for KDA / MLA.
  if parallel_dims.cp_enabled or parallel_dims.pp_enabled:
    raise NotImplementedError("TPU GraphTrainer K3 does not support CP or PP.")
  if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
    raise NotImplementedError("TPU GraphTrainer K3 does not support TP or EP.")
  annotate_module_fqns(model)

  model.parallelize(parallel_dims)

  configure_simple_fsdp_shard_dims(model, parallel_dims)
  model = apply_simple_fsdp(
      model, parallel_dims=parallel_dims, training=training
  )  # pyrefly: ignore[bad-assignment]
  # Creates process groups, so every rank runs it here, right after sharding.
  for holder in model.modules():
    if hasattr(holder, "build_split_meshes"):
      holder.build_split_meshes()
  install_tpu_replicate_computation(model)

  return apply_compile(
      model,
      compile_config=compile_config,
      parallelism=parallelism,
      parallel_dims=parallel_dims,
      dump_folder=dump_folder,
  )


def model_registry(flavor: str) -> ModelSpec:
  """GraphTrainer ``ModelSpec`` for a TPU K3 flavor.

  ``TpuGraphTrainer.Config.from_tpu_trainer_config`` takes only the model
  class, ``parallelize_fn`` and ``pipelining_fn`` from this spec; the field
  values come from the recipe's model config.
  """
  model_spec = tpu_kimi_k3.model_registry(flavor)
  model_config = GraphTrainerKimiK3Model.Config(**{
      field.name: getattr(model_spec.model, field.name)
      for field in dataclasses.fields(model_spec.model)
  })
  return dataclasses.replace(
      model_spec,
      model=model_config,
      parallelize_fn=parallelize_kimi_k3,
      pipelining_fn=None,
  )
