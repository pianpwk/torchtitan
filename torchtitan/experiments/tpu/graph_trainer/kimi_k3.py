"""Kimi K3 (text-only) on the TPU GraphTrainer path.

The model config comes from ``experiments.tpu.kimi_k3.model_registry``; the
recipes swap in the TPU KDA, MLA and LatentMoE layers through
``override.imports``. ``parallelize_kimi_k3`` follows upstream GraphTrainer's
``parallelize_deepseekv3`` (``model.parallelize``, SimpleFSDP, compile) plus
the TPU SimpleFSDP policy of ``graph_trainer/deepseek_v3.py``.
"""

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard
from torchtitan.config import (
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    apply_simple_fsdp,
    get_simple_fsdp_mesh,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    disable_active_parametrization,
    MixedPrecisionPolicy,
)
from torchtitan.experiments.tpu import kimi_k3 as tpu_kimi_k3
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    configure_simple_fsdp_shard_dims,
)
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import (
    install_tpu_replicate_computation,
)
from torchtitan.models.kimi_k3 import KimiK3Model
from torchtitan.models.kimi_k3.kda import KDA
from torchtitan.models.kimi_k3.model import KimiMLAAttention
from torchtitan.models.kimi_k3.moe import KimiLatentMoE
from torchtitan.protocols.model_spec import ModelSpec


class GraphTrainerKimiK3Model(KimiK3Model):
  """``KimiK3Model`` that initializes the sharded, not the gathered, states."""

  @dataclass(kw_only=True, slots=True)
  class Config(KimiK3Model.Config):
    pass

  def init_states(self, *, buffer_device: torch.device | None = None) -> None:
    with disable_active_parametrization():
      super().init_states(buffer_device=buffer_device)


# FSDP placement of the K3 dense parameters whose dim 0 does not divide an FSDP
# degree K3 runs at (64 to 1024 at K3 widths: dim 7168, 96 KDA heads, 448 or
# 896 experts), applied at every degree so placement does not depend on it.
# Keyed by (owner module type, parameter path). The replicated ones are 1-D or
# per-head (KDA's A_log [H] has no dim 1, and SimpleFSDP gives KDA's own
# parameters one placement). Every other dense parameter keeps SimpleFSDP's
# default: dim 0, replicated below the degree; `_check_no_uneven_dim0_shard`
# raises if that leaves an uneven one.
# TODO: pad / unpad uneven dim-0 shards in the TPU SimpleFSDP all-gather
# (TpuReplicateComputation returns the wrong rows for them, unlike DTensor and
# FSDP2) and drop this table.
# TODO: declare these placements in the model's ShardingConfigs at config
# build time instead.
_K3_FSDP_PLACEMENTS: dict[tuple[type[nn.Module], str], Placement] = {
    (KDA, "beta.weight"): Shard(1),  # [H, D]
    (KDA, "A_log"): Replicate(),  # [H]
    (KDA, "dt_bias"): Replicate(),  # [H, K]
    (KimiMLAAttention, "wq_a.weight"): Shard(1),  # [q_lora_rank, D]
    (KimiMLAAttention, "wkv_a.weight"): Shard(1),  # [kv_lora_rank + rope, D]
    (KimiMLAAttention, "q_norm.weight"): Replicate(),  # [q_lora_rank]
    (KimiMLAAttention, "kv_norm.weight"): Replicate(),  # [kv_lora_rank]
    (KimiLatentMoE, "router.gate.weight"): Shard(1),  # [E, D]
    (KimiLatentMoE, "routed_down.weight"): Shard(1),  # [latent, D]
    (KimiLatentMoE, "routed_norm.weight"): Replicate(),  # [latent]
}


def _apply_k3_fsdp_placements(
    model: GraphTrainerKimiK3Model,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
) -> None:
  """Applies ``_K3_FSDP_PLACEMENTS``, before ``apply_simple_fsdp``.

  A ``Shard(1)`` parameter's module gets ``_fsdp_shard_dim``, which
  ``apply_simple_fsdp`` honors; ``Replicate()`` ones are wrapped here, so its
  walk skips them.
  """
  fsdp_mesh = get_simple_fsdp_mesh(parallel_dims)
  degree = fsdp_mesh.size()
  replicated: dict[nn.Module, set[nn.Parameter]] = {}
  for module in model.modules():
    for (owner, path), placement in _K3_FSDP_PLACEMENTS.items():
      if not isinstance(module, owner):
        continue
      holder_path, _, name = path.rpartition(".")
      holder = module.get_submodule(holder_path)
      param = getattr(holder, name)
      if placement.is_replicate():
        replicated.setdefault(holder, set()).add(param)
        continue
      assert list(holder.parameters()) == [param], path
      dim = placement.dim
      if param.shape[dim] >= degree and param.shape[dim] % degree:
        raise ValueError(
            f"{type(module).__name__}.{path} {tuple(param.shape)} is uneven"
            f" on dim {dim} over FSDP {degree}"
        )
      holder._fsdp_shard_dim = dim  # pylint: disable=protected-access
  mp_policy = MixedPrecisionPolicy(
      param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
      reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
  )
  for holder, params in replicated.items():
    data_parallel(
        holder,
        fsdp_mesh,
        "replicate",
        mp_policy=mp_policy,
        ignored_params=set(holder.parameters()) - params,
    )


def _check_no_uneven_dim0_shard(model: nn.Module) -> None:
  """Raises for a dense parameter SimpleFSDP sharded unevenly on dim 0."""
  experts = tuple(
      f"{fqn}.routed_experts."
      for fqn, module in model.named_modules()
      if isinstance(module, KimiLatentMoE)
  )
  for name, param in model.named_parameters():
    if name.startswith(experts) or not isinstance(param, DTensor):
      continue
    # The FSDP mesh axis comes first.
    if param.placements[0] == Shard(0) and (
        param.shape[0] % param.device_mesh.size(0)
    ):
      raise ValueError(
          f"{name} {tuple(param.shape)} is sharded unevenly on dim 0 over FSDP"
          f" {param.device_mesh.size(0)}, which the TPU SimpleFSDP all-gather"
          " gathers wrong; add it to _K3_FSDP_PLACEMENTS"
      )


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
  if parallel_dims.tp_enabled:
    raise ValueError("TP is not supported for TPU K3 yet.")
  if parallel_dims.ep_enabled:
    raise NotImplementedError("TPU GraphTrainer K3 does not support EP.")
  # `_apply_k3_fsdp_placements` replicates over the FSDP mesh only.
  if parallel_dims.dp_replicate_enabled:
    raise NotImplementedError("TPU GraphTrainer K3 does not support HSDP.")
  annotate_module_fqns(model)

  model.parallelize(parallel_dims)
  configure_simple_fsdp_shard_dims(model, parallel_dims)
  _apply_k3_fsdp_placements(model, parallel_dims, training)
  model = apply_simple_fsdp(
      model, parallel_dims=parallel_dims, training=training
  )  # pyrefly: ignore[bad-assignment]
  _check_no_uneven_dim0_shard(model)
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
