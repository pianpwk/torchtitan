"""Kimi K3 (text-only) on the TPU GraphTrainer path.

The model config comes from ``experiments.tpu.kimi_k3.model_registry``; the
recipes swap in the TPU KDA, MLA and LatentMoE layers through
``override.imports``. ``parallelize_kimi_k3`` follows upstream GraphTrainer's
``parallelize_deepseekv3`` (``model.parallelize`` for the declared EP
layout, SimpleFSDP, compile) plus the TPU SimpleFSDP policy of
``graph_trainer/deepseek_v3.py``. Upstream's ``apply_simple_fsdp`` shards the
routed experts on the expert-FSDP mesh only inside ``TransformerBlock``\\ s,
and ``KimiK3TransformerBlock`` is not one, so the experts are sharded here.
"""

import dataclasses
from dataclasses import dataclass
import functools

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Placement, Shard
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
    annotate_moe_ep_regions,
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
from torchtitan.protocols.model_spec import ModelSpec


class GraphTrainerKimiK3Model(KimiK3Model):
  """``KimiK3Model`` that initializes the sharded, not the gathered, states."""

  @dataclass(kw_only=True, slots=True)
  class Config(KimiK3Model.Config):
    pass

  def init_states(self, *, buffer_device: torch.device | None = None) -> None:
    with disable_active_parametrization():
      super().init_states(buffer_device=buffer_device)


def _shard_routed_experts(
    model: GraphTrainerKimiK3Model,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
) -> None:
  """SimpleFSDP for the EP-sharded experts on the expert-FSDP mesh.

  The K3 counterpart of the ``TransformerBlock`` loop in upstream's
  ``apply_simple_fsdp``; it runs first, so that call skips these modules.
  """
  edp_mesh = parallel_dims.get_optional_mesh(["efsdp"])
  assert edp_mesh is not None
  mp_policy = MixedPrecisionPolicy(
      param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
      reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
  )
  for block in model.layers.values():
    if block.moe is None:
      continue
    routed = block.moe.routed_experts
    # Shard the expert dim only when it splits evenly over expert FSDP (56
    # local experts over 16 or 64 eFSDP ranks do not); otherwise dim 1 (the
    # latent / hidden dim of the kernel-native weights).
    num_local_experts = routed.inner_experts.num_experts // parallel_dims.ep
    shard_dim = 1 if num_local_experts % edp_mesh.size() else 0
    routed.inner_experts = data_parallel(
        routed.inner_experts,
        edp_mesh,
        "fully_shard",
        mp_policy=mp_policy,
        shard_dim=shard_dim,
        non_dp_mesh=parallel_dims.get_optional_mesh("ep"),
    )


# SimpleFSDP placements keyed by (owner module type, parameter path), e.g.
# ``{(KDA, "beta.weight"): Shard(1), (KDA, "A_log"): Replicate()}``. A flavor
# sets them for the parameters whose default placement (dim 0, replicated
# below the degree) would be uneven at the FSDP degrees it runs at (see
# ``experiments.tpu.kimi_k3.kimi_k3_tpu_configs``).
FsdpPlacements = dict[tuple[type[nn.Module], str], Placement]


def _apply_k3_fsdp_placements(
    model: GraphTrainerKimiK3Model,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    fsdp_placements: FsdpPlacements,
) -> None:
  """Applies ``fsdp_placements``, before ``apply_simple_fsdp``.

  A ``Shard(d)`` parameter's module gets ``_fsdp_shard_dim``, which
  ``apply_simple_fsdp`` honors (it replicates a parameter whose dim ``d`` is
  smaller than the degree); ``Replicate()`` ones are wrapped here, so its walk
  skips them.
  """
  fsdp_mesh = get_simple_fsdp_mesh(parallel_dims)
  replicated: dict[nn.Module, set[nn.Parameter]] = {}
  for module in model.modules():
    for (owner, path), placement in fsdp_placements.items():
      if not isinstance(module, owner):
        continue
      holder_path, _, name = path.rpartition(".")
      holder = module.get_submodule(holder_path)
      param = getattr(holder, name)
      if placement.is_replicate():
        replicated.setdefault(holder, set()).add(param)
        continue
      assert list(holder.parameters()) == [param], path
      holder._fsdp_shard_dim = placement.dim  # pylint: disable=protected-access
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


def _check_no_uneven_fsdp_shard(model: nn.Module) -> None:
  """Raises for a parameter sharded unevenly on any tensor dim.

  Covers every ``Shard(d)`` placement of every sharded parameter on its own
  mesh (the FSDP mesh for dense parameters; the EP / expert-FSDP meshes for
  routed experts). SimpleFSDP replicates a parameter whose dim is smaller
  than the degree, so that case is never sharded.
  TODO: pad / unpad uneven shards in the TPU SimpleFSDP all-gather
  (TpuReplicateComputation returns the wrong rows for them, unlike DTensor
  and FSDP2), which makes uneven shards legal and this check unnecessary.
  """
  for name, param in model.named_parameters():
    if not isinstance(param, DTensor):
      continue
    num_shards = [1] * param.ndim
    for axis, placement in enumerate(param.placements):
      if isinstance(placement, Shard):
        num_shards[placement.dim] *= param.device_mesh.size(axis)
    for dim, n in enumerate(num_shards):
      if param.shape[dim] % n:
        raise ValueError(
            f"{name} {tuple(param.shape)} is sharded unevenly on dim {dim}"
            f" ({n} shards, placements {param.placements}), which the TPU"
            " SimpleFSDP all-gather gathers wrong; set a placement for it in"
            " the flavor's FSDP placements"
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
    fsdp_placements: FsdpPlacements,
) -> nn.Module:
  """Applies EP, SimpleFSDP and GraphTrainer compile to text-only K3."""
  # GraphTrainer recomputes per `compile.memory_policy`.
  del ac_config
  # The TPU registry has no pipelining_fn; CP is not wired for KDA / MLA.
  if parallel_dims.cp_enabled or parallel_dims.pp_enabled:
    raise NotImplementedError("TPU GraphTrainer K3 does not support CP or PP.")
  if parallel_dims.tp_enabled:
    raise ValueError("TP is not supported for TPU K3 yet.")
  # `_shard_routed_experts` and `_apply_k3_fsdp_placements` use the FSDP
  # meshes only.
  if parallel_dims.dp_replicate_enabled:
    raise NotImplementedError("TPU GraphTrainer K3 does not support HSDP.")
  annotate_moe_ep_regions()
  annotate_module_fqns(model)

  model.parallelize(parallel_dims)
  configure_simple_fsdp_shard_dims(model, parallel_dims)
  if parallel_dims.ep_enabled:
    _shard_routed_experts(model, parallel_dims, training)
  _apply_k3_fsdp_placements(model, parallel_dims, training, fsdp_placements)
  model = apply_simple_fsdp(
      model, parallel_dims=parallel_dims, training=training
  )  # pyrefly: ignore[bad-assignment]
  _check_no_uneven_fsdp_shard(model)
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
  _, _, fsdp_placements = tpu_kimi_k3.kimi_k3_tpu_configs[flavor]
  model_config = GraphTrainerKimiK3Model.Config(**{
      field.name: getattr(model_spec.model, field.name)
      for field in dataclasses.fields(model_spec.model)
  })
  return dataclasses.replace(
      model_spec,
      model=model_config,
      parallelize_fn=functools.partial(
          parallelize_kimi_k3, fsdp_placements=fsdp_placements
      ),
      pipelining_fn=None,
  )
