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

import torch
from torch import nn
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


def _shard_routed_experts(
    model: GraphTrainerKimiK3Model,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
) -> None:
  """SimpleFSDP for the EP-sharded experts on the expert-FSDP mesh.

  The K3 counterpart of the ``TransformerBlock`` loop in upstream's
  ``apply_simple_fsdp``; it runs first, so that call skips these modules.
  """
  if parallel_dims.dp_replicate_enabled:
    raise NotImplementedError("TPU GraphTrainer K3 does not support HSDP")
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


def _shard_uneven_dim0_params(
    model: GraphTrainerKimiK3Model,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
) -> None:
  """SimpleFSDP for the K3 parameters whose dim 0 can be uneven.

  KDA's ``beta`` [H, D], MLA's ``wkv_a`` [576, D] and the router gate [E, D]
  weights shard on dim 1 when dim 0 does not divide the FSDP degree. KDA's own
  ``A_log`` [H] and ``dt_bias`` [H, K] are replicated then: SimpleFSDP gives a
  module one placement, and ``A_log`` has no dim 1. Runs before
  ``apply_simple_fsdp``, whose walk then skips these modules.
  """
  # TODO: pad / unpad uneven dim-0 shards in the TPU SimpleFSDP all-gather (as
  # DTensor and FSDP2 do on GPU) and drop this.
  fsdp_mesh = get_simple_fsdp_mesh(parallel_dims)
  mp_policy = MixedPrecisionPolicy(
      param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
      reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
  )
  for module in model.modules():
    if isinstance(module, KDA):
      targets = [
          (module, ("A_log", "dt_bias"), "replicate"),
          (module.beta, ("weight",), "fully_shard"),
      ]
    elif isinstance(module, KimiMLAAttention):
      targets = [(module.wkv_a, ("weight",), "fully_shard")]
    elif isinstance(module, KimiLatentMoE):
      targets = [(module.router.gate, ("weight",), "fully_shard")]
    else:
      continue
    for holder, names, mode in targets:
      if getattr(holder, names[0]).shape[0] % fsdp_mesh.size() == 0:
        continue
      own = {getattr(holder, name) for name in names}
      data_parallel(
          holder,
          fsdp_mesh,
          mode,
          mp_policy=mp_policy,
          shard_dim=1,
          ignored_params=set(holder.parameters()) - own,
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
  """Applies EP, SimpleFSDP and GraphTrainer compile to text-only K3."""
  # GraphTrainer recomputes per `compile.memory_policy`.
  del ac_config
  # The TPU registry has no pipelining_fn; CP is not wired for KDA / MLA.
  if parallel_dims.cp_enabled or parallel_dims.pp_enabled:
    raise NotImplementedError("TPU GraphTrainer K3 does not support CP or PP.")
  if parallel_dims.tp_enabled:
    raise NotImplementedError("TPU GraphTrainer K3 does not support TP.")
  annotate_moe_ep_regions()
  annotate_module_fqns(model)

  model.parallelize(parallel_dims)

  configure_simple_fsdp_shard_dims(model, parallel_dims)
  if parallel_dims.ep_enabled:
    _shard_routed_experts(model, parallel_dims, training)
  _shard_uneven_dim0_params(model, parallel_dims, training)
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
