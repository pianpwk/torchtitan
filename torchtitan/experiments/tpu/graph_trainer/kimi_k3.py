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

import spmd_types as spmd
import torch
from torch import nn
from torchtitan.config import (
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
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
from torchtitan.experiments.tpu.kimi_k3.sharding import (
    set_kimi_k3_tp_sharding_config,
)
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    configure_simple_fsdp_shard_dims,
)
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import (
    install_tpu_replicate_computation,
)
from torchtitan.models.kimi_k3 import KimiK3Model
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import logger


class GraphTrainerKimiK3Model(KimiK3Model):
  """``KimiK3Model`` that initializes the sharded, not the gathered, states."""

  @dataclass(kw_only=True, slots=True)
  class Config(KimiK3Model.Config):

    def update_from_config(self, *, config, **kwargs) -> None:
      super(GraphTrainerKimiK3Model.Config, self).update_from_config(
          config=config, **kwargs
      )
      if config.parallelism.tensor_parallel_degree > 1:
        set_kimi_k3_tp_sharding_config(
            self, enable_ep=config.parallelism.expert_parallel_degree > 1
        )

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


# Largest parameter `_replicate_uneven_params` replicates instead of raising:
# KDA's per-head `beta` weight (96 x 7168) is the largest that needs it.
_MAX_REPLICATED_NUMEL = 1 << 20


def _replicate_uneven_params(
    model: GraphTrainerKimiK3Model,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
) -> None:
  """Replicates dense parameters whose dim 0 does not divide the FSDP degree.

  SimpleFSDP replicates a parameter only when dim 0 is smaller than the FSDP
  degree; otherwise it shards dim 0 unevenly (KDA's 96-head `A_log`,
  `dt_bias` and `beta` over 64 ranks), which the TPU replicate computation
  does not support (the gathered tensor has the wrong number of rows). At the
  WS 1024 target (FSDP 512) these parameters are below the degree and
  replicated anyway. Runs before `apply_simple_fsdp`, whose walk then skips
  them. Large parameters raise instead of silently costing replicated memory.
  """
  fsdp_mesh = get_simple_fsdp_mesh(parallel_dims)
  degree = fsdp_mesh.size()
  if degree <= 1:
    return
  experts = {
      id(p)
      for block in model.layers.values()
      if block.moe is not None
      for p in block.moe.routed_experts.parameters()
  }
  uneven = set()
  for name, param in model.named_parameters():
    if id(param) in experts or param.ndim == 0:
      continue
    rows = param.shape[0]
    if rows >= degree and rows % degree:
      if param.numel() > _MAX_REPLICATED_NUMEL:
        raise ValueError(
            f"{name} {tuple(param.shape)}: dim 0 does not divide the FSDP"
            f" degree {degree} and it is too large to replicate"
        )
      uneven.add(id(param))
  if not uneven:
    return
  mp_policy = MixedPrecisionPolicy(
      param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
      reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
  )
  ignored = {p for p in model.parameters() if id(p) not in uneven}
  for module in model.modules():
    own = [p for p in module.parameters(recurse=False) if id(p) in uneven]
    if own:
      data_parallel(
          module,
          fsdp_mesh,
          "replicate",
          mp_policy=mp_policy,
          non_dp_mesh=parallel_dims.get_optional_mesh("tp"),
          ignored_params=ignored | {
              p for p in module.parameters() if id(p) not in uneven
          },
      )
  logger.info(
      "K3: replicated %d parameters whose dim 0 does not divide FSDP %d",
      len(uneven),
      degree,
  )


def _check_tp_layers(model: GraphTrainerKimiK3Model) -> None:
  """TP needs the TPU layers, whose forwards run the TP collectives."""
  # Imported here: the layer modules import jax / Tokamax, which must not load
  # before the CLI parses the config (it breaks tyro's type checks).
  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu.kimi_k3.latent_moe import TpuKimiLatentMoE
  from torchtitan.experiments.tpu.kimi_k3.tokamax_kda import TpuKDA
  from torchtitan.experiments.tpu.kimi_k3.tokamax_mla import TpuKimiMLAAttention
  # pylint: enable=g-import-not-at-top
  for name, block in model.layers.items():
    for module, tp_type, override_name in (
        (block.delta_attention, TpuKDA, "tokamax_kda_attention"),
        (block.attention, TpuKimiMLAAttention, "tokamax_mla_attention"),
        (block.moe, TpuKimiLatentMoE, "tpu_latent_moe"),
    ):
      if module is not None and not isinstance(module, tp_type):
        raise ValueError(
            f"TP needs {tp_type.__name__} in layer {name}, got"
            f" {type(module).__name__}; add the {override_name} override"
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
  """Applies TP, EP, SimpleFSDP and GraphTrainer compile to text-only K3."""
  # GraphTrainer recomputes per `compile.memory_policy`.
  del ac_config
  # The TPU registry has no pipelining_fn; CP is not wired for KDA / MLA.
  if parallel_dims.cp_enabled or parallel_dims.pp_enabled:
    raise NotImplementedError("TPU GraphTrainer K3 does not support CP or PP.")
  annotate_moe_ep_regions()
  annotate_module_fqns(model)

  if parallel_dims.tp_enabled:
    _check_tp_layers(model)
  model.parallelize(parallel_dims)
  if parallel_dims.tp_enabled:
    # Parameters `set_kimi_k3_tp_sharding_config` leaves undeclared live on
    # the invariant residual stream (see kimi_k3/sharding.py).
    with set_current_spmd_mesh(parallel_dims.spmd_dense_mesh()):
      for param in model.parameters():
        if not spmd.has_local_type(param):
          spmd.assert_type(param, spmd.I)

  configure_simple_fsdp_shard_dims(model, parallel_dims)
  if parallel_dims.ep_enabled:
    _shard_routed_experts(model, parallel_dims, training)
  _replicate_uneven_params(model, parallel_dims, training)
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
