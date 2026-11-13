"""TPU-local SimpleFSDP support for native non-leading collectives."""

import dataclasses
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan.experiments.graph_trainer.simple_fsdp import ReplicateComputation
from torchtitan.tools.logging import logger


def _all_gather_on_dim(shard: torch.Tensor, dim: int, group_name: str) -> torch.Tensor:
  """Gather ``dim``, requiring the native dimensional operator on TPU."""
  if dim == 0:
    return funcol.all_gather_single(shard, gather_dim=0, group=group_name)
  if shard.device.type == "tpu":
    if not hasattr(torch.ops.tpu, "_all_gather_dim"):
      raise RuntimeError(
          "Non-leading TPU FSDP shards require TorchTPU's "
          "_all_gather_dim operator"
      )
    return torch.ops.tpu._all_gather_dim.default(
        shard,
        dim,
        c10d._get_group_size_by_name(group_name),
        group_name,
    )
  return funcol.all_gather_single(shard, gather_dim=dim, group=group_name)


def _reduce_scatter_on_dim(
    grad: torch.Tensor, dim: int, group_name: str
) -> torch.Tensor:
  """Reduce-scatter ``dim``, requiring its native TPU operator off dim 0."""
  if dim == 0:
    return funcol.reduce_scatter_single(
        grad, reduceOp="sum", scatter_dim=0, group=group_name
    )
  if grad.device.type == "tpu":
    if not hasattr(torch.ops.tpu, "_reduce_scatter_dim"):
      raise RuntimeError(
          "Non-leading TPU FSDP shards require TorchTPU's "
          "_reduce_scatter_dim operator"
      )
    return torch.ops.tpu._reduce_scatter_dim.default(
        grad,
        "sum",
        dim,
        c10d._get_group_size_by_name(group_name),
        group_name,
    )
  return funcol.reduce_scatter_single(
      grad,
      reduceOp="sum",
      scatter_dim=dim,
      group=group_name,
  )


class _ShardToReplicate(torch.autograd.Function):

  @staticmethod
  def forward(
      ctx: Any,
      shard: torch.Tensor,
      group_name: str,
      param_dtype: torch.dtype | None,
      reduce_dtype: torch.dtype | None,
      replicate_group_name: str | None,
      shard_dim: int,
  ) -> torch.Tensor:
    ctx.group_name = group_name
    ctx.replicate_group_name = replicate_group_name
    ctx.reduce_dtype = reduce_dtype
    ctx.shard_dtype = shard.dtype
    ctx.shard_dim = shard_dim
    if param_dtype is not None:
      shard = shard.to(param_dtype)
    return _all_gather_on_dim(shard, shard_dim, group_name)

  @staticmethod
  def backward(ctx: Any, grad: torch.Tensor):
    if ctx.reduce_dtype is not None:
      grad = grad.to(ctx.reduce_dtype)
    out = _reduce_scatter_on_dim(grad, ctx.shard_dim, ctx.group_name)
    if ctx.replicate_group_name is not None:
      out = funcol.all_reduce(out, "sum", group=ctx.replicate_group_name)
    return out.to(ctx.shard_dtype), None, None, None, None, None


def replicate_sharded_tensor(
    shard: torch.Tensor,
    *,
    group_name: str,
    shard_dim: int,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    replicate_group_name: str | None = None,
) -> torch.Tensor:
  """Return a compute replica whose backward returns the local FSDP shard."""
  return _ShardToReplicate.apply(
      shard,
      group_name,
      param_dtype,
      reduce_dtype,
      replicate_group_name,
      shard_dim,
  )


class TpuReplicateComputation(ReplicateComputation):
  """Use native TPU collectives for even, non-leading SimpleFSDP shards."""

  def replicate_compute(self, x: DTensor) -> torch.Tensor:
    non_dp_mesh_dims = x._spec.mesh.ndim - self.device_mesh.ndim
    if non_dp_mesh_dims == 0:
      placements = x._spec.placements
      shard_mesh_dim: int | None = None
      replicate_group_name: str | None = None
      if len(placements) == 1 and isinstance(placements[0], Shard):
        shard_mesh_dim = 0
      elif (
          len(placements) == 2
          and isinstance(placements[0], Replicate)
          and isinstance(placements[1], Shard)
      ):
        replicate_group_name = x.device_mesh.get_group(0).group_name
        shard_mesh_dim = 1

      if shard_mesh_dim is not None:
        shard_placement = placements[shard_mesh_dim]
        assert isinstance(shard_placement, Shard)
        shard_dim = shard_placement.dim
        shard_degree = x.device_mesh.size(shard_mesh_dim)
        if shard_dim != 0 and x.shape[shard_dim] % shard_degree == 0:
          return replicate_sharded_tensor(
              x.to_local(),
              group_name=x.device_mesh.get_group(shard_mesh_dim).group_name,
              shard_dim=shard_dim,
              param_dtype=self.param_dtype,
              reduce_dtype=self.reduce_dtype,
              replicate_group_name=replicate_group_name,
          )
    return super().replicate_compute(x)


def _has_nonleading_shard(param: torch.Tensor) -> bool:
  return isinstance(param, DTensor) and any(
      isinstance(placement, Shard) and placement.dim != 0
      for placement in param.placements
  )


def install_tpu_replicate_computation(model: nn.Module) -> None:
  """Replace SimpleFSDP's parametrization only on the TPU Qwen model.

  Relies on this fork's SimpleFSDP keeping its per-parameter computations in
  ``_simple_fsdp_parametrization_list`` (google-pytorch/torchtitan 7e8acc32b).
  The computations are re-classed in place, so their constructor signature does
  not matter. Raises if a non-leading shard has no list to patch, so an upstream
  change cannot silently fall back to the copying path.
  """
  if any(hasattr(module, "plan_shard_dims") for module in model.modules()):
    # Scan flattens its stacks onto dim 0 so it scales past 512 ranks; keep
    # its collectives on the upstream path.
    logger.info("Layer stacks present; keeping SimpleFSDP's default replication")
    return
  installed = 0
  for module in model.modules():
    # SimpleFSDP's generated parameter properties resolve this private
    # per-parameter list on every access, so patching its elements after
    # wrapping changes the gather implementation without modifying the shared
    # SimpleFSDP module.
    parametrizations = getattr(module, "_simple_fsdp_parametrization_list", None)
    if parametrizations is None:
      unpatched = [
          name
          for name, param in module.named_parameters(recurse=False)
          if _has_nonleading_shard(param)
      ]
      if unpatched:
        raise RuntimeError(
            f"{type(module).__name__} has non-leading FSDP shards {unpatched} "
            "but no _simple_fsdp_parametrization_list to patch; SimpleFSDP's "
            "internals changed, so the TPU native collectives cannot be installed"
        )
      continue
    for computation in parametrizations:
      if isinstance(computation, TpuReplicateComputation):
        continue
      if type(computation) is not ReplicateComputation:
        raise RuntimeError(
            "Unexpected SimpleFSDP parametrization type: "
            f"{type(computation).__name__}"
        )
      computation.__class__ = TpuReplicateComputation
      installed += 1
  logger.info(
      "Installed TPU SimpleFSDP replication on %d parameter(s)", installed
  )


def _simple_fsdp_computation(
    module: nn.Module, name: str
) -> ReplicateComputation | None:
  """The SimpleFSDP computation that gathers ``module.<name>``, if any.

  SimpleFSDP exposes each parameter as a property reading
  ``_simple_fsdp_parametrization_list[i]``, with ``(i, name)`` bound as the
  getter's defaults; read the index from there rather than assuming it
  follows parameter order (ignored parameters are skipped).
  """
  prop = type(module).__dict__.get(name)
  if not isinstance(prop, property) or prop.fget is None:
    return None
  defaults = prop.fget.__defaults__
  if not (defaults and len(defaults) == 2 and defaults[1] == name):
    raise RuntimeError(
        f"{type(module).__name__}.{name} is not a SimpleFSDP property; "
        "SimpleFSDP's parametrization internals changed"
    )
  return module._simple_fsdp_parametrization_list[defaults[0]]


@dataclasses.dataclass(frozen=True)
class Fp8ExpertWeights:
  """Expert weights gathered as FP8; see `gather_fp8_expert_weights`.

  ``weight_q`` holds the e4m3 values of the weights concatenated on
  their last dim, ``[E, K, sum N]``. Pass ``shards`` and ``reduce_grad`` to
  the FP8 GEMM: its backward hands ``reduce_grad`` the full ``dW``, which
  becomes one gradient per shard.
  """

  weight_q: torch.Tensor
  shards: tuple[torch.Tensor, ...]
  group_name: str | None
  reduce_dtype: torch.dtype | None

  def reduce_grad(
      self, grads: tuple[torch.Tensor, ...]
  ) -> tuple[torch.Tensor, ...]:
    """FSDP's backward for each weight's full ``dW`` (the GEMM's backward
    splits ``[W1 | W3]``): cast to ``reduce_dtype``, reduce-scatter over the
    shard group. Per weight, as the BF16 path does: widening the joint
    gradient would hold both FP32 copies at once."""
    if len(grads) != len(self.shards):
      raise ValueError(f"Expected {len(self.shards)} gradients, got {len(grads)}")
    out = []
    for g, shard in zip(grads, self.shards):
      g = g.to(self.reduce_dtype) if self.reduce_dtype is not None else g
      if self.group_name is not None:
        g = funcol.reduce_scatter_single(
            g.contiguous(), reduceOp="sum", scatter_dim=0, group=self.group_name
        )
      out.append(g.to(shard.dtype))
    return tuple(out)

  def reduce_joint_grad(self, grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """`reduce_grad` for the gradient of the concatenated weight
    ``[E, K, sum N]``, e.g. from a kernel that computes ``[dW1 | dW3]`` at
    once: one reduce-scatter of the whole gradient, then the (``group``-times
    smaller) reduced shard is split per weight."""
    if self.reduce_dtype is not None:
      grad = grad.to(self.reduce_dtype)
    if self.group_name is not None:
      grad = funcol.reduce_scatter_single(
          grad.contiguous(), reduceOp="sum", scatter_dim=0, group=self.group_name
      )
    widths = [shard.shape[-1] for shard in self.shards]
    return tuple(
        g.contiguous().to(shard.dtype)
        for g, shard in zip(torch.split(grad, widths, dim=-1), self.shards)
    )


def _gather_fp8(
    shards: tuple[torch.Tensor, ...], group_name: str | None
) -> torch.Tensor:
  from torchtitan.experiments.tpu.kernels.fp8_expert_segment_matmul import fp8_quantize_expert_weight  # pylint: disable=g-import-not-at-top

  # The static scale makes quantization elementwise, so a shard quantizes
  # exactly as the gathered weight would. Nothing here is differentiable; the
  # GEMM routes the gradient to the shards.
  weight_q = torch.cat(
      [fp8_quantize_expert_weight(shard.detach()) for shard in shards], dim=-1
  )
  if group_name is not None:
    weight_q = funcol.all_gather_single(weight_q, gather_dim=0, group=group_name)
  return weight_q


def gather_fp8_expert_weights(
    module: nn.Module,
    names: tuple[str, ...],
    *,
    reduce_dtype: torch.dtype | None = None,
) -> Fp8ExpertWeights:
  """FP8 replicas of ``module``'s expert weights ``names``, concatenated.

  Each weight is ``[E, K, N]`` and, under SimpleFSDP, sharded on ``E`` over a
  1D mesh; the static-scale quantization is elementwise, so it runs on the
  shard and the gather moves FP8. Without SimpleFSDP (or with it inactive)
  the local weights are quantized as they are. Reads the sharded parameters
  directly, so the module's SimpleFSDP properties -- and their BF16 gathers
  -- never run.

  Returns the gathered FP8 values with the shards that receive their gradient;
  see `Fp8ExpertWeights`. The gradients are reduce-scattered in
  ``reduce_dtype`` if given, else in SimpleFSDP's.
  """
  from torchtitan.experiments.graph_trainer import simple_fsdp as upstream_simple_fsdp  # pylint: disable=g-import-not-at-top

  computations = [_simple_fsdp_computation(module, name) for name in names]
  params = [module._parameters[name] for name in names]
  if not upstream_simple_fsdp._active_parametrization or all(
      c is None for c in computations
  ):
    shards = tuple(p.to_local() if isinstance(p, DTensor) else p for p in params)
    return Fp8ExpertWeights(_gather_fp8(shards, None), shards, None, None)
  if any(c is None for c in computations):
    raise ValueError(
        f"Expert weights {names} must all be SimpleFSDP-sharded or none"
    )
  first = computations[0]
  for name, computation, param in zip(names, computations, params):
    if computation.device_mesh is not first.device_mesh or (
        computation.reduce_dtype != first.reduce_dtype
    ):
      raise ValueError(f"Expert weights {names} use different FSDP settings")
    if computation.device_mesh.ndim != 1 or tuple(
        computation.param_sharding
    ) != (Shard(0),):
      raise ValueError(
          f"FP8 expert all-gather needs {name} sharded on its expert dim over "
          f"a 1D FSDP mesh, got {computation.param_sharding} over "
          f"{computation.device_mesh.ndim} dims"
      )
    # Under EP the FSDP placement is a ``_StridedShard`` of dim 0 (FSDP on
    # top of the EP split); SimpleFSDP gathers its local tensor on dim 0 just
    # as for ``Shard(0)``.
    if not (
        isinstance(param, DTensor)
        and isinstance(param.placements[0], (Shard, _StridedShard))
        and param.placements[0].dim == 0
    ):
      raise ValueError(
          f"FP8 expert all-gather needs {name} sharded on dim 0 of its FSDP "
          f"axis, got {getattr(param, 'placements', None)}"
      )
  # ``to_local``'s backward re-wraps the local gradient with the parameter's
  # own (sharded) placements, so it adds no reduction of its own.
  shards = tuple(p.to_local() for p in params)
  group_name = first.device_mesh.get_group().group_name
  return Fp8ExpertWeights(
      _gather_fp8(shards, group_name),
      shards,
      group_name,
      reduce_dtype or first.reduce_dtype,
  )


__all__ = [
    "TpuReplicateComputation",
    "Fp8ExpertWeights",
    "gather_fp8_expert_weights",
    "install_tpu_replicate_computation",
    "replicate_sharded_tensor",
]
