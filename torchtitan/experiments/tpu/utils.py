# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed.algorithms._checkpoint import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.distributed import ParallelDims


__all__ = [
    "ptd_checkpoint_wrapper_with_early_stop",
    "split_optimizers_by_mesh",
    "with_split_optimizers_by_mesh",
]


def ptd_checkpoint_wrapper_with_early_stop(
    module: torch.nn.Module,
    checkpoint_impl: checkpoint_wrapper.CheckpointImpl = checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
    checkpoint_fn=None,
    early_stop: bool = False,
    **checkpoint_fn_kwargs,
) -> torch.nn.Module:
    """Wraps a module with PyTorch Distributed's checkpoint_wrapper.

    This function is a wrapper around
    `torch.distributed.algorithms._checkpoint.checkpoint_wrapper`
    with an added `early_stop` parameter, although early stopping is not
    currently supported in g3.

    Args:
        module: The module to wrap.
        checkpoint_impl: The checkpoint implementation to use.
        checkpoint_fn: Optional function to use for checkpointing.
        early_stop: Whether to enable early stopping.
            Currently raises NotImplementedError if True.
        **checkpoint_fn_kwargs: Additional keyword arguments to pass to
            `checkpoint_fn`.

    Returns:
        The wrapped module.

    Raises:
        NotImplementedError: If `early_stop` is set to True.
    """
    if early_stop:
      raise NotImplementedError(
          "early_stop is not supported by ptd_checkpoint_wrapper."
      )

    checkpoint_fn_kwargs["early_stop"] = early_stop

    return checkpoint_wrapper.checkpoint_wrapper(
        module,
        checkpoint_impl=checkpoint_impl,
        checkpoint_fn=checkpoint_fn,
        **checkpoint_fn_kwargs,
    )


def _contiguous_dist_muon_grads(optimizer, args, kwargs) -> None:
    """Gives every DistMuon gradient its parameter's local memory layout.

    DistMuon requires each gradient to have the parameter's exact storage
    layout (placements and local strides). A SimpleFSDP parameter sharded on
    dim 1 (e.g. K3 routed experts whose local expert count does not divide
    expert FSDP) gets its reduce-scattered gradient from the compiled TPU step
    as a permuted, non-contiguous view of a dim-0-scattered buffer. Copy those
    into the parameter's contiguous layout before the step.
    TODO: Emit the gradient in the parameter's layout inside the compiled step.
    """
    del args, kwargs
    for group in optimizer.param_groups:
        for param in group["params"]:
            grad = param.grad
            if (
                not isinstance(grad, DTensor)
                or grad.to_local().is_contiguous()
                or not param.to_local().is_contiguous()
            ):
                continue
            param.grad = DTensor.from_local(
                grad.to_local().contiguous(),
                grad.device_mesh,
                grad.placements,
                run_check=False,
                shape=param.shape,
                stride=param.stride(),
            )


def split_optimizers_by_mesh(
    optimizers_container: "OptimizersContainer",
) -> bool:
    """Splits each optimizer param group into one group per ``DeviceMesh``.

    Under expert parallelism, routed-expert parameters live on an
    ``(efsdp, ep)`` mesh while dense parameters live on ``(fsdp,)``, and
    torch_tpu's fused and foreach AdamW need every param group to share one
    mesh. A group whose parameters span several meshes is replaced, in place,
    by one group per mesh, in the order the meshes first appear. Each new group
    keeps the original group's options (``lr``, ``betas``, ...) and the entries
    of ``param_names`` that belong to its parameters, so the FQN-keyed
    optimizer state dict keeps the same keys. Groups whose parameters are all
    on one mesh are left as they are. Optimizer state is keyed by parameter
    and is not touched.

    Meshes are compared by equality, not identity: FSDP2 with TP builds a
    separate but equal ``(dp, tp)`` mesh object for each ``fully_shard`` call.
    Parameters that are not DTensors share one group.

    Must run before the LR schedulers are built: ``LambdaLR`` records one base
    LR per param group when it is constructed.

    Args:
        optimizers_container: The ``OptimizersContainer`` to regroup in place.

    Returns:
        ``True`` if any param group was split. ``False`` if every group was
        already on a single mesh; nothing is changed then, so calling this
        again after a split also returns ``False``.

    Raises:
        ValueError: If a group's ``param_names`` and ``params`` differ in
            length.
        RuntimeError: If a group that needs splitting already has an LR
            scheduler attached (it has ``initial_lr``).
    """
    from torchtitan.distributed.flex_shard import DistMuon  # pylint: disable=g-import-not-at-top

    regrouped: list[tuple[torch.optim.Optimizer, list[dict[str, Any]]]] = []
    for optimizer in optimizers_container.optimizers:
        # DistMuon keeps exactly one frozen param group and plans its own
        # per-bucket transports across storage meshes (dense and EP experts).
        if isinstance(optimizer, DistMuon):
            if not getattr(optimizer, "_tpu_contiguous_grads_hook", False):
                optimizer.register_step_pre_hook(_contiguous_dist_muon_grads)
                optimizer._tpu_contiguous_grads_hook = True
            continue
        new_groups: list[dict[str, Any]] = []
        for group in optimizer.param_groups:
            params = list(group["params"])
            names = group.get("param_names")
            if names is not None and len(names) != len(params):
                raise ValueError(
                    f"param_names length ({len(names)}) != params length "
                    f"({len(params)})"
                )
            buckets: dict[DeviceMesh | None, list[int]] = {}
            for i, param in enumerate(params):
                mesh = param.device_mesh if isinstance(param, DTensor) else None
                buckets.setdefault(mesh, []).append(i)
            if len(buckets) <= 1:
                new_groups.append(group)
                continue
            if "initial_lr" in group:
                raise RuntimeError(
                    "split_optimizers_by_mesh must run before the LR "
                    "schedulers are built; they keep one base LR per group"
                )
            options = {
                k: v
                for k, v in group.items()
                if k not in ("params", "param_names")
            }
            for indices in buckets.values():
                new_group = {"params": [params[i] for i in indices]}
                if names is not None:
                    new_group["param_names"] = [names[i] for i in indices]
                new_groups.append({**new_group, **options})
        if len(new_groups) > len(optimizer.param_groups):
            regrouped.append((optimizer, new_groups))
    # Regroup only after every group has been checked, so an error leaves the
    # optimizers unchanged.
    for optimizer, new_groups in regrouped:
        optimizer.param_groups = []
        for group in new_groups:
            optimizer.add_param_group(group)
    return bool(regrouped)


def with_split_optimizers_by_mesh(
    post_optimizer_build_fn: Callable[..., None] | None,
) -> Callable[..., None]:
    """Chains ``split_optimizers_by_mesh`` after ``post_optimizer_build_fn``.

    Both TPU trainers install the result as the model spec's
    ``post_optimizer_build_fn``. The upstream ``Trainer`` calls that hook right
    after it builds the optimizers and before it builds the LR schedulers and
    the checkpointer, so both see the split param groups.

    Args:
        post_optimizer_build_fn: The model's own hook, or ``None``.

    Returns:
        A hook that calls ``post_optimizer_build_fn`` (if set) and then
        ``split_optimizers_by_mesh``.
    """

    def post_optimizer_build_and_split_by_mesh(
        optimizers: "OptimizersContainer",
        model_parts: list[torch.nn.Module],
        parallel_dims: "ParallelDims",
    ) -> None:
        if post_optimizer_build_fn is not None:
            post_optimizer_build_fn(optimizers, model_parts, parallel_dims)
        split_optimizers_by_mesh(optimizers)

    return post_optimizer_build_and_split_by_mesh
