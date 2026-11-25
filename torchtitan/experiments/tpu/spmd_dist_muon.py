# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DistMuon captured as one TPU program per rank (``SpmdDistMuon``).

``SpmdCompiledDistMuon`` runs DistMuon's bucket runtime (pack -> all-to-all ->
Newton-Schulz -> all-to-all -> update, for every bucket) as a single
``torch.compile`` program under ``torch.tpu.distributed.spmd_safe``. TorchTPU
otherwise cuts a program at every collective; inside ``spmd_safe`` the
all-to-alls are fused into the program, and with
``--xla_tpu_enable_async_all_to_all=true`` XLA overlaps one bucket's
all-to-all with another bucket's Newton-Schulz.

``spmd_safe`` is only valid when every rank issues the same collectives in the
same order. Use it for buckets whose per-rank program is identical (e.g.
routed experts whose compute chunks divide evenly over the transport group);
keep owner-balanced buckets (dense ``Owned`` matrices) in an eager DistMuon.
TODO: XLA schedules each rank's program on its own; nothing here checks that
the compiled collective order matches across ranks (a mismatch hangs). That
guarantee belongs in ``spmd_safe``.

    Example (one MoE layer, experts sharded on dim 1 over 4 ranks)::

        bucket w1: pack -> a2a(s2c) -> NS[w1] -> a2a(c2s) -> update w1
        bucket w2:         pack -> a2a(s2c) -> NS[w2] -> a2a(c2s) -> update
        timeline:  | a2a w1 | NS w1          | NS w2         | ...
                            | a2a w2 (async) | a2a w1 back   |
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.components.optimizer.optimizer import (
    build_dist_muon,
    OptimizersContainer,
)
from torchtitan.distributed.flex_shard import DistMuon
from torchtitan.tools.logging import logger

__all__ = ["SpmdCompiledDistMuon", "TpuOptimizersContainer"]


def _is_rank_symmetric(plan: Any) -> bool:
    """Whether every transport participant stores and computes the same shapes.

    Local buckets issue no collectives. A redistributed bucket is captured
    only when each participant's set of storage partitions and of non-empty
    compute partitions has the same shapes (e.g. participant 0 computes
    experts 0-6 of tensor A and participant 8 experts 0-6 of tensor B), i.e.
    its program differs only in offsets and in which tensor feeds a batch.
    """
    if getattr(plan, "group", None) is None:
        return True
    signatures = set()
    for participant in plan.group.participants:
        storage = sorted(
            tuple(r.storage_partition(participant).tensor_shape)
            for r in plan.redistribution_plans
        )
        compute = sorted(
            shape
            for r in plan.redistribution_plans
            if math.prod(shape := tuple(r.compute_partition(participant).tensor_shape))
        )
        signatures.add((tuple(storage), tuple(compute)))
    return len(signatures) == 1


def _run_buckets(
    optimizer: "SpmdCompiledDistMuon", local_grads: dict[str, torch.Tensor]
) -> None:
    # Gradients enter as an explicit input: tracing an optimizer makes Dynamo
    # guard on the identity of every ``param.grad`` it reads, and the trainer
    # allocates new gradients each step (one recompile per step).
    optimizer._step_local_grads = local_grads
    optimizer._redistribution_runtime.run(
        optimizer._captured_plans,
        local_tensor_spec=optimizer._local_tensor_spec,
        prepare=optimizer._prepare_local,
        compute=optimizer._compute_update,
        finalize=optimizer._apply_update,
    )
    optimizer._step_local_grads = None


class SpmdCompiledDistMuon(DistMuon):
    """DistMuon whose bucket runtime runs as one ``spmd_safe`` TPU program.

    Validation and state creation (``_preflight_step``) stay eager. The
    scheduled LR enters the program as a 0-d tensor, so LR warmup and decay do
    not recompile it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Buckets whose per-rank program would differ (e.g. a remainder bucket
        # whose experts do not split evenly over the transport group) run
        # eagerly after the captured program.
        self._captured_plans = tuple(
            p for p in self._bucket_plans if _is_rank_symmetric(p)
        )
        self._eager_plans = tuple(
            p for p in self._bucket_plans if not _is_rank_symmetric(p)
        )
        if self._eager_plans:
            logger.warning(
                f"SpmdDistMuon: {len(self._eager_plans)} of "
                f"{len(self._bucket_plans)} buckets are not rank-symmetric "
                "and run eagerly"
            )
        self._compiled_run: Callable[..., None] | None = None
        self._step_local_grads: dict[str, torch.Tensor] | None = None

    def _local_gradient_and_momentum(self, compute_layout: Any) -> Any:
        if self._step_local_grads is None:
            return super()._local_gradient_and_momentum(compute_layout)
        momentum_state = self.state[compute_layout.param]["momentum_buffer"]
        local_reference = compute_layout.param.to_local()
        if compute_layout.storage_is_compute_ready:
            local_reference = local_reference.detach()
        return (
            self._step_local_grads[compute_layout.fqn].view_as(local_reference),
            momentum_state.to_local().view_as(local_reference),
            momentum_state,
            self._group(compute_layout),
        )

    @torch.no_grad()
    def step(self, closure: None = None) -> None:  # type: ignore[override]
        if closure is not None:
            raise ValueError("SpmdCompiledDistMuon does not support a closure")
        self._preflight_step()
        if self._compiled_run is None:
            self._compiled_run = torch.compile(
                torch.tpu.distributed.spmd_safe(_run_buckets),
                backend="tpu",
                fullgraph=True,
                dynamic=False,
                options={"serializable": False},
            )
        group = self.param_groups[0]
        lr = group["lr"]
        device = self._parameter_compute_layouts[0].param.device
        group["lr"] = torch.tensor(lr, dtype=torch.float32, device=device)
        local_grads = {
            layout.fqn: layout.param.grad.to_local()
            for layout in self._parameter_compute_layouts
        }
        try:
            self._compiled_run(self, local_grads)
        finally:
            group["lr"] = lr
            self._step_local_grads = None
        if self._eager_plans:
            self._redistribution_runtime.run(
                self._eager_plans,
                local_tensor_spec=self._local_tensor_spec,
                prepare=self._prepare_local,
                compute=self._compute_update,
                finalize=self._apply_update,
            )
            self._redistribution_runtime.release_buffers()


def build_spmd_dist_muon(
    params: Iterable[dict[str, Any]], **kwargs: Any
) -> SpmdCompiledDistMuon:
    return build_dist_muon(params, optimizer_cls=SpmdCompiledDistMuon, **kwargs)


class TpuOptimizersContainer(OptimizersContainer):
    """``OptimizersContainer`` that also builds ``SpmdDistMuon``."""

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    @staticmethod
    def _resolve_optimizer_factory(name: str) -> Callable[..., torch.optim.Optimizer]:
        if name == "SpmdDistMuon":
            return build_spmd_dist_muon
        return OptimizersContainer._resolve_optimizer_factory(name)
