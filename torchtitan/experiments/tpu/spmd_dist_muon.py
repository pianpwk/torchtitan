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
XLA schedules each rank's program on its own, so before the first run every
rank reads its post-optimization HLO from the XLA dump (``--xla_dump_to``, one
directory per rank, and ``--xla_dump_hlo_as_text``), extracts the ordered
collective sequence and compares it across ranks through the c10d store. The
step refuses to run on any mismatch, or when the number of all-to-alls found
differs from the bucket plan's.

    Example (one MoE layer, experts sharded on dim 1 over 4 ranks)::

        bucket w1: pack -> a2a(s2c) -> NS[w1] -> a2a(c2s) -> update w1
        bucket w2:         pack -> a2a(s2c) -> NS[w2] -> a2a(c2s) -> update
        timeline:  | a2a w1 | NS w1          | NS w2         | ...
                            | a2a w2 (async) | a2a w1 back   |
"""

import glob
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.components.optimizer.optimizer import (
    build_dist_muon,
    OptimizersContainer,
)
from torchtitan.distributed.flex_shard import DistMuon
from torchtitan.tools.logging import logger

__all__ = ["SpmdCompiledDistMuon", "TpuOptimizersContainer"]


_COLLECTIVE = re.compile(
    r"\b(all-to-all(?:-start|-done)?|all-gather(?:-start|-done)?"
    r"|all-reduce(?:-start|-done)?|reduce-scatter|collective-permute"
    r"(?:-start|-done)?|ragged-all-to-all)\("
)


def _parse_collective(line: str) -> list[str] | None:
    op = _COLLECTIVE.search(line)
    if not op:
        return None
    lhs = line.split("=", 1)
    shape = (
        re.sub(r"\{[^}]*\}", "", lhs[1].split(op.group(1))[0]).strip()
        if len(lhs) > 1
        else ""
    )
    groups = re.search(r"replica_groups=(\{[^ ]*\}|\[[^ ]*\][^ ,]*)", line)
    return [op.group(1), shape, groups.group(1) if groups else ""]


def collective_sequence(hlo_text: str) -> list[list[str]]:
    """Ordered collectives of an HLO module's scheduled ENTRY computation.

    Each entry is ``[op, result shape, replica_groups]``. Synchronous
    collectives appear directly in ENTRY. Asynchronous ones appear as
    ``async-start(...), calls=%computation`` / ``async-done(...)`` wrappers;
    those are resolved through the called computation, since the wrapper's
    own name does not have to name the collective.
    """
    computations = {
        m.group(1): m.group(2)
        for m in re.finditer(
            r"\n(?:ENTRY )?%?([\w.\-]+) [^\n]*\{\n(.*?)\n\}", hlo_text, re.S
        )
    }
    entry = re.search(r"\nENTRY [^\n]*\{\n(.*?)\n\}", hlo_text, re.S)
    body = entry.group(1) if entry else hlo_text
    start_of: dict[str, list[str]] = {}
    sequence = []
    for line in body.splitlines():
        if " async-start(" in line:
            name = re.match(r"\s*(?:ROOT )?%([\w.\-]+)", line)
            call = re.search(r"calls=%([\w.\-]+)", line)
            inner = (
                [
                    c
                    for c in map(
                        _parse_collective,
                        computations.get(call.group(1), "").splitlines(),
                    )
                    if c
                ]
                if call
                else []
            )
            if inner:
                start_of[name.group(1) if name else ""] = inner[0]
                sequence.append(["async-start:" + inner[0][0], *inner[0][1:]])
            continue
        if " async-done(" in line:
            source = re.search(r"async-done\(%([\w.\-]+)", line)
            inner = start_of.get(source.group(1)) if source else None
            if inner:
                sequence.append(["async-done:" + inner[0], *inner[1:]])
            continue
        if collective := _parse_collective(line):
            sequence.append(collective)
    return sequence


def _xla_dump_dir() -> str | None:
    flags = os.environ.get("XLA_FLAGS", "")
    match = re.search(r"--xla_dump_to=(\S+)", flags)
    if match is None or "--xla_dump_hlo_as_text" not in flags:
        return None
    return match.group(1)


def _num_all_to_alls(sequence: list[list[str]]) -> int:
    return sum(op in ("all-to-all", "async-start:all-to-all") for op, *_ in sequence)


@dataclass(slots=True)
class _CollectiveGuard:
    """Refuses a compiled program whose collective order differs across ranks.

    Installed as the TorchTPU ``debug_callback``, which runs after XLA
    compiled the program and before it first executes.
    """

    tag: str
    dump_dir: str
    expected_all_to_alls: int
    compile_start: float = 0.0
    num_compiles: int = 0
    verified_fingerprints: set[str] = field(default_factory=set)

    def __call__(self, debug: Any) -> None:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        index = self.num_compiles
        self.num_compiles += 1
        fingerprints = {
            str(f() if callable(f) else f)
            for f in (
                getattr(e, "fingerprint", None) for e in debug.compiled_executables
            )
        }
        if self.expected_all_to_alls == 0:
            # Only local buckets: the program has no collective to order.
            return
        if fingerprints and fingerprints <= self.verified_fingerprints:
            # A recompile that reused an executable this rank already checked
            # (XLA's cache hit dumps no new HLO).
            return
        files = [
            f
            for f in glob.glob(os.path.join(self.dump_dir, "*after_optimizations.txt"))
            if os.path.getmtime(f) >= self.compile_start - 1
        ]
        texts = [open(f).read() for f in files]
        candidates = [
            (len(t), f, t) for f, t in zip(files, texts) if _COLLECTIVE.search(t)
        ]
        if not candidates:
            raise RuntimeError(
                f"{self.tag}: no post-optimization HLO with collectives in "
                f"{self.dump_dir}; refusing to run the captured optimizer"
            )
        # The captured optimizer is by far the largest module compiled since
        # this compile started.
        _, path, text = max(candidates)
        sequence = collective_sequence(text)
        num_all_to_alls = _num_all_to_alls(sequence)
        digest = hashlib.sha256(json.dumps(sequence).encode()).hexdigest()[:16]
        # TODO: Use a public c10d store accessor once one exists.
        store = dist.distributed_c10d._get_default_store()
        key = f"spmd_dist_muon/{self.tag}/{index}"
        store.set(f"{key}/{rank}", f"{digest}:{num_all_to_alls}:{len(sequence)}")
        peers = [store.get(f"{key}/{r}").decode().split(":") for r in range(world_size)]
        if len({p[0] for p in peers}) != 1:
            raise RuntimeError(
                f"{self.tag}: collective sequences differ across ranks "
                f"(digest:all-to-alls:collectives per rank {peers}); "
                f"rank {rank} HLO {path}"
            )
        if num_all_to_alls != self.expected_all_to_alls:
            raise RuntimeError(
                f"{self.tag}: found {num_all_to_alls} all-to-alls in {path}, "
                f"the bucket plan issues {self.expected_all_to_alls}; refusing"
            )
        self.verified_fingerprints |= fingerprints
        if rank == 0:
            logger.info(
                f"{self.tag} compile {index}: {len(sequence)} collectives "
                f"({num_all_to_alls} all-to-alls, "
                f"{sum(op.startswith('async-start') for op, *_ in sequence)} "
                f"async) in the same order on {world_size} ranks"
            )


def _is_rank_symmetric(plan: Any) -> bool:
    """Whether every transport participant computes the same partition shapes.

    Local buckets issue no collectives. A redistributed bucket is captured
    only when each participant stores and computes identically shaped
    partitions of every parameter, i.e. its program differs only in offsets
    (the collective check still verifies the compiled result).
    """
    participants = getattr(plan, "group", None)
    if participants is None:
        return True
    signatures = {
        tuple(
            (
                tuple(redistribution.storage_partition(p).tensor_shape),
                tuple(redistribution.compute_partition(p).tensor_shape),
            )
            for redistribution in plan.redistribution_plans
        )
        for p in plan.group.participants
    }
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
    not recompile it. See the module docstring for the collective check.
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
        dump_dir = _xla_dump_dir()
        if dump_dir is None:
            raise ValueError(
                "SpmdDistMuon checks the compiled program's collective order "
                "across ranks and needs the XLA dump: set xla_flags "
                "xla_dump_to (one directory per rank) and xla_dump_hlo_as_text"
            )
        self._collective_guard = _CollectiveGuard(
            tag=f"SpmdDistMuon[{len(self._captured_plans)} buckets]",
            dump_dir=dump_dir,
            expected_all_to_alls=sum(
                int(schedule.has_remote_transfers)
                for plan in self._captured_plans
                for schedule in (
                    getattr(plan, "storage_to_compute_schedule", None),
                    getattr(plan, "compute_to_storage_schedule", None),
                )
                if schedule is not None
            ),
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
                options={
                    "debug_callback": self._collective_guard,
                    "serializable": False,
                },
            )
        group = self.param_groups[0]
        lr = group["lr"]
        device = self._parameter_compute_layouts[0].param.device
        group["lr"] = torch.tensor(lr, dtype=torch.float32, device=device)
        local_grads = {
            layout.fqn: layout.param.grad.to_local()
            for layout in self._parameter_compute_layouts
        }
        self._collective_guard.compile_start = time.time()
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
