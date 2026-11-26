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
differs from the bucket plan's. ``collective_check_ranks`` limits the dump and
the comparison to a few ranks (e.g. rank 0 and one rank on the last host) at
scales where a dump on every rank costs too much host disk; the other ranks
wait for the verdict.

    Example (one MoE layer, experts sharded on dim 1 over 4 ranks)::

        bucket w1: pack -> a2a(s2c) -> NS[w1] -> a2a(c2s) -> update w1
        bucket w2:         pack -> a2a(s2c) -> NS[w2] -> a2a(c2s) -> update
        timeline:  | a2a w1 | NS w1          | NS w2         | ...
                            | a2a w2 (async) | a2a w1 back   |
"""

import glob
import hashlib
import json
import math
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
    dump_dir: str | None
    expected_all_to_alls: int
    # Ranks that read their dump and publish a digest; None means every rank.
    check_ranks: tuple[int, ...] | None = None
    compile_start: float = 0.0
    num_compiles: int = 0
    # Executables this rank already checked. Shared by the guards of one
    # optimizer: programs of same-shape buckets compile to the same executable,
    # which XLA serves from its cache without dumping new HLO.
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
        check_ranks = (
            tuple(range(world_size)) if self.check_ranks is None else self.check_ranks
        )
        # TODO: Use a public c10d store accessor once one exists.
        store = dist.distributed_c10d._get_default_store()
        key = f"spmd_dist_muon/{self.tag}/{index}"
        if rank in check_ranks:
            self._publish_digest(store, f"{key}/{rank}")
        peers = [store.get(f"{key}/{r}").decode().split(":") for r in check_ranks]
        if len({p[0] for p in peers}) != 1:
            raise RuntimeError(
                f"{self.tag}: collective sequences differ across ranks "
                f"(digest:all-to-alls:collectives of ranks {check_ranks}: {peers})"
            )
        if int(peers[0][1]) != self.expected_all_to_alls:
            raise RuntimeError(
                f"{self.tag}: found {peers[0][1]} all-to-alls in the captured "
                f"program, the bucket plan issues {self.expected_all_to_alls}; "
                "refusing"
            )
        self.verified_fingerprints |= fingerprints
        if rank == 0:
            logger.info(
                f"{self.tag} compile {index}: {peers[0][2]} collectives "
                f"({peers[0][1]} all-to-alls, {peers[0][3]} async) in the same "
                f"order on ranks {self._describe(check_ranks, world_size)}"
            )

    @staticmethod
    def _describe(check_ranks: tuple[int, ...], world_size: int) -> str:
        if check_ranks == tuple(range(world_size)):
            return f"all {world_size}"
        return ", ".join(map(str, check_ranks))

    def _publish_digest(self, store: Any, key: str) -> None:
        assert self.dump_dir is not None
        # Only DistMuon programs: the dump may hold every program of the step
        # (e.g. a launcher's full dump on rank 0).
        files = [
            f
            for f in glob.glob(os.path.join(self.dump_dir, "*after_optimizations.txt"))
            if "dist_muon" in os.path.basename(f)
            and os.path.getmtime(f) >= self.compile_start - 1
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
        digest = hashlib.sha256(json.dumps(sequence).encode()).hexdigest()[:16]
        num_async = sum(op.startswith("async-start") for op, *_ in sequence)
        logger.info(f"{self.tag}: rank {dist.get_rank()} digest {digest} of {path}")
        store.set(
            key,
            f"{digest}:{_num_all_to_alls(sequence)}:{len(sequence)}:{num_async}",
        )


def _is_rank_symmetric(plan: Any) -> bool:
    """Whether every transport participant stores and computes the same shapes.

    Local buckets issue no collectives. A redistributed bucket is captured
    only when each participant's set of storage partitions and of non-empty
    compute partitions has the same shapes (e.g. participant 0 computes
    experts 0-6 of tensor A and participant 8 experts 0-6 of tensor B), i.e.
    its program differs only in offsets and in which tensor feeds a batch
    (the collective check still verifies the compiled result).
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
    optimizer: "SpmdCompiledDistMuon",
    program: int,
    local_grads: dict[str, torch.Tensor],
) -> None:
    # Gradients enter as an explicit input: tracing an optimizer makes Dynamo
    # guard on the identity of every ``param.grad`` it reads, and the trainer
    # allocates new gradients each step (one recompile per step).
    optimizer._step_local_grads = local_grads
    optimizer._redistribution_runtime.run(
        optimizer._program_plans[program],
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

    ``max_buckets_per_program`` splits the captured buckets into consecutive
    programs of at most that many buckets. One program keeps every bucket's
    exchange buffers and Newton-Schulz temporaries live at once (so the
    all-to-alls can overlap) and returns new parameters and momenta without
    aliasing the inputs, so its HBM grows with the number of buckets; several
    programs bound it, overlapping only within each.
    """

    def __init__(
        self,
        *args: Any,
        collective_check_ranks: tuple[int, ...] | None = None,
        max_buckets_per_program: int | None = None,
        **kwargs: Any,
    ) -> None:
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
            # The eager runtime picks pipeline slots by position among the plans
            # it runs (only the eager ones here); DistMuon reserved them by
            # position among all buckets, so an eager bucket could land in a
            # slot sized for another bucket.
            self._redistribution_runtime.reserve_buffers(
                self._eager_plans, local_tensor_spec=self._local_tensor_spec
            )
        dump_dir = _xla_dump_dir()
        checks_here = (
            collective_check_ranks is None
            or dist.get_rank() in collective_check_ranks
        )
        if dump_dir is None and checks_here:
            raise ValueError(
                "SpmdDistMuon checks the compiled program's collective order "
                "across ranks and needs the XLA dump: set xla_flags "
                "xla_dump_to (one directory per rank) and xla_dump_hlo_as_text"
            )
        size = max_buckets_per_program or max(len(self._captured_plans), 1)
        if size < 1:
            raise ValueError(f"max_buckets_per_program must be >= 1, got {size}")
        self._program_plans = tuple(
            self._captured_plans[i : i + size]
            for i in range(0, len(self._captured_plans), size)
        )
        check_ranks = (
            None
            if collective_check_ranks is None
            else tuple(sorted(set(collective_check_ranks)))
        )
        verified_fingerprints: set[str] = set()
        self._collective_guards = tuple(
            _CollectiveGuard(
                tag=(
                    f"SpmdDistMuon[{len(plans)} buckets]"
                    if len(self._program_plans) == 1
                    else f"SpmdDistMuon[program {i}: {len(plans)} buckets]"
                ),
                dump_dir=dump_dir,
                check_ranks=check_ranks,
                verified_fingerprints=verified_fingerprints,
                expected_all_to_alls=sum(
                    int(schedule.has_remote_transfers)
                    for plan in plans
                    for schedule in (
                        getattr(plan, "storage_to_compute_schedule", None),
                        getattr(plan, "compute_to_storage_schedule", None),
                    )
                    if schedule is not None
                ),
            )
            for i, plans in enumerate(self._program_plans)
        )
        self._compiled_runs: tuple[Callable[..., None], ...] | None = None
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
        if self._compiled_runs is None:
            # One torch.compile per program: each has its own guard and cache.
            self._compiled_runs = tuple(
                torch.compile(
                    torch.tpu.distributed.spmd_safe(_run_buckets),
                    backend="tpu",
                    fullgraph=True,
                    dynamic=False,
                    options={"debug_callback": guard, "serializable": False},
                )
                for guard in self._collective_guards
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
            for program, (run, guard) in enumerate(
                zip(self._compiled_runs, self._collective_guards)
            ):
                guard.compile_start = time.time()
                run(self, program, local_grads)
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
