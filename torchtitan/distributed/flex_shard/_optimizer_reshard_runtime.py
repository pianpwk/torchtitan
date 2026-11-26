# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipelined packed all-to-all redistribution execution.

The packed redistribution and asynchronous compute overlap are informed by
Canzona and its FSDP implementation:

- Canzona: https://arxiv.org/abs/2602.06079
- FSDP-Canzona: https://github.com/liangyuwang/FSDP-Canzona
"""

from __future__ import annotations

import contextlib
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeAlias

import torch
from torch import Tensor

from ._optimizer_reshard_schedule import (
    _BucketPlan,
    _ItemT,
    _LocalBucketPlan,
    _PackedAllToAllSchedule,
    _RedistributionBucketPlan,
    _TensorRegion,
)


__all__: list[str] = []


_NUM_PIPELINE_SLOTS = 2


_BufferKey: TypeAlias = tuple[torch.device, torch.dtype]


@dataclass(slots=True)
class _BufferRequirements:
    storage_exchange_numel: int | None = None
    compute_exchange_numel: int | None = None
    compute_scratch_numel: int | None = None
    storage_scratch_numel: int | None = None

    def include_communication(
        self,
        *,
        storage_exchange_numel: int,
        compute_exchange_numel: int,
    ) -> None:
        self.storage_exchange_numel = max(
            self.storage_exchange_numel or 0,
            storage_exchange_numel,
        )
        self.compute_exchange_numel = max(
            self.compute_exchange_numel or 0,
            compute_exchange_numel,
        )

    def include_compute_scratch(self, numel: int) -> None:
        self.compute_scratch_numel = max(self.compute_scratch_numel or 0, numel)

    def include_storage_scratch(self, numel: int) -> None:
        self.storage_scratch_numel = max(self.storage_scratch_numel or 0, numel)


@dataclass(slots=True)
class _ReservedBuffers:
    storage_exchange: Tensor | None = None
    compute_exchange: Tensor | None = None
    # These remain separate because transfer-stream preparation/finalization
    # can overlap caller-stream compute within the same pipeline slot.
    compute_scratch: Tensor | None = None
    storage_scratch: Tensor | None = None


def _reserve_tensor(
    tensor: Tensor | None,
    *,
    numel: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Prepare a tensor's fixed capacity without retaining backing storage."""
    if tensor is None or tensor.numel() < numel:
        tensor = torch.empty(numel, dtype=dtype, device=device)
    if (storage := tensor.untyped_storage()).nbytes():
        storage.resize_(0)
    return tensor


def _fresh_buffers(device: torch.device) -> bool:
    """Whether buffers are built fresh per use instead of viewing reserved storage.

    A compiled program gets fresh buffers: resizing storage is not traceable,
    and routing every bucket's exchange through one aliased buffer would
    serialize the program; the compiler plans this memory itself. TorchTPU
    eager does too: each write into a slice of a large reserved buffer runs as
    its own program over the whole buffer (K3 full93 at FSDP1024: 1.5 s of
    pack / unpack copies per step for five deep8 buckets), while packing with
    one concatenation and unpacking from views touches only the payload.
    """
    return torch.compiler.is_compiling() or device.type == "tpu"


def _reserved_view(
    reserved: _ReservedBuffers, name: str, numel: int, key: _BufferKey
) -> Tensor:
    device, dtype = key
    if _fresh_buffers(device):
        return torch.empty(numel, dtype=dtype, device=device)
    tensor = getattr(reserved, name)
    assert tensor is not None and tensor.numel() >= numel
    return tensor[:numel]


# Row width of the 2-D view a TPU all-to-all buffer travels as (see
# _execute_packed_all_to_all).
_TPU_ALL_TO_ALL_COLUMNS = 1024


def _execute_packed_all_to_all(
    schedule: _PackedAllToAllSchedule,
    *,
    output: Tensor,
    input: Tensor,
) -> None:
    if not schedule.has_remote_transfers:
        output[: schedule.output_buffer_numel].copy_(
            input[: schedule.input_buffer_numel]
        )
        return
    output = output[: schedule.output_buffer_numel]
    input = input[: schedule.input_buffer_numel]
    output_split_sizes = schedule.output_split_sizes
    input_split_sizes = schedule.input_split_sizes
    # TODO: Drop once XLA TPU handles flat 16-bit all-to-all well. A flat
    # 16-bit buffer gets packed (2, 128) tiles: it compiles for ~1 s per MiB
    # (minutes for DistMuon's buckets) and runs at ~17 GB/s on TPU7x. The same
    # bytes as [rows, 1024] rows compile in ~0.2 s and run at ~70 GB/s.
    if output.device.type == "tpu" and all(
        size % _TPU_ALL_TO_ALL_COLUMNS == 0
        for size in (*output_split_sizes, *input_split_sizes)
    ):
        output = output.view(-1, _TPU_ALL_TO_ALL_COLUMNS)
        input = input.view(-1, _TPU_ALL_TO_ALL_COLUMNS)
        output_split_sizes = [
            size // _TPU_ALL_TO_ALL_COLUMNS for size in output_split_sizes
        ]
        input_split_sizes = [
            size // _TPU_ALL_TO_ALL_COLUMNS for size in input_split_sizes
        ]
    torch.distributed.all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=schedule.process_group,
    )


@dataclass(slots=True)
class _BufferSlot:
    buffers: dict[_BufferKey, _ReservedBuffers] = field(default_factory=dict)
    _tensors: tuple[Tensor, ...] = ()

    def reserve(
        self,
        requirements: dict[_BufferKey, _BufferRequirements],
    ) -> None:
        # Prepare capacities once; storage is grown on each run's caller stream.
        for (device, dtype), required in requirements.items():
            reserved = self.buffers.setdefault((device, dtype), _ReservedBuffers())
            if required.storage_exchange_numel is not None:
                reserved.storage_exchange = _reserve_tensor(
                    reserved.storage_exchange,
                    numel=required.storage_exchange_numel,
                    dtype=dtype,
                    device=device,
                )
            if required.compute_exchange_numel is not None:
                reserved.compute_exchange = _reserve_tensor(
                    reserved.compute_exchange,
                    numel=required.compute_exchange_numel,
                    dtype=dtype,
                    device=device,
                )
            if required.storage_scratch_numel is not None:
                reserved.storage_scratch = _reserve_tensor(
                    reserved.storage_scratch,
                    numel=required.storage_scratch_numel,
                    dtype=dtype,
                    device=device,
                )
            if required.compute_scratch_numel is not None:
                reserved.compute_scratch = _reserve_tensor(
                    reserved.compute_scratch,
                    numel=required.compute_scratch_numel,
                    dtype=dtype,
                    device=device,
                )
        self._tensors = tuple(
            tensor
            for reserved in self.buffers.values()
            for tensor in (
                reserved.storage_exchange,
                reserved.compute_exchange,
                reserved.compute_scratch,
                reserved.storage_scratch,
            )
            if tensor is not None
        )

    def allocate(self) -> None:
        """Regrow cached tensors on the current caller stream before use."""
        for tensor in self._tensors:
            if _fresh_buffers(tensor.device):
                # Fresh buffers per use (see _fresh_buffers).
                continue
            size = tensor.numel() * tensor.element_size()
            if (storage := tensor.untyped_storage()).nbytes() != size:
                storage.resize_(size)

    def release(self) -> None:
        """Free accelerator storage, retaining host storage and tensor metadata."""
        for tensor in self._tensors:
            if _fresh_buffers(tensor.device):
                continue
            if (
                tensor.device.type != "cpu"
                and (storage := tensor.untyped_storage()).nbytes()
            ):
                storage.resize_(0)

    def communication_buffers(
        self, plan: _RedistributionBucketPlan[Any]
    ) -> tuple[Tensor, Tensor]:
        to_compute = plan.storage_to_compute_schedule
        to_storage = plan.compute_to_storage_schedule
        key = (plan.device, plan.compute_dtype)
        reserved = self.buffers[key]
        return (
            _reserved_view(
                reserved,
                "storage_exchange",
                max(
                    to_compute.input_buffer_numel,
                    to_storage.output_buffer_numel,
                ),
                key,
            ),
            _reserved_view(
                reserved,
                "compute_exchange",
                max(
                    to_compute.output_buffer_numel,
                    plan.compute_to_storage_offset + to_storage.input_buffer_numel,
                ),
                key,
            ),
        )

    def compute_buffer(
        self,
        shape: torch.Size | tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        key = (device, dtype)
        return _reserved_view(
            self.buffers[key], "compute_scratch", math.prod(shape), key
        ).view(shape)

    def storage_partition_buffer(
        self,
        shape: torch.Size | tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        key = (device, dtype)
        return _reserved_view(
            self.buffers[key], "storage_scratch", math.prod(shape), key
        ).view(shape)


@dataclass(slots=True)
class _PipelineSlot:
    """Buffer ownership and reusable cross-stream handoff events."""

    buffers: _BufferSlot
    # Transfer producer -> compute consumer.
    compute_input_ready: torch.Event
    # Compute consumer -> transfer return/reuse.
    compute_done: torch.Event
    # Transfer finalization -> subsequent caller work.
    done: torch.Event


@dataclass(slots=True)
class _BucketWork(Generic[_ItemT]):
    plan: _RedistributionBucketPlan[_ItemT]
    slot: _PipelineSlot
    storage_buffer: Tensor
    compute_fragment_buffer: Tensor
    # Compiled runs only: the compute -> storage send buffer, built on its own
    # instead of over the consumed received fragments (see _packed_buffer).
    returned_fragment_buffer: Tensor | None = None


class _NoOpStream:
    def wait_stream(self, stream: Any) -> None:
        return None

    def wait_event(self, event: Any) -> None:
        return None


class _NoOpEvent:
    def record(self, stream: Any = None) -> None:
        return None


class _NoOpStreams:
    """Plain-Python no-op streams and events (TPU).

    Same single-stream semantics as ``torch.cpu``'s streams, but made of plain
    Python objects, so ``torch.compile`` traces the whole runtime into one
    graph (``torch.Stream`` objects cause a graph break).
    """

    _STREAM = _NoOpStream()

    @staticmethod
    def Stream(*args: Any, **kwargs: Any) -> _NoOpStream:  # noqa: N802
        return _NoOpStreams._STREAM

    @staticmethod
    def Event(*args: Any, **kwargs: Any) -> _NoOpEvent:  # noqa: N802
        return _NoOpEvent()

    @staticmethod
    def current_stream(device: Any = None) -> _NoOpStream:
        return _NoOpStreams._STREAM

    @staticmethod
    def stream(stream: Any) -> contextlib.nullcontext:
        return contextlib.nullcontext()


def _stream_device_module(device: torch.device) -> Any:
    """Device module whose streams and events order the transfer pipeline.

    TorchTPU runs every op on one deferred stream and exposes no public
    multi-stream API, so TPU uses no-op streams and events, the same
    single-stream path DistMuon takes on the host. XLA schedules the
    collectives asynchronously within the compiled programs.
    TODO: Use ``torch.accelerator`` streams once TorchTPU implements them.
    """
    if device.type == "tpu":
        return _NoOpStreams
    return torch.get_device_module(device)


@dataclass(slots=True)
class _CommunicationContext:
    device_handle: Any
    transfer_stream: torch.Stream
    slots: tuple[_PipelineSlot, ...]

    @classmethod
    def create(
        cls,
        device: torch.device,
    ) -> _CommunicationContext:
        device_handle = _stream_device_module(device)
        with device_handle.stream(device_handle.current_stream(device)):
            transfer_stream = device_handle.Stream(priority=0)

        def create_slot() -> _PipelineSlot:
            return _PipelineSlot(
                buffers=_BufferSlot(),
                compute_input_ready=device_handle.Event(),
                compute_done=device_handle.Event(),
                done=device_handle.Event(),
            )

        return cls(
            device_handle=device_handle,
            transfer_stream=transfer_stream,
            slots=tuple(create_slot() for _ in range(_NUM_PIPELINE_SLOTS)),
        )


def _include_compute_scratch_requirement(
    requirements: dict[_BufferKey, _BufferRequirements],
    item: _ItemT,
    local_tensor_spec: Callable[[_ItemT], tuple[torch.Size, torch.dtype, torch.device]],
) -> None:
    shape, dtype, device = local_tensor_spec(item)
    requirements.setdefault(
        (device, dtype), _BufferRequirements()
    ).include_compute_scratch(math.prod(shape))


class _BucketedRedistributionRuntime(Generic[_ItemT]):
    """Execute bucket plans with a rolling communication prefetch window.

    ``reserve_buffers`` prepares fixed-capacity tensors after planning or a
    replan. ``run`` allocates their backing storage on the caller stream, and
    ``release_buffers`` frees accelerator storage after transfer completion is
    ordered back to that caller; host storage stays allocated between updates.
    Tensor metadata and cross-stream events stay alive. One slot holds the
    current bucket and the other prefetches the next redistributed bucket.

    ``prepare`` writes an optimizer input into runtime-owned scratch, ``compute``
    updates its runtime-owned input in place, and ``finalize`` consumes a
    runtime-owned result before reuse. Callbacks run under the stream selected
    by the runtime and must not retain tensors, synchronize, or call
    ``Tensor.record_stream()``. The exception: a callback may synchronize when
    the compute buffers live on the host, because the runtime's CPU streams are
    no-ops and the gloo collectives already block the host. Local-only buckets are prefetch barriers, so
    no later redistributed ``prepare`` runs before an intervening local bucket.

    Any exception is fatal: parameters or optimizer state may already be
    updated and communication may be in flight, so callers must not reuse this
    runtime or optimizer.
    """

    def __init__(
        self,
        device: torch.device,
    ) -> None:
        # pyrefly: ignore [read-only]
        self._device = device
        self._context: _CommunicationContext | None = None
        self._local_slot = _BufferSlot()

    def reserve_buffers(
        self,
        plans: Sequence[_BucketPlan[_ItemT]],
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
    ) -> None:
        """Prepare tensor capacities and events once per plan, without storage."""
        context = self._context
        created_context = False
        if context is None and any(
            isinstance(plan, _RedistributionBucketPlan) for plan in plans
        ):
            context = _CommunicationContext.create(self._device)
            created_context = True

        device_handle = (
            context.device_handle
            if context is not None
            else _stream_device_module(self._device)
        )
        compute_stream = device_handle.current_stream(self._device)
        local_requirements: dict[_BufferKey, _BufferRequirements] = {}
        slot_requirements = tuple({} for _ in context.slots) if context else ()
        redistributed_index = 0
        for plan in plans:
            if isinstance(plan, _LocalBucketPlan):
                for item in plan.items:
                    _include_compute_scratch_requirement(
                        local_requirements,
                        item,
                        local_tensor_spec,
                    )
                continue

            assert context is not None
            requirements = slot_requirements[
                redistributed_index % len(slot_requirements)
            ]
            redistributed_index += 1
            to_compute = plan.storage_to_compute_schedule
            to_storage = plan.compute_to_storage_schedule
            wire_requirements = requirements.setdefault(
                (plan.device, plan.compute_dtype), _BufferRequirements()
            )
            storage_requirements = requirements.setdefault(
                (plan.device, plan.dtype), _BufferRequirements()
            )
            wire_requirements.include_communication(
                storage_exchange_numel=max(
                    to_compute.input_buffer_numel,
                    to_storage.output_buffer_numel,
                ),
                compute_exchange_numel=max(
                    to_compute.output_buffer_numel,
                    plan.compute_to_storage_offset + to_storage.input_buffer_numel,
                ),
            )
            participant = plan.group.local_participant
            for redistribution_plan in plan.redistribution_plans:
                storage_partition = redistribution_plan.storage_partition(participant)
                storage_requirements.include_storage_scratch(
                    math.prod(storage_partition.tensor_shape)
                )
                compute_partition = redistribution_plan.compute_partition(participant)
                if compute_numel := math.prod(compute_partition.tensor_shape):
                    wire_requirements.include_compute_scratch(compute_numel)
            for item in plan.unredistributed_items:
                _include_compute_scratch_requirement(
                    requirements,
                    item,
                    local_tensor_spec,
                )

        self._local_slot.reserve(local_requirements)
        if context is not None:
            for slot, requirements in zip(
                context.slots, slot_requirements, strict=True
            ):
                slot.buffers.reserve(requirements)
            if created_context:
                for slot in context.slots:
                    slot.compute_input_ready.record(context.transfer_stream)
                    slot.compute_done.record(compute_stream)
                    slot.done.record(context.transfer_stream)
            self._context = context

    def release_buffers(self) -> None:
        """Release accelerator storage after run() orders transfer completion.

        Freed blocks can serve later eager allocations on their caller stream.
        This does not make them available to a separate CUDA-graph pool.
        Host storage, tensor metadata, communication resources, and plans remain owned.
        """
        self._local_slot.release()
        if self._context is not None:
            for slot in self._context.slots:
                slot.buffers.release()

    def run(
        self,
        plans: Sequence[_BucketPlan[_ItemT]],
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
        prepare: Callable[[_ItemT, Tensor], None],
        compute: Callable[[_ItemT, Tensor], None],
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> None:
        context = self._context
        handle = (
            context.device_handle
            if context is not None
            else _stream_device_module(self._device)
        )
        caller = handle.current_stream(self._device)
        self._local_slot.allocate()
        if context is not None:
            for slot in context.slots:
                slot.buffers.allocate()

        transfer_stream: torch.Stream | None = None
        completed = False
        previous: _BucketWork[_ItemT] | None = None
        prefetched: deque[_BucketWork[_ItemT]] = deque()
        next_plan_to_prefetch = 0
        redistributed_index = 0

        def prefetch_next(slot: _PipelineSlot) -> _BucketWork[_ItemT] | None:
            nonlocal next_plan_to_prefetch
            if next_plan_to_prefetch >= len(plans):
                return None
            next_plan = plans[next_plan_to_prefetch]
            if isinstance(next_plan, _LocalBucketPlan):
                return None
            next_plan_to_prefetch += 1
            assert context is not None and transfer_stream is not None
            return self._enqueue_storage_to_compute(
                next_plan,
                slot,
                context,
                prepare=prepare,
            )

        try:
            for plan_index, plan in enumerate(plans):
                if isinstance(plan, _LocalBucketPlan):
                    assert not prefetched
                    with handle.stream(caller):
                        self._compute_without_redistribution(
                            plan.items,
                            self._local_slot,
                            local_tensor_spec=local_tensor_spec,
                            prepare=prepare,
                            compute=compute,
                            finalize=finalize,
                        )
                    next_plan_to_prefetch = plan_index + 1
                    continue

                assert context is not None
                if transfer_stream is None:
                    context.transfer_stream.wait_stream(caller)
                    transfer_stream = context.transfer_stream
                previous_work = previous
                if prefetched:
                    assert previous_work is not None
                    work = prefetched.popleft()
                    assert work.plan is plan
                    self._enqueue_compute_to_storage(
                        previous_work, context, finalize=finalize
                    )
                else:
                    slot = context.slots[redistributed_index % len(context.slots)]
                    if previous_work is not None and slot is previous_work.slot:
                        self._enqueue_compute_to_storage(
                            previous_work, context, finalize=finalize
                        )
                        work = self._enqueue_storage_to_compute(
                            plan,
                            slot,
                            context,
                            prepare=prepare,
                        )
                    else:
                        work = self._enqueue_storage_to_compute(
                            plan,
                            slot,
                            context,
                            prepare=prepare,
                        )
                        if previous_work is not None:
                            self._enqueue_compute_to_storage(
                                previous_work, context, finalize=finalize
                            )
                    next_plan_to_prefetch = plan_index + 1

                self._compute_bucket(
                    work,
                    work.slot,
                    caller,
                    context,
                    local_tensor_spec=local_tensor_spec,
                    prepare=prepare,
                    compute=compute,
                    finalize=finalize,
                )
                if previous_work is not None:
                    self._release(previous_work, caller)
                previous = work
                redistributed_index += 1

                while len(prefetched) < len(context.slots) - 1:
                    slot = context.slots[
                        (redistributed_index + len(prefetched)) % len(context.slots)
                    ]
                    if future_work := prefetch_next(slot):
                        prefetched.append(future_work)
                    else:
                        break

            assert not prefetched
            if previous is not None:
                assert context is not None
                self._enqueue_compute_to_storage(previous, context, finalize=finalize)
                self._release(previous, caller)
            completed = True
        finally:
            if not completed and transfer_stream is not None:
                # This fatal-only cleanup intentionally has no isolated mock-stream
                # test to keep the unit-test footprint minimal. An exception makes
                # the optimizer and training job unrecoverable.
                # Preserve allocator lifetime ordering for work already queued
                # on either stream without suppressing the active exception.
                transfer_stream.wait_stream(caller)
                caller.wait_stream(transfer_stream)

    @staticmethod
    def _enqueue_storage_to_compute(
        plan: _RedistributionBucketPlan[_ItemT],
        slot: _PipelineSlot,
        context: _CommunicationContext,
        *,
        prepare: Callable[[_ItemT, Tensor], None],
    ) -> _BucketWork[_ItemT]:
        handle = context.device_handle
        transfer = context.transfer_stream
        with handle.stream(transfer):
            (
                storage_buffer,
                compute_fragment_buffer,
            ) = slot.buffers.communication_buffers(plan)
            pieces = _prepare_redistributed(
                plan,
                slot.buffers,
                None if _fresh_buffers(plan.device) else storage_buffer,
                prepare=prepare,
            )
            if pieces is not None:
                storage_buffer = _packed_buffer(
                    pieces,
                    plan.storage_to_compute_schedule.input_buffer_numel,
                    dtype=plan.compute_dtype,
                    device=plan.device,
                )
            work = _BucketWork(plan, slot, storage_buffer, compute_fragment_buffer)
            _execute_packed_all_to_all(
                plan.storage_to_compute_schedule,
                output=compute_fragment_buffer,
                input=storage_buffer,
            )
            slot.compute_input_ready.record(transfer)
        return work

    @staticmethod
    def _compute_bucket(
        work: _BucketWork[_ItemT],
        slot: _PipelineSlot,
        caller_stream: torch.Stream,
        context: _CommunicationContext,
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
        prepare: Callable[[_ItemT, Tensor], None],
        compute: Callable[[_ItemT, Tensor], None],
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> None:
        handle = context.device_handle
        with handle.stream(caller_stream):
            _BucketedRedistributionRuntime._compute_without_redistribution(
                work.plan.unredistributed_items,
                slot.buffers,
                local_tensor_spec=local_tensor_spec,
                prepare=prepare,
                compute=compute,
                finalize=finalize,
            )
            caller_stream.wait_event(slot.compute_input_ready)
            _compute_redistributed(
                work,
                slot.buffers,
                compute=compute,
            )
            slot.compute_done.record(caller_stream)

    @staticmethod
    def _enqueue_compute_to_storage(
        work: _BucketWork[_ItemT],
        context: _CommunicationContext,
        *,
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> None:
        handle = context.device_handle
        transfer = context.transfer_stream
        with handle.stream(transfer):
            transfer.wait_event(work.slot.compute_done)
            returned = work.returned_fragment_buffer
            if returned is not None:
                # Compiled: receive into a fresh buffer rather than over the
                # storage -> compute send buffer.
                work.storage_buffer = torch.empty(
                    work.plan.compute_to_storage_schedule.output_buffer_numel,
                    dtype=work.plan.compute_dtype,
                    device=work.plan.device,
                )
            else:
                returned = work.compute_fragment_buffer[
                    work.plan.compute_to_storage_offset :
                ]
            _execute_packed_all_to_all(
                work.plan.compute_to_storage_schedule,
                output=work.storage_buffer,
                input=returned,
            )
            _finalize_redistributed(work, work.slot.buffers, finalize=finalize)
            work.slot.done.record(transfer)

    @staticmethod
    def _release(work: _BucketWork[_ItemT], caller_stream: torch.Stream) -> None:
        caller_stream.wait_event(work.slot.done)

    @staticmethod
    def _compute_without_redistribution(
        items: Sequence[_ItemT],
        slot: _BufferSlot,
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
        prepare: Callable[[_ItemT, Tensor], None],
        compute: Callable[[_ItemT, Tensor], None],
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> None:
        for item in items:
            shape, dtype, device = local_tensor_spec(item)
            prepared = slot.compute_buffer(shape, dtype=dtype, device=device)
            prepare(item, prepared)
            compute(item, prepared)
            finalize(item, prepared)


def _packed_buffer(
    pieces: list[tuple[int, Tensor]],
    numel: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """A flat send buffer from (offset, flat source) pieces, zeros elsewhere.

    Compiled runs build send buffers with one concatenation. Writing each
    piece into a flat buffer becomes a chain of dynamic-update-slices on the
    whole buffer, and writing returned fragments over received ones that are
    still being read makes XLA copy the whole buffer per write: the 64-rank
    expert buckets of K3 at EP16 x eFSDP64 held ~120 copies of a 147 MiB buffer
    per bucket (66 GiB for seven buckets).
    """
    parts: list[Tensor] = []
    cursor = 0
    for offset, piece in sorted(pieces, key=lambda entry: entry[0]):
        if offset < cursor:
            raise RuntimeError("overlapping packed all-to-all spans")
        if offset > cursor:
            parts.append(torch.zeros(offset - cursor, dtype=dtype, device=device))
        parts.append(piece.reshape(-1).to(dtype))
        cursor = offset + piece.numel()
    if cursor > numel:
        raise RuntimeError("packed all-to-all spans exceed the buffer")
    if cursor < numel:
        parts.append(torch.zeros(numel - cursor, dtype=dtype, device=device))
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _prepare_redistributed(
    plan: _RedistributionBucketPlan[_ItemT],
    slot: _BufferSlot,
    storage_buffer: Tensor | None,
    *,
    prepare: Callable[[_ItemT, Tensor], None],
) -> list[tuple[int, Tensor]] | None:
    """Pack every redistributed item into ``storage_buffer``; without one
    (compiled runs), return the (offset, flat source) pieces instead."""
    schedule = plan.storage_to_compute_schedule
    participant = plan.group.local_participant
    pieces: list[tuple[int, Tensor]] = []
    for index, (item, redistribution_plan) in enumerate(
        zip(
            plan.redistributed_items,
            plan.redistribution_plans,
            strict=True,
        )
    ):
        partition = redistribution_plan.storage_partition(participant)
        prepared = slot.storage_partition_buffer(
            partition.tensor_shape,
            dtype=plan.dtype,
            device=plan.device,
        )
        prepare(item, prepared)
        spans = schedule.input_spans_by_parameter[index]
        prepared_views = tuple(
            _tensor_region_view(prepared, span.region).reshape(-1) for span in spans
        )
        if storage_buffer is None:
            pieces.extend(
                (span.buffer_offset, view)
                for span, view in zip(spans, prepared_views, strict=True)
            )
            continue
        packed_views = tuple(
            storage_buffer[span.buffer_offset : span.buffer_offset + span.numel]
            for span in spans
        )
        _batched_copy_(packed_views, prepared_views)
    return None if storage_buffer is not None else pieces


def _compute_redistributed(
    work: _BucketWork[_ItemT],
    slot: _BufferSlot,
    *,
    compute: Callable[[_ItemT, Tensor], None],
) -> None:
    plan = work.plan
    participant = plan.group.local_participant
    to_compute = plan.storage_to_compute_schedule
    to_storage = plan.compute_to_storage_schedule
    compiling = _fresh_buffers(plan.device)
    returned_fragment_buffer = work.compute_fragment_buffer[
        plan.compute_to_storage_offset :
    ]
    returned_pieces: list[tuple[int, Tensor]] = []
    for index, (item, redistribution_plan) in enumerate(
        zip(
            plan.redistributed_items,
            plan.redistribution_plans,
            strict=True,
        )
    ):
        partition = redistribution_plan.compute_partition(participant)
        if not math.prod(partition.tensor_shape):
            continue
        received_spans = to_compute.output_spans_by_parameter[index]
        received_views = tuple(
            work.compute_fragment_buffer[
                span.buffer_offset : span.buffer_offset + span.numel
            ].view(span.region.shape)
            for span in received_spans
        )
        compute_tensor = _concatenated_regions(
            partition.tensor_shape, received_spans, received_views
        ) if compiling else None
        if compute_tensor is None:
            compute_tensor = slot.compute_buffer(
                partition.tensor_shape,
                dtype=plan.compute_dtype,
                device=plan.device,
            )
            compute_views = tuple(
                _tensor_region_view(compute_tensor, span.region)
                for span in received_spans
            )
            _batched_copy_(compute_views, received_views)

        compute(item, compute_tensor)

        output_spans = to_storage.input_spans_by_parameter[index]
        compute_output_views = tuple(
            _tensor_region_view(compute_tensor, span.region).reshape(-1)
            for span in output_spans
        )
        if compiling:
            returned_pieces.extend(
                (span.buffer_offset, view)
                for span, view in zip(output_spans, compute_output_views, strict=True)
            )
            continue
        packed_views = tuple(
            returned_fragment_buffer[
                span.buffer_offset : span.buffer_offset + span.numel
            ]
            for span in output_spans
        )
        _batched_copy_(packed_views, compute_output_views)
    if compiling:
        work.returned_fragment_buffer = _packed_buffer(
            returned_pieces,
            to_storage.input_buffer_numel,
            dtype=plan.compute_dtype,
            device=plan.device,
        )


def _finalize_redistributed(
    work: _BucketWork[_ItemT],
    slot: _BufferSlot,
    *,
    finalize: Callable[[_ItemT, Tensor], None],
) -> None:
    plan = work.plan
    participant = plan.group.local_participant
    schedule = work.plan.compute_to_storage_schedule
    for index, (item, redistribution_plan) in enumerate(
        zip(
            plan.redistributed_items,
            plan.redistribution_plans,
            strict=True,
        )
    ):
        partition = redistribution_plan.storage_partition(participant)
        spans = schedule.output_spans_by_parameter[index]
        packed_views = tuple(
            work.storage_buffer[
                span.buffer_offset : span.buffer_offset + span.numel
            ].view(span.region.shape)
            for span in spans
        )
        update = (
            _concatenated_regions(partition.tensor_shape, spans, packed_views)
            if _fresh_buffers(plan.device)
            else None
        )
        if update is not None:
            update = update.to(plan.dtype)
        else:
            update = slot.storage_partition_buffer(
                partition.tensor_shape,
                dtype=plan.dtype,
                device=plan.device,
            )
            update_views = tuple(
                _tensor_region_view(update, span.region) for span in spans
            )
            _batched_copy_(update_views, packed_views)
        finalize(item, update)


def _concatenated_regions(
    shape: tuple[int, ...],
    spans: Sequence[Any],
    views: tuple[Tensor, ...],
) -> Tensor | None:
    """The tensor of ``shape`` tiled by ``spans``' regions, by one concatenation.

    Returns None unless the regions are slabs along one dimension that tile it
    in order (then no slice writes are needed); callers copy instead.
    """
    regions = [span.region for span in spans]
    if not regions:
        return None
    if len(regions) == 1 and tuple(regions[0].shape) == tuple(shape):
        return views[0]
    for dim in range(len(shape)):
        if any(
            region.shape[other] != shape[other] or region.offsets[other] != 0
            for region in regions
            for other in range(len(shape))
            if other != dim
        ):
            continue
        order = sorted(range(len(regions)), key=lambda i: regions[i].offsets[dim])
        cursor = 0
        for i in order:
            if regions[i].offsets[dim] != cursor:
                return None
            cursor += regions[i].shape[dim]
        if cursor != shape[dim]:
            return None
        return torch.cat([views[i] for i in order], dim=dim)
    return None


def _batched_copy_(
    destinations: tuple[Tensor, ...],
    sources: tuple[Tensor, ...],
) -> None:
    """Copy aligned region views with a single foreach launch."""
    if len(destinations) != len(sources):
        raise ValueError("destinations and sources must have equal length")
    if not destinations:
        return
    if len(destinations) == 1:
        destinations[0].copy_(sources[0])
        return
    torch._foreach_copy_(destinations, sources)


def _tensor_region_view(tensor: Tensor, region: _TensorRegion) -> Tensor:
    view = tensor[
        tuple(
            slice(offset, offset + size)
            for offset, size in zip(region.offsets, region.shape, strict=True)
        )
    ]
    return view
