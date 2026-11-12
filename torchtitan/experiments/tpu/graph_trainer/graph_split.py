# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Split a GraphTrainer joint FX graph into independently compiled modules.

``compile.graph_split.enabled`` divides the mandatory joint graph into forward
and backward GraphModules connected by explicit residual tensors. The forward
graph can then be divided into sequential layer ranges. The backward graph can remain
whole, group adjacent layer computations with their FSDP collectives, or form
staggered all-gather/compute/reduce-scatter pipeline stages. These subdivisions
are therefore always nested under the forward/backward split.

Each rematerialization region is one transformer layer. Combining regions in
one executable changes only the compiler scheduling scope; it preserves the
per-layer activation-checkpoint boundaries and just-in-time recompute order.
Multi-layer rematerialization instead trades fewer saved or offloaded boundaries
for longer-lived intermediates or repeated recompute and is outside this pass.

The result wrapper retains both the executable GraphModule and the node-to-
partition metadata used to validate that layers and collective chains remain
indivisible.
"""

from collections import Counter, defaultdict
import operator
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch._inductor.fx_passes.bucketing import is_wait_tensor
from torch.fx.passes.split_module import split_module

from torchtitan.experiments.graph_trainer.cpu_offload import _is_view
from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _get_module_fqn,
    _is_backward_node,
)
from torchtitan.experiments.graph_trainer.graph_pp.partition import (
    _forward_mutations_to_materialize,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    is_mutation_node,
)
from torchtitan.experiments.tpu.graph_trainer.collectives import (
    chain_boundary,
    classify_fsdp_roles,
    collect_collective_chains,
    collective_producer,
    is_collective,
    is_gather,
    is_getitem,
    is_reduce_scatter,
    op_name,
)
from torchtitan.experiments.tpu.graph_trainer.passes.expert_all_gathers_under_splash import (
    _is_splash_output,
    apply_expert_all_gathers_under_splash,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _optimization_barrier,
)
from torchtitan.experiments.tpu.graph_trainer.split_cpu_offload import (
  HostDonorSpec,
  initialize_pinned_host_buffer,
  is_ao_transfer,
  place_ao_transfer_pairs,
  thread_host_offload_donors,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
    tpu_backend_compile_partitioned_pass,
    tpu_backend_compile_pass,
)
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
  from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
  from torchtitan.experiments.tpu.graph_trainer.config import (
      GraphSplitterConfig,
  )
  from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
      TpuGraphTrainer,
  )


def _synchronize_tpu(values: list[torch.Tensor], *, wait: bool) -> None:
  """Synchronize lazily so host-only graph-split tests need no TorchTPU."""
  from torch_tpu._internal.sync import synchronize  # pytype: disable=import-error

  synchronize(values, wait=wait)


def _enqueue_forward_graph_outputs(values: list[torch.Tensor]) -> None:
  values = [value for value in values if value.device.type == "tpu"]
  if values:
    _synchronize_tpu(values, wait=False)


def _as_tuple(value: Any) -> tuple[Any, ...]:
  if isinstance(value, tuple):
    return value
  if isinstance(value, list):
    return tuple(value)
  return (value,)


class SplitForwardBackwardModule(nn.Module):

  def __init__(
      self,
      forward_graph: torch.fx.GraphModule,
      backward_graph: torch.fx.GraphModule,
      *,
      forward_output_positions: tuple[int, ...],
      backward_output_positions: tuple[int, ...],
      constant_outputs: dict[int, Any],
      num_direct_forward_outputs: int,
      num_residual_outputs: int,
      num_outputs: int,
      host_donor_specs: tuple[HostDonorSpec, ...] = (),
      host_donor_output_positions: tuple[int, ...] = (),
  ) -> None:
    super().__init__()
    self.forward_graph = forward_graph
    self.backward_graph = backward_graph
    self.forward_output_positions = forward_output_positions
    self.backward_output_positions = backward_output_positions
    self.constant_outputs = constant_outputs
    self.num_direct_forward_outputs = num_direct_forward_outputs
    self.num_residual_outputs = num_residual_outputs
    self.num_outputs = num_outputs
    self.host_donor_specs: tuple[HostDonorSpec, ...] = host_donor_specs
    self.host_donor_output_positions: tuple[int, ...] = (
        host_donor_output_positions
    )
    self._host_donors: tuple[torch.Tensor, ...] | None = None

  def _initialize_host_donors(
      self,
      flat_program_inputs: tuple[Any, ...],
  ) -> tuple[torch.Tensor, ...]:
    """Create one persistent pinned-host tensor per offloaded activation.

    ``host_donor_specs`` describes the slots; ``flat_program_inputs`` is scanned
    only to select their TPU device. An ordinary pinned CPU tensor is not a TPU
    tensor backed by PJRT pinned-host memory. TorchTPU does not currently expose
    a direct allocator for the latter, so each slot is created through AO D2H.
    Materializing one at a time bounds temporary device storage to one
    activation.
    """
    children = tuple(self.forward_graph.children())
    directly_compiled = hasattr(self.forward_graph, "_tpu_executable")
    children_compiled = bool(children) and all(
        any(hasattr(module, "_tpu_executable") for module in child.modules())
        for child in children
    )
    if not directly_compiled and not children_compiled:
      raise RuntimeError(
          "Host-donor threading requires a compiled TPU forward program"
      )
    device = next(
        (
            value.device
            for value in flat_program_inputs
            if isinstance(value, torch.Tensor) and value.device.type == "tpu"
        ),
        None,
    )
    if device is None:
      raise RuntimeError("Cannot initialize host donors without a TPU input")
    donors = []
    for spec in self.host_donor_specs:
      source = torch.empty_strided(
          spec.shape,
          spec.stride,
          dtype=spec.dtype,
          device=device,
      )
      donor = initialize_pinned_host_buffer(source)
      # Bound temporary HBM usage while constructing the persistent host pool.
      _synchronize_tpu([donor], wait=True)
      donors.append(donor)
      del source
    logger.info(
        "Initialized %d persistent pinned-host activation donor buffers",
        len(donors),
    )
    return tuple(donors)

  def forward(self, *flat_inputs: Any) -> tuple[Any, ...]:
    if self.host_donor_specs and self._host_donors is None:
      self._host_donors = self._initialize_host_donors(flat_inputs)
    donor_inputs = self._host_donors or ()
    forward_outputs = _as_tuple(
        self.forward_graph(*flat_inputs, *donor_inputs)
    )
    direct_forward_outputs = forward_outputs[
        : self.num_direct_forward_outputs
    ]
    residual_end = (
        self.num_direct_forward_outputs + self.num_residual_outputs
    )
    residuals = forward_outputs[
        self.num_direct_forward_outputs : residual_end
    ]
    # A compiled child may return lazy outputs without enqueueing its executable.
    # Residual dependencies cover only the forward children that produce those
    # residuals; direct outputs such as loss may come from other children. Enqueue
    # every tensor output before invoking backward so all forward children precede
    # backward collectives. wait=False preserves asynchronous host execution.
    _enqueue_forward_graph_outputs(list(forward_outputs))
    backward_outputs = _as_tuple(
        self.backward_graph(*flat_inputs, *residuals)
    )
    if self.host_donor_output_positions:
      self._host_donors = tuple(
          forward_outputs[position]
          for position in self.host_donor_output_positions
      )

    outputs = [None] * self.num_outputs
    positioned_outputs = chain(
        zip(
            self.forward_output_positions,
            direct_forward_outputs,
            strict=True,
        ),
        zip(
            self.backward_output_positions,
            backward_outputs,
            strict=True,
        ),
        self.constant_outputs.items(),
    )
    for position, value in positioned_outputs:
      outputs[position] = value
    return tuple(outputs)


@dataclass
class SplitForwardBackwardGraph:
  module: SplitForwardBackwardModule
  forward_graph: torch.fx.GraphModule
  backward_graph: torch.fx.GraphModule
  forward_example_inputs: tuple[Any, ...]
  backward_example_inputs: tuple[Any, ...]
  residual_nodes: tuple[torch.fx.Node, ...]


@dataclass
class PartitionedGraph:
  module: torch.fx.GraphModule
  partition_map: dict[torch.fx.Node, int]
  layer_to_partition: dict[int, int]
  collective_layer_to_partition: dict[str, dict[int, int]]


@dataclass(frozen=True)
class GraphSplitter:
  """Validate, partition, and compile one GraphTrainer trace."""

  enabled: bool
  config: "GraphSplitterConfig"
  n_layers: int

  @classmethod
  def from_trainer_config(
      cls,
      config: "TpuGraphTrainer.Config",
      *,
      optimizer_in_graph: bool,
  ) -> "GraphSplitter":
    split_config = config.compile.graph_split
    enabled = split_config.enabled
    if split_config.forward_layers_per_executable == 0 or (
        split_config.forward_layers_per_executable < -1
    ):
      raise ValueError(
          "forward_layers_per_executable must be -1 or positive"
      )
    if split_config.forward_prefetch_parameter_gathers and (
        not enabled or split_config.forward_layers_per_executable < 1
    ):
      raise ValueError(
          "forward_prefetch_parameter_gathers requires a split forward graph "
          "(compile.graph_split.enabled and forward_layers_per_executable > 0)"
      )
    if split_config.backward_partition_mode not in (
        "none",
        "by_layer",
        "fsdp_pipeline",
    ):
      raise ValueError(
          "backward_partition_mode must be none, by_layer, or fsdp_pipeline"
      )
    if split_config.backward_layers_per_executable < 1:
      raise ValueError(
          "backward_layers_per_executable must be at least 1"
      )
    if split_config.backward_partition_mode != "none" and not enabled:
      raise ValueError(
          "backward_partition_mode requires compile.graph_split.enabled"
      )
    if split_config.cpu_offload_prefetch_executables < 0:
      raise ValueError(
          "cpu_offload_prefetch_executables must be nonnegative"
      )
    if split_config.cpu_offload_prefetch_executables and (
        not enabled or split_config.backward_partition_mode == "none"
    ):
      raise ValueError(
          "cpu_offload_prefetch_executables requires a split backward graph"
      )
    if enabled and config.compile.fsdp_barrier_policy != "none":
      raise ValueError(
          "Graph splitting does not currently compose with "
          "compile.fsdp_barrier_policy because the whole-graph collective "
          "placement pass is not partition-boundary aware"
      )
    # Optimizer-in-graph needs a third explicit phase rather than treating its
    # state reads, mutations, and parameter updates as backward.
    if enabled and optimizer_in_graph:
      raise ValueError(
          "TPU GraphTrainer graph split does not support "
          "optimizer_compile='in_graph'"
      )
    # This artifact format stores one transformed FX graph. TorchTPU's XLA and
    # Tier-2 executable caches remain supported for every compiled child.
    if enabled and config.compile.precompile_artifact_dir:
      raise ValueError(
          "TPU GraphTrainer graph split does not support serialized "
          "precompiled FX artifacts"
      )
    return cls(
        enabled=enabled,
        config=split_config,
        n_layers=len(config.model_spec.model.layers) if enabled else 0,
    )

  def split_and_compile(self, traced: "TracedResult") -> torch.fx.GraphModule:
    """Split one transformed joint graph and compile its child executables."""
    if not self.enabled:
      raise RuntimeError("GraphSplitter is disabled")

    if self.config.expert_all_gathers_under_splash:
      apply_expert_all_gathers_under_splash(
          traced.gm,
          recomputed_only=(
              self.config.expert_all_gathers_under_splash_recomputed_only
          ),
          max_gathers_per_layer=(
              self.config.expert_all_gathers_under_splash_max_gathers
          ),
      )
    split = split_forward_backward_graph(traced.gm, traced.example_inputs)

    # Host-slot recurrence belongs to the top-level forward signature, whether
    # that program is compiled whole or subdivided into layer children.
    donor_specs, donor_output_positions, donor_examples = (
        thread_host_offload_donors(split.forward_graph)
    )
    split.module.host_donor_specs = donor_specs
    split.module.host_donor_output_positions = donor_output_positions
    forward_example_inputs = (*split.forward_example_inputs, *donor_examples)
    donor_input_positions = tuple(
        range(len(split.forward_example_inputs), len(forward_example_inputs))
    )

    if self.config.forward_layers_per_executable > 0:
      forward = split_forward_graph_by_layer(
          split.forward_graph,
          self.config.forward_layers_per_executable,
          isolate_prefix=self.config.forward_isolate_prefix,
          prefetch_parameter_gathers=(
              self.config.forward_prefetch_parameter_gathers
          ),
          prefetch_min_gather_bytes=self.config.prefetch_min_gather_bytes,
          prefetch_gathers_under_splash=(
              self.config.forward_prefetch_gathers_under_splash
          ),
      )
      split.forward_graph = forward.module
      split.module.forward_graph = forward.module
      tpu_backend_compile_partitioned_pass(
          forward.module,
          forward_example_inputs,
          donate_internal_buffers=self.config.donate_internal_buffers,
      )
    else:
      tpu_backend_compile_pass(
          split.forward_graph,
          forward_example_inputs,
          donated_input_positions=donor_input_positions,
      )

    mode = self.config.backward_partition_mode
    if mode in ("by_layer", "fsdp_pipeline"):
      backward = split_backward_graph_by_layer(
          split.backward_graph,
          n_layers=self.n_layers,
          layers_per_executable=self.config.backward_layers_per_executable,
          pipeline_collectives=mode == "fsdp_pipeline",
          pipeline_kinds=self.config.pipeline_kinds or ("ag", "rs"),
          pipeline_min_gather_bytes=self.config.prefetch_min_gather_bytes,
          defer_reduction_max_bytes=(
              self.config.backward_defer_reduction_max_bytes
          ),
          cpu_offload_prefetch_executables=(
              self.config.cpu_offload_prefetch_executables
          ),
          deferred_reduction_deadlines=(
              self.config.deferred_reduction_deadlines
          ),
      )
      split.backward_graph = backward.module
      split.module.backward_graph = backward.module
      tpu_backend_compile_partitioned_pass(
          backward.module,
          split.backward_example_inputs,
          is_fwd=False,
          donate_internal_buffers=self.config.donate_internal_buffers,
      )
    else:
      tpu_backend_compile_pass(
          split.backward_graph,
          split.backward_example_inputs,
          is_fwd=False,
      )
    return split.module


def _is_spmd_region_marker(node: torch.fx.Node, marker: str) -> bool:
  return (
      node.op == "call_function"
      and str(getattr(node.target, "overloadpacket", node.target))
      == f"torch_tpu.{marker}"
  )


def _body_nodes(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
  return [
      node
      for node in gm.graph.nodes
      if node.op not in ("placeholder", "output")
  ]


def _group_nodes_by_layer(
    nodes: list[torch.fx.Node],
    *,
    excluded: set[torch.fx.Node] | None = None,
) -> tuple[dict[torch.fx.Node, int], dict[int, set[torch.fx.Node]]]:
  excluded = excluded or set()
  node_layer_ids = {node: _get_layer_id(node) for node in nodes}
  layer_nodes: dict[int, set[torch.fx.Node]] = defaultdict(set)
  for node, layer_id in node_layer_ids.items():
    if layer_id >= 0 and node not in excluded:
      layer_nodes[layer_id].add(node)
  return node_layer_ids, dict(layer_nodes)


def _assign_partition(
    partition_map: dict[torch.fx.Node, int],
    nodes: set[torch.fx.Node],
    partition: int,
    owner: str,
) -> None:
  conflicts = {
      node.name: partition_map[node]
      for node in nodes
      if node in partition_map and partition_map[node] != partition
  }
  if conflicts:
    raise ValueError(f"{owner} has conflicting ownership: {conflicts}")
  partition_map.update(dict.fromkeys(nodes, partition))


def _order_child_boundaries(parent: torch.fx.GraphModule) -> None:
  """Order each child's inputs by last read and outputs by first availability.

  ``split_module`` numbers a child's inputs and outputs in discovery order, and
  only ``parent`` calls the child, so the order carries no meaning. It does
  matter to donation: XLA aliases donated inputs to equal-size outputs pairwise
  in that order, and an output cannot be written before its donor's last read.
  Sorting both sides lets the earliest-freed input back the earliest-ready
  output, e.g. a host-to-device reload that depends on nothing else.

  TODO: drop this ordering once TorchTPU can pass explicit input->output
  aliases to XLA; today it only marks donatable inputs (``jax.buffer_donor``)
  and XLA picks the pairs. To check the pairs XLA picked, dump HLO with
  ``XLA_FLAGS=--xla_dump_to=<dir>`` and read each module's
  ``input_output_alias``.
  """
  calls = parent.graph.find_nodes(op="call_module")
  # Each child is rewritten together with its one call site.
  assert len({call.target for call in calls}) == len(calls), (
      "a split child is called more than once"
  )
  for call in calls:
    child = parent.get_submodule(call.target)
    nodes = list(child.graph.nodes)
    position = {node: index for index, node in enumerate(nodes)}
    depth: dict[torch.fx.Node, int] = {}
    for node in nodes:
      depth[node] = 1 + max(map(depth.get, node.all_input_nodes), default=-1)

    inputs = [node for node in nodes if node.op == "placeholder"]
    assert all(node.op == "placeholder" for node in nodes[: len(inputs)])
    last_read = {
        node: max((position[user] for user in node.users), default=-1)
        for node in inputs
    }
    inputs_by_read = sorted(inputs, key=last_read.__getitem__)
    first_op = nodes[len(inputs)]
    for node in inputs_by_read:
      first_op.prepend(node)
    call.args = tuple(call.args[inputs.index(node)] for node in inputs_by_read)

    output = child.graph.output_node()
    values = output.args[0]
    if isinstance(values, (tuple, list)):
      order = sorted(
          range(len(values)),
          key=lambda i: (depth[values[i]], position[values[i]]),
      )
      output.args = (tuple(values[i] for i in order),)
      new_index = {old: new for new, old in enumerate(order)}
      assert all(is_getitem(user) for user in call.users)
      for user in list(call.users):
        user.args = (call, new_index[user.args[1]])
    child.recompile()
  parent.recompile()


def _materialize_partitioned_graph(
    gm: torch.fx.GraphModule,
    partition_map: dict[torch.fx.Node, int],
    layer_nodes: dict[int, set[torch.fx.Node]],
    collective_layer_to_partition: dict[str, dict[int, int]] | None = None,
) -> PartitionedGraph:
  actual_partitions = sorted(set(partition_map.values()))
  normalized = {
      partition: index for index, partition in enumerate(actual_partitions)
  }
  partition_map = {
      node: normalized[partition] for node, partition in partition_map.items()
  }

  layer_to_partition = {}
  for layer_id, nodes in layer_nodes.items():
    partitions = {partition_map[node] for node in nodes}
    if len(partitions) != 1:
      raise ValueError(
          f"Layer {layer_id} spans partitions {sorted(partitions)}"
      )
    layer_to_partition[layer_id] = next(iter(partitions))

  collective_layer_to_partition = {
      kind: {
          layer_id: normalized[partition]
          for layer_id, partition in layer_partitions.items()
      }
      for kind, layer_partitions in (
          collective_layer_to_partition or {}
      ).items()
  }
  chunked_gm = split_module(gm, None, lambda node: partition_map[node])
  _order_child_boundaries(chunked_gm)
  chunked_gm.graph.lint()
  chunked_gm.recompile()
  return PartitionedGraph(
      module=chunked_gm,
      partition_map=partition_map,
      layer_to_partition=layer_to_partition,
      collective_layer_to_partition=collective_layer_to_partition,
  )


def _preserves_numel(view: torch.fx.Node) -> bool:
  source = view.args[0] if view.args else None
  if not isinstance(source, torch.fx.Node):
    return False
  out, src = view.meta.get("val"), source.meta.get("val")
  return (
      isinstance(out, torch.Tensor)
      and isinstance(src, torch.Tensor)
      and out.numel() == src.numel()
  )


def _gathered_bytes(launch: torch.fx.Node) -> int:
  """Bytes of a collective's output."""
  val = launch.meta["val"]
  values = val if isinstance(val, (tuple, list)) else (val,)
  return sum(v.numel() * v.element_size() for v in values)


def _reduced_bytes(launch: torch.fx.Node) -> int:
  """Bytes of a reduction's (unreduced) inputs.

  A reduce-scatter's output is a 1/N shard, so a large gradient can have a
  small output; deferring it would keep the whole unreduced input alive.
  """
  return sum(
      _gathered_bytes(node)
      for node in launch.all_input_nodes
      if isinstance(node.meta.get("val"), (torch.Tensor, tuple, list))
  )


def split_forward_graph_by_layer(
    gm: torch.fx.GraphModule,
    layers_per_executable: int,
    *,
    isolate_prefix: bool = False,
    prefetch_parameter_gathers: bool = False,
    prefetch_min_gather_bytes: int = 0,
    prefetch_gathers_under_splash: int = 0,
) -> PartitionedGraph:
  """Split forward into ascending, complete transformer-layer ranges.

  With ``prefetch_parameter_gathers``, each parameter all-gather chain (fed
  only by graph inputs, see ``classify_fsdp_roles``) moves to the executable
  before the one holding its first consumer, so XLA can overlap it with that
  executable's compute. Chains whose consumer is in the first executable stay
  put.
  """
  if layers_per_executable < 1:
    raise ValueError(
        "layers_per_executable must be at least 1, got "
        f"{layers_per_executable}"
    )

  body_nodes = _body_nodes(gm)
  ao_nodes = {node for node in body_nodes if is_ao_transfer(node)}
  view_nodes = {node for node in body_nodes if _is_view(node)}
  # Parameter gather chains, keyed by launch; placed after the walk below.
  gather_chains: dict[torch.fx.Node, set[torch.fx.Node]] = {}
  if prefetch_parameter_gathers:
    chains = collect_collective_chains(gm)
    gather_chains = {
        launch: chains[launch]
        for launch, role in classify_fsdp_roles(chains).items()
        if role == "ag"
        and _gathered_bytes(launch) >= prefetch_min_gather_bytes
    }
  gather_chain_nodes: set[torch.fx.Node] = set().union(
      *gather_chains.values()
  )
  node_layer_ids, layer_nodes = _group_nodes_by_layer(
      body_nodes, excluded=ao_nodes | view_nodes | gather_chain_nodes
  )
  layer_ids = sorted(layer_nodes)
  if not layer_ids:
    raise ValueError("Cannot split forward: no layer metadata was found")

  first_layer_index = next(
      index
      for index, node in enumerate(body_nodes)
      if node_layer_ids[node] >= 0
  )
  has_prefix = isolate_prefix and any(
      node.op == "call_function" and not is_ao_transfer(node)
      for node in body_nodes[:first_layer_index]
  )
  partition_offset = int(has_prefix)
  layer_to_target = {
      layer_id: partition_offset + index // layers_per_executable
      for index, layer_id in enumerate(layer_ids)
  }

  partition_map: dict[torch.fx.Node, int] = {}
  current_partition = 0
  target_partition = 0
  current_layer_id = -1
  collective_wait_counts = Counter(
      producer
      for node in body_nodes
      if is_wait_tensor(node)
      and (producer := collective_producer(node)) is not None
  )
  pending_collectives: dict[torch.fx.Node, int] = {}
  spmd_safe_depth = 0
  for node in body_nodes:
    if node in ao_nodes or node in gather_chain_nodes:
      continue
    layer_id = node_layer_ids[node]
    starts_layer = layer_id >= 0 and layer_id != current_layer_id
    if starts_layer:
      current_layer_id = layer_id
      target_partition = max(target_partition, layer_to_target[layer_id])
      if not pending_collectives and spmd_safe_depth == 0:
        current_partition = target_partition

    producer = collective_producer(node)
    if is_wait_tensor(node):
      if producer not in partition_map:
        raise ValueError(
            f"Collective producer for {node.name!r} is not partitioned"
        )
      partition_map[node] = partition_map[producer]
      remaining_waits = pending_collectives[producer] - 1
      if remaining_waits:
        pending_collectives[producer] = remaining_waits
      else:
        del pending_collectives[producer]
    elif is_getitem(node) and producer is not None:
      if producer not in partition_map:
        raise ValueError(
            f"Collective producer for {node.name!r} is not partitioned"
        )
      partition_map[node] = partition_map[producer]
    else:
      partition_map[node] = current_partition
      if is_collective(node) and not str(op_name(node)).startswith("tpu::"):
        pending_collectives[node] = max(1, collective_wait_counts[node])

    if _is_spmd_region_marker(node, "enter_spmd_safe_region"):
      spmd_safe_depth += 1
    elif _is_spmd_region_marker(node, "exit_spmd_safe_region"):
      spmd_safe_depth -= 1
      if spmd_safe_depth < 0:
        raise ValueError("Unbalanced TPU SPMD safe-region exit")

  if pending_collectives:
    raise ValueError(
        f"Forward graph has {len(pending_collectives)} unwaited collectives"
    )
  if spmd_safe_depth:
    raise ValueError("Unbalanced TPU SPMD safe-region entry")

  # Each parameter gather runs one executable ahead of its first consumer. Its
  # chain reads only graph inputs, so moving it earlier keeps every dependency
  # forward.
  num_prefetched = 0
  for launch, chain_nodes in gather_chains.items():
    _, users = chain_boundary(chain_nodes)
    consumer = min(
        (partition_map[user] for user in users if user in partition_map),
        default=0,
    )
    partition = max(consumer - 1, 0)
    num_prefetched += partition < consumer
    partition_map.update(dict.fromkeys(chain_nodes, partition))
  if gather_chains:
    logger.info(
        "Prefetching %d of %d forward parameter all-gathers one executable "
        "ahead",
        num_prefetched,
        len(gather_chains),
    )
  if prefetch_gathers_under_splash > 0 and gather_chains:
    _join_prefetched_gathers_with_splash(
        gm, gather_chains, partition_map, prefetch_gathers_under_splash
    )

  # A view left with its producer layer becomes a second executable output
  # beside its base, which XLA must materialize as a copy. Place each
  # size-preserving view with its earliest consumer so only the base crosses
  # the boundary. Slices stay put: moving one would pass its whole base.
  moved_views: Counter = Counter()
  for node in reversed(body_nodes):
    if node not in view_nodes or not _preserves_numel(node):
      continue
    user_partitions = [partition_map.get(user) for user in node.users]
    if user_partitions and None not in user_partitions:
      target = max(partition_map[node], min(user_partitions))
      if target != partition_map[node]:
        partition_map[node] = target
        moved_views[str(node.target)] += 1
  if moved_views:
    logger.info(
        "Placed %d forward views with their consumers: %s",
        sum(moved_views.values()),
        dict(moved_views),
    )

  place_ao_transfer_pairs(gm, partition_map)

  chunked = _materialize_partitioned_graph(
      gm, partition_map, layer_nodes
  )
  target_count = (
      len(layer_ids) + layers_per_executable - 1
  ) // layers_per_executable + partition_offset
  actual_count = len(set(chunked.partition_map.values()))
  if actual_count != target_count:
    logger.warning(
        "Targeted %d TPU forward executables, produced %d after preserving "
        "layer, collective, and SPMD-safe boundaries",
        target_count,
        actual_count,
    )
  logger.info(
      "Split TPU forward graph into %d executables (prefix=%s): layers=%s",
      actual_count,
      has_prefix,
      chunked.layer_to_partition,
  )
  return chunked


def _join_prefetched_gathers_with_splash(
    gm: torch.fx.GraphModule,
    gather_chains: dict[torch.fx.Node, set[torch.fx.Node]],
    partition_map: dict[torch.fx.Node, int],
    per_partition: int,
) -> None:
  """Finish each forward executable's prefetched small gathers and its first
  ``per_partition`` prefetched expert-bank gathers (>= 256 MiB) under its
  Splash attention.

  A prefetched gather is issued after the executable's own routed-token
  gathers otherwise, and the SparseCore runs collectives in issue order, so
  it idles through attention and then the next layer's weights arrive late.
  Joining a gathered weight with the Splash output in an optimization barrier
  makes the scheduler issue it before attention; the ones read first (the
  next layer's attention and first expert weights) are picked.
  """
  order = {node: index for index, node in enumerate(gm.graph.nodes)}
  splash: dict[int, torch.fx.Node] = {}
  for node in gm.graph.nodes:
    if node in partition_map and _is_splash_output(node):
      splash.setdefault(partition_map[node], node)
  candidates: dict[int, list] = defaultdict(list)
  for launch, chain in gather_chains.items():
    partition = partition_map[launch]
    inputs, users = chain_boundary(chain)
    consumers = [partition_map[user] for user in users if user in partition_map]
    if not consumers or min(consumers) <= partition:
      continue
    anchor = splash.get(partition)
    if anchor is None or any(order[node] > order[anchor] for node in inputs):
      continue
    outputs = sorted(
        (node for node in chain if any(user not in chain for user in node.users)),
        key=order.__getitem__,
    )
    first_use = min(order[user] for user in users)
    # Expert banks (hundreds of MB gathered); the router gate and shared
    # experts are small enough to count as dense here.
    is_moe = _gathered_bytes(launch) >= (256 << 20)
    candidates[partition].append((first_use, chain, outputs, is_moe))

  joined = 0
  for partition, items in candidates.items():
    anchor = splash[partition]
    items.sort(key=lambda item: item[0])
    # Every dense (attention) gather -- small and read first -- and the first
    # ``per_partition`` expert-weight gathers.
    chosen = [item for item in items if not item[3]]
    chosen += [item for item in items if item[3]][:per_partition]
    # Hoist the chosen chains (fed only by nodes before the anchor) to right
    # after it, keeping their order.
    hoisted = sorted(
        (
            node
            for _, chain, _, _ in chosen
            for node in chain
            if order[node] > order[anchor]
        ),
        key=order.__getitem__,
    )
    for node in reversed(hoisted):
      anchor.append(node)
    last = hoisted[-1] if hoisted else anchor
    values = [anchor, *(out for _, _, outputs, _ in chosen for out in outputs)]
    anchor_users = list(anchor.users)
    with gm.graph.inserting_after(last):
      barrier = gm.graph.call_function(_optimization_barrier, args=tuple(values))
    barrier.meta = {**anchor.meta, "val": tuple(v.meta["val"] for v in values)}
    with gm.graph.inserting_after(barrier):
      anchor_out = gm.graph.call_function(operator.getitem, args=(barrier, 0))
    anchor_out.meta = {**anchor.meta}
    partition_map[barrier] = partition
    partition_map[anchor_out] = partition
    for user in anchor_users:
      if user is not barrier:
        user.replace_input_with(anchor, anchor_out)
    joined += len(chosen)
  gm.graph.lint()
  logger.info(
      "Joined %d prefetched forward gathers with their executable's Splash "
      "output",
      joined,
  )


# Backward ops that each deferred expert-gradient reduce-scatter is joined with,
# in order: the routed experts' gather-reduce (dX, ~40% into the layer) and the
# Splash backward (~60%).
_DEFERRED_REDUCTION_ANCHOR_OPS = (
    "tt_moe::ragged_gather_reduce_tc",
    "pallas::mla_splash_attn_bwd",
)


def _join_deferred_reductions_with_anchors(
    gm: torch.fx.GraphModule,
    collective_chains: dict[torch.fx.Node, set[torch.fx.Node]],
    partition_map: dict[torch.fx.Node, int],
    deferred: dict[torch.fx.Node, int],
    max_per_partition: int,
) -> None:
  """Give each backward executable's deferred expert-gradient reductions
  deadlines inside it.

  A reduction deferred from the previous executable reads only that
  executable's outputs, yet the TPU scheduler issues it late and the
  executable ends waiting for it while the SparseCores idled earlier. Joining
  the first ``max_per_partition`` of them (largest first) with an early
  compute value of this executable (``_DEFERRED_REDUCTION_ANCHOR_OPS``, in
  order) in an optimization barrier makes the scheduler issue each one in time
  to finish before that value is read. Values are unchanged.
  """
  order = {node: index for index, node in enumerate(gm.graph.nodes)}
  anchors: dict[int, list[torch.fx.Node]] = defaultdict(list)
  for op_name in _DEFERRED_REDUCTION_ANCHOR_OPS:
    seen: set[int] = set()
    for node in gm.graph.nodes:
      if (
          node.op == "call_function"
          and isinstance(node.target, torch._ops.OpOverload)
          and node.target.name() == op_name
          and node in partition_map
          and partition_map[node] not in seen
      ):
        seen.add(partition_map[node])
        value = node
        if isinstance(node.meta.get("val"), (tuple, list)):
          value = next(
              (u for u in node.users if u.target is operator.getitem and u.args[1] == 0),
              None,
          )
        if value is not None:
          anchors[partition_map[node]].append(value)
  by_partition: dict[int, list[torch.fx.Node]] = defaultdict(list)
  for launch, partition in deferred.items():
    by_partition[partition].append(launch)
  joined = 0
  for partition, launches in by_partition.items():
    launches.sort(key=lambda launch: (-_reduced_bytes(launch), order[launch]))
    for launch, anchor in zip(launches[:max_per_partition], anchors.get(partition, [])):
      chain = collective_chains[launch]
      if any(order[node] > order[anchor] for node in chain):
        continue
      outputs = sorted(
          (node for node in chain if any(user not in chain for user in node.users)),
          key=order.__getitem__,
      )
      anchor_users = list(anchor.users)
      with gm.graph.inserting_after(anchor):
        barrier = gm.graph.call_function(
            _optimization_barrier, args=(anchor, *outputs)
        )
      barrier.meta = {
          **anchor.meta,
          "val": tuple(v.meta["val"] for v in (anchor, *outputs)),
      }
      with gm.graph.inserting_after(barrier):
        anchor_out = gm.graph.call_function(operator.getitem, args=(barrier, 0))
      anchor_out.meta = {**anchor.meta}
      partition_map[barrier] = partition
      partition_map[anchor_out] = partition
      for user in anchor_users:
        user.replace_input_with(anchor, anchor_out)
      joined += 1
  gm.graph.lint()
  logger.info(
      "Joined %d deferred expert-gradient reductions with backward deadlines",
      joined,
  )


def _collect_backward_collective_owners(
    collective_chains: dict[torch.fx.Node, set[torch.fx.Node]],
    n_layers: int,
    pipelined_kinds: tuple[str, ...] = (),
) -> dict[torch.fx.Node, tuple[str, int]]:
  """Assign each backward FSDP collective chain to the layer it serves.

  A parameter gather is owned by the layer of its first consumer, a gradient
  reduction by the layer that produced its input (``classify_fsdp_roles``).
  Collectives with neither role stay with their inputs and are never
  pipelined: activation collectives such as expert-parallel token exchanges
  (Ring-of-Experts' all-gather / reduce-scatter and their transposes) depend on
  the layer's compute.
  """
  owners: dict[torch.fx.Node, tuple[str, int]] = {}
  for launch, role in classify_fsdp_roles(collective_chains).items():
    if role is None:
      if (is_gather(launch) and "ag" in pipelined_kinds) or (
          is_reduce_scatter(launch) and "rs" in pipelined_kinds
      ):
        logger.debug(
            "Backward collective %s is neither a parameter gather nor a "
            "gradient reduction; fsdp_pipeline leaves it in place",
            launch.name,
        )
      continue
    inputs, users = chain_boundary(collective_chains[launch])
    nodes = users if role == "ag" else inputs | {launch}
    layer_ids = {
        i for i in (_get_layer_id(n) for n in nodes) if 0 <= i < n_layers
    }
    if layer_ids:
      owners[launch] = (role, max(layer_ids))
  return owners


def _validate_backward_partitions(
    partition_map: dict[torch.fx.Node, int],
    collective_chains: dict[torch.fx.Node, set[torch.fx.Node]],
) -> None:
  for producer, chain in collective_chains.items():
    partitions = {partition_map[node] for node in chain}
    if len(partitions) != 1:
      details = "; ".join(
          f"{node.name}@{partition_map[node]} <- ["
          + ", ".join(
              f"{i.name}@{partition_map.get(i)}" for i in node.all_input_nodes
          )
          + "] -> ["
          + ", ".join(f"{u.name}@{partition_map.get(u)}" for u in node.users)
          + "]"
          for node in chain
      )
      raise ValueError(
          f"Collective {producer.name!r} spans backward partitions "
          f"{sorted(partitions)}: {details}"
      )
  for node, partition in partition_map.items():
    for input_node in node.all_input_nodes:
      input_partition = partition_map.get(input_node)
      if input_partition is not None and input_partition > partition:
        raise ValueError(
            "Backward split has a reverse dependency: "
            f"{input_node.name} ({input_partition}) -> "
            f"{node.name} ({partition})"
        )


def split_backward_graph_by_layer(
    gm: torch.fx.GraphModule,
    *,
    n_layers: int,
    layers_per_executable: int = 1,
    pipeline_collectives: bool = False,
    pipeline_kinds: tuple[str, ...] = ("ag", "rs"),
    pipeline_min_gather_bytes: int = 0,
    defer_reduction_max_bytes: int = 0,
    cpu_offload_prefetch_executables: int = 0,
    deferred_reduction_deadlines: int = 0,
) -> PartitionedGraph:
  """Split backward by layer, optionally pipelining complete FSDP AG/RS pairs.

  The default keeps attributable FSDP collective chains with their layer and
  also supports backward graphs with no FSDP collectives; other collectives
  stay with their inputs. Pipelining moves the collectives of
  ``pipeline_kinds`` and requires their ownership for every layer so it can
  safely form fill, overlap, and drain stages; the rest stay with their layer.
  Gradient reductions whose unreduced input is smaller than
  ``defer_reduction_max_bytes`` move to the last backward executable, so tiny
  per-layer reductions (lowered as synchronous all-reduces) don't make every
  layer wait for the slowest rank.
  CPU-offload reload pairs may be placed a configured number of complete child
  executables before their first consumer; zero keeps the pair in the consumer
  child.
  """
  if n_layers < 1:
    raise ValueError(f"n_layers must be positive, got {n_layers}")
  if layers_per_executable < 1:
    raise ValueError(
        "layers_per_executable must be at least 1, got "
        f"{layers_per_executable}"
    )

  if pipeline_collectives and not pipeline_kinds:
    raise ValueError(
        "pipeline_collectives needs at least one of pipeline_kinds 'ag', 'rs'"
    )

  body_nodes = _body_nodes(gm)
  ao_nodes = {node for node in body_nodes if is_ao_transfer(node)}
  collective_chains = collect_collective_chains(gm)
  collective_nodes: set[torch.fx.Node] = set().union(
      *collective_chains.values()
  )
  # A recomputed view keeps its producer layer's FQN but is consumed by the
  # layer being differentiated; place views with their first consumer instead.
  view_nodes = {node for node in body_nodes if _is_view(node)}
  _, layer_nodes = _group_nodes_by_layer(
      body_nodes, excluded=collective_nodes | ao_nodes | view_nodes
  )
  expected_layers = set(range(n_layers))
  if set(layer_nodes) != expected_layers:
    missing = sorted(expected_layers - layer_nodes.keys())
    unexpected = sorted(layer_nodes.keys() - expected_layers)
    raise ValueError(
        "Backward split requires compute metadata for every layer; "
        f"missing layers {missing}, unexpected layers {unexpected}"
    )

  layer_ids = sorted(layer_nodes, reverse=True)
  layer_indices = {
      layer_id: index for index, layer_id in enumerate(layer_ids)
  }
  compute_offset = 2 if pipeline_collectives else 1
  raw_layer_to_partition = {
      layer_id: compute_offset + index // layers_per_executable
      for layer_id, index in layer_indices.items()
  }
  owners = _collect_backward_collective_owners(
      collective_chains,
      n_layers,
      pipelined_kinds=pipeline_kinds if pipeline_collectives else (),
  )

  if pipeline_collectives:
    owned_collectives = set(owners.values())
    missing_comms = {}
    for layer_id in layer_ids:
      missing = tuple(
          kind
          for kind in pipeline_kinds
          if (kind, layer_id) not in owned_collectives
      )
      if missing:
        missing_comms[layer_id] = missing
    if missing_comms:
      raise ValueError(
          "FSDP backward pipeline requires AG and RS ownership for every "
          f"layer; missing {missing_comms}"
      )

  def collective_partition(kind: str, layer_id: int) -> int:
    if not pipeline_collectives or kind not in pipeline_kinds:
      return raw_layer_to_partition[layer_id]
    index = layer_indices[layer_id]
    if kind == "ag":
      return 1 if index == 0 else 2 + (index - 1) // layers_per_executable
    if index == len(layer_ids) - 1:
      return 3 + (len(layer_ids) - 1) // layers_per_executable
    return 2 + (index + 1) // layers_per_executable

  partition_map: dict[torch.fx.Node, int] = {}
  raw_collective_partitions: dict[str, dict[int, int]] = {
      "ag": {},
      "rs": {},
  }
  last_partition = max(collective_partition("rs", l) for l in layer_ids)
  deferred_expert_reductions: dict[torch.fx.Node, int] = {}
  for launch, (kind, layer_id) in owners.items():
    if kind == "ag" and _gathered_bytes(launch) < pipeline_min_gather_bytes:
      partition = raw_layer_to_partition[layer_id]
    elif kind == "rs" and _reduced_bytes(launch) < defer_reduction_max_bytes:
      partition = last_partition
    else:
      partition = collective_partition(kind, layer_id)
      if (
          kind == "rs"
          and partition != raw_layer_to_partition[layer_id]
          and _reduced_bytes(launch) >= (256 << 20)
      ):
        deferred_expert_reductions[launch] = partition
    _assign_partition(
        partition_map,
        collective_chains[launch],
        partition,
        f"FSDP {kind} layer {layer_id}",
    )
    raw_collective_partitions[kind][layer_id] = partition

  for layer_id, nodes in layer_nodes.items():
    _assign_partition(
        partition_map,
        nodes,
        raw_layer_to_partition[layer_id],
        f"layer {layer_id}",
    )
  for node in reversed(body_nodes):
    if node in view_nodes and node not in partition_map:
      user_partitions = [
          partition_map[user] for user in node.users if user in partition_map
      ]
      if user_partitions:
        partition_map[node] = min(user_partitions)
  # Collective-chain nodes with no partitioned input (e.g. an arange feeding a
  # gathered index buffer) carry no layer; place them with their consumers
  # below so the chain stays in one partition.
  sourceless_chain_nodes: dict[torch.fx.Node, None] = {}
  for node in body_nodes:
    if node not in partition_map and node not in ao_nodes:
      input_partitions = [
          partition_map[input_node]
          for input_node in node.all_input_nodes
          if input_node in partition_map
          and input_node not in sourceless_chain_nodes
      ]
      if not input_partitions and node in collective_nodes:
        sourceless_chain_nodes[node] = None
      partition_map[node] = max(input_partitions, default=0)
  for node in reversed(sourceless_chain_nodes):
    user_partitions = [
        partition_map[user] for user in node.users if user in partition_map
    ]
    if user_partitions:
      partition_map[node] = min(user_partitions)

  place_ao_transfer_pairs(
      gm,
      partition_map,
      backward_prefetch_executables=cpu_offload_prefetch_executables,
  )
  if deferred_reduction_deadlines and deferred_expert_reductions:
    _join_deferred_reductions_with_anchors(
        gm,
        collective_chains,
        partition_map,
        deferred_expert_reductions,
        deferred_reduction_deadlines,
    )
  _validate_backward_partitions(partition_map, collective_chains)
  chunked = _materialize_partitioned_graph(
      gm,
      partition_map,
      layer_nodes,
      raw_collective_partitions,
  )
  logger.info(
      "Split TPU backward into %d executables with at most %d layers each "
      "(pipeline_collectives=%s): compute=%s collectives=%s",
      len(set(chunked.partition_map.values())),
      layers_per_executable,
      pipeline_collectives,
      chunked.layer_to_partition,
      chunked.collective_layer_to_partition,
  )
  return chunked


def split_forward_backward_graph(
    gm: torch.fx.GraphModule,
    example_inputs: tuple[Any, ...],
) -> SplitForwardBackwardGraph:
  """Partition a make_fx joint graph using ``autograd_backward`` metadata."""
  nodes = list(gm.graph.nodes)
  placeholders = [node for node in nodes if node.op == "placeholder"]
  if len(placeholders) != len(example_inputs):
    raise ValueError(
        "Graph input count does not match example inputs: "
        f"{len(placeholders)} != {len(example_inputs)}"
    )

  output_node = next(node for node in nodes if node.op == "output")
  original_outputs = _as_tuple(output_node.args[0])
  backward_node_set: set[torch.fx.Node] = set()
  for node in nodes:
    if node.op == "output":
      continue
    if _is_backward_node(node) or any(
        input_node in backward_node_set for input_node in node.all_input_nodes
    ):
      backward_node_set.add(node)
  backward_nodes = [node for node in nodes if node in backward_node_set]
  if not backward_nodes:
    raise ValueError("Cannot split a graph with no autograd_backward nodes")

  residual_nodes: list[torch.fx.Node] = []
  residual_set: set[torch.fx.Node] = set()
  backward_recomputed_nodes: set[torch.fx.Node] = set()
  missing = object()

  def make_available_to_backward(node: torch.fx.Node) -> None:
    if node in backward_node_set or node.op in ("placeholder", "get_attr"):
      return
    value = node.meta.get("val", missing)
    if value is missing:
      raise ValueError(
          f"Graph split boundary node {node.name!r} has no example value"
      )
    if isinstance(value, torch.Tensor):
      if node not in residual_set:
        residual_set.add(node)
        residual_nodes.append(node)
      return
    if node in backward_recomputed_nodes:
      return
    backward_recomputed_nodes.add(node)
    for input_node in node.all_input_nodes:
      make_available_to_backward(input_node)

  for node in backward_nodes:
    for input_node in node.all_input_nodes:
      make_available_to_backward(input_node)

  forward_output_positions: list[int] = []
  backward_output_positions: list[int] = []
  constant_outputs: dict[int, Any] = {}
  forward_output_nodes: list[torch.fx.Node] = []
  backward_output_nodes: list[torch.fx.Node] = []
  for position, output in enumerate(original_outputs):
    if not isinstance(output, torch.fx.Node):
      constant_outputs[position] = output
    elif (
        "val" in output.meta
        and not isinstance(output.meta["val"], torch.Tensor)
    ):
      constant_outputs[position] = output.meta["val"]
    elif output in backward_node_set:
      backward_output_positions.append(position)
      backward_output_nodes.append(output)
    else:
      forward_output_positions.append(position)
      forward_output_nodes.append(output)

  # Reuse GraphPP's forward-mutation analysis. ``mutation_saved_values`` are
  # forward-created tensor bases whose mutated contents backward must receive
  # as residuals. ``forward_mutation_outputs`` are otherwise-dead mutation
  # results exposed only to keep state updates (for example, MoE token-count
  # buffer writes) live in the forward graph; the wrapper ignores their values.
  mutation_saved_values, forward_mutation_outputs = (
      _forward_mutations_to_materialize(
          gm,
          fwd_outputs=forward_output_nodes,
          bwd_outputs=backward_output_nodes,
          backward_only_names=set(),
      )
  )
  for node in mutation_saved_values:
    make_available_to_backward(node)
  forward_mutation_outputs = [
      node
      for node in forward_mutation_outputs
      if node not in forward_output_nodes and node not in residual_set
  ]
  forward_roots = [
      *forward_output_nodes,
      *residual_nodes,
      *forward_mutation_outputs,
  ]
  forward_needed: set[torch.fx.Node] = set(placeholders)

  def add_forward_dependencies(node: torch.fx.Node) -> None:
    if node in forward_needed:
      return
    if node in backward_node_set:
      raise ValueError(
          f"Forward output {node.name!r} depends on a backward graph node"
      )
    forward_needed.add(node)
    for input_node in node.all_input_nodes:
      add_forward_dependencies(input_node)

  for root in forward_roots:
    add_forward_dependencies(root)

  discarded_forward_mutations = [
      node.name
      for node in nodes
      if node not in backward_node_set
      and node not in forward_needed
      and is_mutation_node(node)
  ]
  if discarded_forward_mutations:
    raise ValueError(
        "Graph split would discard forward mutation nodes: "
        f"{discarded_forward_mutations}"
    )

  forward_fx = torch.fx.Graph()
  forward_env: dict[torch.fx.Node, torch.fx.Node] = {}
  for node in nodes:
    if node.op == "output" or node not in forward_needed:
      continue
    forward_env[node] = forward_fx.node_copy(node, forward_env.__getitem__)
  forward_fx.output(
      tuple(forward_env[node] for node in forward_roots)
  )
  forward_gm = torch.fx.GraphModule(gm, forward_fx)
  forward_gm.graph.lint()
  forward_gm.recompile()

  backward_fx = torch.fx.Graph()
  backward_env: dict[torch.fx.Node, torch.fx.Node] = {}
  for node in placeholders:
    backward_env[node] = backward_fx.node_copy(node, lambda _: None)
  backward_get_attrs = {
      input_node
      for copied_node in [*backward_recomputed_nodes, *backward_nodes]
      for input_node in copied_node.all_input_nodes
      if input_node.op == "get_attr"
  }
  for node in nodes:
    if node in backward_get_attrs:
      backward_env[node] = backward_fx.node_copy(node, backward_env.__getitem__)
  for residual in residual_nodes:
    placeholder = backward_fx.placeholder(f"residual_{residual.name}")
    placeholder.meta = dict(residual.meta)
    backward_env[residual] = placeholder
  for node in nodes:
    if node in backward_recomputed_nodes or node in backward_node_set:
      backward_env[node] = backward_fx.node_copy(node, backward_env.__getitem__)
  backward_fx.output(tuple(backward_env[node] for node in backward_output_nodes))
  backward_gm = torch.fx.GraphModule(gm, backward_fx)
  backward_gm.graph.lint()
  backward_gm.recompile()

  residual_examples = tuple(node.meta["val"] for node in residual_nodes)
  logger.info(
      "Split TPU graph into %d forward and %d backward nodes with %d tensor "
      "residuals and %d side-effect outputs",
      len(list(forward_gm.graph.nodes)),
      len(list(backward_gm.graph.nodes)),
      len(residual_nodes),
      len(forward_mutation_outputs),
  )
  module = SplitForwardBackwardModule(
      forward_gm,
      backward_gm,
      forward_output_positions=tuple(forward_output_positions),
      backward_output_positions=tuple(backward_output_positions),
      constant_outputs=constant_outputs,
      num_direct_forward_outputs=len(forward_output_nodes),
      num_residual_outputs=len(residual_nodes),
      num_outputs=len(original_outputs),
  )
  return SplitForwardBackwardGraph(
      module=module,
      forward_graph=forward_gm,
      backward_graph=backward_gm,
      forward_example_inputs=example_inputs,
      backward_example_inputs=(*example_inputs, *residual_examples),
      residual_nodes=tuple(residual_nodes),
  )


__all__ = [
    "GraphSplitter",
    "PartitionedGraph",
    "SplitForwardBackwardGraph",
    "SplitForwardBackwardModule",
    "split_backward_graph_by_layer",
    "split_forward_backward_graph",
    "split_forward_graph_by_layer",
]
