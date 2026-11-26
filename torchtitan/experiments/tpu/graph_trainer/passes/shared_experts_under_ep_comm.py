# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run each MoE layer's shared-expert work inside its exposed EP exchanges.

The routed path of an MoE layer waits on its expert-parallel token exchanges
(Ring-of-Experts: the dispatch all-gather and the combine reduce-scatter, and
in the backward their recomputed copies and the combine's transposed
all-gather). The TensorCore has nothing to do while one is in flight when the
next routed kernel needs its result. The shared expert depends on none of
them, but left to itself the scheduler runs it outside those windows.

``apply_shared_experts_under_ep_comm`` splits a layer copy's shared-expert
nodes, in graph order, into one consecutive segment per window (by matmul
FLOPs, an equal share per window) and pins segment ``k`` to window ``k`` with
two optimization barriers:

  start: ``barrier(collective input, *segment inputs)`` -- the segment reads
         its inputs through it, so it cannot start before the collective can.
  end:   ``barrier(collective result, *segment outputs)`` -- the collective's
         consumers and the segment's outside users read through it, so the
         segment must finish before the routed path resumes.

Barriers pass values through, so values are unchanged unless XLA would
otherwise have fused across a segment edge.

A "layer copy" is a layer's forward (one group) or its backward executable
(recompute + autograd backward, one group). Windows are the EP-annotated
floating-point token all-gathers / reduce-scatters of at least
``_MIN_WINDOW_NUMEL`` elements, except the backward's dispatch-transpose
reduce-scatters (they return token gradients at the end of the layer, after
the shared-expert backward is needed anyway). With EP token chunking only the
first window of each kind is used per phase (the first dispatch all-gather and
the last combine reduce-scatter of the forward / recompute, the first
transposed all-gather of the backward): the chunks hide the others under each
other's expert GEMMs.

A window is skipped (and logged) when pinning it would create a cycle: when a
segment node is upstream of the collective's input or downstream of its result
anywhere in the graph (other passes' barriers can join several layers).
"""

from __future__ import annotations

import heapq
import math
import operator
import time
from collections import defaultdict

import torch
from torch._inductor.fx_passes.bucketing import is_wait_tensor

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _get_module_fqn,
    _NOT_IN_LAYERS,
)
from torchtitan.experiments.tpu.graph_trainer.collectives import (
    collect_collective_chains,
    is_gather,
    is_reduce_scatter,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _is_recomputed,
    _optimization_barrier,
)
from torchtitan.tools.logging import logger

# Token exchanges are [tokens, D] activations (>= 16M elements at the K3 shape);
# the route / score exchanges are [tokens * top_k] scalars.
_MIN_WINDOW_NUMEL = 1 << 20

_MATMULS = frozenset(
    {"aten::mm", "aten::addmm", "aten::bmm", "aten::matmul", "aten::linear"}
)
_TAG = "shared_experts_under_ep_comm"


def _group(node: torch.fx.Node) -> tuple[int, bool]:
  """``(layer, backward executable)`` of a node."""
  return (
      _get_layer_id(node),
      _is_recomputed(node) or bool(node.meta.get("autograd_backward")),
  )


def _is_tensor(node: torch.fx.Node) -> bool:
  return isinstance(node.meta.get("val"), torch.Tensor)


def _matmul_flops(node: torch.fx.Node) -> int:
  target = node.target
  if not (
      isinstance(target, torch._ops.OpOverload) and target.name() in _MATMULS
  ):
    return 0
  out = node.meta.get("val")
  lhs = node.args[1] if target.name() == "aten::addmm" else node.args[0]
  if not (isinstance(out, torch.Tensor) and isinstance(lhs, torch.fx.Node)):
    return 0
  return 2 * out.numel() * lhs.meta["val"].shape[-1]


def _window_ends(launch: torch.fx.Node) -> tuple[torch.fx.Node, torch.fx.Node]:
  """``(input, result)`` of a collective: its wait if it has one, else itself."""
  waits = [user for user in launch.users if is_wait_tensor(user)]
  return launch.args[0], waits[0] if waits else launch


def _select_windows(launches: list[torch.fx.Node]) -> list[torch.fx.Node]:
  """The exposed windows of one layer copy, in graph order.

  Phases are the forward / recompute (dispatch all-gathers, then combine
  reduce-scatters) and the backward's transposed all-gathers. Per phase keep
  the first all-gather and the last reduce-scatter.
  """
  phases: dict[str, list[torch.fx.Node]] = defaultdict(list)
  for launch in launches:
    backward = bool(launch.meta.get("autograd_backward")) and not _is_recomputed(
        launch
    )
    phases["bwd" if backward else "fwd"].append(launch)
  chosen = []
  for nodes in phases.values():
    gathers = [n for n in nodes if is_gather(n)]
    scatters = [n for n in nodes if is_reduce_scatter(n)]
    if gathers:
      chosen.append(gathers[0])
    if scatters:
      chosen.append(scatters[-1])
  order = {n: i for i, n in enumerate(launches)}
  return sorted(chosen, key=order.__getitem__)


def _segments(
    shared: list[torch.fx.Node], num_windows: int
) -> list[list[torch.fx.Node]]:
  """Consecutive split by matmul FLOPs: each matmul goes to the window holding
  its FLOP midpoint; every other node joins its earliest shared user's window
  (weight transposes and elementwise producers stay with the matmul that reads
  them), or its preceding matmul's window if it has no shared user."""
  total = sum(_matmul_flops(n) for n in shared) or 1
  window: dict[torch.fx.Node, int] = {}
  done, current = 0, 0
  for node in shared:
    f = _matmul_flops(node)
    if f:
      mid = (done + f / 2) / total
      current = max(0, min(num_windows - 1, math.ceil(mid * num_windows) - 1))
      done += f
    window[node] = current
  for node in reversed(shared):
    if not _matmul_flops(node):
      users = [window[u] for u in node.users if u in window]
      if users:
        window[node] = min(users)
  segments: list[list[torch.fx.Node]] = [[] for _ in range(num_windows)]
  for node in shared:
    segments[window[node]].append(node)
  return segments


def _barrier(
    gm: torch.fx.GraphModule, values: list[torch.fx.Node], tag: str
) -> list[torch.fx.Node]:
  # The caller topologically re-sorts the graph after each window, so the
  # insertion point only has to follow one input.
  with gm.graph.inserting_after(values[-1]):
    barrier = gm.graph.call_function(_optimization_barrier, args=tuple(values))
  barrier.meta = {**values[0].meta, "val": tuple(v.meta["val"] for v in values)}
  barrier.meta[_TAG] = tag
  outputs = []
  insertion_point = barrier
  for index, value in enumerate(values):
    with gm.graph.inserting_after(insertion_point):
      output = gm.graph.call_function(operator.getitem, args=(barrier, index))
    output.meta = {**value.meta, _TAG: tag}
    outputs.append(output)
    insertion_point = output
  return outputs


def _reach(
    start: torch.fx.Node,
    *,
    upstream: bool,
    order: dict[torch.fx.Node, int],
    bound: int,
) -> set[torch.fx.Node]:
  """Nodes reachable from ``start`` through inputs (``upstream``) or users,
  over the whole graph, pruned at graph position ``bound`` (a path between a
  segment and the collective cannot leave the positions between them in a
  topologically sorted graph)."""
  seen, stack = set(), [start]
  while stack:
    node = stack.pop()
    for nxt in node.all_input_nodes if upstream else node.users:
      if nxt in seen:
        continue
      if (order[nxt] < bound) if upstream else (order[nxt] > bound):
        continue
      seen.add(nxt)
      stack.append(nxt)
  return seen


def _stable_topological_sort(gm: torch.fx.GraphModule) -> None:
  """Reorder nodes topologically, keeping the current order where allowed."""
  nodes = list(gm.graph.nodes)
  rank = {node: index for index, node in enumerate(nodes)}
  indegree = {node: len(set(node.all_input_nodes)) for node in nodes}
  ready = [(rank[n], n.name) for n in nodes if indegree[n] == 0]
  by_name = {n.name: n for n in nodes}
  heapq.heapify(ready)
  order = []
  while ready:
    _, name = heapq.heappop(ready)
    node = by_name[name]
    order.append(node)
    for user in node.users:
      indegree[user] -= 1
      if indegree[user] == 0:
        heapq.heappush(ready, (rank[user], user.name))
  if len(order) != len(nodes):
    stuck = {n for n in nodes if indegree[n] > 0}
    node, path, seen = next(iter(stuck)), [], {}
    while node not in seen:
      seen[node] = len(path)
      path.append(node)
      node = next(i for i in node.all_input_nodes if i in stuck)
    cycle = path[seen[node]:]
    raise AssertionError(
        f"{_TAG} created a cycle: "
        + " <- ".join(f"{n.name}[{n.meta.get(_TAG, '')}]" for n in cycle[:40])
    )
  previous = order[0]
  for node in order[1:]:
    previous.append(node)
    previous = node


def apply_shared_experts_under_ep_comm(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
  t0 = time.perf_counter()
  in_chain = set().union(*collect_collective_chains(gm).values())
  shared: dict[tuple, list[torch.fx.Node]] = defaultdict(list)
  launches: dict[tuple, list[torch.fx.Node]] = defaultdict(list)
  for node in gm.graph.nodes:
    if node.op != "call_function" or _get_layer_id(node) == _NOT_IN_LAYERS:
      continue
    group = _group(node)
    if (
        (is_gather(node) or is_reduce_scatter(node))
        and "EP" in (node.meta.get("custom") or {})
        and _is_tensor(node.args[0])
        and node.args[0].meta["val"].is_floating_point()
        and node.args[0].meta["val"].numel() >= _MIN_WINDOW_NUMEL
        and not (
            is_reduce_scatter(node)
            and node.meta.get("autograd_backward")
            and not _is_recomputed(node)
        )
    ):
      launches[group].append(node)
    elif (
        ".shared_experts." in _get_module_fqn(node) + "."
        and node not in in_chain
        # Barriers from other passes (remat layer entry, FSDP gathers) join
        # values of several modules; leave them and their outputs alone.
        and node.target is not _optimization_barrier
        and not (
            node.target is operator.getitem
            and isinstance(node.args[0], torch.fx.Node)
            and node.args[0].target is _optimization_barrier
        )
    ):
      shared[group].append(node)

  pinned, skipped = 0, 0
  for group, group_launches in sorted(launches.items()):
    nodes = shared.get(group)
    if not nodes:
      continue
    windows = _select_windows(group_launches)
    segments = _segments(nodes, len(windows))
    logger.info(
        "%s L%d%s: windows %s; %d shared nodes, segment sizes %s",
        _TAG,
        group[0],
        "b" if group[1] else "f",
        [f"{w.name}:{tuple(w.args[0].meta['val'].shape)}" for w in windows],
        len(nodes),
        [len(s) for s in segments],
    )
    for k, (launch, segment) in enumerate(zip(windows, segments)):
      if not segment:
        continue
      seg = set(segment)
      coll_in, coll_out = _window_ends(launch)
      tag = f"L{group[0]}{'b' if group[1] else 'f'}w{k}"
      order = {n: i for i, n in enumerate(gm.graph.nodes)}
      first = min(order[n] for n in segment)
      last = max(order[n] for n in segment)
      # The segment must not feed the collective's input (the start barrier
      # would close a cycle) nor read anything downstream of its result (the
      # end barrier would).
      upstream = _reach(coll_in, upstream=True, order=order, bound=first)
      downstream = _reach(coll_out, upstream=False, order=order, bound=last)
      bad = [n for n in segment if n in upstream or n in downstream]
      if bad:
        logger.warning(
            "%s: skipping window %s (%s): %d segment nodes are %s it, e.g. %s",
            _TAG,
            tag,
            launch.name,
            len(bad),
            "upstream of" if bad[0] in upstream else "downstream of",
            [n.name for n in bad[:4]],
        )
        skipped += 1
        continue
      ext_in = list(
          dict.fromkeys(
              i
              for n in segment
              for i in n.all_input_nodes
              # Activations only: gating one input of each matmul is enough,
              # and parameters / FSDP-gathered weights stay off the barrier.
              if i not in seg
              and _is_tensor(i)
              and i.op != "placeholder"
              and i not in in_chain
          )
      )
      gate = [coll_in, *(i for i in ext_in if i is not coll_in)]
      if len(gate) > 1:
        start = _barrier(gm, gate, f"{tag}_start")
        for value, gated in zip(gate, start):
          for user in list(value.users):
            if user in seg:
              user.replace_input_with(value, gated)
      outs = [
          n for n in segment if _is_tensor(n) and any(u not in seg for u in n.users)
      ]
      consumers = [u for u in coll_out.users if _group(u) == group]
      end = _barrier(gm, [coll_out, *outs], f"{tag}_end")
      end_nodes = {e.args[0] for e in end} | set(end)
      for user in consumers:
        user.replace_input_with(coll_out, end[0])
      for value, gated in zip(outs, end[1:]):
        for user in list(value.users):
          if user not in seg and user not in end_nodes:
            user.replace_input_with(value, gated)
      _stable_topological_sort(gm)
      pinned += 1

  if not pinned:
    raise ValueError(
        f"{_TAG} pinned no window: no layer has both shared-expert nodes and an"
        f" EP token all-gather / reduce-scatter ({skipped} skipped)"
    )
  gm.graph.lint()
  gm.recompile()
  logger.info(
      "%s: pinned %d windows, skipped %d (%.1f s, %d nodes)",
      _TAG,
      pinned,
      skipped,
      time.perf_counter() - t0,
      len(gm.graph.nodes),
  )
  return gm
