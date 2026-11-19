# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bucket each layer's dense FSDP gradient reduce-scatters into one padded RS.

SimpleFSDP reduce-scatters every parameter gradient on its own, on dim 0 of
the (TP-local) gradient. The per-rank shards of a large model are then small
and oddly shaped (Kimi K3 at FSDP 512 x TP2: 12 x 7168, 14 x 16896,
7 x 7168, ...). Measured with compile-only AOT on a virtual tpu7x:8x8x8, the
TPU7x reduce-scatter takes the multi-axis torus path only for a 2-D per-rank
shard whose rows are a multiple of 8 and that splits into 6 equal pipelines
(two directions per axis); most other shapes run as an all-reduce plus slice
(twice the bytes), some as a synchronous TensorCore op, and a 1-D buffer
always runs as an all-reduce.

This pass keeps parameter storage and shard ownership unchanged and pads only
at the collective. For each bucket (all dense FSDP gradients of one decoder
layer, or a single gradient outside the layers) it builds the RS input so
that rank ``j``'s contiguous chunk holds rank ``j``'s shard of every gradient
in the bucket, as ``Rp`` rows of 1536::

  grad_i [N * a_i, c_i] -> view [N, a_i, c_i] -> pad a_i to a_i'
                        -> view [N, a_i' * c_i / 1536, 1536]
  small / odd grads     -> view [N, n_i] -> cat -> pad -> [N, r, 1536]
  cat on dim 1 -> [N, R, 1536] -> pad rows -> [N, Rp, 1536] -> [N * Rp, 1536]
  reduce-scatter dim 0 -> [Rp, 1536] -> slice rows -> grad shard_i

(``a_i'`` is the smallest row count that fills whole rows of 1536; a
gradient whose padding would exceed 1/32 of it joins the flat block.)

``Rp`` is a multiple of 8, and of the power of two nearest below ``Rp / 32``
from 512 rows on. In AOT at 4, 64 and 512 ranks (f32 and
bf16), every such ``[Rp, 1536]`` shard took the multi-axis path (1536 columns
are 12 tiles of 128 lanes, so any multiple of 8 rows splits 6 ways). Large
row counts with a large odd factor did not (5976 = 8 x 747 rows at 64 ranks
ran on the TensorCore; 39328 rows at 4 ranks ran as an all-reduce), while the
rounded 6016 and 39936 took the torus path. So the rule does not depend on the
group size or dtype. Padding costs under 3.2% of each row-padded gradient,
under 3.2% of the bucket, and fewer than 9 x 1536 elements per rank. The
bucket also replaces a layer's 10-25 RS launches with one.

Two kinds of bucket stay unbucketed (per-parameter RS, as without the pass):
- under half of 8 x 1536 elements per rank: padding would more than double
  their bytes, and they are tiny and latency-bound (a final norm weight at
  FSDP 512);
- over ``_MAX_SHARD_ROWS`` rows per rank: these occur only on small test
  slices at K3 widths (K3's largest bucket is 752 rows at FSDP 512, 6016 at
  FSDP 64). In a compiled 4-rank step, 2 of 5 such buckets (56k-96k rows) ran
  as full-size all-reduces (+6 GiB peak each), although each shape compiled
  alone gives a reduce-scatter.

TODO: on test slices whose dense group runs as a 1-D ring, the compiled
step also rewrote 3 of 5 buckets of 2.5k-6k rows (64 ranks) to all-reduces,
which shapes compiled alone do not predict; check the compiled HLO of the
target before relying on this pass.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from torchtitan.experiments.graph_trainer.common_utils import (
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.tools.logging import logger

_REDUCE_SCATTER = torch.ops._c10d_functional.reduce_scatter_tensor.default
_WAIT_TENSOR = torch.ops._c10d_functional.wait_tensor.default
_aten = torch.ops.aten

# Columns of the bucketed RS buffer and the smallest row multiple of each
# rank's shard. See the module docstring for why these values.
_BUCKET_COLUMNS = 1536
_MIN_ROWS_MULTIPLE = 8
_PADDING_UNIT_NUMEL = _BUCKET_COLUMNS * _MIN_ROWS_MULTIPLE
# Largest per-rank bucket (rows of 1536) to rewrite; see the module docstring.
_MAX_SHARD_ROWS = 16384


def padded_shard_rows(numel: int) -> int:
  """Rows of 1536 for a per-rank bucket of ``numel`` elements, after padding."""
  rows = math.ceil(numel / _BUCKET_COLUMNS)
  # The power of two nearest below rows / 32, and at least 8.
  multiple = max(_MIN_ROWS_MULTIPLE, 1 << max(rows.bit_length() - 6, 0))
  return math.ceil(rows / multiple) * multiple


def _gradient_fqns(wait: torch.fx.Node) -> tuple[str, ...] | None:
  """FQNs of the parameter gradient that ``wait`` feeds through unary ops."""
  node = wait
  while True:
    fqns = node.meta.get("custom", {}).get(PARAMETER_GRADIENT_FQNS_META)
    if fqns:
      return fqns
    if len(node.users) != 1:
      return None
    node = next(iter(node.users))


def _layer_key(fqns: tuple[str, ...]) -> str | None:
  """``layers.<N>`` for a decoder-layer parameter, else None."""
  parts = fqns[0].split(".")
  if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
    return f"layers.{parts[1]}"
  return None


def _meta_without_gradient_tag(node: torch.fx.Node) -> dict:
  meta = dict(node.meta)
  meta.pop("val", None)
  custom = dict(meta.get("custom", {}))
  custom.pop(PARAMETER_GRADIENT_FQNS_META, None)
  if custom:
    meta["custom"] = custom
  else:
    meta.pop("custom", None)
  return meta


class _Member:
  """One SimpleFSDP gradient reduce-scatter and its wait."""

  def __init__(self, rs: torch.fx.Node, wait: torch.fx.Node):
    self.rs = rs
    self.wait = wait
    self.input = rs.args[0]
    self.output_val = rs.meta["val"]
    self.numel = self.output_val.numel()


def apply_fsdp_reduce_scatter_bucketing_pass(
    gm: torch.fx.GraphModule,
    example_inputs,
    *,
    group_name: str,
) -> torch.fx.GraphModule:
  """Rewrite the dense FSDP gradient RS on ``group_name`` into buckets."""
  del example_inputs
  graph = gm.graph

  buckets: dict[tuple, list[_Member]] = defaultdict(list)
  for rs in graph.nodes:
    if rs.op != "call_function" or rs.target is not _REDUCE_SCATTER:
      continue
    _, reduce_op, group_size, rs_group = rs.args[:4]
    if rs_group != group_name:
      continue
    waits = [u for u in rs.users if u.target is _WAIT_TENSOR]
    if reduce_op != "sum" or len(rs.users) != 1 or len(waits) != 1:
      continue
    fqns = _gradient_fqns(waits[0])
    if fqns is None:
      continue
    member = _Member(rs, waits[0])
    layer = _layer_key(fqns)
    key = (
        layer if layer is not None else fqns,
        group_size,
        member.output_val.dtype,
    )
    buckets[key].append(member)

  num_rewritten = 0
  num_skipped = 0
  num_reduce_scatters = 0
  for (name, group_size, dtype), members in buckets.items():
    label = name if isinstance(name, str) else name[0]
    total = sum(m.numel for m in members)
    rows = padded_shard_rows(_bucket_layout(members)[2] * _BUCKET_COLUMNS)
    if total < _PADDING_UNIT_NUMEL // 2 or rows > _MAX_SHARD_ROWS:
      logger.debug(
          "[fsdp_rs_buckets] %s: %d gradient(s), %d elements per rank: left"
          " unbucketed",
          label,
          len(members),
          total,
      )
      num_skipped += len(members)
      continue
    logger.debug(
        "[fsdp_rs_buckets] %s: %d gradient(s), %d elements per rank ->"
        " [%d, %d]",
        label,
        len(members),
        total,
        rows,
        _BUCKET_COLUMNS,
    )
    # Earlier buckets moved nodes, so recompute the order.
    order = {node: i for i, node in enumerate(graph.nodes)}
    members.sort(key=lambda m: order[m.rs])
    last_input = max((m.input for m in members), key=order.__getitem__)
    # Nodes that read a reduced shard (views, casts) before the bucket's last
    # gradient is computed move after the bucket, with their dependents.
    early: set[torch.fx.Node] = set()
    pending = [u for m in members for u in m.wait.users]
    while pending:
      node = pending.pop()
      if node not in early and order[node] <= order[last_input]:
        early.add(node)
        pending.extend(node.users)
    if any(m.input in early for m in members):
      raise ValueError(
          f"FSDP RS bucketing: a gradient of bucket {name} depends on another"
          " gradient's reduced shard"
      )
    tail = _rewrite_bucket(
        graph, members, last_input, group_size, group_name, dtype
    )
    for node in sorted(early, key=order.__getitem__):
      tail.append(node)
      tail = node
    num_rewritten += len(members)
    num_reduce_scatters += 1

  graph.lint()
  gm.recompile()
  logger.info(
      "[fsdp_rs_buckets] %d gradient reduce-scatters -> %d padded buckets;"
      " %d left unbucketed",
      num_rewritten,
      num_reduce_scatters,
      num_skipped,
  )
  return gm


def _padded_leading_rows(m: _Member) -> int | None:
  """Rows ``a'`` to pad the shard's dim 0 to so it fills whole rows of 1536.

  Returns None when that costs more than 1/32 of the shard (small or oddly
  wide gradients); those share one flat padded block instead. Large shards
  must not take the flat path: in a compiled 64-rank step, flattening the
  1280 x 7168 embedding shard to [N, n] made XLA copy it through a loop over
  the N ranks (~50 ms per copy).
  """
  shape = m.output_val.shape
  if not shape or m.numel == 0:
    return None
  rows, cols = shape[0], m.numel // shape[0]
  step = _BUCKET_COLUMNS // math.gcd(cols, _BUCKET_COLUMNS)
  padded = math.ceil(rows / step) * step
  if (padded - rows) * cols * 32 > m.numel:
    return None
  return padded


def _bucket_layout(
    members: list[_Member],
) -> tuple[list[tuple[_Member, int]], list[_Member], int]:
  """Gradients laid out as whole rows (with their padded dim 0), the rest
  (packed into one flat block), and the bucket's total rows of 1536."""
  blocks, rest = [], []
  for m in members:
    padded = _padded_leading_rows(m)
    if padded is None:
      rest.append(m)
    else:
      blocks.append((m, padded))
  rows = sum(
      padded * (m.numel // m.output_val.shape[0]) // _BUCKET_COLUMNS
      for m, padded in blocks
  ) + math.ceil(sum(m.numel for m in rest) / _BUCKET_COLUMNS)
  return blocks, rest, rows


def _rewrite_bucket(
    graph: torch.fx.Graph,
    members: list[_Member],
    last_input: torch.fx.Node,
    group_size: int,
    group_name: str,
    dtype: torch.dtype,
) -> torch.fx.Node:
  """Insert the bucket after ``last_input``; returns its last new node."""
  ref = members[0].output_val
  blocks, rest, rows = _bucket_layout(members)
  padded_rows = padded_shard_rows(rows * _BUCKET_COLUMNS)
  rs_meta = _meta_without_gradient_tag(members[0].rs)
  wait_meta = _meta_without_gradient_tag(members[0].wait)
  n, width = group_size, _BUCKET_COLUMNS

  def add(target, args, shape, meta):
    node = graph.call_function(target, args)
    node.meta.update(meta)
    node.meta["val"] = ref.new_empty(shape, dtype=dtype)
    return node

  def view(source, shape, meta):
    return add(_aten.view.default, (source, list(shape)), shape, meta)

  def contiguous(m: _Member) -> torch.fx.Node:
    val = m.input.meta["val"]
    if val.is_contiguous():
      return m.input
    return add(_aten.clone.default, (m.input,), val.shape, rs_meta)

  # Prepend to a fixed node so the new nodes keep their creation order.
  with graph.inserting_before(last_input.next):
    # Rank j's chunk of a gradient [N * a, ...] is its row block j, so
    # [N, a, cols] holds rank j's shard in [j]; pad a to a' (whole rows of
    # 1536) and reshape to [N, a' * cols / 1536, 1536]. Flattening each shard
    # to [N, n] first and reshaping the concatenation made XLA copy the whole
    # bucket through a loop over the N ranks (tens of ms per layer at 64
    # ranks).
    parts = []
    for m, padded in blocks:
      a = m.output_val.shape[0]
      cols = m.numel // a
      part = view(contiguous(m), (n, a, cols), rs_meta)
      if padded != a:
        part = add(
            _aten.constant_pad_nd.default,
            (part, [0, 0, 0, padded - a], 0.0),
            (n, padded, cols),
            rs_meta,
        )
      block_rows = padded * cols // width
      parts.append(view(part, (n, block_rows, width), rs_meta))
    rest_rows = 0
    if rest:
      # The few gradients that are not whole rows (norms, small odd shapes)
      # share one padded block.
      flat = [view(contiguous(m), (n, m.numel), rs_meta) for m in rest]
      rest_numel = sum(m.numel for m in rest)
      block = (
          add(_aten.cat.default, (flat, 1), (n, rest_numel), rs_meta)
          if len(flat) > 1
          else flat[0]
      )
      rest_rows = math.ceil(rest_numel / width)
      if rest_rows * width != rest_numel:
        block = add(
            _aten.constant_pad_nd.default,
            (block, [0, rest_rows * width - rest_numel], 0.0),
            (n, rest_rows * width),
            rs_meta,
        )
      parts.append(view(block, (n, rest_rows, width), rs_meta))
    bucket = (
        add(_aten.cat.default, (parts, 1), (n, rows, width), rs_meta)
        if len(parts) > 1
        else parts[0]
    )
    if padded_rows != rows:
      bucket = add(
          _aten.constant_pad_nd.default,
          (bucket, [0, 0, 0, padded_rows - rows], 0.0),
          (n, padded_rows, width),
          rs_meta,
      )
    bucket = view(bucket, (n * padded_rows, width), rs_meta)
    rs = add(
        _REDUCE_SCATTER,
        (bucket, "sum", n, group_name),
        (padded_rows, width),
        rs_meta,
    )
    wait = add(_WAIT_TENSOR, (rs,), (padded_rows, width), wait_meta)

    def emit_shard(m: _Member, piece: torch.fx.Node) -> torch.fx.Node:
      shard_meta = _meta_without_gradient_tag(m.wait)
      # A returned gradient must not be a view at a nonzero offset into the
      # bucket (TorchTPU keeps a returned view's layout); clone it.
      shape = piece.meta["val"].shape
      piece = add(_aten.clone.default, (piece,), shape, shard_meta)
      shard = view(piece, m.output_val.shape, shard_meta)
      # The wait may carry the gradient tag; it moves to the new shard.
      shard.meta.update(m.wait.meta)
      shard.meta["val"] = m.output_val
      m.wait.replace_all_uses_with(shard)
      return shard

    row = 0
    for m, padded in blocks:
      a = m.output_val.shape[0]
      cols = m.numel // a
      block_rows = padded * cols // width
      piece = add(
          _aten.slice.Tensor,
          (wait, 0, row, row + block_rows),
          (block_rows, width),
          wait_meta,
      )
      piece = view(piece, (padded, cols), wait_meta)
      if padded != a:
        piece = add(
            _aten.slice.Tensor, (piece, 0, 0, a), (a, cols), wait_meta
        )
      shard = emit_shard(m, piece)
      row += block_rows
    if rest:
      block = add(
          _aten.slice.Tensor,
          (wait, 0, row, row + rest_rows),
          (rest_rows, width),
          wait_meta,
      )
      block = view(block, (rest_rows * width,), wait_meta)
      offset = 0
      for m in rest:
        piece = add(
            _aten.slice.Tensor,
            (block, 0, offset, offset + m.numel),
            (m.numel,),
            wait_meta,
        )
        shard = emit_shard(m, piece)
        offset += m.numel
  for m in members:
    graph.erase_node(m.wait)
    graph.erase_node(m.rs)
  return shard


__all__ = [
    "apply_fsdp_reduce_scatter_bucketing_pass",
    "padded_shard_rows",
]
