# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Guards the per-rank layout of the bucketed FSDP gradient reduce-scatter.

A wrong chunk order or offset would hand each rank another rank's (or another
parameter's) gradient shard without any error; the run would still train.
"""

from __future__ import annotations

import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.experiments.graph_trainer.common_utils import (
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.tpu.graph_trainer.collectives import (
    classify_fsdp_roles,
    collect_collective_chains,
)
from torchtitan.experiments.tpu.graph_trainer.passes.fsdp_reduce_scatter_buckets import (
    apply_fsdp_reduce_scatter_bucketing_pass,
    padded_shard_rows,
)

_RS = torch.ops._c10d_functional.reduce_scatter_tensor.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default
_GROUP = "fsdp"
_N = 4
_RANK = 2

# (parameter FQN, per-rank shard shape, group). Layer 0's three gradients
# share one bucket (wq as whole rows, the other two in the flat block); the
# output projection pads its 64 rows to 66 to fill whole rows of 1536; the
# norm is too small to pad; the expert RS is on another group and stays.
_GRADIENTS = (
    ("layers.0.attention.wq.weight", (24, 512), _GROUP),
    ("layers.0.attention_norm.weight", (5,), _GROUP),
    ("layers.0.feed_forward.w1.weight", (3, 7, 128), _GROUP),
    ("output.weight", (64, 512), _GROUP),
    ("norm.weight", (16,), _GROUP),
    ("layers.0.moe.experts.w1", (8, 64), "efsdp"),
)


def _same_input_reduce_scatter(tensor, reduce_op, group_size, group_name):
  """Rank ``_RANK``'s RS result when every rank holds the same ``tensor``."""
  del reduce_op, group_name
  return group_size * tensor.view(group_size, -1, *tensor.shape[1:])[_RANK]


class TestFsdpReduceScatterBuckets(unittest.TestCase):

  def _make_graph(self) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    outputs = []
    vals = {}
    with FakeTensorMode():
      for fqn, shard_shape, group in _GRADIENTS:
        inp = graph.placeholder(fqn.replace(".", "_"))
        full = (_N * shard_shape[0], *shard_shape[1:])
        inp.meta["val"] = torch.empty(full)
        # Compute between the gradient and its RS, as in a backward graph.
        grad = graph.call_function(torch.ops.aten.mul.Tensor, (inp, 1.0))
        grad.meta["val"] = torch.empty(full)
        rs = graph.call_function(_RS, (grad, "sum", _N, group))
        rs.meta["val"] = torch.empty(shard_shape)
        wait = graph.call_function(_WAIT, (rs,))
        wait.meta["val"] = rs.meta["val"]
        # A reader of the reduced shard placed before the bucket's last
        # gradient exists (as SimpleFSDP's views do); the pass must move it.
        grad_out = graph.call_function(
            torch.ops.aten.view.default, (wait, list(shard_shape))
        )
        grad_out.meta["val"] = rs.meta["val"]
        grad_out.meta["custom"] = {PARAMETER_GRADIENT_FQNS_META: (fqn,)}
        outputs.append(grad_out)
        vals[fqn] = full
    graph.output(tuple(outputs))
    self.full_shapes = vals
    return torch.fx.GraphModule({}, graph)

  def test_bucket_layout_matches_per_parameter_reduce_scatter(self):
    gm = self._make_graph()
    apply_fsdp_reduce_scatter_bucketing_pass(gm, (), group_name=_GROUP)

    rs_nodes = [n for n in gm.graph.nodes if n.target is _RS]
    shapes = sorted(
        (n.args[3], tuple(n.args[0].meta["val"].shape)) for n in rs_nodes
    )
    # Layer 0: 8 rows for wq plus 2 for the flat block, padded to 16; the
    # output projection: 66 x 512 = 22 rows, padded to 24.
    self.assertEqual(padded_shard_rows(10 * 1536), 16)
    self.assertEqual(padded_shard_rows(22 * 1536), 24)
    self.assertEqual(
        shapes,
        sorted([
            (_GROUP, (_N * 16, 1536)),
            (_GROUP, (_N * 24, 1536)),
            (_GROUP, (_N * 16,)),
            ("efsdp", (_N * 8, 64)),
        ]),
    )
    # The bucket's chain is still a gradient reduction for graph splitting.
    roles = classify_fsdp_roles(collect_collective_chains(gm))
    self.assertTrue(all(roles[n] == "rs" for n in rs_nodes))

    output = next(n for n in gm.graph.nodes if n.op == "output")
    for node, (fqn, _, _) in zip(output.args[0], _GRADIENTS):
      self.assertEqual(
          node.meta["custom"][PARAMETER_GRADIENT_FQNS_META], (fqn,)
      )

    for node in gm.graph.nodes:
      if node.target is _RS:
        node.target = _same_input_reduce_scatter
      elif node.target is _WAIT:
        node.target = torch.ops.aten.alias.default
    gm.recompile()
    inputs = [torch.randn(self.full_shapes[fqn]) for fqn, _, _ in _GRADIENTS]
    outputs = gm(*inputs)
    for got, full, (fqn, shard_shape, _) in zip(outputs, inputs, _GRADIENTS):
      rows = shard_shape[0]
      expected = _N * full[_RANK * rows : (_RANK + 1) * rows]
      torch.testing.assert_close(got, expected, rtol=0, atol=0, msg=fqn)
      self.assertEqual(got.storage_offset(), 0, fqn)


if __name__ == "__main__":
  unittest.main()
