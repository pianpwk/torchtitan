# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schedule each layer's expert FSDP all-gathers to finish under attention.

The expert weight all-gathers run on a SparseCore, serially with the
routed-token gathers that follow routing. Left to itself, the TPU layer
scheduler overlaps at most one of them with attention and queues the rest
behind the dispatch gather, where nothing hides them (and the combine
reduce-scatter, queued behind them on the same SparseCore, starts late);
meanwhile the SparseCore idles through attention.

``apply_expert_all_gathers_under_attention`` joins the gathered expert weights
of a layer with that layer's attention kernel output in one optimization
barrier. Nothing that reads the attention output can start until every expert
all-gather is done, so the scheduler issues them during QKV and the attention
kernel. The attention output is a kernel output, materialized either way, so
the barrier changes no values. Anchors are the Splash forward (softmax
attention, e.g. MLA) and the Kimi K3 KDA forward kernel.

Each layer copy that runs the kernel gets its own barrier: the forward, and the
backward copy that recomputes it. A backward copy that reuses a saved output
has no kernel to anchor on and is left alone.

The pass applies to any model whose MoE weights live under a ``moe`` module
(TorchTitan's decoder convention): it matches TorchTPU's dim-d all-gather and
(anchor only) the functional all-gather. Layers without both are left alone; a
graph with none raises, since enabling the pass there is a configuration error.
"""

from __future__ import annotations

import operator
from collections import defaultdict

import torch
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_all_reduce_tensor,
    is_wait_tensor,
)

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _get_module_fqn,
    _NOT_IN_LAYERS,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _is_recomputed,
    _optimization_barrier,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_memory_policy import (
    _is_splash_forward_op,
)
from torchtitan.tools.logging import logger


def _is_gathered_expert_weight(node: torch.fx.Node) -> bool:
    """A gathered expert weight: TorchTPU's dim-d all-gather, or a funcol wait."""
    fqn = _get_module_fqn(node) + "."
    # Routed experts only: their weights feed grouped-matmul kernels. The
    # router and shared-expert weights are small and feed XLA dots, whose
    # fusion a barrier on their input would change.
    if (
        node.op != "call_function"
        or ".moe.routed_experts." not in fqn
        or ".token_dispatcher." in fqn
    ):
        return False
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor) and not val.is_floating_point():
        return False
    launch = node
    if is_wait_tensor(node):
        launch = node.args[0]
        if not (
            isinstance(launch, torch.fx.Node) and is_all_gather_into_tensor(launch)
        ):
            return False
    elif not (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target.name() == "tpu::_all_gather_dim"
    ):
        return False
    # Token traffic annotated "EP" (Ring-of-Experts' routed-token all-gather,
    # traced under the routed experts' FQN) is activation traffic that depends
    # on attention, not a parameter gather.
    return "EP" not in (launch.meta.get("custom") or {})


# Kimi K3's KDA forward kernels (``kimi_k3/tokamax_kda.py``,
# ``kernels/kda.py`` via ``kimi_k3/fused_kda.py``); the training forward
# returns ``[out, *residuals]``.
_KDA_FORWARD_OP_NAMES = frozenset(
    {"tt_kimi_k3::kda_fwd_res", "tt_tpu_kernels::kda_fwd_res"}
)


def _is_attention_kernel(target: object) -> bool:
    return _is_splash_forward_op(target) or (
        isinstance(target, torch._ops.OpOverload)
        and target.name() in _KDA_FORWARD_OP_NAMES
    )


def _is_attention_output(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is operator.getitem
        and node.args[1] == 0
        and isinstance(node.args[0], torch.fx.Node)
        and _is_attention_kernel(node.args[0].target)
    )


def _attention_prefix(node: torch.fx.Node) -> str:
    """FQN of the outermost attention module of ``node`` (``layers.N.*attention.``)."""
    parts = _get_module_fqn(node).split(".")
    end = min(i for i, part in enumerate(parts) if "attention" in part)
    return ".".join(parts[: end + 1]) + "."


def _layer_copy(node: torch.fx.Node) -> tuple[int, bool, bool]:
    """Which copy of which layer ``node`` belongs to."""
    return (
        _get_layer_id(node),
        _is_recomputed(node),
        bool(node.meta.get("autograd_backward")),
    )


def apply_expert_all_gathers_under_attention(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    gathered: dict[tuple, list[torch.fx.Node]] = defaultdict(list)
    anchors: dict[tuple, torch.fx.Node] = {}
    for node in gm.graph.nodes:
        if _get_layer_id(node) == _NOT_IN_LAYERS:
            continue
        if _is_gathered_expert_weight(node):
            gathered[_layer_copy(node)].append(node)
        elif _is_attention_output(node):
            anchors[_layer_copy(node)] = node
        elif (
            is_wait_tensor(node)
            and isinstance(node.args[0], torch.fx.Node)
            and is_all_reduce_tensor(node.args[0])
            and (kernel := anchors.get(_layer_copy(node))) is not None
            and _is_attention_output(kernel)
            and (_get_module_fqn(node) + ".").startswith(_attention_prefix(kernel))
        ):
            # The attention block's tensor-parallel all-reduce: a collective
            # result is materialized and never fused, so joining it (instead
            # of the kernel output) leaves the compiled computation unchanged
            # and also covers the output projection.
            anchors[_layer_copy(node)] = node

    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    barriers = 0
    for layer_copy, weights in gathered.items():
        anchor = anchors.get(layer_copy)
        if anchor is None:
            continue
        values = [anchor, *weights]
        users = {value: list(value.users) for value in values}
        with gm.graph.inserting_after(anchor):
            barrier = gm.graph.call_function(_optimization_barrier, args=tuple(values))
        barrier.meta = {**anchor.meta, "val": tuple(v.meta["val"] for v in values)}

        # The weights' gather chains depend only on parameters; move the parts
        # traced after the anchor ahead of the barrier.
        hoisted: set[torch.fx.Node] = set()

        def hoist(node: torch.fx.Node) -> None:
            if order.get(node, -1) <= order[anchor] or node in hoisted:
                return
            hoisted.add(node)
            for input_node in node.all_input_nodes:
                hoist(input_node)
            barrier.prepend(node)

        for weight in weights:
            hoist(weight)

        insertion_point = barrier
        for index, value in enumerate(values):
            with gm.graph.inserting_after(insertion_point):
                output = gm.graph.call_function(operator.getitem, args=(barrier, index))
            output.meta = {**value.meta}
            insertion_point = output
            for user in users[value]:
                if (
                    _layer_copy(user) == layer_copy
                    and order.get(user, len(order)) > order[anchor]
                ):
                    user.replace_input_with(value, output)
        barriers += 1

    if not barriers:
        raise ValueError(
            "expert_all_gathers_under_attention found no layer with both "
            "gathered MoE weights and an attention kernel output"
        )
    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Joined the expert all-gathers of %d layer copies with their attention "
        "output (%d on the attention all-reduce)",
        barriers,
        sum(is_wait_tensor(a) for a in anchors.values()),
    )
    return gm
