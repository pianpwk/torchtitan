# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-step module statistics returned from the TPU compiled step.

The TPU compiled step (StaticCompiler) lowers the training step functionally,
so an in-place write to a module buffer inside it is dropped. A module that
accumulates a statistic in its forward (e.g. a router histogram) subclasses
``StepOutputModule`` instead:

* Inside ``emitting_step_outputs()``, which ``TpuGraphTrainer`` enters around
  the step it traces, the forward keeps its fixed-shape value rather than
  writing the buffer, and the trainer returns ``step_output()`` as an extra
  output of the step.
* After each step, outside the graph, the trainer hands that value to
  ``consume_step_output``, which adds it to the buffer, so gradient
  accumulation sums every microbatch.
* Outside the context (eager), the forward writes its buffer directly.

TODO: upstream this statistic-as-output form, so modules need no TPU-specific
path.
"""

import contextlib
from collections.abc import Iterator

import torch

_emitting_step_outputs = False


@contextlib.contextmanager
def emitting_step_outputs() -> Iterator[None]:
    """While active, ``StepOutputModule`` forwards keep their step outputs."""
    global _emitting_step_outputs
    previous, _emitting_step_outputs = _emitting_step_outputs, True
    try:
        yield
    finally:
        _emitting_step_outputs = previous


def step_outputs_emitted() -> bool:
    """Whether a ``StepOutputModule`` forward must keep its step output."""
    return _emitting_step_outputs


class StepOutputModule:
    """Mixin for a module whose forward accumulates a per-step statistic."""

    def step_output(self) -> torch.Tensor:
        """The latest forward's statistic, as a compiled-step output."""
        raise NotImplementedError

    def consume_step_output(self, value: torch.Tensor) -> None:
        """Applies a compiled step's ``step_output`` outside the graph."""
        raise NotImplementedError
