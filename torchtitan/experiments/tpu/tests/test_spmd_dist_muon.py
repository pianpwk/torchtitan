"""SpmdCompiledDistMuon (captured bucket runtime) vs the eager DistMuon on CPU.

The TPU pieces are stubbed: ``torch.compile`` runs ``aot_eager`` (counting
frames), ``torch.tpu.distributed.spmd_safe`` is the identity, and the
transfer pipeline uses no-op streams as on TPU. Everything else is the real
``SpmdCompiledDistMuon.step``: the captured buckets and the 0-d LR tensor.
"""

import types
from unittest import mock

import torch
import torch._dynamo.testing
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.flex_shard import BucketConfig, ComputeLayout, DistMuon
from torchtitan.distributed.flex_shard import _optimizer_reshard_runtime
from torchtitan.experiments.tpu import spmd_dist_muon


class SpmdCompiledDistMuonTest(DTensorTestBase):
  @property
  def world_size(self) -> int:
    return 2

  @property
  def device_type(self) -> str:
    return "cpu"

  def _run_steps(self, optimizer_cls, mesh) -> list[torch.Tensor]:
    """Three steps with fresh gradients and a changing LR."""
    rank, n = self.rank, self.world_size
    # Kernel-native expert stacks of two MoE layers, stored dim-1 sharded and
    # updated per expert (dim-0 compute layout).
    shapes = {"w1_EDF": (2 * n, 16, 8), "w2_EFD": (2 * n, 8, 16)}
    torch.manual_seed(0)
    names, params = [], []
    for layer in (1, 2):
      for weight, shape in shapes.items():
        full = torch.randn(shape) * 0.02
        names.append(f"layers.{layer}.moe.routed_experts.inner_experts.{weight}")
        params.append(
            torch.nn.Parameter(
                DTensor.from_local(
                    full.chunk(n, dim=1)[rank].contiguous(),
                    mesh,
                    [Shard(1)],
                    run_check=False,
                    shape=full.shape,
                    stride=full.stride(),
                )
            )
        )
    layout = ComputeLayout(
        shardings_by_mesh_axis={"fsdp": Shard(0)},
        compute_dtype=torch.bfloat16,
    )
    optimizer = optimizer_cls(
        [{"params": params, "param_names": names}],
        compute_sharding_by_fqn={name: layout for name in names},
        bucket_configs=tuple(
            BucketConfig(name=name, patterns=(name,)) for name in names
        ),
        lr=8e-4,
        weight_decay=0.1,
        adjust_lr_fn="match_rms_adamw",
    )
    for step in range(3):
      torch.manual_seed(100 * step + rank)
      for p in params:
        p.grad = DTensor.from_local(
            torch.randn(p.to_local().shape) * 1e-3,
            p.device_mesh,
            p.placements,
            run_check=False,
            shape=p.shape,
            stride=p.stride(),
        )
      optimizer.param_groups[0]["lr"] = 8e-4 * (step + 1) / 3
      optimizer.step()
      for p in params:
        p.grad = None
    return [p.to_local().clone() for p in params]

  @with_comms
  def test_captured_matches_eager(self):
    mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))
    counter = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
    real_compile = torch.compile

    def compile_aot_eager(fn, **kwargs):
      del kwargs  # TPU backend and options.
      return real_compile(fn, backend=counter, fullgraph=True, dynamic=False)

    with torch.no_grad():
      eager = self._run_steps(DistMuon, mesh)
      with (
          mock.patch.object(torch, "compile", compile_aot_eager),
          mock.patch.object(
              torch,
              "tpu",
              types.SimpleNamespace(
                  distributed=types.SimpleNamespace(spmd_safe=lambda f: f)
              ),
              create=True,
          ),
          mock.patch.object(
              _optimizer_reshard_runtime,
              "_stream_device_module",
              lambda device: _optimizer_reshard_runtime._NoOpStreams,  # pylint: disable=protected-access
          ),
      ):
        captured = self._run_steps(spmd_dist_muon.SpmdCompiledDistMuon, mesh)
    # One program across the LR changes (the LR is a 0-d tensor input).
    self.assertEqual(counter.frame_count, 1)
    # Equal up to the FP32 rounding of the tensor-LR update (relative
    # Frobenius error <= 1.1e-8 here; 7e-9 at K3-like bucket shapes).
    for c, e in zip(captured, eager, strict=True):
      self.assertLess(float((c - e).norm() / e.norm()), 3e-8)


if __name__ == "__main__":
  run_tests()
