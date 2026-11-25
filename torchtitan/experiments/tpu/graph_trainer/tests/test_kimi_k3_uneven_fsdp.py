"""K3-width SimpleFSDP placements (graph_trainer/kimi_k3.py) at K3's degrees.

The TPU SimpleFSDP all-gather does not pad uneven shards, so a parameter that
is sharded unevenly gathers back with the wrong rows. The K3-width flavors'
placements plus SimpleFSDP's default (and, under EP, the expert sharding) must
leave none at the FSDP and EP degrees the K3 recipes run at. The sharding runs
on meta tensors over a fake process group; ``_check_no_uneven_fsdp_shard`` is
the check ``parallelize_kimi_k3`` runs.
"""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.config import apply_overrides
from torchtitan.config.manager import ConfigManager
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
)
from torchtitan.experiments.tpu.graph_trainer import kimi_k3
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    configure_simple_fsdp_shard_dims,
)


class K3FsdpPlacementsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("_e448_ep8", "slice_e448", 8),
      ("_e896_ep16", "slice_e896", 16),
  )
  def test_no_uneven_shard_at_k3_widths(self, recipe: str, ep: int):
    for degree in (64, 128, 512, 1024):
      with self.subTest(fsdp=degree):
        dist.init_process_group(
            "fake", store=FakeStore(), rank=0, world_size=degree
        )
        try:
          self._shard_and_check(recipe, ep, degree)
        finally:
          dist.destroy_process_group()

  def _shard_and_check(self, recipe: str, ep: int, degree: int) -> None:
    cfg = ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.kimi_k3",
        f"--config=graph_trainer_kimi_k3_{recipe}",
        "--compile.no-compile_grad_norm",
        f"--parallelism.expert_parallel_degree={ep}",
    ])
    cfg.model_spec.model.update_from_config(
        config=cfg, global_batch_size=degree, local_global_batch_size=1
    )
    apply_overrides(cfg.override, cfg)
    # The flavor's placements, as model_registry binds them to parallelize.
    fsdp_placements = cfg.model_spec.parallelize_fn.keywords["fsdp_placements"]
    with (
        mock.patch("torchtitan.distributed.parallel_dims.device_type", "cpu"),
        # Ring-of-Experts checks for a TPU when it binds the EP mesh.
        mock.patch.object(
            TpuAllToAllTokenDispatcher,
            "ring_of_experts_unsupported_reason",
            return_value=None,
        ),
    ):
      parallel_dims = ParallelDims(
          dp_replicate=1,
          dp_shard=degree,
          cp=1,
          tp=1,
          pp=1,
          ep=ep,
          world_size=degree,
      )
      with dist_utils.get_spmd_context(parallel_dims=parallel_dims)():
        with torch.device("meta"):
          model = cfg.model_spec.model.build()
        # The sharding steps of `parallelize_kimi_k3`.
        model.parallelize(parallel_dims)
        configure_simple_fsdp_shard_dims(model, parallel_dims)
        kimi_k3._shard_routed_experts(model, parallel_dims, cfg.training)  # pylint: disable=protected-access
        kimi_k3._apply_k3_fsdp_placements(  # pylint: disable=protected-access
            model, parallel_dims, cfg.training, fsdp_placements
        )
        model = apply_simple_fsdp(
            model, parallel_dims=parallel_dims, training=cfg.training
        )
        kimi_k3._check_no_uneven_fsdp_shard(model)  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
