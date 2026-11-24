"""K3's explicit FSDP placements (graph_trainer/kimi_k3.py) at K3 widths.

The TPU SimpleFSDP all-gather does not pad uneven shards, so a dense parameter
that SimpleFSDP shards unevenly gathers back with the wrong rows, silently.
``_K3_FSDP_PLACEMENTS`` plus SimpleFSDP's default (dim 0, replicated below the
degree) must leave none at the FSDP degrees K3 runs at.
"""

from absl.testing import absltest
import torch

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.graph_trainer.kimi_k3 import _K3_FSDP_PLACEMENTS  # pylint: disable=protected-access
from torchtitan.models import kimi_k3


class K3FsdpPlacementsTest(absltest.TestCase):

  def test_no_uneven_shard_at_k3_widths(self):
    cfg = kimi_k3.model_registry("Kimi-K3").model
    # One layer of each kind: KDA + dense FFN, KDA + MoE, MLA + MoE.
    cfg.layers = cfg.layers[:4]
    cfg.vision_encoder = None
    with torch.device("meta"):
      model = cfg.build()
    shard_dims = {}  # Parameter FQN -> table dim (None: replicated).
    for fqn, module in model.named_modules():
      for (owner, path), placement in _K3_FSDP_PLACEMENTS.items():
        if isinstance(module, owner):
          shard_dims[f"{fqn}.{path}"] = (
              None if placement.is_replicate() else placement.dim
          )
    params = dict(model.named_parameters())
    self.assertContainsSubset(shard_dims, params)  # Every path resolves.
    for degree in (64, 128, 256, 512, 1024):
      for name, p in params.items():
        if "routed_experts" in name:  # SimpleFSDP's expert path.
          continue
        dim = shard_dims.get(name, 0)
        self.assertFalse(
            dim is not None and p.shape[dim] >= degree and p.shape[dim] % degree,
            f"FSDP {degree}: {name} {tuple(p.shape)} uneven on dim {dim}",
        )


if __name__ == "__main__":
  absltest.main()
