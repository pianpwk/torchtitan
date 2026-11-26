"""Tests the TPU attention residual against an fp64 reference of upstream math.

``attention_residual`` replaces upstream ``_apply_attention_residual``'s
concat + fp32 ``matmul`` with elementwise passes (see
``kimi_k3/attention_residual.py``). The reference is upstream's formula
written out in fp64 on CPU from the same bf16 inputs: RMS-normalized keys,
scores against ``norm.weight * proj.weight``, a softmax over the block values
and the probability-weighted sum. Checked: the output and the gradients of
the prefix, the block residual, the norm weight and the projection weight,
on TPU in bf16, for 0, 1 and 8 earlier block values (8 is K3's largest).
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.experiments.tpu.kimi_k3.attention_residual import (
    attention_residual,
)
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import RMSNorm

_EPS = 1e-6


def _reference64(prefix_TD, block_residual_TND, norm_w_D, proj_w_1D):
  values = torch.cat((block_residual_TND, prefix_TD.unsqueeze(1)), dim=1)
  keys = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + _EPS)
  scores = (keys * (norm_w_D * proj_w_1D.squeeze(0))).sum(-1)
  return (torch.softmax(scores, -1).unsqueeze(-1) * values).sum(1)


def _rel_l2(actual, expected):
  actual, expected = actual.double().cpu(), expected.double()
  return ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()


class AttentionResidualTest(base_device_test.BaseAcceleratorDeviceTest):

  @parameterized.parameters(0, 1, 8)
  def test_matches_fp64_reference(self, num_block_values):
    tokens, dim = 256, 512
    g = torch.Generator().manual_seed(num_block_values)
    prefix = (3 * torch.randn(tokens, dim, generator=g)).bfloat16()
    block = torch.randn(tokens, num_block_values, dim, generator=g).bfloat16()
    norm_w = (1 + 0.1 * torch.randn(dim, generator=g)).bfloat16()
    proj_w = (0.05 * torch.randn(1, dim, generator=g)).bfloat16()
    grad_out = torch.randn(tokens, dim, generator=g).bfloat16()

    dev = self.accelerator_device
    norm = RMSNorm.Config(normalized_shape=dim, eps=_EPS).build()
    proj = Linear.Config(in_features=dim, out_features=1, bias=False).build()
    with torch.no_grad():
      norm.weight.copy_(norm_w.float())
      proj.weight.copy_(proj_w.float())
    norm, proj = norm.to(dev, torch.bfloat16), proj.to(dev, torch.bfloat16)
    # detach(): on CPU, .to() returns the input itself, which the reference
    # below must keep as a separate leaf.
    prefix_d = prefix.to(dev).detach().requires_grad_()
    block_d = block.to(dev).detach().requires_grad_()
    out = attention_residual(prefix_d, tuple(block_d.unbind(1)), proj, norm)
    out.backward(grad_out.to(dev))
    self.assertEqual(out.dtype, torch.bfloat16)

    ref_in = [
        t.double().requires_grad_() for t in (prefix, block, norm_w, proj_w)
    ]
    ref = _reference64(*ref_in)
    ref.backward(grad_out.double())
    actual = {
        "out": out.detach(),
        "d_prefix": prefix_d.grad,
        "d_norm_weight": norm.weight.grad,
        "d_proj_weight": proj.weight.grad,
    }
    expected = {
        "out": ref.detach(),
        "d_prefix": ref_in[0].grad,
        "d_norm_weight": ref_in[2].grad,
        "d_proj_weight": ref_in[3].grad,
    }
    if num_block_values:
      actual["d_block_residual"] = block_d.grad
      expected["d_block_residual"] = ref_in[1].grad
    for key, exp in expected.items():
      err = _rel_l2(actual[key], exp)
      # bf16 inputs and gradients: ~4e-3 per element; the weight gradients
      # are fp32 sums over the tokens.
      self.assertLess(err, 1e-2, msg=f"{key}: rel L2 {err:.2e}")


if __name__ == "__main__":
  absltest.main()
