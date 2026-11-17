"""Accelerator tests for the ``fp8_full`` expert-weight quantization and its
FP8 all-gather (``gather_fp8_expert_weights``)."""

from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import gather_fp8_expert_weights
from torchtitan.experiments.tpu.kernels import fp8_full
from torchtitan.experiments.tpu.testing import base_device_test


def _static_dequantized(x: torch.Tensor) -> torch.Tensor:
  """``x`` through MaxText's static ``fixed,-224,224`` e4m3 quantization."""
  scale = fp8_full.STATIC_SCALE
  fmax = torch.finfo(torch.float8_e4m3fn).max
  x = x.detach().float().cpu()
  return (x / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn).float() * scale


class Fp8FullExpertWeightTest(base_device_test.BaseAcceleratorDeviceTest):

  def _weights(self):
    torch.manual_seed(0)
    # [E, K, N]; BF16-representable so the reference sees the device's values.
    a = (torch.randn(4, 256, 128) * 0.5).bfloat16().float()
    b = (torch.randn(4, 256, 64) * 0.5).bfloat16().float()
    return a, b

  def test_quantize_expert_weight_is_static(self):
    a, _ = self._weights()
    q = fp8_full.quantize_expert_weight(a.to(self.accelerator_device))
    self.assertEqual(q.dtype, torch.float8_e4m3fn)
    torch.testing.assert_close(
        q.cpu().float() * fp8_full.STATIC_SCALE, _static_dequantized(a)
    )

  def test_unsharded_gather_concatenates_and_splits_gradients(self):
    a, b = self._weights()
    module = torch.nn.Module()
    module.a = torch.nn.Parameter(a.to(self.accelerator_device))
    module.b = torch.nn.Parameter(b.to(self.accelerator_device))
    weights = gather_fp8_expert_weights(
        module, ("a", "b"), reduce_dtype=torch.bfloat16
    )
    self.assertEqual(tuple(weights.weight_q.shape), (4, 256, 192))
    torch.testing.assert_close(
        weights.weight_q.cpu().float() * fp8_full.STATIC_SCALE,
        _static_dequantized(torch.cat([a, b], dim=-1)),
    )
    # Without SimpleFSDP nothing is reduce-scattered: the joint gradient is
    # cast to reduce_dtype, split per weight and returned in each one's dtype.
    grad = torch.randn(4, 256, 192)
    grad_a, grad_b = weights.reduce_joint_grad(grad.to(self.accelerator_device))
    self.assertEqual(grad_a.dtype, torch.float32)
    torch.testing.assert_close(grad_a.cpu(), grad[..., :128].bfloat16().float())
    torch.testing.assert_close(grad_b.cpu(), grad[..., 128:].bfloat16().float())


if __name__ == "__main__":
  absltest.main()
