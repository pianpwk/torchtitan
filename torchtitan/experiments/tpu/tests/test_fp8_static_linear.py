"""Accelerator tests for `Fp8StaticLinear` (MaxText ``fp8_full`` dense GEMMs)."""

from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.components.fp8_static_linear import Fp8StaticLinear
from torchtitan.experiments.tpu.components.fp8_static_linear import Fp8StaticLinearConverter
from torchtitan.experiments.tpu.kernels.fp8_expert_segment_matmul import STATIC_SCALE
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.models.common.linear import Linear


def _static_dequantized(x: torch.Tensor) -> torch.Tensor:
  fmax = torch.finfo(torch.float8_e4m3fn).max
  x = x.detach().float().cpu()
  return (x / STATIC_SCALE).clamp(-fmax, fmax).to(
      torch.float8_e4m3fn
  ).float() * STATIC_SCALE


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
  got, want = got.detach().float().cpu(), want.detach().float().cpu()
  return ((got - want).norm() / want.norm()).item()


class Fp8StaticLinearTest(base_device_test.BaseAcceleratorDeviceTest):

  def _layer_and_inputs(self):
    torch.manual_seed(0)
    layer = Fp8StaticLinear.Config(in_features=512, out_features=384).build()
    with torch.no_grad():
      # BF16-representable, so the reference quantizes the device's values.
      layer.weight.copy_(torch.randn(384, 512).bfloat16() * 0.1)
    x = torch.randn(2, 128, 512).bfloat16()
    dout = (torch.randn(2, 128, 384) * 1e-3).bfloat16()
    return layer, x, dout

  def test_forward_matches_static_dequantized_reference(self):
    layer, x, _ = self._layer_and_inputs()
    want = _static_dequantized(x) @ _static_dequantized(layer.weight).T
    layer = layer.to(self.accelerator_device).bfloat16()
    out = layer(x.to(self.accelerator_device))
    self.assertEqual(out.dtype, torch.bfloat16)
    self.assertEqual(tuple(out.shape), (2, 128, 384))
    self.assertLess(_rel_err(out, want), 1e-2)

  def test_backward_is_close_to_fp32(self):
    layer, x, dout = self._layer_and_inputs()
    x_r = x.float().requires_grad_()
    w_r = layer.weight.detach().float().requires_grad_()
    want_dx, want_dw = torch.autograd.grad(x_r @ w_r.T, (x_r, w_r), dout.float())
    layer = layer.to(self.accelerator_device).bfloat16()
    x_d = x.to(self.accelerator_device).requires_grad_()
    dx, dw = torch.autograd.grad(
        layer(x_d), (x_d, layer.weight), dout.to(self.accelerator_device)
    )
    self.assertLess(_rel_err(dx, want_dx), 0.1)
    self.assertLess(_rel_err(dw, want_dw), 0.1)


class Fp8StaticLinearConverterTest(absltest.TestCase):

  def test_swaps_matching_linears_only(self):
    config = Linear.Config(in_features=128, out_features=64)
    converter = Fp8StaticLinearConverter.Config(fqns=("",)).build()
    self.assertIsInstance(converter.convert(config), Fp8StaticLinear.Config)
    with self.assertRaisesRegex(ValueError, "No Linear matches"):
      Fp8StaticLinearConverter.Config(
          fqns=("",), filter_fqns=("",)
      ).build().convert(config)


if __name__ == "__main__":
  absltest.main()
