"""Tests the Qwen3.5 GatedDeltaNet on the fused Pallas KDA kernels."""

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import base_device_test
import torchtitan.experiments.tpu.qwen3_5 as qwen3_5_tpu
from torchtitan.experiments.tpu.qwen3_5 import fused_gated_delta
from torchtitan.experiments.tpu.qwen3_5 import gated_delta
from torchtitan.models.common.attention import segment_ids_from_positions


def _relative_l2_error(test: torch.Tensor, reference: torch.Tensor) -> float:
  test, reference = test.double().cpu(), reference.double().cpu()
  return float((test - reference).norm() / reference.norm().clamp_min(1e-30))


def _segment_ids(doc_lens: tuple[tuple[int, ...], ...]) -> torch.Tensor:
  positions = torch.stack(
      [torch.cat([torch.arange(n) for n in lens]) for lens in doc_lens]
  )
  return segment_ids_from_positions(positions.reshape(-1)).reshape(
      positions.shape
  )


class FusedGatedDeltaTest(base_device_test.BaseAcceleratorDeviceTest):
  """``FusedGatedDeltaKernel`` vs the XLA ``chunked_gated_delta``."""

  def _require_tpu(self) -> torch.device:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("The fused KDA kernels need a TPU.")
    return self.accelerator_device

  def test_override_arguments_and_validation(self):
    kernel_cfg = fused_gated_delta.fused_gated_delta_kernel(
        gated_delta.TpuGatedDeltaKernel.Config(chunk_size=32),
        min_log_decay=-6.0,
    )
    self.assertIsInstance(
        kernel_cfg, fused_gated_delta.FusedGatedDeltaKernel.Config
    )
    self.assertEqual(
        (kernel_cfg.chunk_size, kernel_cfg.min_log_decay), (32, -6.0)
    )
    for bad in (
        dict(min_log_decay=-9.0),
        dict(min_log_decay=0.0),
        dict(chunk_size=8),
    ):
      with self.assertRaises(ValueError, msg=str(bad)):
        fused_gated_delta.FusedGatedDeltaKernel.Config(**bad)

  @parameterized.named_parameters(
      # Grouped key heads (Hk=2 shared by H=4), many documents per row incl.
      # length-1 documents and documents spanning chunks; decays in (-6, 0).
      dict(
          testcase_name="_gqa_documents",
          doc_lens=((100, 28, 1, 1, 200, 3, 179), (512,)),
          init_gates=False,
      ),
      # Qwen3.5's initialization decays (down to about -70): clamped at -8.
      dict(
          testcase_name="_init_gates",
          doc_lens=((512,), (512,)),
          init_gates=True,
      ),
  )
  def test_kernel_matches_fp64_xla(self, doc_lens, init_gates):
    """Kernel (bf16) vs the XLA recurrence in fp64 on the same inputs."""
    dev = self._require_tpu()
    torch.manual_seed(0)
    batch, seq_len = len(doc_lens), sum(doc_lens[0])
    key_heads, heads, dim = 2, 4, 128
    segment_ids = _segment_ids(doc_lens)
    q = torch.randn(batch, seq_len, key_heads, dim).bfloat16()
    k = torch.randn(batch, seq_len, key_heads, dim).bfloat16()
    v = torch.randn(batch, seq_len, heads, dim).bfloat16()
    if init_gates:
      a_log = torch.empty(heads).uniform_(0.01, 16.0).log()
      g = -a_log.exp() * F.softplus(torch.randn(batch, seq_len, heads) + 1.0)
    else:
      g = -F.softplus(2.0 * torch.randn(batch, seq_len, heads))
    beta = torch.sigmoid(torch.randn(batch, seq_len, heads)).bfloat16()
    grad_out = torch.randn(batch, seq_len, heads, dim).bfloat16()
    kernel = fused_gated_delta.FusedGatedDeltaKernel.Config().build()

    def run(fn, device, dtype):
      # The model passes fp32 decays with bf16 q / k / v / beta.
      g_dtype = torch.float64 if dtype == torch.float64 else torch.float32
      inputs = [
          x.to(device, d).requires_grad_()
          for x, d in zip(
              (q, k, v, g, beta), (dtype, dtype, dtype, g_dtype, dtype)
          )
      ]
      out = fn(*inputs, segment_ids.to(device))
      out.backward(grad_out.to(device, out.dtype))
      return [out] + [x.grad for x in inputs]

    def reference(q_in, k_in, v_in, g_in, beta_in, seg):
      repeats = heads // key_heads
      return gated_delta.chunked_gated_delta(
          q_in.repeat_interleave(repeats, dim=2),
          k_in.repeat_interleave(repeats, dim=2),
          v_in,
          g_in.clamp_min(fused_gated_delta.DEFAULT_MIN_LOG_DECAY),
          beta_in,
          segment_ids=seg,
      )

    def fused(q_in, k_in, v_in, g_in, beta_in, seg):
      # The kernels, not the parent's XLA path, must run these inputs.
      with mock.patch.object(
          gated_delta.TpuGatedDeltaKernel,
          "forward",
          side_effect=AssertionError("fell back to the XLA recurrence"),
      ):
        return kernel(q_in, k_in, v_in, g_in, beta_in, segment_ids_BS=seg)

    expected = run(reference, "cpu", torch.float64)
    actual = run(fused, dev, torch.bfloat16)
    self.assertEqual(actual[0].dtype, torch.bfloat16)
    clamped = g < fused_gated_delta.DEFAULT_MIN_LOG_DECAY
    self.assertTrue(bool((actual[4].cpu()[clamped] == 0).all()))
    for name, a_t, e_t in zip(
        ("out", "dq", "dk", "dv", "dg", "dbeta"), actual, expected, strict=True
    ):
      self.assertTrue(torch.isfinite(a_t).all().item(), msg=name)
      # TPU7x: 1.7e-3 (out) to 4e-3 (dg).
      self.assertLess(_relative_l2_error(a_t, e_t), 1e-2, msg=name)
    if not init_gates:
      # Control: without documents the state leaks into row 0's second one.
      leaky = run(lambda *x: reference(*x[:5], None), "cpu", torch.float64)
      self.assertGreater(
          _relative_l2_error(leaky[0][0, 100:104], expected[0][0, 100:104]),
          5e-2,
      )

  def test_gated_delta_net_module_matches_xla(self):
    """Full 35B TpuGatedDeltaNet (conv, GQA, gates, norm), fused vs XLA."""
    dev = self._require_tpu()
    torch.manual_seed(0)
    base_cfg = qwen3_5_tpu.model_registry("35B-A3B", seq_len=256).model
    xla_cfg = copy.deepcopy(base_cfg.layers[0].delta_net)
    fused_cfg = copy.deepcopy(xla_cfg)
    fused_cfg.inner_gated_delta_net.kernel = (
        fused_gated_delta.FusedGatedDeltaKernel.Config()
    )
    xla_gdn = xla_cfg.build()
    xla_gdn.init_states(buffer_device=torch.device("cpu"))
    # Decays above the -8 clamp, so both paths compute the same function.
    with torch.no_grad():
      xla_gdn.A_log.uniform_(-3.0, -0.7)
      xla_gdn.dt_bias.uniform_(-0.5, 0.5)
    fused_gdn = fused_cfg.build()
    fused_gdn.load_state_dict(copy.deepcopy(xla_gdn.state_dict()))
    gdns = []
    for mod in (xla_gdn, fused_gdn):
      mod = mod.to(device=dev, dtype=torch.bfloat16)
      mod.A_log.data = mod.A_log.data.float()
      mod.dt_bias.data = mod.dt_bias.data.float()
      gdns.append(mod)

    batch, seq_len = 2, 256
    doc_lens = ((96, 1, 63, 96), (160, 3, 93))
    positions = torch.cat(
        [torch.cat([torch.arange(n) for n in lens]) for lens in doc_lens]
    ).to(dev)
    x_base = torch.randn(
        batch * seq_len,
        xla_cfg.in_proj_q.in_features,
        dtype=torch.bfloat16,
        device=dev,
    )
    grad_out = torch.randn_like(x_base)

    def run_module(mod):
      x_in = x_base.detach().clone().requires_grad_()
      with set_unflatten_seq_len(seq_len):
        out = mod(x_in, positions=positions)
      out.backward(grad_out)
      return out, x_in.grad, {n: p.grad for n, p in mod.named_parameters()}

    expected_out, expected_dx, expected_grads = run_module(gdns[0])
    actual_out, actual_dx, actual_grads = run_module(gdns[1])
    self.assertLess(_relative_l2_error(actual_out, expected_out), 1.5e-2)
    self.assertLess(_relative_l2_error(actual_dx, expected_dx), 1.5e-2)
    for name, e_grad in expected_grads.items():
      a_grad = actual_grads[name]
      self.assertTrue(torch.isfinite(a_grad).all().item(), msg=name)
      bound = 5e-2 if name in ("A_log", "dt_bias", "in_proj_a.weight") else 2e-2
      self.assertLess(
          _relative_l2_error(a_grad, e_grad), bound, msg=f"grad {name}"
      )


if __name__ == "__main__":
  absltest.main()
