"""Accelerator tests for `TpuMaxTextMLAAttention` (Lineage fused FP8 q/kv up
kernels + Tokamax Splash) against upstream DeepSeek MLA in FP32 on CPU.

DeepSeek-V3 671B dims except 16 heads and one 1024-token sequence. The FP8
numerics (``fp8_full``: static e4m3 scale for the up projections' inputs and
weights, e5m2 gradients) bound the agreement; a wrong rope permutation, k / v
split or gradient routing gives errors near 1.
"""

import copy

from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.deepseek_v3.maxtext_mla_attention import TpuMaxTextMLAAttention
from torchtitan.experiments.tpu.deepseek_v3.maxtext_mla_attention import TpuMaxTextMLAAttentionConverter
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.models.deepseek_v3 import model_registry

_SEQ = 1024
_HEADS = 16


def _attention_config():
  cfg = copy.deepcopy(
      model_registry("671B", seq_len=_SEQ, attn_backend="sdpa")
      .model.layers[0].attention
  )
  cfg.n_heads = _HEADS
  cfg.wq_b.out_features = _HEADS * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
  cfg.wkv_b.out_features = _HEADS * (cfg.qk_nope_head_dim + cfg.v_head_dim)
  cfg.wo.in_features = _HEADS * cfg.v_head_dim
  return cfg


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
  got, want = got.detach().float().cpu(), want.detach().float().cpu()
  return ((got - want).norm() / want.norm()).item()


class TpuMaxTextMLAAttentionTest(base_device_test.BaseAcceleratorDeviceTest):

  def test_matches_fp32_upstream_mla(self):
    cfg = _attention_config()
    torch.manual_seed(0)
    ref = cfg.build()
    with torch.no_grad():
      for prm in ref.parameters():
        if prm.ndim == 2:
          # BF16-representable, so both sides see the same weights.
          prm.copy_((torch.randn_like(prm) * 0.02).bfloat16())
        else:
          prm.fill_(1.0)
    model = TpuMaxTextMLAAttentionConverter(
        TpuMaxTextMLAAttentionConverter.Config()
    ).convert(copy.deepcopy(cfg)).build()
    self.assertIsInstance(model, TpuMaxTextMLAAttention)
    model.load_state_dict(ref.state_dict())

    x = torch.randn(_SEQ, cfg.dim).bfloat16()
    dout = (torch.randn(_SEQ, cfg.dim) * 1e-2).bfloat16()
    positions = torch.arange(_SEQ)
    segment_ids = torch.zeros(1, _SEQ, dtype=torch.int32)

    x_r = x.float().requires_grad_()
    out_r = ref(x_r, None, positions)
    grads_r = torch.autograd.grad(
        out_r, [x_r, *ref.parameters()], dout.float())

    dev = self.accelerator_device
    model = model.to(dev)
    with torch.no_grad():
      for prm in model.parameters():
        prm.data = prm.data.bfloat16()
    x_d = x.to(dev).requires_grad_()
    with set_unflatten_seq_len(_SEQ):
      out = model(x_d, None, positions.to(dev), segment_ids.to(dev))
    grads = torch.autograd.grad(
        out, [x_d, *model.parameters()], dout.to(dev))

    self.assertLess(_rel_err(out, out_r), 0.1)
    names = ["x"] + [n for n, _ in model.named_parameters()]
    for name, g, g_r in zip(names, grads, grads_r):
      with self.subTest(name):
        self.assertTrue(bool(torch.isfinite(g.float()).all()))
        self.assertLess(_rel_err(g, g_r), 0.15)


if __name__ == "__main__":
  absltest.main()
