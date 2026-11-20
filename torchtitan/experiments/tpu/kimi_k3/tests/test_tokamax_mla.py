"""Tests Kimi K3's Tokamax Splash gated MLA against an fp64 PyTorch reference.

The reference is K3's gated MLA written out with plain matmuls (q / kv LoRA
projections, RMSNorms, the expanded per-head K = [k_nope, shared k_rope]
without RoPE, softmax scale ``(qk_nope + qk_rope) ** -0.5``, a dense softmax
under an explicit causal-within-document mask, the sigmoid output gate and the
output projection), evaluated in fp64 on CPU from the same (rounded) inputs,
with autograd for every input and parameter gradient. It shares no code with
Splash or upstream's ``KimiMLAAttention``. Both tests use bf16 and packed
documents of odd lengths (one of 1 token), with row 1 starting mid-document.
"""

import json

from absl import logging
from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type

# Before `torchtitan.models.kimi_k3` (installs the `attn_gym` stub).
from torchtitan.experiments.tpu.kimi_k3 import tokamax_mla
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.models import kimi_k3
from torchtitan.models.common.attention import segment_ids_from_positions

# 2 rows x 512 tokens of packed documents.
_BATCH, _SEQ_LEN = 2, 512
# Relative L2 bound vs the fp64 reference in bf16. TPU7x: <= 5.6e-3 for Splash
# alone, <= 8.6e-3 for the layer.
_BOUND = 3e-2


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
  a, e = actual.double().cpu().flatten(), expected.double().cpu().flatten()
  return float((a - e).norm() / e.norm().clamp_min(1e-30))


def _positions() -> torch.Tensor:
  """``[B * S]`` positions of the packed rows.

  Documents of odd lengths (one of 1 token); row 1's first document continues
  at position 128, so only the row boundary separates it from row 0's last.
  """
  rows = []
  for i, lens in enumerate(((100, 37, 247, 128), (300, 1, 211))):
    row = [torch.arange(n) for n in lens]
    if i == 1:
      row[0] = row[0] + 128
    rows.append(torch.cat(row))
  return torch.cat(rows)


def _reference_attention(q_BSHK, k_BSHK, v_BSHV, doc_ids_BS, scale):
  """Dense softmax attention, causal within each row (and document)."""
  seq_len = q_BSHK.shape[1]
  allowed_BSS = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()[None]
  if doc_ids_BS is not None:
    allowed_BSS = allowed_BSS & (doc_ids_BS[:, :, None] == doc_ids_BS[:, None])
  scores_BHSS = torch.einsum("bqhk,bshk->bhqs", q_BSHK, k_BSHK) * scale
  scores_BHSS = scores_BHSS.masked_fill(~allowed_BSS[:, None], float("-inf"))
  return torch.einsum("bhqs,bshv->bqhv", scores_BHSS.softmax(-1), v_BSHV)


def _reference_layer(p, x_TD, doc_ids_BS, *, batch, cfg):
  """K3's ``KimiMLAAttention.forward`` in fp64, from its parameters ``p``."""
  num_tokens = x_TD.shape[0]
  seq_len = num_tokens // batch
  heads = cfg.n_heads
  nope, rope, v_dim = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim

  def lin(name, inp):
    return inp @ p[f"{name}.weight"].T

  def rms_norm(name, inp, eps):
    return (
        inp
        * torch.rsqrt(inp.pow(2).mean(-1, keepdim=True) + eps)
        * p[f"{name}.weight"]
    )

  q = lin("wq_b", rms_norm("q_norm", lin("wq_a", x_TD), cfg.q_norm.eps))
  q_BSHK = q.reshape(batch, seq_len, heads, nope + rope)
  kv_a = lin("wkv_a", x_TD)
  latent, k_rope = kv_a[:, : cfg.kv_lora_rank], kv_a[:, cfg.kv_lora_rank :]
  kv = lin("wkv_b", rms_norm("kv_norm", latent, cfg.kv_norm.eps))
  kv = kv.reshape(batch, seq_len, heads, nope + v_dim)
  k_rope_BSHK = k_rope.reshape(batch, seq_len, 1, rope).expand(
      -1, -1, heads, -1
  )
  k_BSHK = torch.cat([kv[..., :nope], k_rope_BSHK], dim=-1)
  v_BSHV = kv[..., nope:]
  out_BSHV = _reference_attention(
      q_BSHK,
      k_BSHK,
      v_BSHV,
      doc_ids_BS,
      (nope + rope) ** -0.5,
  )
  out_TC = out_BSHV.reshape(num_tokens, heads * v_dim)
  return lin("wo", out_TC * torch.sigmoid(lin("gate", x_TD)))


class TokamaxMLATest(base_device_test.BaseAcceleratorDeviceTest):
  """``TokamaxSplashInnerAttention`` / ``TpuKimiMLAAttention`` vs fp64."""

  def _require_tpu(self) -> None:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Tokamax Splash needs a TPU.")

  def _check(self, name, actual, expected) -> None:
    results = {k: _rel_l2(actual[k], e) for k, e in expected.items()}
    logging.info("MLA_NUMERICS %s", json.dumps({"case": name, **results}))
    for key, err in results.items():
      self.assertTrue(torch.isfinite(actual[key]).all().item(), msg=key)
      self.assertLess(err, _BOUND, msg=f"{name} {key}: {err}")

  def test_kernel_matches_reference(self):
    """Splash fwd + dq/dk/dv with K3's head dims, 4 heads, bf16."""
    self._require_tpu()
    dev = self.accelerator_device
    cfg = kimi_k3.model_registry("Kimi-K3").model
    mla = next(layer.attention for layer in cfg.layers if layer.attention)
    q_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim
    shape = (_BATCH, _SEQ_LEN, 4)
    torch.manual_seed(0)
    inputs = dict(
        q=torch.randn(*shape, q_dim, device=dev).bfloat16(),
        k=torch.randn(*shape, q_dim, device=dev).bfloat16(),
        v=torch.randn(*shape, mla.v_head_dim, device=dev).bfloat16(),
    )
    grad_out = torch.randn(*shape, mla.v_head_dim, device=dev).bfloat16()
    doc_ids_T = segment_ids_from_positions(_positions())
    doc_ids_BS = doc_ids_T.reshape(_BATCH, _SEQ_LEN)
    scale = q_dim**-0.5

    inner = tokamax_mla.TokamaxSplashInnerAttention.Config().build()
    leaves = {n: t.detach().clone().requires_grad_() for n, t in inputs.items()}
    with set_unflatten_seq_len(_SEQ_LEN):
      out = inner(
          *(leaves[n].flatten(0, 1) for n in "qkv"),
          attention_masks=doc_ids_T.to(dev),
          scale=scale,
      )
    out.backward(grad_out.flatten(0, 1))
    actual = {"out": out.detach().unflatten(0, (_BATCH, _SEQ_LEN))}
    actual.update({f"d{n}": leaves[n].grad for n in "qkv"})

    ref = {
        n: t.detach().double().cpu().requires_grad_()
        for n, t in inputs.items()
    }
    ref_out = _reference_attention(
        ref["q"], ref["k"], ref["v"], doc_ids_BS, scale
    )
    ref_out.backward(grad_out.double().cpu())
    expected = {"out": ref_out.detach()}
    expected.update({f"d{n}": ref[n].grad for n in "qkv"})
    self._check("kernel_bf16", actual, expected)
    # Negative control: ignoring the documents moves the output well past the
    # bound, so the check above does see the document mask.
    with torch.no_grad():
      causal_only = _reference_attention(
          ref["q"], ref["k"], ref["v"], None, scale
      )
    self.assertGreater(_rel_l2(actual["out"], causal_only), 10 * _BOUND)

  def test_layer_matches_reference(self):
    """Debug-flavor ``TpuKimiMLAAttention`` vs an fp64 reference layer, bf16.

    Checks the plumbing around Splash: the LoRA projections and norms, the
    expanded K, ``positions`` -> document ids, the output gate and every
    parameter and input gradient.
    """
    self._require_tpu()
    dev = self.accelerator_device
    model_cfg = kimi_k3.model_registry("debugmodel").model
    cfg = tokamax_mla.tokamax_mla_attention(
        next(layer.attention for layer in model_cfg.layers if layer.attention)
    )
    torch.manual_seed(0)
    layer = cfg.build()
    layer.init_states(buffer_device=torch.device("cpu"))
    with torch.no_grad():
      # Fan-in scaled weights (larger than init) so attention logits have
      # O(1) spread and the softmax is far from uniform.
      for p in layer.parameters():
        if p.ndim >= 2:
          p.normal_(0.0, p.shape[-1] ** -0.5)
        else:
          p.uniform_(0.5, 1.5)
    params64 = {
        n: p.detach().bfloat16().double() for n, p in layer.named_parameters()
    }
    layer = layer.to(device=dev, dtype=torch.bfloat16)

    positions = _positions()
    num_tokens = _BATCH * _SEQ_LEN
    x = torch.randn(num_tokens, cfg.dim).bfloat16()
    grad_out = torch.randn(num_tokens, cfg.dim).bfloat16()
    x_dev = x.to(dev).requires_grad_()
    with set_unflatten_seq_len(_SEQ_LEN):
      out = layer(x_dev, positions=positions.to(dev))
    out.backward(grad_out.to(dev))
    actual = {"out": out.detach(), "dx": x_dev.grad}
    actual.update({f"d{n}": p.grad for n, p in layer.named_parameters()})

    p64 = {n: p.clone().requires_grad_() for n, p in params64.items()}
    x64 = x.double().requires_grad_()
    doc_ids_BS = segment_ids_from_positions(positions).reshape(
        _BATCH, _SEQ_LEN
    )
    out64 = _reference_layer(p64, x64, doc_ids_BS, batch=_BATCH, cfg=cfg)
    out64.backward(grad_out.double())
    expected = {"out": out64.detach(), "dx": x64.grad}
    expected.update({f"d{n}": p.grad for n, p in p64.items()})
    self._check("layer_debug_bf16", actual, expected)


if __name__ == "__main__":
  absltest.main()
