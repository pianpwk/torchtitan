"""Tests Kimi K3's Tokamax Splash gated MLA against an fp64 PyTorch reference.

The reference is K3's gated MLA written out with plain matmuls (q / kv LoRA
projections, RMSNorms, the expanded per-head K = [k_nope, shared k_rope]
without RoPE, softmax scale ``(qk_nope + qk_rope) ** -0.5``, a dense softmax
under an explicit causal-within-document mask, the sigmoid output gate and the
output projection), evaluated in fp64 on CPU from the same (rounded) inputs,
with autograd for every input and parameter gradient. It shares no code with
Splash or upstream's ``KimiMLAAttention``.
"""

from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type

# Before `torchtitan.models.kimi_k3` (installs the `attn_gym` stub).
from torchtitan.experiments.tpu.kimi_k3 import tokamax_mla
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.models import kimi_k3
from torchtitan.models.common.attention import segment_ids_from_positions

# Relative L2 bound vs the fp64 reference for the bf16 layer. TPU7x: <= 5.6e-3
# for Splash alone, <= 8.6e-3 for the layer at K3's head dims.
_BOUND = 3e-2


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
  a, e = actual.double().cpu(), expected.double().cpu()
  return float((a - e).norm() / e.norm().clamp_min(1e-30))


def _k3_mla_config(flavor: str) -> kimi_k3.KimiMLAAttention.Config:
  model = kimi_k3.model_registry(flavor).model
  return next(layer.attention for layer in model.layers if layer.attention)


def _positions(doc_lens_per_row, row1_offset: int = 0) -> torch.Tensor:
  """Flattened ``[B * S]`` positions of packed documents.

  Row 1's first document starts at position ``row1_offset``: it continues a
  document from the previous row, so only the row boundary separates them.
  """
  rows = []
  for i, lens in enumerate(doc_lens_per_row):
    row = [torch.arange(n) for n in lens]
    if i == 1:
      row[0] = row[0] + row1_offset
    rows.append(torch.cat(row))
  return torch.cat(rows)


def _reference_attention(q_BSHK, k_BSHK, v_BSHV, doc_ids_BS, scale):
  """Dense softmax attention, causal within each row and document."""
  seq_len = q_BSHK.shape[1]
  allowed_BSS = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()[None]
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
      q_BSHK, k_BSHK, v_BSHV, doc_ids_BS, (nope + rope) ** -0.5
  )
  out_TC = out_BSHV.reshape(num_tokens, heads * v_dim)
  return lin("wo", out_TC * torch.sigmoid(lin("gate", x_TD)))


class TokamaxMLATest(base_device_test.BaseAcceleratorDeviceTest):
  """``TokamaxSplashInnerAttention`` / ``TpuKimiMLAAttention`` vs fp64."""

  def _require_tpu(self) -> None:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Tokamax Splash needs a TPU.")

  def test_layer_matches_reference(self):
    """K3-flavor ``TpuKimiMLAAttention`` (via the override) vs fp64.

    Checks the plumbing around Splash: the LoRA projections and norms, the
    expanded K at K3's head dims (192 / 128), ``positions`` -> document ids
    (documents crossing a 1024-token Splash block, a row starting
    mid-document), the output gate and every parameter and input gradient.
    """
    self._require_tpu()
    dev = self.accelerator_device
    cfg = tokamax_mla.tokamax_mla_attention(_k3_mla_config("Kimi-K3"))

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

    seq_len = 2048
    docs = ((700, 37, 900, 411), (1200, 848))
    batch = len(docs)
    positions = _positions(docs, row1_offset=docs[0][-1])
    x = torch.randn(batch * seq_len, cfg.dim).bfloat16()
    grad_out = torch.randn(batch * seq_len, cfg.dim).bfloat16()

    x_dev = x.to(dev).requires_grad_()
    with set_unflatten_seq_len(seq_len):
      out = layer(x_dev, positions=positions.to(dev))
    out.backward(grad_out.to(dev))
    actual = {"out": out.detach(), "dx": x_dev.grad}
    actual.update({f"d{n}": p.grad for n, p in layer.named_parameters()})

    p64 = {n: p.clone().requires_grad_() for n, p in params64.items()}
    x64 = x.double().requires_grad_()
    out64 = _reference_layer(
        p64,
        x64,
        segment_ids_from_positions(positions).reshape(batch, seq_len),
        batch=batch,
        cfg=cfg,
    )
    out64.backward(grad_out.double())
    expected = {"out": out64.detach(), "dx": x64.grad}
    expected.update({f"d{n}": p.grad for n, p in p64.items()})
    for key, e in expected.items():
      self.assertTrue(torch.isfinite(actual[key]).all().item(), msg=key)
      self.assertLess(_rel_l2(actual[key], e), _BOUND, msg=key)

if __name__ == "__main__":
  absltest.main()
