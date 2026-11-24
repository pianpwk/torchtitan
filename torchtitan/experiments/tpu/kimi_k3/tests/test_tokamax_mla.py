"""Tests Kimi K3's Tokamax Splash gated MLA against an fp64 PyTorch reference.

The reference is K3's gated MLA written out with plain matmuls (q / kv LoRA
projections, RMSNorms, the expanded per-head K = [k_nope, shared k_rope]
without RoPE, softmax scale ``(qk_nope + qk_rope) ** -0.5``, a dense softmax
under an explicit causal-within-document mask, the sigmoid output gate and the
output projection), evaluated in fp64 on CPU from the same (rounded) inputs,
with autograd for every input and parameter gradient. It shares no code with
Splash or upstream's ``KimiMLAAttention``.
"""

import json

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchtitan.config.override import apply_overrides, OverrideConfig
from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type

# Before `torchtitan.models.kimi_k3` (installs the `attn_gym` stub).
from torchtitan.experiments.tpu.kimi_k3 import tokamax_mla
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.models import kimi_k3
from torchtitan.models.common.attention import segment_ids_from_positions

_MLA_OVERRIDE = (
    "torchtitan.experiments.tpu.kimi_k3.tokamax_mla.tokamax_mla_attention"
)
_KDA_OVERRIDE = (
    "torchtitan.experiments.tpu.kimi_k3.tokamax_kda.tokamax_kda_attention"
)
_SMALL_DOCS = ((100, 37, 247, 128), (300, 1, 211))
_K3_DOCS = ((1000, 37, 2048, 700, 311), (1500, 2596))
# Relative L2 bounds vs the fp64 reference, per dtype. TPU7x: <= 4.2e-3
# (fp32) / 5.6e-3 (bf16) for Splash alone, <= 4.1e-3 / 8.6e-3 for the layer.
# Splash's fp32 error is bf16-like: its Mosaic matmuls run at default
# precision (fp32 operands rounded to bf16), as do torch_tpu's fp32 matmuls
# unless the layer test raises them to "highest" (6.2e-3 -> 2.7e-3 on the
# debug layer's output), which leaves Splash as the only bf16-precision step.
_BOUNDS = {torch.float32: 1e-2, torch.bfloat16: 3e-2}


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
  """Relative L2, max abs, max abs / max |expected|, and cosine similarity."""
  a, e = actual.double().cpu().flatten(), expected.double().cpu().flatten()
  diff = (a - e).abs()
  return dict(
      rel_l2=float((a - e).norm() / e.norm().clamp_min(1e-30)),
      max_abs=float(diff.max()),
      max_rel=float(diff.max() / e.abs().max().clamp_min(1e-30)),
      cos=float(F.cosine_similarity(a, e, dim=0)),
  )


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


def _attention_heads(q_BSHK, k_BSHK, v_BSHV, allowed_BSS, scale):
  scores_BHSS = torch.einsum("bqhk,bshk->bhqs", q_BSHK, k_BSHK) * scale
  scores_BHSS = scores_BHSS.masked_fill(~allowed_BSS[:, None], float("-inf"))
  return torch.einsum("bhqs,bshv->bqhv", scores_BHSS.softmax(-1), v_BSHV)


def _reference_attention(
    q_BSHK, k_BSHK, v_BSHV, doc_ids_BS, scale, heads_per_chunk=None
):
  """Dense softmax attention, causal within each row and document.

  ``heads_per_chunk`` recomputes the scores per head chunk in the backward
  (checkpointing), to bound the fp64 reference's memory at full K3 shapes.
  """
  seq_len = q_BSHK.shape[1]
  causal_SS = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
  allowed_BSS = causal_SS[None]
  if doc_ids_BS is not None:
    allowed_BSS = allowed_BSS & (doc_ids_BS[:, :, None] == doc_ids_BS[:, None])
  if heads_per_chunk is None:
    return _attention_heads(q_BSHK, k_BSHK, v_BSHV, allowed_BSS, scale)
  outs = []
  for h in range(0, q_BSHK.shape[2], heads_per_chunk):
    sl = slice(h, h + heads_per_chunk)
    outs.append(
        checkpoint(
            _attention_heads,
            q_BSHK[:, :, sl],
            k_BSHK[:, :, sl],
            v_BSHV[:, :, sl],
            allowed_BSS,
            scale,
            use_reentrant=False,
        )
    )
  return torch.cat(outs, dim=2)


def _reference_layer(p, x_TD, doc_ids_BS, *, batch, cfg, heads_per_chunk):
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
      heads_per_chunk=heads_per_chunk,
  )
  out_TC = out_BSHV.reshape(num_tokens, heads * v_dim)
  return lin("wo", out_TC * torch.sigmoid(lin("gate", x_TD)))


class TokamaxMLATest(base_device_test.BaseAcceleratorDeviceTest):
  """``TokamaxSplashInnerAttention`` / ``TpuKimiMLAAttention`` vs fp64."""

  def _require_tpu(self) -> None:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("Tokamax Splash needs a TPU.")

  def _check(self, name, actual, expected, bound) -> None:
    results = {k: _errors(actual[k], e) for k, e in expected.items()}
    logging.info("MLA_NUMERICS %s", json.dumps({"case": name, **results}))
    for key, err in results.items():
      self.assertTrue(torch.isfinite(actual[key]).all().item(), msg=key)
      self.assertLess(err["rel_l2"], bound, msg=f"{name} {key}: {err}")

  def _compare_kernel(
      self, name, *, batch, seq_len, heads, cfg, dtype, docs, ref_heads=None
  ):
    """Splash fwd + dq/dk/dv vs the fp64 reference (on ``ref_heads``)."""
    dev = self.accelerator_device
    q_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    shape = (batch, seq_len, heads)
    torch.manual_seed(0)
    inputs = dict(
        q=torch.randn(*shape, q_dim, device=dev).to(dtype),
        k=torch.randn(*shape, q_dim, device=dev).to(dtype),
        v=torch.randn(*shape, cfg.v_head_dim, device=dev).to(dtype),
    )
    grad_out = torch.randn(*shape, cfg.v_head_dim, device=dev).to(dtype)
    doc_ids_BS = None
    doc_ids_T = None
    if docs:
      positions = _positions(docs, row1_offset=docs[0][-1])
      doc_ids_T = segment_ids_from_positions(positions)
      doc_ids_BS = doc_ids_T.reshape(batch, seq_len)
    scale = q_dim**-0.5

    inner = tokamax_mla.TokamaxSplashInnerAttention.Config().build()
    leaves = {n: t.detach().clone().requires_grad_() for n, t in inputs.items()}
    with set_unflatten_seq_len(seq_len):
      out = inner(
          *(leaves[n].flatten(0, 1) for n in "qkv"),
          attention_masks=None if doc_ids_T is None else doc_ids_T.to(dev),
          scale=scale,
      )
    out.backward(grad_out.flatten(0, 1))
    self.assertEqual(out.dtype, dtype)
    sel = slice(None) if ref_heads is None else ref_heads
    actual = {"out": out.detach().unflatten(0, (batch, seq_len))[:, :, sel]}
    actual.update({f"d{n}": leaves[n].grad[:, :, sel] for n in "qkv"})

    ref = {
        n: t[:, :, sel].detach().double().cpu().requires_grad_()
        for n, t in inputs.items()
    }
    ref_out = _reference_attention(
        ref["q"], ref["k"], ref["v"], doc_ids_BS, scale
    )
    ref_out.backward(grad_out[:, :, sel].double().cpu())
    expected = {"out": ref_out.detach()}
    expected.update({f"d{n}": ref[n].grad for n in "qkv"})
    self._check(name, actual, expected, _BOUNDS[dtype])
    if doc_ids_BS is not None:
      # Negative control: ignoring the documents moves the output well past
      # the bound, so the check above does see the document mask.
      with torch.no_grad():
        causal_only = _reference_attention(
            ref["q"], ref["k"], ref["v"], None, scale
        )
      self.assertGreater(
          _errors(actual["out"], causal_only)["rel_l2"], 10 * _BOUNDS[dtype]
      )

  @parameterized.named_parameters(
      dict(testcase_name="_bf16_docs", dtype=torch.bfloat16, docs=True),
  )
  def test_kernel_matches_reference(self, dtype: torch.dtype, docs: bool):
    """Splash with K3's head dims, 2 rows x 512 tokens, 4 heads."""
    self._require_tpu()
    self._compare_kernel(
        f"kernel_small{'_docs' if docs else ''}_{str(dtype)[6:]}",
        batch=2,
        seq_len=512,
        heads=4,
        cfg=_k3_mla_config("Kimi-K3"),
        dtype=dtype,
        # Documents of odd lengths (one of 1 token); row 1 starts mid-document.
        docs=_SMALL_DOCS if docs else None,
    )

  @parameterized.named_parameters(
      dict(testcase_name="_bf16", dtype=torch.bfloat16),
  )
  def test_k3_shape_matches_reference(self, dtype: torch.dtype):
    """All 96 K3 heads, 2 rows x 4096 tokens, several documents per row."""
    self._require_tpu()
    cfg = _k3_mla_config("Kimi-K3")
    self.assertEqual(
        (
            cfg.n_heads,
            cfg.qk_nope_head_dim,
            cfg.qk_rope_head_dim,
            cfg.v_head_dim,
        ),
        (96, 128, 64, 128),
    )
    # Heads are independent: the reference checks a subset of the 96.
    self._compare_kernel(
        f"kernel_k3_4096x2_{str(dtype)[6:]}",
        batch=2,
        seq_len=4096,
        heads=cfg.n_heads,
        cfg=cfg,
        dtype=dtype,
        docs=_K3_DOCS,
        ref_heads=[0, 33, 66, 95],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="_debug_bf16", flavor="debugmodel", dtype=torch.bfloat16
      ),
      dict(testcase_name="_k3_bf16", flavor="Kimi-K3", dtype=torch.bfloat16),
  )
  def test_layer_matches_reference(self, flavor: str, dtype: torch.dtype):
    """``TpuKimiMLAAttention`` (via the override) vs an fp64 reference layer.

    Checks the plumbing around Splash: the LoRA projections and norms, the
    expanded K, ``positions`` -> document ids, the output gate and every
    parameter and input gradient.
    """
    self._require_tpu()
    dev = self.accelerator_device
    model_cfg = kimi_k3.model_registry(flavor).model
    replaced = apply_overrides(
        OverrideConfig(imports=[_MLA_OVERRIDE, _KDA_OVERRIDE]), model_cfg
    )
    num_mla = sum(layer.attention is not None for layer in model_cfg.layers)
    self.assertEqual(
        sum("tokamax_mla_attention" in r for r in replaced), num_mla
    )
    cfg = next(layer.attention for layer in model_cfg.layers if layer.attention)
    self.assertIsInstance(cfg, tokamax_mla.TpuKimiMLAAttention.Config)
    self.assertIsInstance(
        cfg.inner_attention, tokamax_mla.TokamaxSplashInnerAttention.Config
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
        n: p.detach().to(dtype).double() for n, p in layer.named_parameters()
    }
    layer = layer.to(device=dev, dtype=dtype)

    is_k3 = flavor == "Kimi-K3"
    seq_len = 4096 if is_k3 else 512
    docs = _K3_DOCS if is_k3 else _SMALL_DOCS
    batch = len(docs)
    positions = _positions(docs, row1_offset=docs[0][-1])
    x = torch.randn(batch * seq_len, cfg.dim).to(dtype)
    grad_out = torch.randn(batch * seq_len, cfg.dim).to(dtype)

    x_dev = x.to(dev).requires_grad_()
    precision = torch.get_float32_matmul_precision()
    if dtype == torch.float32:
      torch.set_float32_matmul_precision("highest")
    try:
      with set_unflatten_seq_len(seq_len):
        out = layer(x_dev, positions=positions.to(dev))
      out.backward(grad_out.to(dev))
    finally:
      torch.set_float32_matmul_precision(precision)
    self.assertEqual(out.dtype, dtype)
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
        heads_per_chunk=8 if is_k3 else None,
    )
    out64.backward(grad_out.double())
    expected = {"out": out64.detach(), "dx": x64.grad}
    expected.update({f"d{n}": p.grad for n, p in p64.items()})
    self._check(
        f"layer_{'k3' if is_k3 else 'debug'}_{str(dtype)[6:]}",
        actual,
        expected,
        _BOUNDS[dtype],
    )


if __name__ == "__main__":
  absltest.main()
