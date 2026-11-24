"""Tests Kimi K3's Tokamax KDA modules against an fp64 PyTorch reference.

The reference is K3's KDA math written out token by token (the gate
``lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))``, ``sigmoid`` beta,
L2-normalized Q/K and the delta rule restarted at every document), evaluated in
fp64 on CPU from the same (rounded) inputs, with autograd for every input
gradient. It shares no code with Tokamax or ``attn_gym``.
"""

from absl.testing import absltest
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type

# Before `torchtitan.models.kimi_k3` (installs the `attn_gym` stub).
from torchtitan.experiments.tpu.kimi_k3 import tokamax_kda
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.models import kimi_k3
from torchtitan.models.common.attention import segment_ids_from_positions

_LOWER_BOUND = -5.0


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
  a, e = actual.double().cpu(), expected.double().cpu()
  return float((a - e).norm() / e.norm().clamp_min(1e-30))


def _k3_kda_config(flavor: str) -> kimi_k3.KDA.Config:
  model = kimi_k3.model_registry(flavor).model
  return next(
      layer.delta_attention
      for layer in model.layers
      if layer.delta_attention is not None
  )


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


def _reference_kda(
    q_BSHK,
    k_BSHK,
    v_BSHV,
    raw_gate_BSHK,
    raw_beta_BSH,
    A_log_H,
    dt_bias_HK,
    *,
    lower_bound,
    doc_ids_BS=None,
) -> torch.Tensor:
  """K3's KDA in the inputs' dtype (use fp64), token by token."""
  gate_BSHK = lower_bound * torch.sigmoid(
      A_log_H.exp().unsqueeze(-1) * (raw_gate_BSHK + dt_bias_HK)
  )
  beta_BSH = torch.sigmoid(raw_beta_BSH)
  scale = q_BSHK.shape[-1] ** -0.5
  q_BSHK = q_BSHK * torch.rsqrt((q_BSHK * q_BSHK).sum(-1, keepdim=True) + 1e-6)
  k_BSHK = k_BSHK * torch.rsqrt((k_BSHK * k_BSHK).sum(-1, keepdim=True) + 1e-6)
  batch, seq_len, heads, d_k = q_BSHK.shape
  state_BHKV = q_BSHK.new_zeros(batch, heads, d_k, v_BSHV.shape[-1])
  outs = []
  for t in range(seq_len):
    if doc_ids_BS is not None and t > 0:
      keep_B = (doc_ids_BS[:, t] == doc_ids_BS[:, t - 1]).to(state_BHKV.dtype)
      state_BHKV = state_BHKV * keep_B[:, None, None, None]
    state_BHKV = state_BHKV * gate_BSHK[:, t].exp().unsqueeze(-1)
    k_BHK = k_BSHK[:, t]
    pred_BHV = torch.einsum("bhk,bhkv->bhv", k_BHK, state_BHKV)
    delta_BHV = beta_BSH[:, t].unsqueeze(-1) * (v_BSHV[:, t] - pred_BHV)
    state_BHKV = state_BHKV + k_BHK.unsqueeze(-1) * delta_BHV.unsqueeze(-2)
    outs.append(torch.einsum("bhk,bhkv->bhv", q_BSHK[:, t] * scale, state_BHKV))
  return torch.stack(outs, dim=1)


class TokamaxKDATest(base_device_test.BaseAcceleratorDeviceTest):
  """``TokamaxKDAKernel`` / ``TpuKDA`` on one TPU core vs the fp64 reference."""

  def _require_mosaic_kda(self) -> None:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("The Mosaic KDA kernels need a TPU.")
    if not tokamax_kda.supports_current_tpu():
      self.skipTest("The Mosaic KDA kernels need TPU v6 or newer.")
    if not tokamax_kda.has_tokamax_kda():
      self.skipTest(
          tokamax_kda.tokamax_kda_import_error()
          or "Tokamax KDA op is not installed."
      )

  def test_layer_matches_reference(self):
    """Debug-flavor ``TpuKDA`` (via the override) vs an fp64 reference layer.

    Checks the plumbing around the kernel: the projections, the per-document
    causal conv (reference: ``F.conv1d`` per document), ``positions`` ->
    segment ids, the output gate / norm and every parameter gradient.
    """
    self._require_mosaic_kda()
    torch.manual_seed(0)
    dev = self.accelerator_device
    tpu_cfg = tokamax_kda.tokamax_kda_attention(
        _k3_kda_config("debugmodel"), max_num_segments=8
    )
    layer = tpu_cfg.build()
    layer.init_states(buffer_device=torch.device("cpu"))
    with torch.no_grad():
      # Larger projections than init, so the gate and conv matter.
      for p in layer.parameters():
        if p.ndim >= 2:
          p.normal_(0.0, p.shape[-1] ** -0.5)
      layer.dt_bias.normal_(0.0, 0.5)
      layer.output_norm.weight.uniform_(0.5, 1.5)
    params64 = {n: p.detach().double() for n, p in layer.named_parameters()}
    layer = layer.to(device=dev, dtype=torch.bfloat16)
    # Kept in fp32 like K3's mixed precision does for the gate parameters.
    layer.A_log.data = layer.A_log.data.float()
    layer.dt_bias.data = layer.dt_bias.data.float()
    for name in params64:
      if name not in ("A_log", "dt_bias"):
        params64[name] = params64[name].bfloat16().double()

    batch, seq_len = 2, 512
    positions = _positions(
        ((100, 37, 247, 128), (300, 1, 211)), row1_offset=100
    )
    dim = tpu_cfg.q_proj.in_features
    x = torch.randn(batch * seq_len, dim, dtype=torch.bfloat16)
    grad_out = torch.randn(batch * seq_len, dim, dtype=torch.bfloat16)

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
        positions,
        batch=batch,
        num_heads=tpu_cfg.num_heads,
        head_dim=tpu_cfg.head_dim,
        eps=tpu_cfg.output_norm.eps,
    )
    out64.backward(grad_out.double())
    expected = {"out": out64.detach(), "dx": x64.grad}
    expected.update({f"d{n}": p.grad for n, p in p64.items()})

    for key, e in expected.items():
      self.assertTrue(torch.isfinite(actual[key]).all().item(), msg=key)
      # bf16 inputs: <= 5.1e-3 for the kernel's output and q/k/v/beta
      # gradients, but the gate's gradient chain loses more in the Mosaic
      # kernels' bf16 arithmetic (TPU7x: dt_bias 9.2e-2, a sum over every
      # token of the raw-gate gradient, at cosine >= 0.996; the same values
      # upcast to fp32 give <= 1.5e-3). The projections add bf16 rounding
      # (TPU7x: <= 2.1e-2 outside the gate chain).
      if key == "ddt_bias":
        bound = 1.5e-1
      elif key in ("dA_log", "dforget_a.weight", "dforget_b.weight"):
        bound = 5e-2
      else:
        bound = 3e-2
      self.assertLess(_rel_l2(actual[key], e), bound, msg=key)


def _reference_layer(p, x_TD, positions, *, batch, num_heads, head_dim, eps):
  """K3's ``KDA.forward`` in fp64 with a per-document ``F.conv1d``."""
  num_tokens = x_TD.shape[0]
  seq_len = num_tokens // batch
  doc_BS = segment_ids_from_positions(positions).reshape(batch, seq_len)

  def lin(name, inp):
    return inp @ p[f"{name}.weight"].T

  def conv_silu(x_TC, weight_C1W):
    x_BSC = x_TC.reshape(batch, seq_len, -1)
    width = weight_C1W.shape[-1]
    rows = []
    for b in range(batch):
      starts = [0] + [
          s for s in range(1, seq_len) if doc_BS[b, s] != doc_BS[b, s - 1]
      ]
      pieces = []
      for s0, s1 in zip(starts, starts[1:] + [seq_len]):
        seg_CS = x_BSC[b, s0:s1].T.unsqueeze(0)
        y = F.conv1d(
            seg_CS, weight_C1W, padding=width - 1, groups=weight_C1W.shape[0]
        )
        pieces.append(y[0, :, : s1 - s0].T)
      rows.append(torch.cat(pieces))
    return F.silu(torch.stack(rows)).unflatten(-1, (num_heads, head_dim))

  q = conv_silu(lin("q_proj", x_TD), p["q_conv.weight"])
  k = conv_silu(lin("k_proj", x_TD), p["k_conv.weight"])
  v = conv_silu(lin("v_proj", x_TD), p["v_conv.weight"])
  raw_gate = lin("forget_b", lin("forget_a", x_TD)).reshape(
      batch, seq_len, num_heads, head_dim
  )
  raw_beta = lin("beta", x_TD).reshape(batch, seq_len, num_heads)
  out_BSHV = _reference_kda(
      q,
      k,
      v,
      raw_gate,
      raw_beta,
      p["A_log"],
      p["dt_bias"],
      lower_bound=_LOWER_BOUND,
      doc_ids_BS=doc_BS,
  )
  out_THV = out_BSHV.reshape(num_tokens, num_heads, head_dim)
  gate_THV = lin("output_gate", x_TD).view_as(out_THV)
  normed = F.rms_norm(out_THV, (head_dim,), p["output_norm.weight"], eps)
  return lin("output_proj", (normed * gate_THV.sigmoid()).flatten(-2))


if __name__ == "__main__":
  absltest.main()
