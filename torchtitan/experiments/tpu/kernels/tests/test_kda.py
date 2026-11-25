"""Tests the fused Pallas KDA kernels (``kernels/kda.py``) against fp64.

The reference is the KDA / Gated DeltaNet recurrence written out token by
token in fp64 on CPU from the same (rounded) inputs, with autograd for every
input gradient. It shares no code with the kernels.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.experiments.tpu.kernels import kda
from torchtitan.experiments.tpu.testing import base_device_test

# Relative L2 bound vs fp64 for the output, the final state and every gradient.
# TPU7x, bf16 inputs: <= 4e-3 on these cases (fuzzing over adversarial gates,
# keys, magnitudes and documents: <= 6e-3 outside beta ~ 1 with aligned keys
# and no decay, which is ill-conditioned for any bf16 implementation).
_TOL = 1e-2


def _reference(
    q, k, v, g, beta, a_log, dt_bias, segment_ids, initial_state, lower_bound
):
  """fp64 token recurrence; returns ``(out, final_state)``."""
  H = v.shape[2]
  q = q.repeat_interleave(H // q.shape[2], dim=2)
  k = k.repeat_interleave(H // k.shape[2], dim=2)
  B, S, _, K = q.shape
  if a_log is not None:
    g = lower_bound * torch.sigmoid(a_log.exp()[:, None] * (g + dt_bias))
  else:
    g = g.clamp_min(lower_bound)[..., None].expand(B, S, H, K)
  q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K**-0.5
  k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
  state = (
      torch.zeros(B, H, K, v.shape[-1], dtype=q.dtype)
      if initial_state is None
      else initial_state
  )
  outs = []
  for t in range(S):
    if t > 0:
      new_doc = (segment_ids[:, t] != segment_ids[:, t - 1])[
          :, None, None, None
      ]
      state = torch.where(new_doc, torch.zeros_like(state), state)
    elif initial_state is None:
      state = torch.zeros_like(state)
    state = g[:, t].exp()[..., None] * state
    kv = torch.einsum("bhk,bhkv->bhv", k[:, t], state)
    state = (
        state
        + (beta[:, t, :, None, None] * k[:, t, :, :, None])
        * (v[:, t] - kv)[:, :, None, :]
    )
    outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
  return torch.stack(outs, 1), state


def _rel(actual, expected):
  actual, expected = actual.double().cpu(), expected.double().cpu()
  return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


class KDATest(base_device_test.BaseAcceleratorDeviceTest):

  @parameterized.named_parameters(
      # K3's in-kernel gate, packed documents not aligned to chunks
      # (including documents of length 1), S not a multiple of the chunk.
      dict(testcase_name="_k3_gate", gate="k3", S=200, K=128, V=128),
      # One decay per head (Gated DeltaNet), clamped at the lower bound, with
      # two value heads per q / k head, head dims padded to 128 lanes, and
      # initial and final state.
      dict(
          testcase_name="_per_head_decay_gqa_small_heads",
          gate="head",
          S=128,
          K=64,
          V=192,
          state=True,
          key_heads=2,
      ),
  )
  def test_matches_fp64_recurrence(
      self, gate, S, K, V, state=False, key_heads=4
  ):
    dev = self.accelerator_device
    if dev.type != "tpu":
      self.skipTest("Pallas kernels are TPU-only.")
    torch.manual_seed(0)
    B, H, lower = 2, 4, -5.0
    bf = lambda x: x.to(torch.bfloat16).double()
    q = bf(torch.randn(B, S, key_heads, K))
    k = bf(torch.randn(B, S, key_heads, K))
    v = bf(torch.randn(B, S, H, V))
    beta = torch.rand(B, S, H, dtype=torch.float64)
    a_log = dt_bias = None
    if gate == "k3":
      g = bf(torch.randn(B, S, H, K))
      a_log = torch.rand(H, dtype=torch.float64) * 2 - 1
      dt_bias = torch.rand(H, K, dtype=torch.float64) * 11 - 10  # long memory
    else:
      g = -torch.exp(torch.rand(B, S, H, dtype=torch.float64) * 10 - 8)
    seg = torch.zeros(B, S, dtype=torch.int32)
    for b, cuts in enumerate(([1, 2, 3, 63, 64, 100], [17, 128, 129])):
      for c in cuts:
        if c < S:
          seg[b, c:] += 1
    init = torch.randn(B, H, K, V, dtype=torch.float64) if state else None
    leaves = dict(
        q=q, k=k, v=v, g=g, beta=beta, a_log=a_log, dt_bias=dt_bias, init=init
    )
    names = [n for n, x in leaves.items() if x is not None]
    grad_o = bf(torch.randn(B, S, H, V))
    grad_s = torch.randn(B, H, K, V, dtype=torch.float64)

    ref = {n: leaves[n].clone().requires_grad_() for n in names}
    o_r, s_r = _reference(
        ref["q"],
        ref["k"],
        ref["v"],
        ref["g"],
        ref["beta"],
        ref.get("a_log"),
        ref.get("dt_bias"),
        seg,
        ref.get("init"),
        lower,
    )
    loss = (o_r * grad_o).sum() + ((s_r * grad_s).sum() if state else 0)
    grads_r = torch.autograd.grad(loss, [ref[n] for n in names])

    dtype = dict(
        q=torch.bfloat16,
        k=torch.bfloat16,
        v=torch.bfloat16,
        g=torch.bfloat16 if gate == "k3" else torch.float32,
    )
    dut = {
        n: leaves[n].to(dev, dtype.get(n, torch.float32)).requires_grad_()
        for n in names
    }
    o, s = kda.kda_torch(
        dut["q"],
        dut["k"],
        dut["v"],
        dut["g"],
        dut["beta"],
        lower_bound=lower,
        a_log=dut.get("a_log"),
        dt_bias=dut.get("dt_bias"),
        segment_ids=seg.to(dev),
        initial_state=dut.get("init"),
        output_final_state=state,
    )
    loss = (o.float() * grad_o.to(dev, torch.float32)).sum()
    if state:
      loss = loss + (s * grad_s.to(dev, torch.float32)).sum()
    grads = torch.autograd.grad(loss, [dut[n] for n in names])

    self.assertEqual(o.dtype, torch.bfloat16)
    self.assertEqual(o.shape, (B, S, H, V))
    self.assertLess(_rel(o, o_r), _TOL, "out")
    if state:
      self.assertLess(_rel(s, s_r), _TOL, "final_state")
    for n, a, e in zip(names, grads, grads_r):
      self.assertEqual(a.shape, e.shape, n)
      self.assertLess(_rel(a, e), _TOL, f"d{n}")


if __name__ == "__main__":
  absltest.main()
