"""Accelerator tests for the fused FP8 Ring-of-Experts expert kernels."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from torchtitan.experiments.tpu.kernels import fp8_roe_experts
from torchtitan.experiments.tpu.testing import base_device_test

_S = fp8_roe_experts.STATIC_SCALE
# DeepSeek-V3 widths (the gmmv2 tiles are measured for D = 7168), few rows.
_N, _TOP_K, _NUM_EXPERTS, _E, _D, _F = 1024, 8, 64, 8, 7168, 2048


def _q(a):
  return jnp.clip(a.astype(jnp.float32) / _S, -448, 448).astype(
      jnp.float8_e4m3fn
  )


def _dq(a):
  return a.astype(jnp.float32) * _S


def _rel(got, want):
  got = jnp.asarray(got, jnp.float32)
  return float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))


class Fp8RoeExpertsTest(base_device_test.BaseAcceleratorDeviceTest):

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(0)
    ids = np.stack(
        [rng.choice(_NUM_EXPERTS, _TOP_K, replace=False) for _ in range(_N)]
    )
    tok, slot = np.nonzero(ids < _E)
    exp = ids[tok, slot]
    order = np.lexsort((tok, exp))
    self.tok, exp = tok[order], exp[order]
    self.gs = np.bincount(exp, minlength=_E).astype(np.int32)
    self.nv = len(self.tok)
    rows = -(-self.nv * 5 // 4 // 256) * 256
    token_ids = np.zeros(rows, np.int32)
    token_ids[: self.nv] = self.tok
    k = jax.random.split(jax.random.key(0), 5)
    self.x_q = _q(jax.random.normal(k[0], (_N, _D)))
    self.w13_q = _q(jax.random.normal(k[1], (_E, _D, 2 * _F)) * 0.05)
    self.w2_q = _q(jax.random.normal(k[2], (_E, _F, _D)) * 0.05)
    self.scores = jax.random.uniform(k[3], (rows,), jnp.float32)
    self.dy = jax.random.normal(k[4], (_N, _D), jnp.float32) * 0.01
    self.args = (
        self.x_q.reshape(_N, _D // 128, 128),
        jnp.asarray(token_ids),
        jnp.asarray(self.gs),
        self.scores,
        self.w13_q,
        self.w2_q,
    )

  def _expert_rows(self, rows, w13, w2, scores, quantize_hidden):
    out, start = [], 0
    for e in range(_E):
      n = int(self.gs[e])
      g = rows[start : start + n] @ w13[e]
      h = jax.nn.silu(g[:, :_F]) * g[:, _F:] * scores[start : start + n, None]
      if quantize_hidden:
        h = _dq(_q(h))
      out.append(h @ w2[e])
      start += n
    return jnp.concatenate(out)

  def test_forward_matches_dequantized_reference(self):
    y, gate_out = jax.jit(fp8_roe_experts.experts_fwd)(*self.args)
    rows = _dq(self.x_q)[self.tok]
    o = self._expert_rows(
        rows, _dq(self.w13_q), _dq(self.w2_q), self.scores, quantize_hidden=True
    )
    want = jnp.zeros((_N, _D)).at[self.tok].add(
        o.astype(jnp.bfloat16).astype(jnp.float32)
    )
    self.assertEqual(y.dtype, jnp.bfloat16)
    self.assertEqual(gate_out.shape, (self.args[1].shape[0], 2 * _F))
    self.assertLess(_rel(y.reshape(_N, _D), want), 0.05)
    untouched = np.setdiff1d(np.arange(_N), self.tok)
    if untouched.size:
      self.assertEqual(float(jnp.abs(y.reshape(_N, _D)[untouched]).max()), 0.0)

  def test_backward_is_close_to_fp32(self):
    _, gate_out = jax.jit(fp8_roe_experts.experts_fwd)(*self.args)
    dy3 = self.dy.astype(jnp.bfloat16).reshape(_N, _D // 128, 128)
    dx, dscores, dw13, dw2 = jax.jit(fp8_roe_experts.experts_bwd)(
        dy3, *self.args, gate_out
    )

    def loss(rows, w13, w2, scores):
      o = self._expert_rows(rows, w13, w2, scores, quantize_hidden=False)
      return jnp.sum(o * self.dy[self.tok])

    want = jax.grad(loss, argnums=(0, 1, 2, 3))(
        _dq(self.x_q)[self.tok], _dq(self.w13_q), _dq(self.w2_q), self.scores
    )
    nv = self.nv
    self.assertLess(_rel(dx.reshape(-1, _D)[:nv], want[0]), 0.1)
    self.assertLess(_rel(dw13, want[1]), 0.1)
    self.assertLess(_rel(dw2, want[2]), 0.1)
    self.assertLess(_rel(dscores[:nv], want[3][:nv]), 0.1)


if __name__ == "__main__":
  absltest.main()
