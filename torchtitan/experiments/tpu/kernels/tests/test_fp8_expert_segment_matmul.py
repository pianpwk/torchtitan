"""Accelerator tests for the FP8 expert grouped matmuls."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.kernels import fp8_expert_segment_matmul as fp8_esm

_TILING = (128, 0, 0)
_TILINGS = dict(fwd_tiling=_TILING, dlhs_tiling=_TILING, drhs_tiling=_TILING)
_GROUP_SIZES = (0, 103, 17, 136, 0, 256)


def _dequantized(x: torch.Tensor, dims, qdtype=torch.float8_e4m3fn):
    """``x`` through absmax FP8 quantization over ``dims``, in FP32 on CPU."""
    x = x.detach().float().cpu()
    amax = x.abs().amax(dim=dims, keepdim=True)
    scale = (
        torch.where(amax > 0, amax, torch.ones_like(amax))
        / torch.finfo(qdtype).max
    )
    return (x / scale).to(qdtype).float() * scale


def _static_dequantized(x: torch.Tensor) -> torch.Tensor:
    """``x`` through MaxText's static ``fixed,-224,224`` e4m3 quantization."""
    scale = fp8_esm.STATIC_SCALE
    fmax = torch.finfo(torch.float8_e4m3fn).max
    x = x.detach().float().cpu()
    return (x / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn).float() * scale


def _grouped_reference(lhs, rhs, group_sizes):
    """FP32 ``lhs[rows of g] @ rhs[g]`` for every group ``g``."""
    out, start = [], 0
    for g, size in enumerate(group_sizes):
        out.append(lhs[start : start + size] @ rhs[g])
        start += size
    return torch.cat(out)


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.detach().float().cpu(), want.detach().float().cpu()
    return ((got - want).norm() / want.norm()).item()


def _to_torch(x: jax.Array) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x.astype(jnp.float32)))


class Fp8GmmTest(base_device_test.BaseAcceleratorDeviceTest):
    """The JAX-level GEMMs, run by JAX directly."""

    def _operands(self, k=256, n=256):
        keys = jax.random.split(jax.random.key(0), 3)
        m = sum(_GROUP_SIZES)
        lhs = jax.random.normal(keys[0], (m, k), jnp.bfloat16)
        rhs = jax.random.normal(
            keys[1], (len(_GROUP_SIZES), k, n), jnp.bfloat16
        ) * jnp.bfloat16(0.05)
        dout = jax.random.normal(keys[2], (m, n), jnp.bfloat16)
        sizes = jnp.array(_GROUP_SIZES, jnp.int32)
        return lhs, rhs, dout, sizes

    def test_fwd_matches_dequantized_reference(self):
        lhs, rhs, _, sizes = self._operands()
        out = fp8_esm.fp8_gmm_fwd(lhs, rhs, sizes, _TILING)

        want = _grouped_reference(
            _dequantized(_to_torch(lhs), dims=1),
            _dequantized(_to_torch(rhs), dims=(0, 1)),
            _GROUP_SIZES,
        )
        self.assertEqual(out.dtype, jnp.bfloat16)
        self.assertLess(_rel_err(_to_torch(out), want), 5e-3)

    def test_bwd_is_close_to_f32(self):
        lhs, rhs, dout, sizes = self._operands()
        dlhs, drhs = fp8_esm.fp8_gmm_bwd(
            dout, lhs, rhs, sizes, _TILING, _TILING
        )

        lhs_r = _to_torch(lhs).requires_grad_()
        rhs_r = _to_torch(rhs).requires_grad_()
        want = _grouped_reference(lhs_r, rhs_r, _GROUP_SIZES)
        want_dlhs, want_drhs = torch.autograd.grad(
            want, (lhs_r, rhs_r), _to_torch(dout)
        )
        self.assertEqual(drhs.dtype, jnp.bfloat16)
        self.assertLess(_rel_err(_to_torch(dlhs), want_dlhs), 0.1)
        self.assertLess(_rel_err(_to_torch(drhs), want_drhs), 0.1)
        # Experts that received no rows get exactly zero weight gradient.
        for g, size in enumerate(_GROUP_SIZES):
            if size == 0:
                self.assertEqual(float(jnp.abs(drhs[g]).max()), 0.0)


def _bf16_ulp_distance(a, b):
    """Elementwise distance in BF16 ulps (ordered sign-magnitude bits)."""
    def ordered(x):
        bits = np.asarray(x).view(np.uint16).astype(np.int32)
        return np.where(bits & 0x8000, -(bits & 0x7FFF), bits)
    return np.abs(ordered(a) - ordered(b))


class Fp8SwigluKernelTest(base_device_test.BaseAcceleratorDeviceTest):
    """The SwiGLU Pallas kernels, run by JAX directly."""

    # Not a multiple of the 256-row tile, so the last tile is partial.
    _ROWS, _HIDDEN = 600, 256

    def _gate_up(self):
        key = jax.random.key(2)
        return jax.random.normal(
            key, (self._ROWS, 2 * self._HIDDEN), jnp.bfloat16
        )

    def test_swiglu_quantize_matches_xla(self):
        gate_up = self._gate_up()
        q, scale = fp8_esm.swiglu_quantize(gate_up, fused=True)
        want_q, want_scale = fp8_esm.swiglu_quantize(gate_up, fused=False)
        self.assertEqual(q.dtype, jnp.float8_e4m3fn)
        self.assertEqual(scale.shape, (self._ROWS, 1))
        np.testing.assert_allclose(
            np.asarray(scale), np.asarray(want_scale), rtol=1e-6
        )
        # The FP32 -> FP8 rounding may differ by one FP8 ulp per element.
        got = _to_torch(q) * _to_torch(scale)
        want = _to_torch(want_q) * _to_torch(want_scale)
        self.assertLess(_rel_err(got, want), 1e-2)

        gate, up = _to_torch(gate_up).chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        self.assertLess(_rel_err(got, _dequantized(hidden, dims=1)), 1e-2)

    def test_swiglu_grad_matches_autodiff(self):
        gate_up = self._gate_up()
        d_hidden = jax.random.normal(
            jax.random.key(3), (self._ROWS, self._HIDDEN), jnp.bfloat16
        )
        got = fp8_esm.swiglu_grad(gate_up, d_hidden, fused=True)
        unfused = fp8_esm.swiglu_grad(gate_up, d_hidden, fused=False)
        self.assertEqual(got.dtype, jnp.bfloat16)
        self.assertEqual(got.shape, gate_up.shape)
        # Mosaic and XLA lower sigmoid differently (equal on JAX 0.11.1, up to
        # 3 FP32 ulp apart on 0.12), so a result on a BF16 rounding tie can
        # land on the neighboring value.
        self.assertLessEqual(_bf16_ulp_distance(got, unfused).max(), 1)

        gu_r = _to_torch(gate_up).requires_grad_()
        g_r, u_r = gu_r.chunk(2, dim=-1)
        (want,) = torch.autograd.grad(
            F.silu(g_r) * u_r, (gu_r,), _to_torch(d_hidden)
        )
        self.assertLess(_rel_err(_to_torch(got), want), 1e-2)


class Fp8ExpertSegmentMatmulTest(base_device_test.BaseAcceleratorDeviceTest):

    def _operands(self, k=256, n=256, group_sizes=_GROUP_SIZES):
        torch.manual_seed(0)
        m = sum(group_sizes)
        lhs = torch.randn(m, k, dtype=torch.bfloat16)
        rhs = torch.randn(len(group_sizes), k, n, dtype=torch.bfloat16) * 0.05
        dout = torch.randn(m, n, dtype=torch.bfloat16)
        sizes = torch.tensor(group_sizes, dtype=torch.int32)
        return lhs, rhs, dout, sizes

    def _to_dev(self, *tensors, grad=False):
        return [
            t.to(self.accelerator_device).requires_grad_(grad) for t in tensors
        ]

    def test_forward_matches_dequantized_reference(self):
        lhs, rhs, _, sizes = self._operands()
        lhs_d, rhs_d, sizes_d = self._to_dev(lhs, rhs, sizes)
        out = fp8_esm.fp8_segment_matmul_pallas_2d(
            lhs_d, rhs_d, sizes_d, **_TILINGS
        ).cpu()

        # lhs is scaled per row; rhs per output column, shared by all experts.
        want = _grouped_reference(
            _dequantized(lhs, dims=1),
            _dequantized(rhs, dims=(0, 1)),
            _GROUP_SIZES,
        )
        self.assertEqual(out.dtype, torch.bfloat16)
        # XLA's FP32 -> FP8 convert may round a few elements one FP8 ulp
        # differently from torch's, so compare in norm. How many differ
        # varies across TPU generations, so use the 1e-2 bound of the other
        # forward checks here.
        self.assertLess(_rel_err(out, want), 1e-2)

    def test_backward_is_close_to_bf16(self):
        lhs, rhs, dout, sizes = self._operands()
        lhs_d, rhs_d = self._to_dev(lhs, rhs, grad=True)
        (sizes_d,) = self._to_dev(sizes)
        out = fp8_esm.fp8_segment_matmul_pallas_2d(
            lhs_d, rhs_d, sizes_d, **_TILINGS
        )
        dlhs, drhs = torch.autograd.grad(
            out, (lhs_d, rhs_d), dout.to(self.accelerator_device)
        )

        lhs_r, rhs_r = (
            lhs.float().requires_grad_(),
            rhs.float().requires_grad_(),
        )
        want = _grouped_reference(lhs_r, rhs_r, _GROUP_SIZES)
        want_dlhs, want_drhs = torch.autograd.grad(
            want, (lhs_r, rhs_r), dout.float()
        )

        self.assertEqual(drhs.dtype, torch.bfloat16)
        self.assertLess(_rel_err(dlhs, want_dlhs), 0.1)
        self.assertLess(_rel_err(drhs, want_drhs), 0.1)
        # Experts that received no rows get exactly zero weight gradient.
        drhs = drhs.cpu()
        for g, size in enumerate(_GROUP_SIZES):
            if size == 0:
                torch.testing.assert_close(drhs[g], torch.zeros_like(drhs[g]))

    def test_tiny_gradient_rows_stay_finite(self):
        # A dout row whose (dout * rhs_scale) amax / 57344 is subnormal in FP32
        # gets a zero scale on TPU (flush-to-zero) unless the scale is floored;
        # the row then quantizes to inf/NaN and, through dW's per-column
        # scales, poisons every expert's weight gradient.
        lhs, rhs, dout, sizes = self._operands()
        dout[5] = dout[5] * 1e-32
        lhs_d, rhs_d = self._to_dev(lhs, rhs, grad=True)
        (sizes_d,) = self._to_dev(sizes)
        out = fp8_esm.fp8_segment_matmul_pallas_2d(
            lhs_d, rhs_d, sizes_d, **_TILINGS
        )
        dlhs, drhs = torch.autograd.grad(
            out, (lhs_d, rhs_d), dout.to(self.accelerator_device)
        )
        self.assertTrue(torch.isfinite(dlhs.float()).all())
        self.assertTrue(torch.isfinite(drhs.float()).all())

        lhs_r, rhs_r = (
            lhs.float().requires_grad_(),
            rhs.float().requires_grad_(),
        )
        want = _grouped_reference(lhs_r, rhs_r, _GROUP_SIZES)
        want_drhs = torch.autograd.grad(want, (rhs_r,), dout.float())[0]
        self.assertLess(_rel_err(drhs, want_drhs), 0.1)

    def test_paired_matches_unpaired(self):
        lhs, rhs_a, dout_a, sizes = self._operands()
        _, rhs_b, dout_b, _ = self._operands(n=256)
        rhs_b = rhs_b.flip(0) * 0.5
        dout_b = dout_b.flip(0)
        dev = self.accelerator_device

        def unpaired():
            x, a, b = self._to_dev(lhs, rhs_a, rhs_b, grad=True)
            out_a = fp8_esm.fp8_segment_matmul_pallas_2d(
                x, a, sizes.to(dev), **_TILINGS
            )
            out_b = fp8_esm.fp8_segment_matmul_pallas_2d(
                x, b, sizes.to(dev), **_TILINGS
            )
            grads = torch.autograd.grad(
                (out_a, out_b), (x, a, b), (dout_a.to(dev), dout_b.to(dev))
            )
            return (out_a, out_b), grads

        def paired():
            x, a, b = self._to_dev(lhs, rhs_a, rhs_b, grad=True)
            out_a, out_b = fp8_esm.fp8_paired_segment_matmul_pallas_2d(
                x, a, b, sizes.to(dev), **_TILINGS
            )
            grads = torch.autograd.grad(
                (out_a, out_b), (x, a, b), (dout_a.to(dev), dout_b.to(dev))
            )
            return (out_a, out_b), grads

        # Column scales are per column, so concatenating [a | b] leaves every
        # forward operand unchanged. dInput is one GEMM over 2N instead of two
        # GEMMs and a sum, so it only matches to rounding.
        (pa, pb), (pdx, pda, pdb) = paired()
        (ua, ub), (udx, uda, udb) = unpaired()
        torch.testing.assert_close(pa.cpu(), ua.cpu())
        torch.testing.assert_close(pb.cpu(), ub.cpu())
        torch.testing.assert_close(pda.cpu(), uda.cpu())
        torch.testing.assert_close(pdb.cpu(), udb.cpu())
        # dY is quantized per row over [a | b] instead of per half.
        self.assertLess(_rel_err(pdx, udx), 0.06)

    def test_paired_unsplit_is_concatenation(self):
        lhs, rhs, _, sizes = self._operands()
        dev = self.accelerator_device
        x, a, b = self._to_dev(lhs, rhs, rhs.flip(0))
        split = fp8_esm.fp8_paired_segment_matmul_pallas_2d(
            x, a, b, sizes.to(dev), **_TILINGS
        )
        joint = fp8_esm.fp8_paired_segment_matmul_pallas_2d(
            x, a, b, sizes.to(dev), split=False, **_TILINGS
        )
        torch.testing.assert_close(
            joint.cpu(), torch.cat(split, dim=-1).cpu(), rtol=0, atol=0
        )


    @parameterized.parameters(True, False)
    def test_swiglu_matches_separate_ops(self, fused_quant):
        torch.manual_seed(1)
        group_sizes = _GROUP_SIZES
        m, f, n = sum(group_sizes), 256, 256
        gate_up = torch.randn(m, 2 * f, dtype=torch.bfloat16)
        w2 = torch.randn(len(group_sizes), f, n, dtype=torch.bfloat16) * 0.05
        dout = torch.randn(m, n, dtype=torch.bfloat16)
        sizes = torch.tensor(group_sizes, dtype=torch.int32)
        dev = self.accelerator_device

        gu, w = self._to_dev(gate_up, w2, grad=True)
        out = fp8_esm.fp8_swiglu_segment_matmul_pallas_2d(
            gu, w, sizes.to(dev), fused_quant=fused_quant, **_TILINGS
        )
        d_gu, d_w = torch.autograd.grad(out, (gu, w), dout.to(dev))

        # The op quantizes silu(gate) * up from FP32 instead of from BF16, so
        # compare against an FP32 reference of the hidden.
        gate, up = gate_up.float().chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        want = _grouped_reference(
            _dequantized(hidden, dims=1),
            _dequantized(w2, dims=(0, 1)),
            group_sizes,
        )
        self.assertLess(_rel_err(out, want), 1e-2)

        gu_r = gate_up.float().requires_grad_()
        w_r = w2.float().requires_grad_()
        g_r, u_r = gu_r.chunk(2, dim=-1)
        want_out = _grouped_reference(F.silu(g_r) * u_r, w_r, group_sizes)
        want_dgu, want_dw = torch.autograd.grad(
            want_out, (gu_r, w_r), dout.float()
        )
        self.assertLess(_rel_err(d_gu, want_dgu), 0.1)
        self.assertLess(_rel_err(d_w, want_dw), 0.1)

    def _qrhs(self, rhs):
        """``rhs`` quantized on device, and a leaf that receives its gradient."""
        leaf = rhs.to(self.accelerator_device).requires_grad_()
        return fp8_esm.fp8_quantize_expert_weight(leaf.detach()), leaf

    def _qrhs_mm(self, lhs, q, leaf, sizes):
        return fp8_esm.fp8_qrhs_segment_matmul(
            lhs,
            q,
            sizes,
            weight_inputs=(leaf,),
            reduce_grad=lambda grads: grads,
            **_TILINGS,
        )

    def test_quantize_expert_weight_is_static(self):
        _, rhs, _, _ = self._operands()
        q = self._qrhs(rhs)[0]
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        got = q.cpu().float() * fp8_esm.STATIC_SCALE
        self.assertLess(_rel_err(got, _static_dequantized(rhs)), 1e-2)

    def test_qrhs_forward_matches_dequantized_reference(self):
        lhs, rhs, _, sizes = self._operands()
        q, leaf = self._qrhs(rhs)
        lhs_d, sizes_d = self._to_dev(lhs, sizes)
        out = self._qrhs_mm(lhs_d, q, leaf, sizes_d)
        want = _grouped_reference(
            _static_dequantized(lhs), _static_dequantized(rhs), _GROUP_SIZES
        )
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertLess(_rel_err(out, want), 1e-2)

    def test_qrhs_backward_is_close_to_bf16(self):
        lhs, rhs, dout, sizes = self._operands()
        q, leaf = self._qrhs(rhs)
        (lhs_d,) = self._to_dev(lhs, grad=True)
        (sizes_d,) = self._to_dev(sizes)
        out = self._qrhs_mm(lhs_d, q, leaf, sizes_d)
        dlhs, drhs = torch.autograd.grad(
            out, (lhs_d, leaf), dout.to(self.accelerator_device)
        )
        lhs_r = lhs.float().requires_grad_()
        rhs_r = rhs.float().requires_grad_()
        want = _grouped_reference(lhs_r, rhs_r, _GROUP_SIZES)
        want_dlhs, want_drhs = torch.autograd.grad(
            want, (lhs_r, rhs_r), dout.float()
        )
        self.assertEqual(drhs.dtype, torch.bfloat16)
        self.assertEqual(drhs.shape, rhs.shape)
        self.assertLess(_rel_err(dlhs, want_dlhs), 0.1)
        self.assertLess(_rel_err(drhs, want_drhs), 0.1)
        drhs = drhs.cpu()
        for g, size in enumerate(_GROUP_SIZES):
            if size == 0:
                torch.testing.assert_close(drhs[g], torch.zeros_like(drhs[g]))

    def test_qrhs_ignores_rows_outside_groups(self):
        # Ring-of-Experts leaves rows past sum(group_sizes) unspecified. They
        # must not reach dW, whose quantization takes a per-column amax over
        # every row.
        lhs, rhs, dout, sizes = self._operands()
        q, leaf = self._qrhs(rhs)
        pad = 128
        lhs_pad = torch.cat([lhs, torch.full((pad, lhs.shape[1]), float("inf"))]).bfloat16()
        dout_pad = torch.cat([dout, torch.full((pad, dout.shape[1]), float("nan"))]).bfloat16()
        (lhs_d,) = self._to_dev(lhs_pad, grad=True)
        out = self._qrhs_mm(lhs_d, q, leaf, sizes.to(self.accelerator_device))
        dlhs, drhs = torch.autograd.grad(
            out, (lhs_d, leaf), dout_pad.to(self.accelerator_device)
        )
        self.assertTrue(torch.isfinite(drhs.float()).all())
        self.assertTrue(torch.isfinite(dlhs[: lhs.shape[0]].float()).all())
        self.assertTrue(torch.isfinite(out[: lhs.shape[0]].float()).all())

    def test_unsharded_expert_weight_gather_routes_gradients(self):
        from torchtitan.experiments.tpu.graph_trainer.simple_fsdp import gather_fp8_expert_weights  # pylint: disable=g-import-not-at-top

        lhs, rhs_a, dout, sizes = self._operands()
        rhs_b = rhs_a.flip(0) * 0.5
        module = torch.nn.Module()
        module.a = torch.nn.Parameter(rhs_a.float().to(self.accelerator_device))
        module.b = torch.nn.Parameter(rhs_b.float().to(self.accelerator_device))
        weights = gather_fp8_expert_weights(module, ("a", "b"))
        self.assertEqual(tuple(weights.weight_q.shape), (len(_GROUP_SIZES), 256, 512))
        (lhs_d,) = self._to_dev(lhs)
        out = fp8_esm.fp8_qrhs_segment_matmul(
            lhs_d,
            weights.weight_q,
            sizes.to(self.accelerator_device),
            weight_inputs=weights.shards,
            reduce_grad=weights.reduce_grad,
            **_TILINGS,
        )
        dout2 = torch.cat([dout, dout.flip(0)], dim=-1)
        out.backward(dout2.to(self.accelerator_device))
        rhs_r = torch.cat([rhs_a, rhs_b], dim=-1).float().requires_grad_()
        want = _grouped_reference(lhs.float(), rhs_r, _GROUP_SIZES)
        (want_d,) = torch.autograd.grad(want, (rhs_r,), dout2.float())
        self.assertEqual(module.a.grad.dtype, torch.float32)
        self.assertLess(_rel_err(module.a.grad, want_d[..., :256]), 0.1)
        self.assertLess(_rel_err(module.b.grad, want_d[..., 256:]), 0.1)


if __name__ == "__main__":
    absltest.main()
