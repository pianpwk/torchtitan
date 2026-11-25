"""Accelerator tests for the fused grouped GLU matmul (kernels/gmm_glu.py).

The reference is the unfused path (per-expert W1 and W3 matmuls rounded to the
dtype, the activation in FP32) in FP64 on the same BF16 inputs; the kernel
must stay within 2x the eager BF16 path's error of it plus the BF16 rounding
floor, forward and every gradient.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu.kernels import gmm_glu
from torchtitan.experiments.tpu.testing import base_device_test


def _torch_act(act, gate, up):
    """``act`` on torch tensors, as upstream ``kimi_k3.moe._situ_glu``."""
    if isinstance(act, gmm_glu.SwiGLU):
        return F.silu(gate) * up
    gate = act.beta * torch.tanh(gate / act.beta) * torch.sigmoid(gate)
    if act.linear_beta is not None:
        up = act.linear_beta * torch.tanh(up / act.linear_beta)
    return gate * up


def _unfused(x, wg, wu, sizes, dtype, act=gmm_glu.SwiGLU()):
    out, start = torch.zeros(x.shape[0], wg.shape[2], dtype=dtype), 0
    for e, size in enumerate(sizes):
        rows = slice(start, start + size)
        h1 = x[rows].to(dtype) @ wg[e].to(dtype)
        h3 = x[rows].to(dtype) @ wu[e].to(dtype)
        compute = torch.float64 if dtype == torch.float64 else torch.float32
        out[rows] = _torch_act(act, h1.to(compute), h3.to(compute)).to(dtype)
        start += size
    return out


# K3's SiTU-GLU (beta 4, linear_beta 25) and the variant without linear_beta.
_SITU_K3 = gmm_glu.SiTUGLU(4.0, 25.0)
_SITU = gmm_glu.SiTUGLU(4.0)


class GmmGluTest(base_device_test.BaseAcceleratorDeviceTest):
    def setUp(self):
        super().setUp()
        if self.accelerator_device.type != "tpu":
            self.skipTest("Pallas gmm_glu kernels are TPU-only.")

    @parameterized.parameters(
        # empty expert, size-1 group, padding
        ((180, 0, 300, 45, 1), 600, gmm_glu.SwiGLU()),
        ((180, 0, 300, 45, 1), 600, _SITU_K3),
        ((180, 0, 300, 45, 1), 600, _SITU),
        ((0, 0, 777, 0), 777, gmm_glu.SwiGLU()),  # every row on one expert
        ((0, 0, 777, 0), 777, _SITU_K3),
        ((0, 0, 0), 64, _SITU_K3),  # no routed rows
    )
    def test_matches_reference(self, sizes, rows, act):
        g = torch.Generator().manual_seed(0)
        e, k, n = len(sizes), 256, 256
        used = sum(sizes)
        x = torch.randn(rows, k, generator=g).bfloat16()
        wg, wu = (
            (torch.randn(e, k, n, generator=g) / 16).bfloat16() for _ in range(2)
        )
        dy = torch.randn(rows, n, generator=g).bfloat16()
        dy[used:] = 0

        def grads(dtype):
            leaves = [p.detach().clone().requires_grad_() for p in (x, wg, wu)]
            out = _unfused(*leaves, sizes, dtype, act)
            out.backward(dy.to(dtype))
            return [out.detach().double()] + [p.grad.double() for p in leaves]

        golden, eager = grads(torch.float64), grads(torch.bfloat16)

        device = self.accelerator_device
        x_d = x.clone()
        x_d[used:] = float("nan")  # Rows past the groups must not be read.
        leaves = [p.to(device).requires_grad_() for p in (x_d, wg, wu)]
        out = gmm_glu.grouped_glu(
            *leaves,
            torch.tensor(sizes, dtype=torch.int32, device=device),
            act=act,
        )
        out.backward(dy.to(device))
        target = [out.detach().cpu().double()] + [
            p.grad.cpu().double() for p in leaves
        ]
        for name, gold, ref, tgt in zip(
            ("y", "dx", "dwg", "dwu"), golden, eager, target
        ):
            if name in ("y", "dx"):
                self.assertTrue(
                    torch.all(tgt[used:] == 0), f"{name} rows past groups"
                )
                gold, ref, tgt = gold[:used], ref[:used], tgt[:used]
            if gold.numel() == 0:
                continue
            self.assertTrue(torch.isfinite(tgt).all(), name)
            ref_err = (ref - gold).abs().max().item()
            tgt_err = (tgt - gold).abs().max().item()
            floor = (gold.bfloat16().double() - gold).abs().max().item()
            self.assertLessEqual(tgt_err, 2 * ref_err + floor, name)

    def test_bad_group_sizes_stay_in_bounds(self):
        """Negative sizes and an oversubscribed total read and write no row
        past lhs: the result is the capacity-truncated product."""
        device = self.accelerator_device
        g = torch.Generator().manual_seed(1)
        x = torch.randn(512, 128, generator=g).bfloat16()
        wg, wu = (
            (torch.randn(3, 128, 128, generator=g) / 12).bfloat16()
            for _ in range(2)
        )
        out = gmm_glu.grouped_glu(
            x.to(device),
            wg.to(device),
            wu.to(device),
            torch.tensor([300, -5, 10**6], dtype=torch.int32, device=device),
            act=gmm_glu.SwiGLU(),
        )
        expected = _unfused(x, wg, wu, (300, 0, 212), torch.float32)
        torch.testing.assert_close(
            out.cpu().float(), expected, rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    absltest.main()
