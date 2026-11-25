"""Accelerator tests for the Tokamax gmm_v2 / tgmm_v2 expert grouped matmul.

The reference is the per-expert matmul in FP64 on the same BF16 inputs; the
kernel must stay within 2x the eager BF16 path's error of it plus the BF16
rounding floor, forward and both gradients.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch

from torchtitan.experiments.tpu.kernels import gmm_v2_segment_matmul
from torchtitan.experiments.tpu.testing import base_device_test


def _per_expert(lhs, rhs_EKN, sizes, dtype):
    out, start = torch.zeros(lhs.shape[0], rhs_EKN.shape[2], dtype=dtype), 0
    for e, size in enumerate(sizes):
        rows = slice(start, start + size)
        out[rows] = lhs[rows].to(dtype) @ rhs_EKN[e].to(dtype)
        start += size
    return out


class GmmV2SegmentMatmulTest(base_device_test.BaseAcceleratorDeviceTest):
    def setUp(self):
        super().setUp()
        if self.accelerator_device.type != "tpu":
            self.skipTest("Tokamax gmm_v2 kernels are TPU-only.")

    @parameterized.parameters(False, True)
    def test_matches_reference(self, transpose_rhs):
        # An empty expert, a size-1 group, rows past the groups, and a row
        # count that is not a multiple of the BF16 sublane tiling (16).
        sizes, rows, k, n = (180, 0, 300, 45, 1), 600 + 8, 256, 384
        used = sum(sizes)
        g = torch.Generator().manual_seed(0)
        lhs = torch.randn(rows, k, generator=g).bfloat16()
        rhs_EKN = (torch.randn(len(sizes), k, n, generator=g) / 16).bfloat16()
        dy = torch.randn(rows, n, generator=g).bfloat16()
        dy[used:] = 0

        def grads(dtype):
            leaves = [
                p.detach().clone().requires_grad_() for p in (lhs, rhs_EKN)
            ]
            out = _per_expert(*leaves, sizes, dtype)
            out.backward(dy.to(dtype))
            return [out.detach().double()] + [p.grad.double() for p in leaves]

        golden, eager = grads(torch.float64), grads(torch.bfloat16)

        device = self.accelerator_device
        lhs_d = lhs.clone()
        lhs_d[used:] = float("nan")  # Rows past the groups must not be read.
        lhs_d = lhs_d.to(device).requires_grad_()
        rhs_d = (
            (
                rhs_EKN.transpose(-2, -1).contiguous()
                if transpose_rhs
                else rhs_EKN
            )
            .to(device)
            .requires_grad_()
        )
        out = gmm_v2_segment_matmul.grouped_mm_v2(
            lhs_d,
            rhs_d,
            torch.tensor(sizes, dtype=torch.int32, device=device),
            transpose_rhs,
        )
        out.backward(dy.to(device))
        grad_rhs = rhs_d.grad.cpu()
        if transpose_rhs:
            grad_rhs = grad_rhs.transpose(-2, -1)
        target = [
            out.detach().cpu().double(),
            lhs_d.grad.cpu().double(),
            grad_rhs.double(),
        ]
        self.assertEqual(rhs_d.grad.dtype, torch.bfloat16)
        for name, gold, ref, tgt in zip(
            ("y", "dx", "dw"), golden, eager, target
        ):
            if name in ("y", "dx"):
                self.assertTrue(
                    torch.all(tgt[used:] == 0), f"{name} rows past groups"
                )
                gold, ref, tgt = gold[:used], ref[:used], tgt[:used]
            self.assertTrue(torch.isfinite(tgt).all(), name)
            ref_err = (ref - gold).abs().max().item()
            tgt_err = (tgt - gold).abs().max().item()
            floor = (gold.bfloat16().double() - gold).abs().max().item()
            self.assertLessEqual(tgt_err, 2 * ref_err + floor, name)


if __name__ == "__main__":
    absltest.main()
