# Fused chunked KDA / Gated DeltaNet (`kda.py`)

One Pallas forward kernel and one backward kernel for the gated delta rule,
per head (``q``, ``k`` L2-normalized in the kernel)::

    S_t = (I - beta_t k_t k_t^T) diag(exp(g_t)) S_{t-1} + beta_t k_t v_t^T
    o_t = scale * S_t^T q_t

Each kernel scans the chunks of a row in order (backward: in reverse) with the
``[K, V]`` states of a group of heads in VMEM. Inputs stay in the model's
``[B, S, H * K]`` layout; packed documents are masked inside chunks (no padding
to chunk boundaries, no bound on documents per row). The algorithm and the
precision of every matmul are described in the module docstring. Qwen3.5's
GatedDeltaNet selects it with the `fused_gated_delta_kernel` override
(`qwen3_5/fused_gated_delta.py`).

## Entry points

| Function | Framework | Notes |
| --- | --- | --- |
| `kda(q, k, v, g, beta, *, lower_bound, ...)` | JAX | `custom_vjp`; returns `(o, final_state)` |
| `kda_torch(...)` | PyTorch on TPU (`torch_tpu` `jax_op`) | same arguments; autograd; no residuals saved under `no_grad` |
| `kda_fused_fwd` / `kda_fused_bwd` | JAX | kernel layout `[B, S, H * K]`, `S % chunk == 0`; for callers that manage residuals |

Example (Gated DeltaNet, one decay per head, chunked prefill)::

    o, state = kda.kda(q, k, v, g_BSH, beta, lower_bound=-8.0,
                       segment_ids=None, initial_state=prev_state,
                       output_final_state=True)

## Modes

| Mode | Status |
| --- | --- |
| Per-channel log decay `g [B, S, H, K]` (KDA) | supported |
| One log decay per head `g [B, S, H]` (Gated DeltaNet) | supported; broadcast to `[B, S, H, K]` fp32 in the wrapper (extra HBM traffic) |
| K3 / Kimi gate in-kernel: `g = lower_bound * sigmoid(exp(a_log) * (raw + dt_bias))` | supported (`a_log`, `dt_bias`); grads for both |
| Packed documents (`segment_ids [B, S]`, any count, any lengths incl. 1) | supported |
| Non-packed rows (`segment_ids=None`) | supported |
| `initial_state [B, H, K, V]` (enters each row's first document) | supported, differentiable |
| `final_state` output (state after each row's last token) | supported, differentiable |
| Forward only, no residuals (inference) | supported (`kda_torch` under `no_grad`; `kda` without `jax.vjp`) |
| `S` not a multiple of the chunk | supported (tail padded in the wrapper; pad tokens have decay 1) |
| Head dims `K, V <= 256` | supported; dims padded to multiples of 128 in the wrapper (64 and 192 pay the padding) |
| Chunk sizes | 16, 32, 64 (default), 128 |
| Grouped q / k heads (`q, k [B, S, Hk, K]`, `H % Hk == 0`; value head `h` reads key head `h // (H / Hk)`, as Qwen3.5 GDN) | supported; repeated in the wrapper and dq / dk summed per group (in-kernel sharing would save the repeat traffic) |
| Per-document initial / final states in a packed row | missing: lay one document per row |
| Context parallel | missing in the kernel: compose with `initial_state` / `final_state` across ranks |

## Shapes and dtypes

- `q`, `k`: `[B, S, Hk, K]` (`Hk` divides `H`); `v`: `[B, S, H, V]`; bf16 or fp32 (all equal).
  The MXU runs bf16 operands with fp32 accumulation; fp32 inputs are not
  computed in full fp32.
- `beta`: `[B, S, H]`, used in fp32. `a_log`: `[H]`, `dt_bias`: `[H, K]`, fp32.
- `lower_bound` in `[-8, 0)`: the gate's lower bound, or the clamp floor of a
  log-decay `g` (`g < lower_bound` is computed as `lower_bound` and gets no
  gradient). The in-chunk decay factorization needs `|g| <= 8` per token: a
  token with `g = -8` keeps `exp(-8) = 3.4e-4` of the state instead of less.
- Output in `q`'s dtype; states fp32.

## Precision

Measured on TPU7x vs an fp64 token recurrence on the same rounded inputs
(relative L2):

- typical inputs: output ~1.7e-3, every gradient 2-4e-3 (Tokamax Mosaic: output
  4.6e-3, gate grads 6e-2 to 0.3).
- error is flat in sequence length (512 to 64K tokens in one document).
- worst case, `beta ~ 1` with aligned keys and no decay (the WY inverse has
  entries near +-1 and the recurrence cancels): output 5-7e-3, grads up to
  2.7e-2 (Tokamax xla_chunked 2.8e-2, mosaic 5.8e-2). This regime is
  ill-conditioned: a 2^-9 relative perturbation of `v` moves the exact output
  by ~2e-3.
- fp32 inputs get the same (bf16-operand) precision as bf16 inputs.
- chunk 128: gate gradients ~2x less accurate than chunk 64 (6e-3).
- a gate gradient that is exactly zero (e.g. every document of length 1)
  comes out as rounding noise (~1e-5 of a typical gate gradient).

## Known limits

- Head dims above 256 are rejected (VMEM).
- The backward saves the per-chunk input states in bf16 and the WY inverse in
  fp32: `B * S / C * H * (K * V * 2 + C * C * 4)` bytes.
- Tests: `kernels/tests/test_kda.py`.
