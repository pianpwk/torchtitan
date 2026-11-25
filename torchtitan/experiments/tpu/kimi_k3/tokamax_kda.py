"""Kimi K3's Kimi Delta Attention (KDA) on TPU with Tokamax's Mosaic kernels.

Upstream ``torchtitan.models.kimi_k3.kda`` runs the gate, the Q/K L2 norm, the
varlen causal ``conv1d`` and the chunked delta rule with CUDA-only
``attn_gym`` kernels. The ``tokamax_kda_attention`` override swaps every
``KDA`` layer for ``TpuKDA``, whose modules compute the same function on TPU:

* ``TpuKDA``: ``KDA`` that passes ``positions`` (instead of ``cu_seqlens``)
  down, so packed documents restart the convolution and the recurrence.
* ``TpuInnerKDA``: unflattens ``[T]`` to ``[B, S]`` rows
  (``unflatten_shim``) and runs the depthwise causal ``conv1d`` + SiLU per row
  (``qwen3_5.gated_delta.causal_conv_silu``), restarting at every document.
* ``TokamaxKDAKernel``: K3's gate and beta in PyTorch (fp32, the formula of
  ``attn_gym``'s ``bound_gate``), then Tokamax's Mosaic TPU KDA forward and
  backward kernels, which also L2-normalize Q/K in-kernel as ``attn_gym``'s
  ``l2norm`` does (fp32, ``eps=1e-6``, rounded to the input dtype).

The short convolution and the output norm are selected by config type; the
reference paths are the defaults:

* ``TpuInnerKDA.Config.conv``: ``CausalConvSilu`` (XLA, the reference) or
  ``PallasCausalConvSilu`` (with documents, the Pallas
  ``causal_conv1d.causal_conv_silu_segmented``).
* ``TpuKDA.Config.output_norm``: upstream's ``KimiRMSNormGated`` (XLA) or
  ``PallasGatedHeadRMSNorm`` (one Pallas pass per direction,
  ``kernels/rms_norm_gated.py`` with the sigmoid gate).

``tokamax_kda_attention`` keeps the reference conv and norm;
``tokamax_kda_attention_pallas_conv_norm`` selects the Pallas ones, and so does
``fused_kda.fused_kda_attention``.

Why ``KDA`` and ``InnerKDA`` are forked (subclassed with a copied forward)
rather than only the kernel swapped:

* Upstream ``KDA.forward`` accepts only ``VarlenMetadata`` and passes
  ``cu_seqlens`` down. Its shape changes with the document count, so TPU
  cannot compile it statically, and upstream does not pass ``positions`` to
  its inner modules. The TPU path needs static per-token segment ids derived
  from ``positions``, so ``TpuKDA`` overrides the forward to route
  ``positions`` down.
* Upstream ``InnerKDA`` runs one flat ``[1, T]`` stream with varlen kernels;
  Tokamax takes ``[B, S]`` rows plus segment ids. ``TpuInnerKDA`` reshapes to
  rows, runs the conv per row with document restarts and calls the kernel
  with ``segment_ids_BS``.

``qwen3_5/gated_delta.py`` forks Qwen3.5's GDN the same way
(``TpuGatedDeltaNet`` / ``TpuInnerGatedDeltaNet``).

TODO: drop this fork once upstream passes documents to the kernel / inner
attention as a static segment-id mask derived from positions; then only the
kernel (KDAKernel) / inner_attention config needs swapping.

No decay clamp: K3's gate is ``lower_bound * sigmoid(...)`` with
``lower_bound`` in ``[-5, 0)`` (``KDAKernel.Config``), so every log decay is in
``[-5, 0]`` (fp32 rounding keeps ``lower_bound * s`` >= ``lower_bound`` for
``s <= 1``), inside the Mosaic backward kernel's range (it overflows below
about -5.9; see ``qwen3_5.tokamax_gated_delta.MIN_SUPPORTED_LOG_DECAY``).

As in ``qwen3_5.tokamax_gated_delta``, the forward op returns the backward
kernels' residuals (the leaves of the ``jax.vjp`` closure) and the backward op
runs only the backward kernels on them. Packed documents use Tokamax's
``segment_ids`` / ``max_num_segments``: the kernels lay every document out from
a 64-token chunk boundary, so each row always costs
``seq_len + 63 * max_num_segments`` kernel tokens, and documents beyond
``max_num_segments`` share one sequence (in the convolution too).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
import functools

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from torchtitan.config.override import derive, override

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.experiments.tpu.components import unflatten_shim
from torchtitan.experiments.tpu.kernels import causal_conv1d
from torchtitan.experiments.tpu.kernels.rms_norm_gated import rms_norm_gated
from torchtitan.experiments.tpu.qwen3_5.gated_delta import causal_conv_silu, per_row_segment_ids
from torchtitan.experiments.tpu.qwen3_5.tokamax_gated_delta import (
    MIN_NUM_SEGMENTS,
    MIN_TOKAMAX_VERSION,
    TOKENS_PER_SEGMENT,
    has_tokamax_kda,
    supports_current_tpu,
    tokamax_kda_import_error,
)
from torchtitan.models.common.attention import segment_ids_from_positions
from torchtitan.models.kimi_k3.kda import InnerKDA, KDA, KDAKernel, KimiRMSNormGated
from torchtitan.protocols.module import Module

# Shape suffixes:
# T = packed tokens, B = rows, S = tokens per row, D = model dimension,
# C = projection channels, H = heads, K = query/key head dimension,
# V = value head dimension, W = convolution kernel width.


def _heads_first(x: jax.Array) -> jax.Array:
  """``[B, S, H, ...]`` -> ``[H, B, S, ...]`` (Tokamax KDA's layout)."""
  return jnp.moveaxis(x, 2, 0)


def _kda(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    segment_ids: jax.Array,
    max_num_segments: int,
) -> jax.Array:
  """Mosaic KDA on ``[B, S, H, *]``; ``max_num_segments == 0``: no documents."""
  # pylint: disable-next=g-import-not-at-top
  from tokamax import ops as tokamax_ops

  out, _ = tokamax_ops.kimi_delta_attention(
      _heads_first(q),
      _heads_first(k),
      _heads_first(v),
      _heads_first(g),
      _heads_first(beta),
      use_qk_l2norm=True,
      segment_ids=segment_ids if max_num_segments else None,
      max_num_segments=max_num_segments or None,
      implementation="mosaic",
  )
  return jnp.moveaxis(out, 0, 2)


def _kda_vjp(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    segment_ids: jax.Array,
    max_num_segments: int,
):
  """``(out, vjp)`` w.r.t. ``q, k, v, g, beta``."""
  return jax.vjp(
      functools.partial(
          _kda, segment_ids=segment_ids, max_num_segments=max_num_segments
      ),
      q,
      k,
      v,
      g,
      beta,
  )


def _input_spec(*tensors: torch.Tensor) -> str:
  """Encodes input shapes and dtypes as one static ``jax_op`` argument."""
  return ";".join(
      ",".join(str(d) for d in t.shape) + ":" + str(t.dtype).split(".")[-1]
      for t in tensors
  )


def _parse_input_spec(spec: str) -> list[jax.ShapeDtypeStruct]:
  structs = []
  for item in spec.split(";"):
    shape, dtype = item.split(":")
    structs.append(
        jax.ShapeDtypeStruct(
            tuple(int(d) for d in shape.split(",") if d), jnp.dtype(dtype)
        )
    )
  return structs


@pallas.jax_op("tt_kimi_k3::kda_fwd")
def _kda_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    segment_ids: jax.Array,
    max_num_segments: int,
) -> jax.Array:
  """Forward only (no residuals), for calls that need no gradient."""
  return _kda(q, k, v, g, beta, segment_ids, max_num_segments)


@pallas.jax_op("tt_kimi_k3::kda_fwd_res")
def _kda_fwd_res(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    segment_ids: jax.Array,
    max_num_segments: int,
) -> list[jax.Array]:
  """Returns ``[out, *residuals]``."""
  out, vjp = _kda_vjp(q, k, v, g, beta, segment_ids, max_num_segments)
  return [out, *jax.tree_util.tree_leaves(vjp)]


@pallas.jax_op("tt_kimi_k3::kda_bwd_res")
def _kda_bwd_res(
    residuals: Sequence[jax.Array],
    grad_out: jax.Array,
    input_spec: str,
    max_num_segments: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Runs the backward kernels on ``_kda_fwd_res``'s residuals."""
  structs = _parse_input_spec(input_spec)
  # The VJP closure's structure, traced abstractly from the same input shapes.
  _, vjp_struct = jax.eval_shape(
      functools.partial(_kda_vjp, max_num_segments=max_num_segments), *structs
  )
  leaves = jax.tree_util.tree_leaves(vjp_struct)
  expected = [(tuple(l.shape), l.dtype) for l in leaves]
  actual = [(tuple(r.shape), r.dtype) for r in residuals]
  if expected != actual:
    raise ValueError(
        "Tokamax KDA residuals do not match the VJP's structure:"
        f" expected {expected}, got {actual}"
    )
  vjp = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(vjp_struct), list(residuals)
  )
  return vjp(grad_out.astype(structs[0].dtype))


def _setup_context(ctx, inputs, output):
  *tensors, max_num_segments = inputs
  residuals = output[1:]
  ctx.mark_non_differentiable(*residuals)
  ctx.set_materialize_grads(False)
  ctx.save_for_backward(*residuals)
  ctx.input_spec = _input_spec(*tensors)
  ctx.max_num_segments = max_num_segments


def _backward(ctx, grad_outputs):
  grad_out = grad_outputs[0]
  if grad_out is None:
    return (None,) * 7
  grads = _kda_bwd_res(
      list(ctx.saved_tensors),
      grad_out.contiguous(),
      ctx.input_spec,
      ctx.max_num_segments,
  )
  return (*grads, None, None)


_kda_fwd_res.register_autograd(_backward, setup_context=_setup_context)


def tokamax_kda(
    q_BSHK: torch.Tensor,
    k_BSHK: torch.Tensor,
    v_BSHV: torch.Tensor,
    gate_BSHK: torch.Tensor,
    beta_BSH: torch.Tensor,
    segment_ids_BS: torch.Tensor | None = None,
    max_num_segments: int | None = None,
) -> torch.Tensor:
  """KDA with Tokamax's Mosaic TPU kernels.

  Args:
    q_BSHK: Un-normalized queries (bf16 or fp32).
    k_BSHK: Un-normalized keys, ``q_BSHK.dtype``.
    v_BSHV: Values, ``q_BSHK.dtype``.
    gate_BSHK: Per-channel log decay in ``[-5, 0]`` (fp32).
    beta_BSH: Delta-rule rate in ``[0, 1]`` (fp32).
    segment_ids_BS: Optional per-row document ids in ``[1,
      max_num_segments]`` (``per_row_segment_ids``); the recurrence restarts
      at every change. Without them, ``S`` must be a multiple of 64.
    max_num_segments: Static bound on ``segment_ids_BS``; required with it.

  Returns:
    ``[B, S, H, V]`` in ``q_BSHK.dtype``: L2-normalized Q/K, query scale
    ``K ** -0.5``, as ``attn_gym``'s ``chunk_kda(l2norm(q), l2norm(k), ...)``.
  """
  if q_BSHK.device.type != "tpu":
    raise ValueError(
        f"tokamax_kda runs on TPU only, got device {q_BSHK.device.type}."
    )
  if segment_ids_BS is None:
    max_num_segments = 0
    # `jax_op` takes tensors only; the kernels ignore this one.
    segment_ids_BS = q_BSHK.new_zeros((1,), dtype=torch.int32)
  elif max_num_segments is None or max_num_segments < 1:
    raise ValueError(
        "tokamax_kda: segment_ids_BS require max_num_segments >= 1, got"
        f" {max_num_segments}."
    )
  # `jax_op` reads its operands by shape alone, so they must be dense.
  inputs = [
      x.contiguous()
      for x in (q_BSHK, k_BSHK, v_BSHV, gate_BSHK, beta_BSH, segment_ids_BS)
  ]
  needs_grad = torch.is_grad_enabled() and any(
      x.requires_grad for x in inputs[:5]
  )
  if not needs_grad:
    return _kda_fwd(*inputs, max_num_segments)
  return _kda_fwd_res(*inputs, max_num_segments)[0]


class TokamaxKDAKernel(KDAKernel):
  """``KDAKernel`` on TPU: K3's gate in PyTorch, then ``tokamax_kda``."""

  @dataclass(kw_only=True, slots=True)
  class Config(KDAKernel.Config):
    # Packed documents per row kept apart (Tokamax's `max_num_segments`);
    # later documents of a row share one sequence in the convolution and the
    # recurrence. Each row costs `seq_len + 63 * max_num_segments` kernel
    # tokens. None: one document per `TOKENS_PER_SEGMENT` tokens of the row,
    # at least `MIN_NUM_SEGMENTS` (8 per 4K row: +12% kernel tokens).
    max_num_segments: int | None = None

    def __post_init__(self):
      KDAKernel.Config.__post_init__(self)
      if self.max_num_segments is not None and self.max_num_segments < 1:
        raise ValueError(
            f"max_num_segments must be positive, got {self.max_num_segments}"
        )

  def __init__(self, config: Config):
    super().__init__(config)
    # Fail at model construction rather than at the first traced forward.
    if supports_current_tpu() and not has_tokamax_kda():
      raise ImportError(
          f"TokamaxKDAKernel needs tokamax>={MIN_TOKAMAX_VERSION} (the Kimi"
          " Delta Attention op with the Mosaic TPU backend): pip install -r"
          " torchtitan/experiments/tpu/requirements.txt"
          f" ({tokamax_kda_import_error()})"
      )
    self.max_num_segments = config.max_num_segments

  def num_segments(self, seq_len: int) -> int:
    """Documents per ``seq_len``-token row kept apart."""
    if self.max_num_segments is not None:
      return min(self.max_num_segments, max(1, seq_len))
    return min(
        max(MIN_NUM_SEGMENTS, seq_len // TOKENS_PER_SEGMENT), max(1, seq_len)
    )

  def document_ids(self, segment_ids_BS: torch.Tensor) -> torch.Tensor:
    """Per-row ids for the conv and the kernel; extra documents share the last."""
    return per_row_segment_ids(
        segment_ids_BS, self.num_segments(segment_ids_BS.shape[1])
    )

  def forward(  # pyrefly: ignore [bad-override]
      self,
      q_BSHK: torch.Tensor,
      k_BSHK: torch.Tensor,
      v_BSHV: torch.Tensor,
      raw_gate_BSHK: torch.Tensor,
      raw_beta_BSH: torch.Tensor,
      A_log_H: torch.Tensor,
      dt_bias_HK: torch.Tensor,
      *,
      segment_ids_BS: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """K3's KDA on ``[B, S]`` rows; ``segment_ids_BS`` from ``document_ids``."""
    # `attn_gym.linear.kda.bound_gate`:
    # lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias)), in fp32.
    gate_BSHK = self.lower_bound * torch.sigmoid(
        A_log_H.float().exp().unsqueeze(-1)
        * (raw_gate_BSHK.float() + dt_bias_HK.float())
    )
    return tokamax_kda(
        q_BSHK,
        k_BSHK,
        v_BSHV,
        gate_BSHK,
        raw_beta_BSH.float().sigmoid(),
        segment_ids_BS,
        max_num_segments=(
            None
            if segment_ids_BS is None
            else self.num_segments(q_BSHK.shape[1])
        ),
    )


class CausalConvSilu(Module):
  """Per-row depthwise causal ``conv1d`` + SiLU, restarting at every document.

  The reference path: ``qwen3_5.gated_delta.causal_conv_silu`` in XLA.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Module.Config):
    pass

  def __init__(self, config: Config):
    del config  # No options.
    super().__init__()

  def forward(
      self,
      x_BSC: torch.Tensor,
      weight_C1W: torch.Tensor,
      segment_ids_BS: torch.Tensor | None,
  ) -> torch.Tensor:
    return causal_conv_silu(x_BSC, weight_C1W, segment_ids_BS)


class PallasCausalConvSilu(CausalConvSilu):
  """``CausalConvSilu`` on the Pallas segmented kernel when documents are given.

  K3's KDA always has packed documents, where the XLA path takes shifted
  slices (misaligned copies, several masked passes);
  ``causal_conv1d.causal_conv_silu_segmented`` runs it in one Pallas pass per
  direction. TPU7x, B=8, S=4096, C=6144, bf16, 8 documents per row: fwd
  1.68 -> 0.82 ms, bwd 4.97 -> 1.86 ms per tensor. Forward bits equal the XLA
  path; the backward (dw summation order) is not verified bitwise.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(CausalConvSilu.Config):
    pass

  def forward(
      self,
      x_BSC: torch.Tensor,
      weight_C1W: torch.Tensor,
      segment_ids_BS: torch.Tensor | None,
  ) -> torch.Tensor:
    if segment_ids_BS is None:
      return super().forward(x_BSC, weight_C1W, segment_ids_BS)
    return causal_conv1d.causal_conv_silu_segmented(
        x_BSC, weight_C1W, segment_ids_BS
    )


class PallasGatedHeadRMSNorm(KimiRMSNormGated):
  """``KimiRMSNormGated`` as one Pallas pass per direction.

  Left to XLA, the per-head RMSNorm + sigmoid gate fuses into the
  ``output_proj`` / ``output_gate`` GEMMs (57-65% of peak instead of ~77%)
  and adds fp32 ``[T, H, V]`` passes; ``kernels/rms_norm_gated.py``
  (``gate_activation="sigmoid"``) computes the same fp32 math in one pass per
  direction (same weight and eps) on the flat ``[T, H * V]`` heads, with no
  relayout. Not verified bitwise vs the XLA path (dweight summation order).
  """

  @dataclass(kw_only=True, slots=True)
  class Config(KimiRMSNormGated.Config):
    pass

  def forward(self, x_THV: torch.Tensor, gate_THV: torch.Tensor) -> torch.Tensor:
    return rms_norm_gated(
        x_THV.flatten(-2),
        gate_THV.flatten(-2),
        self.weight,
        self.eps,
        gate_activation="sigmoid",
    ).view_as(x_THV)


class TpuInnerKDA(InnerKDA):
  """``InnerKDA`` that convolves each ``[B, S]`` row on its own.

  Given ``positions``, the convolution and the recurrence also restart at
  every document packed into a row (``positions == 0``), as upstream's varlen
  kernels do at ``cu_seqlens``. ``conv`` selects the convolution.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(InnerKDA.Config):
    conv: CausalConvSilu.Config = field(default_factory=CausalConvSilu.Config)

  def __init__(self, config: Config):
    super().__init__(config)
    self.conv = config.conv.build()

  def forward(  # pyrefly: ignore [bad-override]
      self,
      query_TC: torch.Tensor,
      key_TC: torch.Tensor,
      value_TC: torch.Tensor,
      raw_gate_THK: torch.Tensor,
      raw_beta_TH: torch.Tensor,
      conv_q_weight_C1W: torch.Tensor,
      conv_k_weight_C1W: torch.Tensor,
      conv_v_weight_C1W: torch.Tensor,
      A_log_H: torch.Tensor,
      dt_bias_HK: torch.Tensor,
      *,
      positions: torch.Tensor | None,
  ) -> torch.Tensor:
    num_tokens = query_TC.shape[0]
    # TT-UNFLATTEN-SHIM: unflatten [T, ...] to [B, S, ...] for conv + kernel.
    rows = [
        unflatten_shim.unflatten_tensor(x, expected_flat_ndim=x.ndim)
        for x in (query_TC, key_TC, value_TC, raw_gate_THK, raw_beta_TH)
    ]
    if rows[0].ndim == query_TC.ndim:
      # Without set_unflatten_seq_len, the [T] token stream is one row.
      rows = [x.unsqueeze(0) for x in rows]
    query_BSC, key_BSC, value_BSC, raw_gate_BSHK, raw_beta_BSH = rows
    segment_ids_BS = None
    if positions is not None:
      if positions.numel() != num_tokens:
        raise ValueError(
            "TpuInnerKDA needs one position per token, got positions of shape"
            f" {tuple(positions.shape)} for {num_tokens} tokens."
        )
      segment_ids_BS = self.kernel.document_ids(
          segment_ids_from_positions(positions.reshape(-1)).reshape(
              query_BSC.shape[:2]
          )
      )
    q_BSHK, k_BSHK, v_BSHV = (
        self.conv(x_BSC, weight_C1W, segment_ids_BS).unflatten(
            -1, (-1, self.head_dim)
        )
        for x_BSC, weight_C1W in (
            (query_BSC, conv_q_weight_C1W),
            (key_BSC, conv_k_weight_C1W),
            (value_BSC, conv_v_weight_C1W),
        )
    )
    out_BSHV = self.kernel(
        q_BSHK,
        k_BSHK,
        v_BSHV,
        raw_gate_BSHK,
        raw_beta_BSH,
        A_log_H,
        dt_bias_HK,
        segment_ids_BS=segment_ids_BS,
    )
    # TT-UNFLATTEN-SHIM: fold [B, S, ...] back to [T, ...].
    return out_BSHV.reshape(num_tokens, *out_BSHV.shape[2:])


class TpuKDA(KDA):
  """``KDA`` that passes ``positions`` to ``TpuInnerKDA``.

  Identical to upstream ``KDA.forward`` except that packed documents come from
  ``positions`` (static ``[B, S]`` segment ids) instead of the varlen
  ``attention_masks``' ``cu_seqlens``, whose shape changes with the number of
  documents.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(KDA.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks=None,
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    del attention_masks  # Documents come from `positions`.
    if x_TD.ndim != 2:
      raise ValueError(
          f"KDA input must have shape [T, D], got {tuple(x_TD.shape)}."
      )
    raw_beta_TH = self.beta(x_TD)
    num_tokens = x_TD.shape[0]
    raw_gate_THK = self.forget_b(self.forget_a(x_TD)).reshape(
        num_tokens, self.num_heads, self.head_dim
    )
    out_THV = self.inner_kda(
        self.q_proj(x_TD),
        self.k_proj(x_TD),
        self.v_proj(x_TD),
        raw_gate_THK,
        raw_beta_TH,
        self.q_conv.weight,
        self.k_conv.weight,
        self.v_conv.weight,
        self.A_log,
        self.dt_bias,
        positions=positions,
    )
    output_gate_THV = self.output_gate(x_TD).view_as(out_THV)
    return self.output_proj(
        self.output_norm(out_THV, output_gate_THV).flatten(-2)
    )


def tpu_kda_config(
    cfg: KDA.Config,
    kernel: type[KDAKernel.Config],
    *,
    conv: CausalConvSilu.Config,
    output_norm: type[KimiRMSNormGated.Config],
    **kernel_fields,
) -> TpuKDA.Config:
  """``TpuKDA.Config`` for an upstream ``KDA.Config`` with these modules.

  ``kernel`` and ``output_norm`` are derived from upstream's configs
  (``kernel_fields`` set on the kernel).
  """
  inner = cfg.inner_kda
  if not isinstance(inner, InnerKDA.Config) or not isinstance(
      inner.kernel, KDAKernel.Config
  ):
    raise ValueError(
        "TPU KDA overrides expect upstream InnerKDA / KDAKernel configs,"
        f" got {type(inner).__name__}"
    )
  return derive(
      cfg,
      TpuKDA.Config,
      inner_kda=derive(
          inner,
          TpuInnerKDA.Config,
          kernel=derive(inner.kernel, kernel, **kernel_fields),
          conv=conv,
      ),
      output_norm=derive(cfg.output_norm, output_norm),
  )


@override(
    target=KDA.Config,
    exact=True,
    description="Run Kimi K3's KDA layers with Tokamax's TPU kernels.",
)
def tokamax_kda_attention(
    cfg: KDA.Config, *, max_num_segments: int | None = None
) -> TpuKDA.Config:
  """``TpuKDA.Config`` with ``TokamaxKDAKernel`` for an upstream ``KDA.Config``.

  The reference conv and output norm. For example
  ``(TOKAMAX_KDA_OVERRIDE, {"max_num_segments": 16})`` in
  ``override.imports`` keeps up to 16 packed documents per row apart.

  Args:
    cfg: The upstream ``KDA.Config`` to replace.
    max_num_segments: See ``TokamaxKDAKernel.Config``.

  Returns:
    The TPU layer's config.
  """
  return tpu_kda_config(
      cfg,
      TokamaxKDAKernel.Config,
      conv=CausalConvSilu.Config(),
      output_norm=KimiRMSNormGated.Config,
      max_num_segments=max_num_segments,
  )


@override(
    target=KDA.Config,
    exact=True,
    description=(
        "Run Kimi K3's KDA layers with Tokamax's TPU kernels and the Pallas"
        " short conv / output norm."
    ),
)
def tokamax_kda_attention_pallas_conv_norm(
    cfg: KDA.Config, *, max_num_segments: int | None = None
) -> TpuKDA.Config:
  """``tokamax_kda_attention`` with ``PallasCausalConvSilu`` and
  ``PallasGatedHeadRMSNorm`` (an option; use it in its place)."""
  return tpu_kda_config(
      cfg,
      TokamaxKDAKernel.Config,
      conv=PallasCausalConvSilu.Config(),
      output_norm=PallasGatedHeadRMSNorm.Config,
      max_num_segments=max_num_segments,
  )
