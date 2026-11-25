"""Pallas TPU kernels for GatedDeltaNet's depthwise causal ``conv1d`` + SiLU.

Qwen3.5's GatedDeltaNet runs a width-4 depthwise causal convolution followed by
SiLU over the ``[B, S, C]`` query, key and value projections. Written as shifted
slices in PyTorch (``qwen3_5.gated_delta._causal_conv``), XLA lowers every
sequence shift to a sublane-misaligned copy: on TPU7x the forward and backward
of one ``[8, 8192, 4096]`` bf16 convolution take ~6 ms, about 7x the HBM
traffic bound. These kernels keep a ``[S, 128]`` column strip in VMEM, shift it
with sublane rotations and fuse the SiLU (and its derivative), so each pass
reads and writes every element once.

``causal_conv_silu`` runs whole rows and ``causal_conv_silu_segmented``
restarts the convolution at every packed document via each token's (clamped)
position in its document; see ``qwen3_5.gated_delta.causal_conv_silu`` for the
dispatch.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch.distributed.tensor import DTensor
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels import pallas_vmem

# Channels per kernel block (one lane tile). At S=8192 on TPU7x, one [8192, 128]
# fp32 strip is 4 MiB, so keeping S bounded by TPU generation and dtype keeps
# the backward's live strips inside scoped VMEM.
_BLOCK_C = 128
_MAX_WIDTH = 8
# Segmented kernels: rows x 128 lanes of one fp32 strip (_segmented_block_c).
_SEG_FWD_BUDGET_ROWS = 16384
_SEG_BWD_BUDGET_ROWS = 8192
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


def _vmem_limit_bytes() -> int | None:
  """Scoped VMEM limit for both kernels: 90% of the core's VMEM.

  Set explicitly so the kernels do not depend on the caller's
  ``xla_tpu_scoped_vmem_limit_kib`` (the backward keeps several fp32
  ``[seq_len, 128]`` strips live). Returns ``None`` when TPU device info is
  unavailable (e.g. CPU shape evaluation).
  """
  try:
    return int(0.9 * pltpu.get_tpu_info().vmem_capacity_bytes)
  except (AttributeError, RuntimeError, ValueError):
    return None


def _max_supported_seq_len(dtype: torch.dtype, width: int = 4) -> int:
  """Maximum whole-strip ``seq_len`` the kernels support (conservative).

  The bounds were measured against the default scoped VMEM (16 MiB on v5/v6,
  >= 32 MiB on TPU7x); the kernels request ``_vmem_limit_bytes()``, which is
  larger, so the bounds keep headroom.
  """
  gen = pallas_vmem.tpu_generation()
  if gen is not None and gen >= 7:
    # TPU7x has >= 32 MiB default scoped VMEM (63.94 MiB in 35B recipes);
    # wider kernels (> 4 taps) keep more rolled strips live in backward.
    base = 8192 if dtype == torch.bfloat16 else 4096
  else:
    # TPU v5/v6 default to 16 MiB scoped VMEM.
    # TODO: Benchmark and raise the v5/v6 sequence-length caps now
    # that _vmem_limit_bytes() requests 90% of physical VMEM.
    base = 2048 if dtype == torch.bfloat16 else 1024
  return base // 2 if width > 4 else base


def _tap_valid(rows: jax.Array, doc_pos: jax.Array | None, shift: int):
  """``[S, 1]`` mask: token ``t`` reads ``t - shift`` (same row and document)."""
  # doc_pos (position within the document, clamped) is >= shift exactly when
  # t - shift is in t's document; rows is the same for a whole row.
  return (rows if doc_pos is None else doc_pos) >= shift


def _shift_down(
    x: jax.Array, shift: int, rows: jax.Array, doc_pos: jax.Array | None = None
) -> jax.Array:
  """``y[t] = x[t - shift]`` along axis 0, zeros where the tap is invalid."""
  if shift == 0:
    return x
  rolled = pltpu.roll(x, shift=shift, axis=0)
  return jnp.where(_tap_valid(rows, doc_pos, shift), rolled, 0.0)


def _shift_up(x: jax.Array, shift: int, rows: jax.Array) -> jax.Array:
  """``y[t] = x[t + shift]`` along axis 0, zeros for ``t >= S - shift``."""
  if shift == 0:
    return x
  seq_len = x.shape[0]
  rolled = pltpu.roll(x, shift=seq_len - shift, axis=0)
  return jnp.where(rows < seq_len - shift, rolled, 0.0)


def _conv(
    x: jax.Array, w: jax.Array, rows: jax.Array, doc_pos: jax.Array | None
) -> jax.Array:
  """Causal depthwise conv of a ``[S, Cb]`` fp32 strip with ``[W, Cb]`` taps."""
  width = w.shape[0]
  acc = x * w[width - 1 : width]
  for k in range(width - 1):
    acc = acc + _shift_down(x, width - 1 - k, rows, doc_pos) * w[k : k + 1]
  return acc


def _fwd_kernel(x_ref, w_ref, *refs):
  *pos_ref, o_ref = refs
  x = x_ref[0].astype(jnp.float32)
  # `[S, 1]` row ids broadcast across lanes; a full `[S, 128]` int32 iota
  # would be one more live strip.
  rows = jax.lax.broadcasted_iota(jnp.int32, (x.shape[0], 1), 0)
  doc_pos = pos_ref[0][0] if pos_ref else None
  z = _conv(x, w_ref[...], rows, doc_pos)
  o_ref[0] = (z * jax.nn.sigmoid(z)).astype(o_ref.dtype)


def _bwd_kernel(x_ref, w_ref, g_ref, *refs):
  *pos_ref, dx_ref, dw_ref = refs
  b = pl.program_id(1)
  w = w_ref[...]
  width = w.shape[0]
  x = x_ref[0].astype(jnp.float32)
  # `[S, 1]` row ids broadcast across lanes; a full `[S, 128]` int32 iota
  # would be one more live strip.
  rows = jax.lax.broadcasted_iota(jnp.int32, (x.shape[0], 1), 0)
  doc_pos = pos_ref[0][0] if pos_ref else None
  # x_taps[k][t] = x[t - (W - 1 - k)] (masked), shared by the recomputed
  # convolution and the dw sums.
  x_taps = [_shift_down(x, width - 1 - k, rows, doc_pos) for k in range(width)]
  # Same summation order as `_conv`, so z matches the forward's.
  z = x_taps[width - 1] * w[width - 1 : width]
  for k in range(width - 1):
    z = z + x_taps[k] * w[k : k + 1]
  sig = jax.nn.sigmoid(z)
  gz = g_ref[0].astype(jnp.float32) * (sig * (1.0 + z * (1.0 - sig)))
  # dx[t] = sum_k w[k] * gz[t + W - 1 - k]: an anti-causal convolution. With
  # documents, output t + s reads input t only if its tap s is valid, so mask
  # gz at its own position before shifting it up.
  dx = gz * w[width - 1 : width]
  for k in range(width - 1):
    shift = width - 1 - k
    gz_k = gz
    if doc_pos is not None:
      gz_k = jnp.where(_tap_valid(rows, doc_pos, shift), gz, 0.0)
    dx = dx + _shift_up(gz_k, shift, rows) * w[k : k + 1]
  dx_ref[0] = dx.astype(dx_ref.dtype)
  # dw[k] = sum_t x[t - (W - 1 - k)] * gz[t], accumulated over the batch.
  dw = jnp.concatenate(
      [jnp.sum(x_taps[k] * gz, axis=0, keepdims=True) for k in range(width)],
      axis=0,
  )

  @pl.when(b == 0)
  def _():
    dw_ref[...] = jnp.zeros_like(dw_ref)

  dw_ref[...] += dw


def _segmented_block_c(seq_len: int, channels: int, budget_rows: int) -> int:
  """Channel block for the segmented kernels: the document mask is per row, so
  a wider block amortizes it over more lanes. ``budget_rows * 128`` fp32
  elements bound one live strip (forward 16K rows, backward 8K)."""
  lane_tiles = max(1, budget_rows // seq_len)
  block_c = _BLOCK_C << (lane_tiles.bit_length() - 1)
  while channels % block_c:
    block_c //= 2
  return block_c


def _specs(batch: int, seq_len: int, channels: int, width: int, block_c: int):
  """BlockSpecs and grid ``(channels // 128, batch)``.

  ``batch`` is the innermost grid axis so ``dw_ref``'s block index ``(0, c)``
  stays constant across ``b = 0 .. batch - 1`` in backward, keeping ``dw_ref``
  resident in VMEM across the batch before a single HBM writeback.
  """
  strip = pl.BlockSpec((1, seq_len, block_c), lambda c, b: (b, 0, c))
  taps = pl.BlockSpec((width, block_c), lambda c, b: (0, c))
  grid = (channels // block_c, batch)
  return strip, taps, grid


def _doc_pos_spec(seq_len: int):
  """``[B, S, 1]`` int32 document positions, one row per grid step."""
  return pl.BlockSpec((1, seq_len, 1), lambda c, b: (b, 0, 0))


def causal_conv_silu_fwd_jax(
    x: jax.Array, w_WC: jax.Array, doc_pos_BS1: jax.Array | None = None
) -> jax.Array:
  """``silu(causal_conv(x))`` for ``x`` ``[B, S, C]``, fp32 taps ``[W, C]``.

  ``doc_pos_BS1``: optional ``[B, S, 1]`` int32 position of each token in its
  document, clamped to ``W - 1`` (see ``doc_positions``).
  """
  batch, seq_len, channels = x.shape
  width = w_WC.shape[0]
  extra = [] if doc_pos_BS1 is None else [doc_pos_BS1]
  block_c = (
      _BLOCK_C
      if doc_pos_BS1 is None
      else _segmented_block_c(seq_len, channels, _SEG_FWD_BUDGET_ROWS)
  )
  strip, taps, grid = _specs(batch, seq_len, channels, width, block_c)
  return pl.pallas_call(
      _fwd_kernel,
      grid=grid,
      in_specs=[strip, taps] + [_doc_pos_spec(seq_len)] * len(extra),
      out_specs=strip,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel"),
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      name="causal_conv1d_silu_fwd",
  )(x, w_WC, *extra)


def causal_conv_silu_bwd_jax(
    x: jax.Array,
    w_WC: jax.Array,
    g: jax.Array,
    doc_pos_BS1: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Returns ``(dx, dw_WC)`` for the output gradient ``g`` ``[B, S, C]``."""
  batch, seq_len, channels = x.shape
  width = w_WC.shape[0]
  extra = [] if doc_pos_BS1 is None else [doc_pos_BS1]
  block_c = (
      _BLOCK_C
      if doc_pos_BS1 is None
      else _segmented_block_c(seq_len, channels, _SEG_BWD_BUDGET_ROWS)
  )
  strip, taps, grid = _specs(batch, seq_len, channels, width, block_c)
  return pl.pallas_call(
      _bwd_kernel,
      grid=grid,
      in_specs=[strip, taps, strip] + [_doc_pos_spec(seq_len)] * len(extra),
      out_specs=[strip, taps],
      out_shape=[
          jax.ShapeDtypeStruct(x.shape, x.dtype),
          jax.ShapeDtypeStruct(w_WC.shape, jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      name="causal_conv1d_silu_bwd",
  )(x, w_WC, g, *extra)


@pallas.jax_op("tt_qwen35::causal_conv1d_silu_fwd")
def _fwd_op(x: jax.Array, w_WC: jax.Array) -> jax.Array:
  return causal_conv_silu_fwd_jax(x, w_WC)


@pallas.jax_op("tt_qwen35::causal_conv1d_silu_bwd")
def _bwd_op(
    x: jax.Array, w_WC: jax.Array, g: jax.Array
) -> tuple[jax.Array, jax.Array]:
  return causal_conv_silu_bwd_jax(x, w_WC, g)


def _setup_context(ctx, inputs, output):
  del output
  ctx.set_materialize_grads(False)
  x, w_WC = inputs
  ctx.save_for_backward(x, w_WC)


def _backward(ctx, grad_out):
  need_dx, need_dw = ctx.needs_input_grad
  if grad_out is None or not (need_dx or need_dw):
    return None, None
  x, w_WC = ctx.saved_tensors
  dx, dw_WC = _bwd_op(x, w_WC, grad_out.contiguous())
  return dx if need_dx else None, dw_WC if need_dw else None


_fwd_op.register_autograd(_backward, setup_context=_setup_context)


@pallas.jax_op("tt_qwen35::causal_conv1d_silu_seg_fwd")
def _seg_fwd_op(
    x: jax.Array, w_WC: jax.Array, doc_pos_BS1: jax.Array
) -> jax.Array:
  return causal_conv_silu_fwd_jax(x, w_WC, doc_pos_BS1)


@pallas.jax_op("tt_qwen35::causal_conv1d_silu_seg_bwd")
def _seg_bwd_op(
    x: jax.Array, w_WC: jax.Array, g: jax.Array, doc_pos_BS1: jax.Array
) -> tuple[jax.Array, jax.Array]:
  return causal_conv_silu_bwd_jax(x, w_WC, g, doc_pos_BS1)


def _seg_setup_context(ctx, inputs, output):
  del output
  ctx.set_materialize_grads(False)
  ctx.save_for_backward(*inputs)


def _seg_backward(ctx, grad_out):
  need_dx, need_dw, _ = ctx.needs_input_grad
  if grad_out is None or not (need_dx or need_dw):
    return None, None, None
  x, w_WC, doc_pos_BS1 = ctx.saved_tensors
  dx, dw_WC = _seg_bwd_op(x, w_WC, grad_out.contiguous(), doc_pos_BS1)
  return dx if need_dx else None, dw_WC if need_dw else None, None


_seg_fwd_op.register_autograd(_seg_backward, setup_context=_seg_setup_context)


def doc_positions(segment_ids_BS: torch.Tensor, width: int) -> torch.Tensor:
  """``[B, S, 1]`` int32 position of each token in its document, clamped to
  ``width - 1``. Documents are contiguous runs of equal ``segment_ids_BS``, so
  the count of ``s in [1, width)`` with ``segment_ids[t - s] == segment_ids[t]``
  is that clamped position."""
  pos = torch.zeros_like(segment_ids_BS, dtype=torch.int32)
  for shift in range(1, width):
    same = segment_ids_BS[:, shift:] == segment_ids_BS[:, :-shift]
    pos = pos + torch.nn.functional.pad(same.to(torch.int32), (shift, 0))
  return pos.unsqueeze(-1)


def unsupported_reason(
    x_BSC: torch.Tensor, weight_C1W: torch.Tensor
) -> str | None:
  """Reason ``x_BSC`` and ``weight_C1W`` cannot use Pallas, or ``None``."""
  if isinstance(x_BSC, DTensor) or isinstance(weight_C1W, DTensor):
    return "DTensor inputs are not supported"
  if x_BSC.device.type != "tpu" or weight_C1W.device != x_BSC.device:
    return (
        f"inputs must be on the same TPU device, got {x_BSC.device} and"
        f" {weight_C1W.device}"
    )
  if (
      x_BSC.dtype not in _SUPPORTED_DTYPES
      or weight_C1W.dtype not in _SUPPORTED_DTYPES
  ):
    return (
        f"unsupported dtype x={x_BSC.dtype}, weight={weight_C1W.dtype}"
        f" (supported: {_SUPPORTED_DTYPES})"
    )
  if x_BSC.ndim != 3 or weight_C1W.ndim != 3:
    return (
        f"expected 3D x_BSC and weight_C1W, got {x_BSC.ndim}D and"
        f" {weight_C1W.ndim}D"
    )
  batch, seq_len, channels = x_BSC.shape
  width = weight_C1W.shape[2]
  max_s = _max_supported_seq_len(x_BSC.dtype, width)
  if batch <= 0:
    return f"batch size must be positive, got {batch}"
  if not (0 < seq_len <= max_s and seq_len % 8 == 0):
    return (
        f"seq_len={seq_len} must be a positive multiple of 8 and <="
        f" {max_s} for dtype={x_BSC.dtype}, width={width}"
    )
  if channels <= 0 or channels % _BLOCK_C != 0:
    return (
        f"channels={channels} must be a positive multiple of {_BLOCK_C}"
    )
  if weight_C1W.shape != (channels, 1, width):
    return (
        f"weight shape {tuple(weight_C1W.shape)} must match"
        f" (channels={channels}, 1, width={width})"
    )
  if not 1 <= width <= min(seq_len, _MAX_WIDTH):
    return (
        f"kernel width={width} must be in [1, min(seq_len={seq_len},"
        f" {_MAX_WIDTH})]"
    )
  return None


def supports(x_BSC: torch.Tensor, weight_C1W: torch.Tensor) -> bool:
  """Whether ``x_BSC`` and ``weight_C1W`` satisfy the Pallas kernel limits."""
  return unsupported_reason(x_BSC, weight_C1W) is None


def causal_conv_silu(
    x_BSC: torch.Tensor, weight_C1W: torch.Tensor
) -> torch.Tensor:
  """``silu`` of a depthwise causal ``conv1d`` over whole ``[B, S, C]`` rows.

  Args:
    x_BSC: ``[B, S, C]`` TPU tensor with ``C % 128 == 0`` and ``S % 8 == 0``.
    weight_C1W: ``[C, 1, W]`` depthwise ``Conv1d`` weight. The taps run in fp32,
      as in ``F.conv1d``'s accumulation.

  Returns:
    ``[B, S, C]`` output in ``x_BSC.dtype``.
  """
  reason = unsupported_reason(x_BSC, weight_C1W)
  if reason is not None:
    raise ValueError(
        f"Unsupported inputs for Pallas causal_conv_silu ({reason}):"
        f" x={tuple(x_BSC.shape)} ({x_BSC.dtype}, {x_BSC.device}),"
        f" weight={tuple(weight_C1W.shape)} ({weight_C1W.dtype},"
        f" {weight_C1W.device})"
    )
  w_WC = weight_C1W[:, 0, :].t().to(torch.float32).contiguous()
  return _fwd_op(x_BSC.contiguous(), w_WC)


def causal_conv_silu_segmented(
    x_BSC: torch.Tensor, weight_C1W: torch.Tensor, segment_ids_BS: torch.Tensor
) -> torch.Tensor:
  """``causal_conv_silu`` that restarts at every document of ``segment_ids_BS``.

  Same limits as ``causal_conv_silu``; ``segment_ids_BS`` is ``[B, S]``.
  """
  reason = unsupported_reason(x_BSC, weight_C1W)
  if reason is None and segment_ids_BS.shape != x_BSC.shape[:2]:
    reason = (
        f"segment_ids shape {tuple(segment_ids_BS.shape)} must be"
        f" {tuple(x_BSC.shape[:2])}"
    )
  if reason is not None:
    raise ValueError(
        f"Unsupported inputs for causal_conv_silu_segmented ({reason})"
    )
  w_WC = weight_C1W[:, 0, :].t().to(torch.float32).contiguous()
  return _seg_fwd_op(
      x_BSC.contiguous(),
      w_WC,
      doc_positions(segment_ids_BS, weight_C1W.shape[2]),
  )
