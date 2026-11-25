"""Pallas TPU kernels for gated per-head RMSNorm (``RMSNormGated``).

GatedDeltaNet normalizes every ``[T, H, 128]`` head output in FP32, scales it by
a learned weight, rounds to the input dtype and multiplies by ``silu(gate)``
(``qwen3_5.gdn.RMSNormGated``). XLA splits this into several elementwise and
reduction fusions with FP32 intermediates in HBM; under full rematerialization
they run twice forward and once backward per layer. These kernels process a
block of rows in VMEM per pass, so each pass reads its inputs and writes its
outputs once.

``gate_activation="silu"`` follows ``RMSNormGated.forward``;
``gate_activation="sigmoid"`` follows Kimi K3's ``KimiRMSNormGated`` (sigmoid
gate, rounded once at the end). The last dim of ``x`` may hold several heads
(``[T, H * V]`` with a ``[V]`` weight): each ``V``-wide group is normalized on
its own, without reshaping to ``[T * H, V]``, which on TPU is a relayout copy
(3-4x the kernel's own time at ``T = 16384, H = 96, V = 128``).
"""

from typing import Literal

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
# pyformat: disable
# With torch_tpu, torch must be imported before spmd_types to avoid a circular
# import when torch autoloads torch_tpu -> torch.distributed.fsdp -> spmd_types.
import torch  # isort: skip
import spmd_types as spmd
# pyformat: enable
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
from torch_tpu._internal import pallas

_BLOCK_ROWS = 2048
_BLOCK_DIM = 128
_MAX_DIM = 512
# Lanes per block when the last dim holds several heads (12 heads of 128).
_MAX_BLOCK_LANES = 1536
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)

GateActivation = Literal["silu", "sigmoid"]
_GATE_ACTIVATIONS = ("silu", "sigmoid")


def _gate_and_grad(
    g: jax.Array, gate_activation: GateActivation
) -> tuple[jax.Array, jax.Array]:
  sig = jax.nn.sigmoid(g)
  if gate_activation == "sigmoid":
    return sig, sig * (1.0 - sig)
  return g * sig, sig * (1.0 + g * (1.0 - sig))


def _heads(x_ref, head_dim: int):
  """Lane slices of the heads in a ``[rows, heads * head_dim]`` block."""
  return [
      pl.ds(h * head_dim, head_dim) for h in range(x_ref.shape[1] // head_dim)
  ]


def _fwd_kernel(eps, gate_activation, x_ref, g_ref, w_ref, o_ref):
  w = w_ref[...]
  for sl in _heads(x_ref, w.shape[1]):
    x = x_ref[:, sl].astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    normed = w * (x * rstd)
    if gate_activation == "silu":
      # RMSNormGated rounds the scaled norm to the input dtype before the gate.
      normed = normed.astype(o_ref.dtype).astype(jnp.float32)
    gate, _ = _gate_and_grad(g_ref[:, sl].astype(jnp.float32), gate_activation)
    o_ref[:, sl] = (normed * gate).astype(o_ref.dtype)


def _bwd_kernel(
    eps, gate_activation, x_ref, g_ref, w_ref, dy_ref, dx_ref, dg_ref, dw_ref
):
  w = w_ref[...]

  @pl.when((pl.program_id(0) == 0) & (pl.program_id(1) == 0))
  def _():
    dw_ref[...] = jnp.zeros_like(dw_ref)

  for sl in _heads(x_ref, w.shape[1]):
    x = x_ref[:, sl].astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    n = x * rstd
    scaled = w * n
    if gate_activation == "silu":
      scaled = scaled.astype(x_ref.dtype).astype(jnp.float32)
    gate, dgate = _gate_and_grad(
        g_ref[:, sl].astype(jnp.float32), gate_activation
    )
    dy = dy_ref[:, sl].astype(jnp.float32)
    dg_ref[:, sl] = (dy * scaled * dgate).astype(dg_ref.dtype)
    d_scaled = dy * gate
    dw_ref[...] += jnp.sum(d_scaled * n, axis=0, keepdims=True)
    dn = d_scaled * w
    dx = rstd * (dn - n * jnp.mean(dn * n, axis=-1, keepdims=True))
    dx_ref[:, sl] = dx.astype(dx_ref.dtype)


def _vmem_limit_bytes() -> int | None:
  """Scoped VMEM limit for both kernels: 90% of the core's VMEM."""
  try:
    return int(0.9 * pltpu.get_tpu_info().vmem_capacity_bytes)
  except (AttributeError, RuntimeError, ValueError):
    return None


def _blocks(dim: int, head_dim: int, *dtypes) -> tuple[int, int]:
  """``(block_rows, block_lanes)``: whole heads per block, rows a power of 2.

  One head per row (``dim == head_dim``): ``<= 2048`` rows keeping each
  ``(block_rows, dim)`` buffer <= 1 MiB: the backward's 5 double-buffered arrays
  (x, gate, dy, dx, dg) take <= 10 MiB, plus ~8 fp32 temporaries (<= 8 MiB at
  2048 x 128). Several heads per row: up to ``_MAX_BLOCK_LANES`` lanes of whole
  heads and 512 rows (1.5 MiB bf16 buffers; the temporaries are per head). The
  kernels request `_vmem_limit_bytes()`, so this does not rely on the default
  scoped VMEM.
  """
  elem_bytes = max(jnp.dtype(dt).itemsize for dt in dtypes)
  block_lanes = head_dim
  if dim > head_dim:
    block_lanes = head_dim * max(1, _MAX_BLOCK_LANES // head_dim)
    while dim % block_lanes:
      block_lanes -= head_dim
  block = _BLOCK_ROWS if block_lanes == head_dim else 512
  while block > 128 and block * head_dim * elem_bytes > 1024 * 1024:
    block //= 2
  return block, block_lanes


def _check_rows(rows: int, block_rows: int) -> None:
  if rows <= 0 or rows % block_rows != 0:
    raise ValueError(
        f"rows ({rows}) must be a positive multiple of block_rows"
        f" ({block_rows})"
    )


def rms_norm_gated_fwd_jax(
    x: jax.Array,
    gate: jax.Array,
    w: jax.Array,
    eps: float,
    gate_activation: GateActivation = "silu",
) -> jax.Array:
  """``x``, ``gate`` ``[rows, H * V]``; ``w`` ``[1, V]`` FP32."""
  rows, dim = x.shape
  block_rows, block_lanes = _blocks(dim, w.shape[1], x.dtype, gate.dtype)
  _check_rows(rows, block_rows)
  tile = pl.BlockSpec((block_rows, block_lanes), lambda i, j: (i, j))
  return pl.pallas_call(
      lambda *refs: _fwd_kernel(eps, gate_activation, *refs),
      grid=(rows // block_rows, dim // block_lanes),
      in_specs=[tile, tile, pl.BlockSpec(w.shape, lambda i, j: (0, 0))],
      out_specs=tile,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel"),
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      name="rms_norm_gated_fwd",
  )(x, gate, w)


def rms_norm_gated_bwd_jax(
    x: jax.Array,
    gate: jax.Array,
    w: jax.Array,
    dy: jax.Array,
    eps: float,
    gate_activation: GateActivation = "silu",
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns ``(dx, dg, dw [1, V])``, dw accumulated in FP32 across blocks."""
  rows, dim = x.shape
  block_rows, block_lanes = _blocks(
      dim, w.shape[1], x.dtype, gate.dtype, dy.dtype
  )
  _check_rows(rows, block_rows)
  tile = pl.BlockSpec((block_rows, block_lanes), lambda i, j: (i, j))
  weight = pl.BlockSpec(w.shape, lambda i, j: (0, 0))
  return pl.pallas_call(
      lambda *refs: _bwd_kernel(eps, gate_activation, *refs),
      grid=(rows // block_rows, dim // block_lanes),
      in_specs=[tile, tile, weight, tile],
      out_specs=[tile, tile, weight],
      out_shape=[
          jax.ShapeDtypeStruct(x.shape, x.dtype),
          jax.ShapeDtypeStruct(gate.shape, gate.dtype),
          jax.ShapeDtypeStruct(w.shape, jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary"),
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      name="rms_norm_gated_bwd",
  )(x, gate, w, dy)


@pallas.jax_op("tt_qwen35::rms_norm_gated_fwd")
def _fwd_op(
    x: jax.Array,
    gate: jax.Array,
    w: jax.Array,
    eps: float,
    gate_activation: str,
) -> jax.Array:
  return rms_norm_gated_fwd_jax(x, gate, w, eps, gate_activation)


@pallas.jax_op("tt_qwen35::rms_norm_gated_bwd")
def _bwd_op(
    x: jax.Array,
    gate: jax.Array,
    w: jax.Array,
    dy: jax.Array,
    eps: float,
    gate_activation: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  return rms_norm_gated_bwd_jax(x, gate, w, dy, eps, gate_activation)


def _setup_context(ctx, inputs, output):
  del output
  ctx.set_materialize_grads(False)
  x, gate, w, eps, gate_activation = inputs
  ctx.save_for_backward(x, gate, w)
  ctx.eps = eps
  ctx.gate_activation = gate_activation


def _backward(ctx, dy):
  need_dx, need_dg, need_dw, _, _ = ctx.needs_input_grad
  if dy is None or not (need_dx or need_dg or need_dw):
    return None, None, None, None, None
  x, gate, w = ctx.saved_tensors
  dx, dg, dw = _bwd_op(
      x, gate, w, dy.contiguous(), ctx.eps, ctx.gate_activation
  )
  return (
      dx if need_dx else None,
      dg if need_dg else None,
      dw.reshape(w.shape) if need_dw else None,
      None,
      None,
  )


_fwd_op.register_autograd(_backward, setup_context=_setup_context)


@torch._dynamo.assume_constant_result
def _is_spmd_type_checking() -> bool:
  return spmd.is_type_checking()


def _native_rms_norm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gate_activation: GateActivation,
) -> torch.Tensor:
  dtype, head_dim = x.dtype, weight.shape[0]
  xf = x.float().unflatten(-1, (-1, head_dim))
  xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
  gf = gate.float().unflatten(-1, (-1, head_dim))
  if gate_activation == "sigmoid":
    out = weight.float() * xf * torch.sigmoid(gf)
  else:
    out = (weight.float() * xf).to(dtype) * F.silu(gf)
  return out.to(dtype).flatten(-2)


class _RmsNormGatedSpmdFn(torch.autograd.Function):
  """``rms_norm_gated`` wrapper registered with ``spmd.register_decomposition``."""

  @staticmethod
  def forward(
      ctx,
      x: torch.Tensor,
      gate: torch.Tensor,
      weight: torch.Tensor,
      eps: float,
      gate_activation: GateActivation,
  ) -> torch.Tensor:
    ctx.set_materialize_grads(False)
    dim = x.shape[-1]
    ctx.orig_shape = x.shape
    ctx.weight_shape = weight.shape
    ctx.weight_dtype = weight.dtype
    ctx.eps = float(eps)
    ctx.gate_activation = gate_activation
    with spmd.no_typecheck():
      x_2d = x.reshape(-1, dim).contiguous()
      gate_2d = gate.reshape(-1, dim).contiguous()
      w_2d = weight.to(torch.float32).reshape(1, -1).contiguous()
      out = _fwd_op(x_2d, gate_2d, w_2d, ctx.eps, gate_activation)
      out = out.reshape(x.shape)
    ctx.save_for_backward(x_2d, gate_2d, w_2d)
    return out

  @staticmethod
  def backward(ctx, dy):
    need_dx, need_dg, need_dw, _, _ = ctx.needs_input_grad
    if dy is None or not (need_dx or need_dg or need_dw):
      return None, None, None, None, None
    x_2d, gate_2d, w_2d = ctx.saved_tensors
    dim = ctx.orig_shape[-1]
    with spmd.no_typecheck():
      dx_2d, dg_2d, dw_2d = _bwd_op(
          x_2d,
          gate_2d,
          w_2d,
          dy.reshape(-1, dim).contiguous(),
          ctx.eps,
          ctx.gate_activation,
      )
      dx = dx_2d.reshape(ctx.orig_shape) if need_dx else None
      dg = dg_2d.reshape(ctx.orig_shape) if need_dg else None
      dw = (
          dw_2d.reshape(ctx.weight_shape).to(ctx.weight_dtype)
          if need_dw
          else None
      )
    return dx, dg, dw, None, None


spmd.register_decomposition(_RmsNormGatedSpmdFn, _native_rms_norm_gated)


def _has_sharded_norm_dim(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor
) -> bool:
  for t in (x, gate):
    ps = spmd.get_partition_spec(t)
    if ps is not None and len(ps) > 0 and ps[-1] is not None:
      return True
  w_ps = spmd.get_partition_spec(weight)
  return w_ps is not None and len(w_ps) > 0 and w_ps[0] is not None


def supports(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor) -> bool:
  """Whether ``(x, gate, weight)`` satisfy the Pallas kernel limits."""
  if (
      isinstance(x, DTensor)
      or isinstance(gate, DTensor)
      or isinstance(weight, DTensor)
  ):
    return False
  if _is_spmd_type_checking() and _has_sharded_norm_dim(x, gate, weight):
    return False
  if (
      x.device.type != "tpu"
      or gate.device != x.device
      or weight.device != x.device
  ):
    return False
  if (
      x.dtype not in _SUPPORTED_DTYPES
      or gate.dtype not in _SUPPORTED_DTYPES
      or weight.dtype not in _SUPPORTED_DTYPES
  ):
    return False
  if x.ndim < 2 or gate.shape != x.shape or weight.ndim != 1:
    return False
  dim, head_dim = x.shape[-1], weight.shape[0]
  if head_dim <= 0 or head_dim > _MAX_DIM or head_dim % _BLOCK_DIM != 0:
    return False
  if dim % head_dim != 0:
    return False
  rows = x.numel() // dim
  return rows > 0 and rows % _BLOCK_ROWS == 0


def rms_norm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    gate_activation: GateActivation = "silu",
) -> torch.Tensor:
  """Gated RMSNorm over every ``weight``-sized group of ``x``'s last dim.

  ``x [..., V]`` (``RMSNormGated``) or ``x [..., H * V]`` with a ``[V]``
  weight (one norm per head, no relayout); ``gate`` has ``x``'s shape.

  Args:
    x: Input.
    gate: Output gate input.
    weight: ``[V]`` per-channel scale.
    eps: Added to the mean square.
    gate_activation: ``"silu"`` (``RMSNormGated``, which rounds the scaled norm
      to ``x.dtype`` before the gate) or ``"sigmoid"`` (``KimiRMSNormGated``,
      rounded once).

  Returns:
    The output, in ``x``'s shape and dtype.
  """
  if gate_activation not in _GATE_ACTIVATIONS:
    raise ValueError(
        f"gate_activation must be one of {_GATE_ACTIVATIONS}, got"
        f" {gate_activation!r}"
    )
  if not supports(x, gate, weight):
    raise ValueError(
        "Unsupported inputs for Pallas rms_norm_gated:"
        f" x={tuple(x.shape)} ({x.dtype}, {x.device}),"
        f" gate={tuple(gate.shape)} ({gate.dtype}, {gate.device}),"
        f" weight={tuple(weight.shape)} ({weight.dtype}, {weight.device})"
    )
  if eps < 0:
    raise ValueError(f"eps must be non-negative, got {eps}")
  if _is_spmd_type_checking():
    return _RmsNormGatedSpmdFn.apply(
        x, gate, weight, float(eps), gate_activation
    )
  dim = x.shape[-1]
  w = weight.to(torch.float32).reshape(1, -1).contiguous()
  out = _fwd_op(
      x.reshape(-1, dim).contiguous(),
      gate.reshape(-1, dim).contiguous(),
      w,
      float(eps),
      gate_activation,
  )
  return out.reshape(x.shape)
