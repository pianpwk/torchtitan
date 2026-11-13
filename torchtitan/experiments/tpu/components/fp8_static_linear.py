# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dense linears with MaxText's ``fp8_full`` numerics.

As for the FP8 routed experts (``kernels/fp8_expert_segment_matmul.py``): the
input and the weight take one static e4m3 scale (calibration
``fixed,-224,224``) and the output gradient one absmax e5m2 scale per tensor,
shared by dInput and dWeight. The GEMMs are XLA dots on FP8 operands; with
static scales the quantization is elementwise, so XLA fuses it into the dots
and no FP8 copy is written to HBM.

Unlike torchao's ``Float8Linear`` this needs no amax of the forward operands
and does not gather FP8 shards (whose layouts trip a libtpu fusion check on
DeepSeek-V3's MLA projections).

Shape legend: M = rows (tokens), K = in features, N = out features.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

from torchtitan.experiments.tpu.kernels import fp8_scale
from torchtitan.experiments.tpu.kernels.fp8_expert_segment_matmul import (
    BWD_DTYPE,
    STATIC_SCALE,
    static_quantize,
)
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.tools.logging import logger


def _dot(a: jax.Array, b: jax.Array, contract: tuple[int, int]) -> jax.Array:
  return jax.lax.dot_general(
      a, b, ((contract[:1], contract[1:]), ((), ())),
      preferred_element_type=jnp.float32,
  )


@pallas.jax_op("tt_fp8::static_linear_fwd")
def _static_linear_fwd(x_MK: jax.Array, w_NK: jax.Array) -> jax.Array:
  out = _dot(static_quantize(x_MK), static_quantize(w_NK), (1, 1))
  return (out * (STATIC_SCALE * STATIC_SCALE)).astype(x_MK.dtype)


@pallas.jax_op("tt_fp8::static_linear_bwd")
def _static_linear_bwd(
    dout_MN: jax.Array, x_MK: jax.Array, w_NK: jax.Array
) -> tuple[jax.Array, jax.Array]:
  dout_f32 = dout_MN.astype(jnp.float32)
  dout_scale = fp8_scale.scale_from_amax(jnp.max(jnp.abs(dout_f32)), BWD_DTYPE)
  fmax = float(jnp.finfo(BWD_DTYPE).max)
  dout_q = jnp.clip(dout_f32 / dout_scale, -fmax, fmax).astype(BWD_DTYPE)
  out_scale = dout_scale * STATIC_SCALE
  dx_MK = _dot(dout_q, static_quantize(w_NK), (1, 0)) * out_scale
  dw_NK = _dot(dout_q, static_quantize(x_MK), (0, 0)) * out_scale
  return dx_MK.astype(x_MK.dtype), dw_NK.astype(w_NK.dtype)


class _Fp8StaticLinearFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x_MK, w_NK):
    ctx.save_for_backward(x_MK, w_NK)
    return _static_linear_fwd(x_MK, w_NK)

  @staticmethod
  def backward(ctx, dout_MN):
    return _static_linear_bwd(dout_MN.contiguous(), *ctx.saved_tensors)


class Fp8StaticLinear(Linear):
  """`Linear` (no bias) whose GEMMs run in FP8 with ``fp8_full`` numerics."""

  @dataclass(kw_only=True, slots=True)
  class Config(Linear.Config):
    pass

  def __init__(self, config: Config):
    if config.bias:
      raise ValueError("Fp8StaticLinear does not support bias")
    super().__init__(config)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    weight = self.weight
    x_MK = x.reshape(-1, x.shape[-1])
    out_MN = _Fp8StaticLinearFunction.apply(x_MK, weight.to(x.dtype))
    return out_MN.reshape(*x.shape[:-1], out_MN.shape[-1])


class Fp8StaticLinearConverter(ModelConfigConverter):
  """Swap ``Linear.Config``s whose FQN contains any of ``fqns`` (and none of
  ``filter_fqns``) for `Fp8StaticLinear`."""

  @dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    fqns: tuple[str, ...]
    filter_fqns: tuple[str, ...] = ()

  def __init__(self, config: Config):
    self.fqns = list(config.fqns)
    self.filter_fqns = list(config.filter_fqns)

  def convert(self, model_config):
    swapped = []
    for fqn, linear_config, parent, attr in list(
        model_config.traverse(Linear.Config)
    ):
      if type(linear_config) is not Linear.Config:
        continue
      if not any(f in fqn for f in self.fqns) or any(
          f in fqn for f in self.filter_fqns
      ):
        continue
      new_config = Fp8StaticLinear.Config(
          in_features=linear_config.in_features,
          out_features=linear_config.out_features,
          bias=linear_config.bias,
          param_init=linear_config.param_init,
      )
      if parent is None:
        model_config = new_config
      elif isinstance(parent, list):
        parent[attr] = new_config
      else:
        setattr(parent, attr, new_config)
      swapped.append(fqn)
    if not swapped:
      raise ValueError(f"No Linear matches {self.fqns}")
    logger.info(
        "Swapped %d linears to Fp8StaticLinear (e.g. %s)", len(swapped), swapped[0]
    )
    return model_config


__all__ = ["Fp8StaticLinear", "Fp8StaticLinearConverter"]
