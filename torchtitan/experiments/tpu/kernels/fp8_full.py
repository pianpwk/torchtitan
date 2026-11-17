# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MaxText's ``quantization=fp8_full`` numerics: shared constants and helpers.

Our implementation of MaxText's recipe (Lineage DeepSeek-V3,
``maxtext/experimental/lineage/quantization.py``), not ported code. Weights and
activations take one *static* per-tensor e4m3 scale -- calibration
``fixed,-224,224``, i.e. ``clip(x / 0.5)`` -- and output gradients one dynamic
absmax e5m2 scale per tensor, so dInput and dWeight share a single e5m2 copy
of dY. A static scale makes quantization elementwise: it fuses into the
consuming dot, and a weight quantizes on its FSDP shard exactly as the
gathered weight would, so the all-gather can move FP8.
"""

import jax
import jax.numpy as jnp

FWD_DTYPE = jnp.float8_e4m3fn
BWD_DTYPE = jnp.float8_e5m2

# ``fixed,-224,224``: 224 / 448 (e4m3's largest finite value).
STATIC_SCALE = 224.0 / float(jnp.finfo(FWD_DTYPE).max)


def static_quantize(x: jax.Array) -> jax.Array:
    """``clip(x / STATIC_SCALE)`` in e4m3; saturates, never inf / NaN."""
    fmax = float(jnp.finfo(FWD_DTYPE).max)
    return jnp.clip(x.astype(jnp.float32) / STATIC_SCALE, -fmax, fmax).astype(
        FWD_DTYPE
    )


__all__ = [
    "BWD_DTYPE",
    "FWD_DTYPE",
    "STATIC_SCALE",
    "static_quantize",
]
