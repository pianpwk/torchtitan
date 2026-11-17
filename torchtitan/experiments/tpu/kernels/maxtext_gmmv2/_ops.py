class RaggedGatherReduceMetadata:  # placeholder for the import smoke
  pass
def ragged_gather_reduce_tc(*a, **k): raise NotImplementedError
def ragged_silu_mul_bwd(*a, **k): raise NotImplementedError
import jax
import jax.numpy as jnp
def silu_mul_tile(x, y, c, out_dtype):
  return (jax.nn.silu(x) * y * c).astype(out_dtype)
def quantize_tile(x, inv_scale, qtype):
  q_max = float(jnp.finfo(qtype).max)
  return jnp.clip(x.astype(jnp.float32) * inv_scale, -q_max, q_max).astype(qtype)
