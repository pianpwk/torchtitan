"""Kimi K3 on TPU.

``override.imports`` entries:

* ``tokamax_kda.tokamax_kda_attention``: Kimi Delta Attention with Tokamax's
  Mosaic TPU kernels.
* ``tokamax_mla.tokamax_mla_attention``: gated MLA with Tokamax Splash.
* ``latent_moe.tpu_latent_moe``: LatentMoE with MegaBlox experts.

``model_registry`` builds a text-only K3 config for the TPU lane; the recipes
in ``config_registry`` add the three overrides. This module imports neither
``jax`` nor ``torch_tpu``, so a GPU run can build the same flavors.
"""

import dataclasses

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
import torchtitan.models.kimi_k3 as upstream_kimi_k3
from torchtitan.models.kimi_k3 import KimiK3Model
from torchtitan.protocols.model_spec import ModelSpec

__all__ = ["kimi_k3_tpu_configs", "model_registry"]


def _tiny() -> KimiK3Model.Config:
  """A text-only K3 small enough for one TPU core, with every K3 layer type.

  8 layers in K3's pattern: layer 0 is KDA with the dense SiTU-GLU FFN, then
  KDA + LatentMoE and every 4th layer gated MLA + LatentMoE (3:1, as K3's
  ``range(3, 92, 4)``). Attention residual blocks of 4 layers (two blocks).
  K3's head dims (KDA 128, MLA qk 128 + 64 and v 128), latent dim =
  dim / 2, sigmoid router without expert-bias balancing, shared experts and
  an untied LM head. The vocabulary covers the test tokenizer (2020 tokens).
  """
  dim = 512
  # pylint: disable-next=protected-access
  cfg = upstream_kimi_k3._kimi_k3_config(
      dim=dim,
      vocab_size=2048,
      num_layers=8,
      full_attention_layers={3, 7},
      attn_res_block_size=4,
      num_heads=4,
      q_lora_rank=256,
      kv_lora_rank=128,
      qk_nope_head_dim=128,
      qk_rope_head_dim=64,
      v_head_dim=128,
      kda_head_dim=128,
      conv_kernel_size=4,
      dense_hidden_dim=2048,
      latent_dim=256,
      expert_hidden_dim=256,
      num_experts=32,
      top_k=4,
      num_shared_experts=2,
      vision_encoder=None,  # pytype: disable=wrong-arg-types
      attn_backend="flex",
  )
  for layer in cfg.layers:
    if layer.moe is not None:
      # No sign-based expert-bias balancing (upstream's load_balance_coeff).
      layer.moe.load_balance_coeff = None
  return cfg


# flavor -> (config builder, max context length)
kimi_k3_tpu_configs = {
    "tiny": (_tiny, 4096),
}


def model_registry(flavor: str, *, seq_len: int | None = None) -> ModelSpec:
  """Returns a text-only ``ModelSpec`` for a TPU K3 flavor.

  Args:
    flavor: A key of ``kimi_k3_tpu_configs``.
    seq_len: Context length; defaults to the flavor's maximum.

  Returns:
    Upstream's spec (flex MLA and the standard dispatcher, which the TPU
    overrides replace) without a vision encoder.
  """
  get_config, max_context_len = kimi_k3_tpu_configs[flavor]
  context_len = seq_len or max_context_len
  if context_len > max_context_len:
    raise ValueError(
        f"Requested seq_len {context_len} exceeds max context length "
        f"{max_context_len} for flavor {flavor}"
    )
  model_spec = upstream_kimi_k3.model_registry("debugmodel")
  return dataclasses.replace(
      model_spec,
      flavor=flavor,
      model=get_config(),
      max_context_length=context_len,
  )
