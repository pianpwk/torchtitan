"""Kimi K3 on TPU.

``override.imports`` entries:

* ``tokamax_kda.tokamax_kda_attention``: Kimi Delta Attention with Tokamax's
  Mosaic TPU kernels.
* ``tokamax_mla.tokamax_mla_attention``: gated MLA with Tokamax Splash.
* ``latent_moe.tpu_latent_moe``: LatentMoE with gmm_v2 experts.

The ``tiny`` flavor balances experts by quantile balancing alone
(``quantile_balance``; auxiliary-loss-free, as in the Kimi K3 report).

``model_registry`` builds a text-only K3 config for the TPU lane; the recipes
in ``config_registry`` add the three overrides. This module imports neither
``jax`` nor ``torch_tpu``, so a GPU run can build the same flavors.
"""

import dataclasses

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.config.override import derive
from torchtitan.experiments.tpu.kimi_k3 import quantile_balance
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
  dim / 2, sigmoid router whose expert bias is set by quantile balancing
  (upstream K3's router and 1000 bins), shared experts and an untied LM head.
  The vocabulary covers the test tokenizer (2020 tokens).
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
      layer.moe = derive(
          layer.moe,
          quantile_balance.QuantileBalancedKimiLatentMoE.Config,
          load_balance_coeff=None,
          router=derive(
              layer.moe.router,
              quantile_balance.QuantileBalancedTopKRouter.Config,
              num_bins=1000,
          ),
      )
  return cfg


# flavor -> (config builder, max context length, TPU GraphTrainer SimpleFSDP
# placements; see graph_trainer/kimi_k3.py)
kimi_k3_tpu_configs = {
    # SimpleFSDP's default placements are even up to FSDP 64.
    "tiny": (_tiny, 4096, {}),
}


def model_registry(flavor: str, *, seq_len: int | None = None) -> ModelSpec:
  """Returns a text-only ``ModelSpec`` for a TPU K3 flavor.

  Args:
    flavor: A key of ``kimi_k3_tpu_configs``.
    seq_len: Context length; defaults to the flavor's maximum.

  Returns:
    Upstream's spec (flex MLA and the standard dispatcher, which the TPU
    overrides replace) without a vision encoder, with the quantile-balancing
    hook (a no-op without quantile-balanced layers).
  """
  get_config, max_context_len, _ = kimi_k3_tpu_configs[flavor]
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
      post_optimizer_build_fn=quantile_balance.register_moe_quantile_balancing_hook,
  )
