"""Kimi K3 on TPU.

``override.imports`` entries:

* ``tokamax_kda.tokamax_kda_attention``: Kimi Delta Attention with Tokamax's
  Mosaic TPU kernels.
* ``tokamax_mla.tokamax_mla_attention``: gated MLA with Tokamax Splash.
* ``latent_moe.tpu_latent_moe``: LatentMoE with MegaBlox experts and
  Ring-of-Experts expert parallelism.

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

  Model: 67,631,896 parameters, 29,096,728 activated per token (routed
  experts at top_k / num_experts; embedding and LM head included).
  - 8 layers in K3's 3:1 pattern (K3's ``range(3, 92, 4)``): 0 KDA + dense
    FFN; 1, 2, 4, 5, 6 KDA + LatentMoE; 3, 7 gated MLA + LatentMoE.
    Attention residual blocks of 4 layers.
  - dim 512; vocab 2048 (covers the test tokenizer's 2020 tokens); untied LM
    head.
  - KDA: 4 heads x 128. MLA: 4 heads, qk 128 + 64 (no RoPE), v 128, q / kv
    LoRA ranks 256 / 128.
  - LatentMoE: 32 routed experts, top-4, expert hidden 256 on the latent dim
    256; 2 shared experts (hidden 512). Dense FFN hidden 2048.
  - Router: sigmoid, expert bias from auxiliary-loss-free quantile balancing
    (upstream K3's router, 1000 bins), as in the Kimi K3 report.
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
  # No sign-based expert-bias balancing: it reads `tokens_per_expert_E`,
  # which upstream's MoE forward accumulates in place, an update the TPU
  # compiled step does not keep.
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


def _k3_slice(num_experts: int) -> KimiK3Model.Config:
  """Three K3 layers at the released widths, one of each layer type.

  Model: ``k3_slice_e448`` 34,163,726,784 parameters (5,625,682,368
  activated), ``k3_slice_e64`` 8,791,071,168 (5,620,177,344 activated);
  routed experts at top_k / num_experts, embedding and LM head included.
  - 3 layers: 0 KDA + dense FFN, 1 KDA + LatentMoE, 2 gated MLA + LatentMoE.
  - dim 7168; vocab 163840; untied LM head.
  - KDA: 96 heads x 128. MLA: 96 heads, qk 128 + 64 (no RoPE), v 128, q / kv
    LoRA ranks 1536 / 512.
  - LatentMoE: ``num_experts`` routed experts, top-16, expert hidden 3072 on
    the latent dim 3584; 2 shared experts (hidden 6144). Dense FFN hidden
    33792.
  - Router: upstream's balanced round-robin assignment
    (``_debug_force_load_balance``), no expert-bias balancing.

  Every width is Kimi-K3's; only the layer count and the expert count change.
  The expert count sets the experts per EP rank: 448 experts over EP8 keep the
  target's 56 experts per core (896 over EP16). The forced routing gives every
  EP rank exactly its share of the routes, so capacity factor 1.0 drops
  nothing, and it is balanced by construction (quantile balancing rejects
  forced routing); this flavor is for comm / memory benchmarks, not training.
  """
  dim = 7168
  # pylint: disable-next=protected-access
  cfg = upstream_kimi_k3._kimi_k3_config(
      dim=dim,
      vocab_size=163840,
      num_layers=3,
      full_attention_layers={2},
      attn_res_block_size=12,
      num_heads=96,
      q_lora_rank=1536,
      kv_lora_rank=512,
      qk_nope_head_dim=128,
      qk_rope_head_dim=64,
      v_head_dim=128,
      kda_head_dim=128,
      conv_kernel_size=4,
      dense_hidden_dim=33792,
      latent_dim=3584,
      expert_hidden_dim=3072,
      num_experts=num_experts,
      top_k=16,
      num_shared_experts=2,
      vision_encoder=None,  # pytype: disable=wrong-arg-types
      attn_backend="flex",
  )
  for layer in cfg.layers:
    if layer.moe is not None:
      # pylint: disable-next=protected-access
      layer.moe.router._debug_force_load_balance = True
      layer.moe.load_balance_coeff = None
  return cfg


# flavor -> (config builder, max context length)
kimi_k3_tpu_configs = {
    "tiny": (_tiny, 4096),
    # K3 widths, 3 layers: 56 experts per core on EP8 (4x4x4 x ring) or a
    # smaller expert count for debugging on small slices.
    "k3_slice_e448": (lambda: _k3_slice(448), 4096),
    "k3_slice_e64": (lambda: _k3_slice(64), 4096),
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
      post_optimizer_build_fn=quantile_balance.register_moe_quantile_balancing_hook,
  )
