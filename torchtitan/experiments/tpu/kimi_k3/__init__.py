"""Kimi K3 on TPU.

``override.imports`` entries:

* ``tokamax_kda.tokamax_kda_attention``: Kimi Delta Attention with Tokamax's
  Mosaic TPU kernels.
* Options, each in place of the entry above it: ``tokamax_kda.
  tokamax_kda_attention_pallas_conv_norm`` (Pallas short conv and output
  norm) and ``fused_kda.fused_kda_attention`` (the fused Pallas KDA kernels,
  ``kernels/kda.py``, with the Pallas conv and norm).
* ``tokamax_mla.tokamax_mla_attention``: gated MLA with Tokamax Splash.
* ``latent_moe.tpu_latent_moe``: LatentMoE with MegaBlox experts and
  Ring-of-Experts expert parallelism.
* Option, in place of it: ``latent_moe.tpu_latent_moe_gmm_glu`` (W1 / W3 and
  the SiTU-GLU fused in ``kernels/gmm_glu.py``, W2 on Tokamax's ``gmm_v2`` /
  ``tgmm_v2``).

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


def _k3_slice(
    num_experts: int,
    num_layers: int = 3,
    full_attention_layers: set[int] | None = None,
) -> KimiK3Model.Config:
  """K3 layers at the released widths (three by default, one of each type).

  Layer 0 is KDA + dense FFN, layer 1 KDA + LatentMoE, layer 2 gated MLA +
  LatentMoE. Every width is Kimi-K3's (dim 7168, vocab 163840, 96 heads,
  latent 3584, expert hidden 3072, top-16, 2 shared experts, dense hidden
  33792); only the layer count and the expert count change. The expert count
  sets the experts per EP rank: 448 experts over EP8 keep the target's 56
  experts per core (896 over EP16). The router uses upstream's balanced
  round-robin assignment (``_debug_force_load_balance``), so every EP rank
  receives exactly its share of the routes and capacity factor 1.0 drops
  nothing (set through ``debug.moe_force_load_balance`` in the
  ``graph_trainer_kimi_k3_slice_*`` / ``deep8`` configs, which the trainer
  copies onto every router); no expert-bias balancing. This flavor is for
  comm / memory benchmarks, not training.

  With ``num_layers`` > 3 the layers follow K3's own mix: gated MLA at every
  fourth layer (3, 7, ...), KDA elsewhere, layer 0 dense, unless
  ``full_attention_layers`` gives the MLA layers explicitly.
  """
  dim = 7168
  # pylint: disable-next=protected-access
  cfg = upstream_kimi_k3._kimi_k3_config(
      dim=dim,
      vocab_size=163840,
      num_layers=num_layers,
      full_attention_layers=(
          full_attention_layers
          if full_attention_layers is not None
          else {2} if num_layers == 3 else set(range(3, num_layers, 4))
      ),
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
      layer.moe.load_balance_coeff = None
  return cfg


# flavor -> (config builder, max context length)
kimi_k3_tpu_configs = {
    "tiny": (_tiny, 4096),
    # K3 widths, 3 layers: 56 experts per core on EP8 (4x4x4 x ring) or a
    # smaller expert count for debugging on small slices.
    "k3_slice_e448": (lambda: _k3_slice(448), 4096),
    "k3_slice_e64": (lambda: _k3_slice(64), 4096),
    # 896 experts: 56 per core on EP16 (8x8x8 x ring).
    "k3_slice_e896": (lambda: _k3_slice(896), 4096),
    # Eight layers in K3's KDA:MLA mix (MLA at 3 and 7).
    "k3_deep8_e448": (lambda: _k3_slice(448, num_layers=8), 4096),
    "k3_deep8_e896": (lambda: _k3_slice(896, num_layers=8), 4096),
    # Half of K3's 93 layers at full widths: K3 has MLA at range(3, 92, 4) plus
    # the last layer (24 MLA + 68 KDA + MoE + layer 0 dense); halved, MLA at
    # range(3, 46, 4) plus the last layer 46: 12 MLA + 34 KDA + MoE + 1 dense.
    "k3_half47_e896": (
        lambda: _k3_slice(
            896, num_layers=47, full_attention_layers=set(range(3, 46, 4)) | {46}
        ),
        4096,
    ),
    # Full-depth K3 (93 layers, MLA at range(3, 92, 4) plus layer 92) at the
    # released widths: 24 MLA + MoE, 68 KDA + MoE, 1 KDA + dense.
    "k3_full93_e896": (
        lambda: _k3_slice(
            896, num_layers=93, full_attention_layers=set(range(3, 92, 4)) | {92}
        ),
        4096,
    ),
    # 12 layers at K3 widths (9 KDA + 3 MLA), 64 experts: optimizer-bucketing
    # benchmarks with many same-shape matrices on 4x4x4.
    "k3_slice_l12_e64": (lambda: _k3_slice(64, num_layers=12), 4096),
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
