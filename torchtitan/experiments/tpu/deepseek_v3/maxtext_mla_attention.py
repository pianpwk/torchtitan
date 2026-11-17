# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek MLA attention on MaxText Lineage's fused FP8 q/kv up kernels (TPU).

Same parameters and state dict as the upstream ``Attention``. ``wq_a``,
``wkv_a`` and ``wo`` run as before; ``q_norm``, ``wq_b``, ``kv_norm``,
``wkv_b``, the rope, the q / k concat, the k_pe broadcast, the transposes and
Splash run as one op (``kernels/maxtext_lineage_mla_attention.py``), with the
up projections in FP8 (``fp8_full``) regardless of the recipe's linears.

Legend: T tokens (B * S), B batch, S sequence, N heads, V v head dim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.config import derive
from torchtitan.experiments.tpu.components.unflatten_shim import (
    get_unflatten_seq_len,
)
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.deepseek_v3.model import Attention
from torchtitan.protocols.model import ModelConfigConverter


class TpuMaxTextMLAAttention(Attention):
  """``Attention`` whose q/kv up projections and Splash run as
  ``maxtext_mla_attention``."""

  @dataclass(kw_only=True, slots=True)
  class Config(Attention.Config):
    pass

  def __init__(self, config: Config):
    super().__init__(config)
    if self.q_lora_rank == 0:
      raise ValueError("TpuMaxTextMLAAttention needs q_lora_rank > 0")
    if self.q_norm.eps != self.kv_norm.eps:
      raise ValueError("TpuMaxTextMLAAttention needs equal q / kv norm eps")

  def forward(
      self,
      x: torch.Tensor,
      attention_masks: AttentionMasksType,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ):
    from torchtitan.experiments.tpu.kernels.maxtext_lineage_mla_attention import (  # pylint: disable=g-import-not-at-top
        maxtext_mla_attention,
        rope_half_tables,
    )

    if attention_masks is not None:
      raise ValueError("TpuMaxTextMLAAttention takes segment_ids, not masks")
    if positions is None or segment_ids is None:
      raise ValueError("TpuMaxTextMLAAttention needs positions and segment_ids")
    num_tokens = x.shape[0]
    seq = get_unflatten_seq_len() or segment_ids.shape[-1]
    batch = num_tokens // seq

    q_down_BSC = self.wq_a(x).view(batch, seq, -1)
    kv_down_BSL = self.wkv_a(x).view(batch, seq, -1)
    k_pe_T1R = kv_down_BSL[..., self.kv_lora_rank:].reshape(num_tokens, 1, -1)
    cis_T1R = self.rope._reshape_cache(k_pe_T1R, positions)  # pylint: disable=protected-access
    cos_BSR, sin_BSR = rope_half_tables(cis_T1R.squeeze(1), batch, seq)
    o_BSNV = maxtext_mla_attention(
        q_down_BSC.contiguous(),
        kv_down_BSL.contiguous(),
        self.wq_b.weight,
        self.wkv_b.weight,
        self.q_norm.weight,
        self.kv_norm.weight,
        cos_BSR.contiguous(),
        sin_BSR.contiguous(),
        segment_ids.reshape(batch, seq).to(torch.int32).contiguous(),
        num_heads=self.n_heads,
        qk_nope_head_dim=self.qk_nope_head_dim,
        qk_rope_head_dim=self.qk_rope_head_dim,
        v_head_dim=self.v_head_dim,
        softmax_scale=self.softmax_scale,
        eps=self.q_norm.eps,
    )
    return self.wo(o_BSNV.reshape(num_tokens, -1))


class TpuMaxTextMLAAttentionConverter(ModelConfigConverter):
  """Select ``TpuMaxTextMLAAttention`` for every DeepSeek ``Attention``."""

  @dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    pass

  def __init__(self, config: Config):
    pass

  def convert(self, model_config):
    for _, config, parent, field_name in list(
        model_config.traverse(Attention.Config)
    ):
      if type(config) is not Attention.Config:
        continue
      replacement = derive(config, TpuMaxTextMLAAttention.Config)
      if parent is None:
        model_config = replacement
      elif isinstance(parent, list):
        parent[field_name] = replacement
      else:
        setattr(parent, field_name, replacement)
    return model_config


__all__ = ["TpuMaxTextMLAAttention", "TpuMaxTextMLAAttentionConverter"]
