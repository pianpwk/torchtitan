"""Kimi K3's block attention residual, written for XLA on TPU.

Upstream ``_apply_attention_residual`` (``torchtitan/models/kimi_k3/model.py``)
concatenates the block values into ``[T, N, D]``, copies them to fp32 and mixes
them with ``matmul(probs[T, 1, N], values[T, N, D])``. On TPU that ``matmul``
is a batched dot with M = 1 and K = N (2..10): it runs at a fraction of the
HBM bandwidth, and at the default fp32 matmul precision it rounds the
probabilities to bf16 (about 2.4 bf16 ulp mean error on the output against an
fp64 reference, vs 0.36 here).

``attention_residual`` computes the same function as elementwise passes that
XLA fuses into reductions over the bf16 values: per value the RMS-normalized
score (fp32), a softmax over the N + 1 values, and the mix as an unrolled fp32
multiply-add. No ``[T, N, D]`` concatenation or fp32 copy is materialized and
nothing lowers to a dot. The backward is written by hand
(``_AttentionResidual``) so that it saves only small ``[T, N + 1]`` fp32
tensors besides the bf16 values.
Microbenchmark, one core, T = 32768, D = 7168, fwd + bwd per call: 20.1 -> 3.9
ms at 2 values, 28.7 -> 13.4 ms at 5, 45.5 -> 20.2 ms at 9.

``TpuKimiK3TransformerBlock`` is upstream's block with this function. It is
an option: a recipe calls ``use_tpu_attention_residual`` on its model config
to build every layer with it; the default stays upstream's block. Not bitwise
equal to upstream's formula (see above).
"""

from dataclasses import dataclass

import torch
from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import RMSNorm

# Before `torchtitan.models.kimi_k3`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.config.override import derive
from torchtitan.models.kimi_k3 import KimiK3Model
from torchtitan.models.kimi_k3.model import KimiK3TransformerBlock

# Shape suffixes: T = tokens, N = earlier block values, V = N + 1 values mixed,
# D = model dim.


class _AttentionResidual(torch.autograd.Function):
  """The mix with a hand-written backward.

  Saves the bf16 values and three ``[T, V]`` fp32 tensors (V = N + 1 values):
  softmax probabilities, 1 / rms and scores. The backward then needs one
  pass over the block values for the score gradients and one fused pass that
  writes the ``[T, N, D]`` block gradient (autograd's version builds it from
  N per-value ``select`` gradients and was 1.5x slower at N = 8).
  """

  @staticmethod
  def forward(ctx, prefix_sum_TD, block_residual_TND, score_weight_D, eps):
    values = [
        block_residual_TND[:, i] for i in range(block_residual_TND.shape[1])
    ] + [prefix_sum_TD]
    inv_rms, scores = [], []
    for value_TD in values:
      value_float = value_TD.float()
      inv_rms_T = torch.rsqrt(value_float.pow(2).mean(dim=-1) + eps)
      inv_rms.append(inv_rms_T)
      scores.append(
          (value_float * inv_rms_T.unsqueeze(-1) * score_weight_D).sum(dim=-1)
      )
    inv_rms_TV = torch.stack(inv_rms, dim=-1)
    scores_TV = torch.stack(scores, dim=-1)
    probs_TV = torch.softmax(scores_TV, dim=-1)
    output_TD = probs_TV[:, 0:1] * values[0].float()
    for i in range(1, len(values)):
      output_TD = output_TD + probs_TV[:, i : i + 1] * values[i].float()
    ctx.save_for_backward(
        prefix_sum_TD, block_residual_TND, score_weight_D, probs_TV,
        inv_rms_TV, scores_TV,
    )
    return output_TD.to(prefix_sum_TD.dtype)

  @staticmethod
  def backward(ctx, grad_TD):
    (prefix_sum_TD, block_residual_TND, score_weight_D, probs_TV, inv_rms_TV,
     scores_TV) = ctx.saved_tensors
    n = block_residual_TND.shape[1]
    dim = prefix_sum_TD.shape[-1]
    grad_float_TD = grad_TD.float()
    block_float_TND = block_residual_TND.float()
    prefix_float_TD = prefix_sum_TD.float()
    # d out / d probs, then through the softmax.
    grad_probs_TV = torch.cat(
        (
            (block_float_TND * grad_float_TD.unsqueeze(1)).sum(dim=-1),
            (prefix_float_TD * grad_float_TD).sum(dim=-1, keepdim=True),
        ),
        dim=-1,
    )
    grad_scores_TV = probs_TV * (
        grad_probs_TV - (probs_TV * grad_probs_TV).sum(dim=-1, keepdim=True)
    )
    # score = inv_rms * <value, w>: d score / d value
    #   = inv_rms * (w - score * inv_rms * value / D).
    a_TV = grad_scores_TV * inv_rms_TV
    c_TV = scores_TV * inv_rms_TV / dim
    grad_block_TND = (
        probs_TV[:, :n, None] * grad_float_TD.unsqueeze(1)
        + a_TV[:, :n, None]
        * (score_weight_D - c_TV[:, :n, None] * block_float_TND)
    ).to(block_residual_TND.dtype)
    grad_prefix_TD = (
        probs_TV[:, n:] * grad_float_TD
        + a_TV[:, n:] * (score_weight_D - c_TV[:, n:] * prefix_float_TD)
    ).to(prefix_sum_TD.dtype)
    grad_weight_D = (a_TV[:, :n, None] * block_float_TND).sum(dim=(0, 1)) + (
        a_TV[:, n:] * prefix_float_TD
    ).sum(dim=0)
    return grad_prefix_TD, grad_block_TND, grad_weight_D, None


def attention_residual(
    prefix_sum_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection: Linear,
    norm: RMSNorm,
) -> torch.Tensor:
  """Upstream ``_apply_attention_residual`` (same math, fp32 accumulation)."""
  assert norm.eps is not None
  score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
  return _AttentionResidual.apply(
      prefix_sum_TD, block_residual_TND, score_weight_D, norm.eps
  )


class TpuKimiK3TransformerBlock(KimiK3TransformerBlock):
  """``KimiK3TransformerBlock`` with ``attention_residual``.

  ``forward`` is upstream's with ``_apply_attention_residual`` replaced. The
  block is subclassed because upstream's forward calls that module-level
  function directly; there is no config field or submodule to swap.

  TODO: drop this subclass and ``use_tpu_attention_residual`` once upstream
  adopts this form or makes the attention residual a pluggable hook.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(KimiK3TransformerBlock.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      block_residual_TND: torch.Tensor,
      attention_masks=None,
      positions: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_sum_TD = x_TD

    if block_residual_TND.shape[1] > 0:
      assert self.attention_res_proj is not None
      assert self.attention_res_norm is not None
      x_TD = attention_residual(
          prefix_sum_TD,
          block_residual_TND,
          self.attention_res_proj,
          self.attention_res_norm,
      )

    opens_block = self.layer_id % self.attn_res_block_size == 0
    if opens_block:
      block_residual_TND = torch.cat(
          (block_residual_TND, prefix_sum_TD.unsqueeze(1)), dim=1
      )

    h_TD = self.attention_norm(x_TD)
    layer_mask = (
        attention_masks[self.attn_mask_key]
        if attention_masks is not None
        else None
    )
    if self.attention is not None:
      h_TD = self.attention(h_TD, layer_mask, positions)
    else:
      assert self.delta_attention is not None
      h_TD = self.delta_attention(h_TD, layer_mask, positions)
    prefix_sum_TD = h_TD if opens_block else prefix_sum_TD + h_TD

    h_TD = attention_residual(
        prefix_sum_TD,
        block_residual_TND,
        self.ffn_res_proj,
        self.ffn_res_norm,
    )
    h_TD = self.ffn_norm(h_TD)
    if self.moe is not None:
      h_TD = self.moe(h_TD)
    else:
      assert self.feed_forward is not None
      h_TD = self.feed_forward(h_TD)
    return prefix_sum_TD + h_TD, block_residual_TND


def use_tpu_attention_residual(model_cfg: KimiK3Model.Config) -> None:
  """Builds every layer of ``model_cfg`` as ``TpuKimiK3TransformerBlock``.

  Recipes call this at config build time to opt in. It is not an override:
  the block config is the parent of the KDA / MLA / MoE configs that the
  layer overrides replace, and nested overrides are rejected.
  """
  model_cfg.layers = [
      derive(layer, TpuKimiK3TransformerBlock.Config)
      for layer in model_cfg.layers
  ]
