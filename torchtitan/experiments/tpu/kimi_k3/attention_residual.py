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

``TpuKimiK3TransformerBlock`` is upstream's block with this function, and it
carries the block residual as a tuple of ``[T, D]`` values instead of one
stacked ``[T, N, D]`` tensor. Each value is the input of the layer that opened
its block, which full remat already saves (and offloads) as a layer boundary,
so a new block costs no copy, no stacked state is saved or offloaded per
block, and the backward never allocates a ``[T, N, D]`` buffer: with graph
split, the stacked state and its gradient were contiguous buffers of up to
N x 224 MiB (K3, 16,384 tokens) that each block opener reallocated, and on a
full HBM they made the runtime wait for the device to drain or defragment
memory before the next executable could start. ``TpuKimiK3Model`` threads the
tuple through the layers and stacks it only inside the final residual.

It is an option: a recipe builds its model config with
``use_tpu_attention_residual``; the default stays upstream's model. Not
bitwise equal to upstream's formula (see above).
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_local_context
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
  writes the N block value gradients (autograd's version builds them from N
  per-value ``select`` gradients and was 1.5x slower at N = 8). The block
  values are never stacked: every reduction reads them one by one, so XLA
  keeps no fp32 copy of them.
  """

  @staticmethod
  def forward(ctx, prefix_sum_TD, score_weight_D, eps, *block_values_TD):
    values = list(block_values_TD) + [prefix_sum_TD]
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
        prefix_sum_TD, score_weight_D, probs_TV, inv_rms_TV, scores_TV,
        *block_values_TD,
    )
    return output_TD.to(prefix_sum_TD.dtype)

  @staticmethod
  def backward(ctx, grad_TD):
    (prefix_sum_TD, score_weight_D, probs_TV, inv_rms_TV, scores_TV,
     *block_values_TD) = ctx.saved_tensors
    n = len(block_values_TD)
    dim = prefix_sum_TD.shape[-1]
    grad_float_TD = grad_TD.float()
    prefix_float_TD = prefix_sum_TD.float()
    grad_prefix_probs_T1 = (prefix_float_TD * grad_float_TD).sum(
        dim=-1, keepdim=True
    )
    # d out / d probs, then through the softmax. One reduction per value: a
    # stacked fp32 [T, N, D] operand read by two reductions is materialized
    # (2 x 3 GiB at full93 N = 7).
    grad_probs_TV = torch.cat(
        [
            (v.float() * grad_float_TD).sum(dim=-1, keepdim=True)
            for v in block_values_TD
        ]
        + [grad_prefix_probs_T1],
        dim=-1,
    )
    grad_scores_TV = probs_TV * (
        grad_probs_TV - (probs_TV * grad_probs_TV).sum(dim=-1, keepdim=True)
    )
    # score = inv_rms * <value, w>: d score / d value
    #   = inv_rms * (w - score * inv_rms * value / D).
    a_TV = grad_scores_TV * inv_rms_TV
    c_TV = scores_TV * inv_rms_TV / dim
    grad_block_values_TD = tuple(
        (
            probs_TV[:, i : i + 1] * grad_float_TD
            + a_TV[:, i : i + 1]
            * (score_weight_D - c_TV[:, i : i + 1] * value_TD.float())
        ).to(value_TD.dtype)
        for i, value_TD in enumerate(block_values_TD)
    )
    grad_prefix_TD = (
        probs_TV[:, n:] * grad_float_TD
        + a_TV[:, n:] * (score_weight_D - c_TV[:, n:] * prefix_float_TD)
    ).to(prefix_sum_TD.dtype)
    weighted_TD = a_TV[:, n:] * prefix_float_TD
    for i, value_TD in enumerate(block_values_TD):
      weighted_TD = weighted_TD + a_TV[:, i : i + 1] * value_TD.float()
    grad_weight_D = weighted_TD.sum(dim=0)
    return grad_prefix_TD, grad_weight_D, None, *grad_block_values_TD


def attention_residual(
    prefix_sum_TD: torch.Tensor,
    block_values_TD: tuple[torch.Tensor, ...],
    projection: Linear,
    norm: RMSNorm,
) -> torch.Tensor:
  """Upstream ``_apply_attention_residual`` (same math, fp32 accumulation).

  ``block_values_TD`` are the N earlier block values, each ``[T, D]``
  (upstream's ``block_residual_TND[:, i]``).
  """
  assert norm.eps is not None
  score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
  return _AttentionResidual.apply(
      prefix_sum_TD, score_weight_D, norm.eps, *block_values_TD
  )


class TpuKimiK3TransformerBlock(KimiK3TransformerBlock):
  """``KimiK3TransformerBlock`` with ``attention_residual``.

  ``forward`` is upstream's with ``_apply_attention_residual`` replaced and
  the block residual carried as a tuple of ``[T, D]`` values: a block opener
  appends its input ``x_TD`` instead of concatenating it. The block is
  subclassed because upstream's forward calls that module-level function
  directly; there is no config field or submodule to swap.

  TODO: drop this subclass, ``TpuKimiK3Model`` and
  ``use_tpu_attention_residual`` once upstream adopts this form or makes the
  attention residual a pluggable hook.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(KimiK3TransformerBlock.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      block_values_TD: tuple[torch.Tensor, ...],
      attention_masks=None,
      positions: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    prefix_sum_TD = x_TD

    if block_values_TD:
      assert self.attention_res_proj is not None
      assert self.attention_res_norm is not None
      x_TD = attention_residual(
          prefix_sum_TD,
          block_values_TD,
          self.attention_res_proj,
          self.attention_res_norm,
      )

    opens_block = self.layer_id % self.attn_res_block_size == 0
    if opens_block:
      block_values_TD = (*block_values_TD, prefix_sum_TD)

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
        block_values_TD,
        self.ffn_res_proj,
        self.ffn_res_norm,
    )
    h_TD = self.ffn_norm(h_TD)
    if self.moe is not None:
      h_TD = self.moe(h_TD)
    else:
      assert self.feed_forward is not None
      h_TD = self.feed_forward(h_TD)
    return prefix_sum_TD + h_TD, block_values_TD


class _ContiguousGrad(torch.autograd.Function):
  """Identity whose backward returns a contiguous gradient.

  The gradient of a ``torch.stack`` input is a strided view of the stacked
  gradient. Graph split passes a view across executables as its whole
  storage, so N block values would each carry the ``[T, N + 1, D]`` buffer
  (N x 2 GiB for full93); a contiguous ``[T, D]`` gradient carries 224 MiB.
  """

  @staticmethod
  def forward(ctx, value_TD):
    return value_TD.view_as(value_TD)

  @staticmethod
  def backward(ctx, grad_TD):
    return grad_TD.contiguous()


def _output_attention_residual(
    prefix_sum_TD: torch.Tensor,
    block_values_TD: tuple[torch.Tensor, ...],
    projection: Linear,
    norm: RMSNorm,
) -> torch.Tensor:
  """Upstream ``_apply_attention_residual`` on the stacked block values.

  The model's final residual keeps upstream's formula (concat + fp32
  ``matmul``), as before this module carried the values as a tuple.

  TODO: use ``attention_residual`` here too; it changes the final residual's
  numerics (see the module docstring), so it is a separate change.
  """
  assert norm.eps is not None
  values_TND = torch.stack(
      (*(_ContiguousGrad.apply(v) for v in block_values_TD), prefix_sum_TD),
      dim=1,
  )
  values_float = values_TND.float()
  variance = values_float.pow(2).mean(dim=-1, keepdim=True)
  keys_TND = values_float * torch.rsqrt(variance + norm.eps)
  score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
  scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
  probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
  output_TD = torch.matmul(probs_T1N, values_float).squeeze(1)
  return output_TD.to(values_TND.dtype)


class TpuKimiK3Model(KimiK3Model):
  """``KimiK3Model`` whose layers pass the block residual as a value tuple.

  ``forward`` is upstream's except for the block residual: it starts as an
  empty tuple, and the final residual stacks the values inside the
  executable that computes it (``_output_attention_residual``).
  """

  @dataclass(kw_only=True, slots=True)
  class Config(KimiK3Model.Config):
    pass

  def forward(  # pyrefly: ignore [bad-override]
      self,
      tokens: torch.Tensor,
      *,
      pixel_values: torch.Tensor | None = None,
      grid_thw: torch.Tensor | None = None,
      pixel_values_videos: torch.Tensor | None = None,
      grid_thw_videos: torch.Tensor | None = None,
      special_tokens: dict[str, int] | None = None,
      positions: torch.Tensor | None = None,
      attention_masks=None,
  ) -> torch.Tensor:
    if pixel_values_videos is not None or grid_thw_videos is not None:
      raise NotImplementedError("Kimi K3 v1 supports images but not videos.")
    if self.tok_embeddings is not None:
      with spmd_local_context("dp"):
        h_TD = self._prepare_multimodal_embeds(
            tokens,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            special_tokens=special_tokens,
        )
    else:
      h_TD = tokens

    if spmd.is_type_checking():
      spmd.assert_type(h_TD, {MeshAxisName.DP: spmd.S(0)})

    block_values_TD = ()
    for layer in self.layers.values():
      h_TD, block_values_TD = layer(
          h_TD,
          block_values_TD,
          attention_masks,
          positions,
      )

    h_TD = _output_attention_residual(
        h_TD,
        block_values_TD,
        self.output_res_proj,
        self.output_res_norm,
    )
    h_TD = self.norm(h_TD) if self.norm is not None else h_TD
    if self._skip_lm_head:
      return h_TD
    return self.lm_head(h_TD) if self.lm_head is not None else h_TD


def use_tpu_attention_residual(
    model_cfg: KimiK3Model.Config,
    *,
    config_cls: type[TpuKimiK3Model.Config],
) -> TpuKimiK3Model.Config:
  """``model_cfg`` as ``config_cls`` built from ``TpuKimiK3TransformerBlock``s.

  ``config_cls`` is ``TpuKimiK3Model.Config`` or a subclass that also keeps
  the recipe's model class behavior (the graph trainer's K3 model). Recipes
  call this at config build time to opt in. It is not an override: the block
  config is the parent of the KDA / MLA / MoE configs that the layer
  overrides replace, and nested overrides are rejected.
  """
  return derive(
      model_cfg,
      config_cls,
      layers=[
          derive(layer, TpuKimiK3TransformerBlock.Config)
          for layer in model_cfg.layers
      ],
  )
