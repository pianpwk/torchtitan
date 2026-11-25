"""Tests Kimi K3's LatentMoE on TPU against an fp64 PyTorch reference.

The reference is K3's LatentMoE written out with plain tensor ops and a dense
loop over experts: FP32-style router (sigmoid of ``x @ gate.T``), gating
weights gathered from the unbiased scores at the selected experts and divided
by their sum (+1e-20), the latent down projection, per-expert SiTU-GLU
experts, score weighting after W2, a per-token sum, RMSNorm, the up projection
and the SiTU-GLU shared experts. It runs in fp64 on CPU from the same rounded
inputs and parameters, with autograd for the input and every parameter. It
shares no code with the TPU modules or upstream's ``KimiLatentMoE``.

Top-k selection is discontinuous, so the reference takes the expert ids the
device chose and checks separately that they are the fp64 top-k of the biased
scores up to near-ties.
"""

import dataclasses
import json

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
from torchtitan.config.override import apply_overrides, OverrideConfig
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
)
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type

# Before `torchtitan.models.kimi_k3` (installs the `attn_gym` stub).
from torchtitan.experiments.tpu.kimi_k3 import latent_moe
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.models import kimi_k3
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.kimi_k3.moe import KimiLatentMoE

_OVERRIDE = "torchtitan.experiments.tpu.kimi_k3.latent_moe.tpu_latent_moe"
# Relative L2 bound vs the fp64 reference. The experts run in BF16 whatever
# the layer dtype (as upstream's), so fp32 layers are bounded like bf16 ones.
_BOUND = 3e-2
# Selected experts may differ from the fp64 top-k only where the fp64 biased
# scores of the k-th and (k+1)-th expert are this close.
_TIE_GAP = 2e-3


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
  """Relative L2, max abs, max abs / max |expected|, and cosine similarity."""
  a, e = actual.double().cpu().flatten(), expected.double().cpu().flatten()
  diff = (a - e).abs()
  return dict(
      rel_l2=float((a - e).norm() / e.norm().clamp_min(1e-30)),
      max_abs=float(diff.max()),
      max_rel=float(diff.max() / e.abs().max().clamp_min(1e-30)),
      cos=float(F.cosine_similarity(a, e, dim=0)),
  )


def _moe_config(
    flavor: str, override: str | tuple, *, num_experts: int, top_k: int
) -> KimiLatentMoE.Config:
  """The flavor's LatentMoE config with ``num_experts`` / ``top_k`` replaced.

  Only the expert count and top-k change; dims, activations, norms and the
  router are the flavor's.
  """
  model_cfg = kimi_k3.model_registry(flavor).model
  replaced = apply_overrides(OverrideConfig(imports=[override]), model_cfg)
  assert len(replaced) == sum(l.moe is not None for l in model_cfg.layers)
  cfg = next(layer.moe for layer in model_cfg.layers if layer.moe)
  routed = cfg.routed_experts
  router = cfg.router
  return dataclasses.replace(
      cfg,
      num_experts=num_experts,
      router=dataclasses.replace(
          router,
          num_experts=num_experts,
          top_k=top_k,
          gate=dataclasses.replace(router.gate, out_features=num_experts),
      ),
      routed_experts=dataclasses.replace(
          routed,
          inner_experts=dataclasses.replace(
              routed.inner_experts, num_experts=num_experts
          ),
          token_dispatcher=dataclasses.replace(
              routed.token_dispatcher, num_experts=num_experts, top_k=top_k
          ),
      ),
  )


def _build_layer(cfg: KimiLatentMoE.Config, *, seed: int) -> KimiLatentMoE:
  """Build ``cfg`` on CPU with seeded, fan-in scaled weights and bias."""
  torch.manual_seed(seed)
  layer = cfg.build()
  layer.init_states(buffer_device=torch.device("cpu"))
  with torch.no_grad():
    for name, p in layer.named_parameters():
      if name == "router.gate.weight":
        # Spread the router logits (std 2) so the top-k has clear gaps.
        p.normal_(0.0, 2.0 * p.shape[-1] ** -0.5)
      elif p.ndim >= 2:
        # Kernel-native expert weights are (E, in, out); the rest (out, in).
        fan_in = p.shape[-2] if "routed_experts" in name else p.shape[-1]
        p.normal_(0.0, fan_in**-0.5)
      else:
        p.uniform_(0.5, 1.5)
    layer.expert_bias_E.normal_(0.0, 0.1)
  return layer


def _situ_glu64(gate, up, beta, linear_beta):
  gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
  return gate * (linear_beta * torch.tanh(up / linear_beta))


def _reference(p, x_TD, expert_ids_TK, cfg, keep_TK=None):
  """K3's LatentMoE forward in fp64 from parameters ``p`` and the given routes.

  ``keep_TK`` (bool) drops routes after the gating weights are normalized, as
  a capacity-limited dispatcher does.
  """
  experts = cfg.routed_experts.inner_experts
  beta, linear_beta = experts.beta, experts.linear_beta
  scores_TE = torch.sigmoid(x_TD @ p["router.gate.weight"].T)
  weights_TK = scores_TE.gather(1, expert_ids_TK)
  weights_TK = weights_TK / (weights_TK.sum(-1, keepdim=True) + 1e-20)
  if keep_TK is not None:
    weights_TK = weights_TK * keep_TK
  latent_TL = x_TD @ p["routed_down.weight"].T
  routed_TL = torch.zeros_like(latent_TL)
  for e in range(experts.num_experts):
    token_R, slot_R = (expert_ids_TK == e).nonzero(as_tuple=True)
    if token_R.numel() == 0:
      continue
    lat_RL = latent_TL[token_R]
    h_RF = _situ_glu64(
        lat_RL @ p["routed_experts.inner_experts.w1_EDF"][e],
        lat_RL @ p["routed_experts.inner_experts.w3_EDF"][e],
        beta,
        linear_beta,
    )
    y_RL = h_RF @ p["routed_experts.inner_experts.w2_EFD"][e]
    routed_TL = routed_TL.index_add(
        0, token_R, y_RL * weights_TK[token_R, slot_R][:, None]
    )
  eps = cfg.routed_norm.eps
  normed_TL = (
      routed_TL
      * torch.rsqrt(routed_TL.pow(2).mean(-1, keepdim=True) + eps)
      * p["routed_norm.weight"]
  )
  out_TD = normed_TL @ p["routed_up.weight"].T
  shared = cfg.shared_experts
  h_TF = _situ_glu64(
      x_TD @ p["shared_experts.w1.weight"].T,
      x_TD @ p["shared_experts.w3.weight"].T,
      shared.beta,
      shared.linear_beta,
  )
  return out_TD + h_TF @ p["shared_experts.w2.weight"].T


def _check_routing(x_TD, ids_TK, p, bias_E) -> float:
  """Asserts the device's experts are the fp64 biased top-k up to near-ties.

  Returns the fraction of tokens whose expert set equals the fp64 top-k.
  """
  top_k = ids_TK.shape[1]
  biased_TE = torch.sigmoid(x_TD @ p["router.gate.weight"].T) + bias_E
  sorted_TE = biased_TE.sort(-1, descending=True).values
  ref_TK = biased_TE.topk(top_k, -1).indices
  same_T = (ref_TK.sort(-1).values == ids_TK.sort(-1).values).all(-1)
  gap_T = sorted_TE[:, top_k - 1] - sorted_TE[:, top_k]
  bad = (~same_T) & (gap_T > _TIE_GAP)
  assert not bad.any(), f"{int(bad.sum())} tokens routed off the fp64 top-k"
  return float(same_T.double().mean())


def _to_device(layer, device, dtype):
  """Moves ``layer`` to ``device``, parameters in ``dtype``.

  The MoE buffers (expert bias, token counts) stay FP32, as in training.
  """
  layer = layer.to(device=device, dtype=dtype)
  layer.expert_bias_E = layer.expert_bias_E.float()
  layer.tokens_per_expert_E = layer.tokens_per_expert_E.float()
  return layer


def _params64(layer, dtype):
  return {n: p.detach().to(dtype).double() for n, p in layer.named_parameters()}


def _run_reference(params64, x, ids, grad_out, cfg, keep=None):
  p64 = {n: t.clone().requires_grad_() for n, t in params64.items()}
  x64 = x.double().requires_grad_()
  out64 = _reference(p64, x64, ids, cfg, keep)
  out64.backward(grad_out.double())
  expected = {"out": out64.detach(), "dx": x64.grad}
  expected.update({f"d{n}": t.grad for n, t in p64.items()})
  return expected


def _log(tag, name, results):
  print(f"{tag} {json.dumps({'case': name, **results})}", flush=True)


class LatentMoETest(base_device_test.BaseAcceleratorDeviceTest):
  """``KimiLatentMoE`` with the TPU experts, one core, vs fp64."""

  def _require_tpu(self) -> None:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("MegaBlox grouped matmul needs a TPU.")

  def test_override_replaces_every_latent_moe(self):
    model_cfg = kimi_k3.model_registry("Kimi-K3").model
    replaced = apply_overrides(OverrideConfig(imports=[_OVERRIDE]), model_cfg)
    moes = [layer.moe for layer in model_cfg.layers if layer.moe]
    self.assertLen(moes, 92)
    self.assertLen(replaced, 92)
    for moe in moes:
      routed = moe.routed_experts
      self.assertIs(type(moe), KimiLatentMoE.Config)
      self.assertIs(type(routed), RoutedExperts.Config)
      self.assertIs(
          type(routed.inner_experts), latent_moe.TpuKimiGroupedExperts.Config
      )
      self.assertEqual(
          (routed.inner_experts.beta, routed.inner_experts.linear_beta),
          (4.0, 25.0),
      )
      self.assertIs(
          type(routed.token_dispatcher), TpuAllToAllTokenDispatcher.Config
      )

  @parameterized.named_parameters(
      dict(testcase_name="_debug_bf16", flavor="debugmodel",
           dtype=torch.bfloat16, num_experts=32, top_k=4, tokens=2048),
      # K3's dims (dim 7168, latent 3584, expert hidden 3072, shared 6144)
      # and top-16, with 32 experts so the fp64 reference fits in memory.
      dict(testcase_name="_k3dims_bf16", flavor="Kimi-K3",
           dtype=torch.bfloat16, num_experts=32, top_k=16, tokens=1024),
  )
  def test_layer_matches_reference(
      self, flavor, dtype, num_experts, top_k, tokens
  ):
    """Output, dx and every parameter gradient; training-mode bookkeeping."""
    self._require_tpu()
    dev = self.accelerator_device
    cfg = _moe_config(
        flavor, _OVERRIDE, num_experts=num_experts, top_k=top_k
    )
    layer = _build_layer(cfg, seed=0)
    params64 = _params64(layer, dtype)
    bias_E = layer.expert_bias_E.clone().double()
    layer = _to_device(layer, dev, dtype)
    layer.train()
    dim = cfg.router.gate.in_features
    x = torch.randn(tokens, dim).to(dtype)
    grad_out = torch.randn(tokens, dim).to(dtype)

    x_dev = x.to(dev).requires_grad_()
    ids = []
    hook = layer.router.register_forward_hook(
        lambda m, i, o: ids.append(o[1].detach().cpu())
    )
    out = layer(x_dev)
    out.backward(grad_out.to(dev))
    hook.remove()
    self.assertEqual(out.dtype, dtype)
    ids_TK = ids[0]
    agree = _check_routing(x.double(), ids_TK, params64, bias_E)
    counts_E = torch.bincount(ids_TK.flatten(), minlength=num_experts)
    torch.testing.assert_close(
        layer.tokens_per_expert_E.cpu(), counts_E.float(), rtol=0, atol=0
    )

    actual = {"out": out.detach(), "dx": x_dev.grad}
    actual.update({f"d{n}": p.grad for n, p in layer.named_parameters()})
    expected = _run_reference(params64, x, ids_TK, grad_out, cfg)
    results = {k: _errors(actual[k], e) for k, e in expected.items()}
    results["routing_agreement"] = agree
    _log("MOE_NUMERICS", f"single_{flavor}_{str(dtype)[6:]}", results)
    for key, e in expected.items():
      self.assertTrue(torch.isfinite(actual[key]).all().item(), msg=key)
      self.assertLess(results[key]["rel_l2"], _BOUND, msg=f"{key}: {results}")

  def test_rows_past_the_groups_are_ignored(self):
    """Rows outside every expert group (capacity padding) never leak.

    MegaBlox leaves those output rows uninitialized; the AG/RS dispatcher
    depends on the experts producing correct rows inside the groups and on
    the padding rows' input gradient being dropped.
    """
    self._require_tpu()
    dev = self.accelerator_device
    cfg = _moe_config(
        "debugmodel", _OVERRIDE, num_experts=4, top_k=1
    ).routed_experts.inner_experts
    torch.manual_seed(0)
    experts = cfg.build()
    experts.init_states(buffer_device=torch.device("cpu"))
    experts = experts.to(dev)
    rows, valid = 2048, 1300
    counts = torch.tensor([300, 0, 700, 300], dtype=torch.int32)
    x = torch.randn(rows, cfg.dim, device=dev).bfloat16()
    grad = torch.randn(valid, cfg.dim, device=dev).bfloat16()
    results = []
    for rows_in in (valid, rows):
      experts.zero_grad()
      out = experts(x[:rows_in], counts.to(dev))[:valid]
      out.backward(grad)
      results.append([out.detach()] + [p.grad for p in experts.parameters()])
    for full, padded in zip(*results):
      torch.testing.assert_close(padded, full, rtol=0, atol=0)


if __name__ == "__main__":
  absltest.main()
