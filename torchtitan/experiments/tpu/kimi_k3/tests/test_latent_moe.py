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

The EP test runs on 8 TPU cores: every rank routes its own tokens, and the
same layer runs without EP (all experts on every rank) and with the
Ring-of-Experts transport (drop-free, drop-free in two token chunks, and with
a tight capacity that drops routes); each is compared to the reference over
the global batch. The upstream all-to-all transport is not tested: its
variable splits give asymmetric all-to-all buffers, which torch_tpu rejects.
"""

import dataclasses
import json

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchtitan.config.override import apply_overrides, OverrideConfig
from torchtitan.distributed.parallel_dims import init_device_mesh
from torchtitan.distributed.spmd_types import set_spmd_meshes
from torchtitan.experiments.tpu.components.grouped_experts import (
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.components.ring_of_experts import (
    ring_of_experts_capacity,
)
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExperts,
)
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.distributed import distributed

# Before `torchtitan.models.kimi_k3` (installs the `attn_gym` stub).
from torchtitan.experiments.tpu.kimi_k3 import latent_moe
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.testing import base_distributed_device_test
from torchtitan.models import kimi_k3
from torchtitan.models.kimi_k3.moe import KimiLatentMoE

_OVERRIDE = "torchtitan.experiments.tpu.kimi_k3.latent_moe.tpu_latent_moe"
_EP = 8
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
  # print, not absl logging: EP workers are spawned without absl's handlers.
  print(f"{tag} {json.dumps({'case': name, **results})}", flush=True)


class LatentMoETest(base_device_test.BaseAcceleratorDeviceTest):
  """``KimiLatentMoE`` with the TPU experts, one core, vs fp64."""

  def _require_tpu(self) -> None:
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("MegaBlox grouped matmul needs a TPU.")

  def test_override_replaces_every_latent_moe(self):
    model_cfg = kimi_k3.model_registry("Kimi-K3").model
    override = (_OVERRIDE, {"capacity_factor": 2.0, "num_moe_token_chunks": 2})
    replaced = apply_overrides(OverrideConfig(imports=[override]), model_cfg)
    moes = [layer.moe for layer in model_cfg.layers if layer.moe]
    self.assertLen(moes, 92)
    self.assertLen(replaced, 92)
    for moe in moes:
      routed = moe.routed_experts
      self.assertIs(type(moe), latent_moe.TpuKimiLatentMoE.Config)
      self.assertIs(type(routed), TpuRoutedExperts.Config)
      self.assertEqual(routed.num_moe_token_chunks, 2)
      self.assertIs(
          type(routed.inner_experts), latent_moe.TpuKimiGroupedExperts.Config
      )
      self.assertEqual(
          (routed.inner_experts.beta, routed.inner_experts.linear_beta),
          (4.0, 25.0),
      )
      dispatcher = routed.token_dispatcher
      self.assertIs(type(dispatcher), TpuAllToAllTokenDispatcher.Config)
      self.assertEqual(
          (dispatcher.ep_transport, dispatcher.capacity_factor),
          ("ring_of_experts", 2.0),
      )
    # The generic expert converter keeps the TPU K3 experts, and refuses
    # upstream's K3 experts rather than turning their SiTU into SiLU.
    converter = TpuGroupedExpertsConverter.Config().build()
    converter.convert(model_cfg)
    self.assertIs(
        type(moes[0].routed_experts.inner_experts),
        latent_moe.TpuKimiGroupedExperts.Config,
    )
    with self.assertRaisesRegex(ValueError, "KimiGroupedExperts"):
      converter.convert(kimi_k3.model_registry("Kimi-K3").model)

  @parameterized.named_parameters(
      dict(testcase_name="_debug_bf16", flavor="debugmodel",
           dtype=torch.bfloat16, num_experts=32, top_k=4, tokens=2048),
      dict(testcase_name="_debug_fp32", flavor="debugmodel",
           dtype=torch.float32, num_experts=32, top_k=4, tokens=2048),
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


def _ep_worker(device, rank, world_size, flavor, num_experts, top_k, tokens):
  """Runs the layer without EP and with AG/RS EP on this rank's tokens."""
  dtype = torch.bfloat16
  mesh = init_device_mesh(
      "tpu", (world_size,), ("ep",), enable_tpu_topo_mesh=True
  )
  set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh)
  # The topology-aware EP order need not be the global rank order: gather in
  # EP order.
  group = mesh.get_group()
  ep_rank = mesh.get_local_rank()
  num_local = num_experts // world_size
  local = slice(ep_rank * num_local, (ep_rank + 1) * num_local)
  dim = None
  modes = {
      "single": (_OVERRIDE, False),
      "ep_roe": (_OVERRIDE, True),
      "ep_roe_chunks2": ((_OVERRIDE, {"num_moe_token_chunks": 2}), True),
      # One routed row per local route on average: busy ranks drop routes.
      "ep_roe_capacity1": ((_OVERRIDE, {"capacity_factor": 1.0}), True),
  }
  runs = {}
  for mode, (override, use_ep) in modes.items():
    cfg = _moe_config(flavor, override, num_experts=num_experts, top_k=top_k)
    layer = _build_layer(cfg, seed=0)
    params64 = _params64(layer, dtype)
    bias_E = layer.expert_bias_E.clone().double()
    dim = cfg.router.gate.in_features
    torch.manual_seed(1)
    x_global = torch.randn(world_size * tokens, dim).to(dtype)
    grad_global = torch.randn(world_size * tokens, dim).to(dtype)
    mine = slice(ep_rank * tokens, (ep_rank + 1) * tokens)
    experts = layer.routed_experts.inner_experts
    if use_ep:
      with torch.no_grad():
        for name in ("w1_EDF", "w2_EFD", "w3_EDF"):
          full = getattr(experts, name)
          setattr(experts, name, torch.nn.Parameter(full[local].clone()))
    layer = _to_device(layer, device, dtype)
    if use_ep:
      layer.routed_experts.token_dispatcher.wire_meshes(ep_mesh=mesh)
    ids = []
    hook = layer.router.register_forward_hook(
        lambda m, i, o: ids.append(o[1].detach())
    )
    x_dev = x_global[mine].to(device).requires_grad_()
    out = layer(x_dev)
    out.backward(grad_global[mine].to(device))
    hook.remove()

    ids_GK = torch.empty(
        world_size * tokens, top_k, dtype=ids[0].dtype, device=device
    )
    dist.all_gather_into_tensor(ids_GK, ids[0].contiguous(), group=group)
    out_G = torch.empty(world_size * tokens, dim, dtype=dtype, device=device)
    dist.all_gather_into_tensor(out_G, out.detach().contiguous(), group=group)
    dx_G = torch.empty_like(out_G)
    dist.all_gather_into_tensor(dx_G, x_dev.grad.contiguous(), group=group)
    grads = {}
    for name, p in layer.named_parameters():
      g = p.grad.float()
      if use_ep and "routed_experts" in name:
        full = torch.empty(
            world_size * g.shape[0], *g.shape[1:], device=device
        )
        dist.all_gather_into_tensor(full, g.contiguous(), group=group)
        grads[name] = full.cpu()
      else:
        dist.all_reduce(g, group=group)
        grads[name] = g.cpu()
    ids_GK = ids_GK.cpu()
    keep = None
    if mode == "ep_roe_capacity1":
      keep = _capacity_keep(
          ids_GK, layer.routed_experts.token_dispatcher, world_size, tokens
      )
    agree = _check_routing(x_global.double(), ids_GK, params64, bias_E)
    actual = {"out": out_G.cpu(), "dx": dx_G.cpu()}
    actual.update({f"d{n}": g for n, g in grads.items()})
    expected = _run_reference(
        params64, x_global, ids_GK, grad_global, cfg, keep
    )
    results = {k: _errors(actual[k], e) for k, e in expected.items()}
    results["routing_agreement"] = agree
    if keep is not None:
      results["dropped_routes"] = int((~keep).sum())
    runs[mode] = (actual, ids_GK)
    if rank == 0:
      _log("MOE_EP_NUMERICS", f"{mode}_{flavor}_E{num_experts}_K{top_k}",
           results)
    for key in expected:
      assert torch.isfinite(actual[key]).all(), f"{mode} {key}"
      assert results[key]["rel_l2"] < _BOUND, f"{mode} {key}: {results[key]}"
    if keep is not None:
      assert not keep.all(), "the tight capacity should drop routes"

  # Every mode routes the same per-rank tokens, so the ids are identical and
  # the modes differ only in the order of their sums.
  base, base_ids = runs["single"]
  for mode in ("ep_roe", "ep_roe_chunks2"):
    actual, ids_GK = runs[mode]
    assert torch.equal(ids_GK, base_ids), mode
    results = {k: _errors(actual[k], base[k]) for k in base}
    if rank == 0:
      _log("MOE_EP_VS_SINGLE", f"{mode}_{flavor}_E{num_experts}_K{top_k}",
           results)
    for key, err in results.items():
      assert err["rel_l2"] < 1e-2, f"{mode} vs single {key}: {err}"


def _capacity_keep(ids_GK, dispatcher, world_size, tokens):
  """Which gathered routes a capacity-limited Ring-of-Experts rank computes.

  Each rank takes the routes to its experts in (local expert, gathered route)
  order and keeps the first ``max_R``.
  """
  num_routes, top_k = ids_GK.numel(), ids_GK.shape[1]
  num_local = dispatcher.num_experts // world_size
  max_rows = ring_of_experts_capacity(
      tokens * top_k, world_size, dispatcher.capacity_factor
  )
  keep = torch.zeros(num_routes, dtype=torch.bool)
  flat = ids_GK.flatten()
  for r in range(world_size):
    mine = ((flat // num_local) == r).nonzero().flatten()
    order = mine[torch.argsort(flat[mine] - r * num_local, stable=True)]
    keep[order[:max_rows]] = True
  return keep.reshape(ids_GK.shape)


class LatentMoEDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """EP over 8 TPU cores (2x2x1 slice) vs single device and fp64."""

  @parameterized.named_parameters(
      dict(testcase_name="_debug", flavor="debugmodel", num_experts=32,
           top_k=4, tokens=512),
      # K3's routing ratio: top-16, 16 experts per rank (K3 at EP16 has 56).
      dict(testcase_name="_debug_k16", flavor="debugmodel", num_experts=128,
           top_k=16, tokens=512),
  )
  def test_ep_matches_reference(self, flavor, num_experts, top_k, tokens):
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("EP test needs 8 TPU cores.")
    distributed.run_distributed(
        _EP,
        self.accelerator_device_type,
        _ep_worker,
        flavor,
        num_experts,
        top_k,
        tokens,
    )


if __name__ == "__main__":
  absltest.main()
