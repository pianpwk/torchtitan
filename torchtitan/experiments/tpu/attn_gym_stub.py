"""Lets upstream models that import ``attn_gym`` import without it.

Upstream ``torchtitan.models.qwen3_5.gdn`` and ``torchtitan.models.kimi_k3.kda``
import the CUDA/Triton-only ``attn_gym`` package at module top level. Importing
this module registers stub ``attn_gym`` modules (the ones those models import)
when the package is not installed, so they import on CPU and TPU. The stub ops
raise if called; the TPU replacements (``qwen3_5.gated_delta``,
``kimi_k3.tokamax_kda``) never call them. Import this module before the
upstream model.
"""

import importlib.util
import sys
import types

_STUB_MODULES = (
    "attn_gym",
    "attn_gym.linear",
    # Kimi K3 (``models/kimi_k3/kda.py``).
    "attn_gym.linear.kda",
    "attn_gym.linear.kda.fwd",
    "attn_gym.linear.kda.fwd.triton",
    "attn_gym.linear.kda.fwd.triton.l2norm_fwd",
    "attn_gym.linear.short_conv",
)


def _stub_op(module_name: str):
  def stub_attr(name: str):
    if name.startswith("__"):
      raise AttributeError(name)

    def unavailable(*args, **kwargs):
      del args, kwargs
      raise RuntimeError(
          f"{module_name}.{name} is CUDA-only; on TPU, the model's TPU modules"
          " replace the layers that call it."
      )

    return unavailable

  return stub_attr


if "attn_gym" not in sys.modules and not importlib.util.find_spec("attn_gym"):
  for _name in _STUB_MODULES:
    _module = types.ModuleType(_name)
    _module.__path__ = []  # type: ignore[attr-defined]
    _module.__getattr__ = _stub_op(_name)  # type: ignore[method-assign]
    sys.modules[_name] = _module
