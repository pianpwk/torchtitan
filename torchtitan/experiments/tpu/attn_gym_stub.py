"""Lets upstream models that import ``attn_gym`` import without it.

Upstream ``torchtitan.models.qwen3_5.gdn`` and ``torchtitan.models.kimi_k3.kda``
import the CUDA/Triton-only ``attn_gym`` package (and submodules such as
``attn_gym.linear.kda``) at module top level. Importing this module, when the
package is not installed, registers an import hook that serves a stub module
for ``attn_gym`` and every ``attn_gym.*`` submodule, so those models import on
CPU and TPU. Any attribute of a stub module is a function that raises when
called; the TPU replacements (``qwen3_5.gated_delta``,
``kimi_k3.tokamax_kda``) never call them. Import this module before the
upstream model.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types


def _stub_op(module_name: str, name: str):
  if name.startswith("__"):
    raise AttributeError(name)

  def unavailable(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        f"{module_name}.{name} is CUDA-only; on TPU, the model's TPU modules"
        " replace the layers that call it."
    )

  return unavailable


class _AttnGymStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
  """Serves a stub package for ``attn_gym`` and each of its submodules."""

  def find_spec(self, fullname, path=None, target=None):
    del path, target
    if fullname != "attn_gym" and not fullname.startswith("attn_gym."):
      return None
    return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

  def create_module(self, spec):
    module = types.ModuleType(spec.name)
    module.__path__ = []  # type: ignore[attr-defined]
    module.__getattr__ = lambda name: _stub_op(spec.name, name)  # type: ignore[method-assign]
    return module

  def exec_module(self, module):
    del module


if "attn_gym" not in sys.modules and not importlib.util.find_spec("attn_gym"):
  sys.meta_path.append(_AttnGymStubFinder())
