"""``Module.__call__`` hook orchestration extracted for ``module.py`` size limits."""

from autopilot.core.module.operator import ModuleCallOperator
from autopilot.core.types import Datum
from typing import Any


def module_call(module: Any, *args: Any, **kwargs: Any) -> Any:
  """Run forward pre-hooks, forward via ``ModuleCallOperator``, then post-hooks.

  Returns:
    Forward output, possibly rewritten by post-hooks.
  """
  for hook in module._forward_pre_hooks.values():
    result = hook(module, args, kwargs)
    if result is not None and isinstance(result, tuple) and len(result) == 2:
      args, kwargs = result
  output = ModuleCallOperator.apply(module, *args, **kwargs)
  for hook in module._forward_hooks.values():
    result = hook(module, args, output)
    if result is not None:
      if isinstance(result, Datum) and isinstance(output, Datum) and output.grad_fn is not None:
        object.__setattr__(result, 'grad_fn', output.grad_fn)
      output = result
  return output
