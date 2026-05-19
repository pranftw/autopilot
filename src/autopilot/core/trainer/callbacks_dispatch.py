"""Lightning-style callback hook dispatch for Trainer."""

from typing import Any


def dispatch_callbacks(trainer: Any, hook_name: str, **kwargs: Any) -> list[Any]:
  """Invoke ``hook_name`` on each callback, passing ``trainer`` and ``module``.

  Args:
    trainer: Trainer instance whose ``_callbacks`` and ``_module`` are used.
    hook_name: Callback method name (e.g. ``'on_epoch_start'``).
    **kwargs: Extra keyword arguments forwarded to each hook.

  Returns:
    List of non-``None`` return values from matching callback methods.
  """
  results: list[Any] = []
  for callback in trainer._callbacks:
    method = getattr(callback, hook_name, None)
    if method and callable(method):
      result = method(trainer=trainer, module=trainer._module, **kwargs)
      if result is not None:
        results.append(result)
  return results


def should_stop_at(hook_method: Any, **kwargs: Any) -> bool:
  """Return True if any hook result is a dict with ``stop`` equal to ``True`` (identity).

  Args:
    hook_method: Zero-arg callable that returns callback hook results (typically a list).
    **kwargs: Forwarded to ``hook_method``.

  Returns:
    Whether any entry is a dict whose ``'stop'`` key is exactly ``True``.
  """
  hook_results = hook_method(**kwargs)
  if not isinstance(hook_results, list):
    return False
  return any(isinstance(entry, dict) and entry.get('stop') is True for entry in hook_results)
