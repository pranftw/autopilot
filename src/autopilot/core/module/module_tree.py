"""Module subtree traversal and train/eval/apply helpers."""

from autopilot.core.parameter import Parameter
from collections.abc import Iterator
from typing import Any


def module_children(module: Any) -> Iterator[Any]:
  """Yield immediate child modules."""
  yield from module._modules.values()


def module_named_children(module: Any) -> Iterator[tuple[str, Any]]:
  """Yield ``(name, child)`` pairs for immediate children."""
  yield from module._modules.items()


def module_modules(module: Any) -> Iterator[Any]:
  """Yield self then descendants depth-first."""
  yield module
  for child in module._modules.values():
    yield from module_modules(child)


def module_named_modules(module: Any, prefix: str | None = None) -> Iterator[tuple[str, Any]]:
  """Yield ``(dotted_name, module)`` pairs for self and descendants."""
  pfx = '' if prefix is None else prefix
  yield pfx, module
  for name, child in module._modules.items():
    child_prefix = f'{pfx}.{name}' if pfx else name
    yield from module_named_modules(child, child_prefix)


def module_parameters(module: Any, recurse: bool = True) -> Iterator[Parameter]:
  """Yield owned parameters, optionally recurring into children."""
  yield from module._parameters.values()
  if recurse:
    for child in module._modules.values():
      yield from module_parameters(child, recurse=True)


def module_named_parameters(
  module: Any, prefix: str | None = None, recurse: bool = True
) -> Iterator[tuple[str, Parameter]]:
  """Yield ``(dotted_name, parameter)`` pairs."""
  pfx = '' if prefix is None else prefix
  for name, param in module._parameters.items():
    full_name = f'{pfx}.{name}' if pfx else name
    yield full_name, param
  if recurse:
    for mod_name, child in module._modules.items():
      child_prefix = f'{pfx}.{mod_name}' if pfx else mod_name
      yield from module_named_parameters(child, child_prefix, recurse=True)


def module_train(module: Any, mode: bool = True) -> Any:
  """Set training mode recursively.

  Args:
    module: Root ``Module`` subtree.
    mode: Training flag (``True`` enables training).

  Returns:
    ``module`` for call-site chaining.
  """
  module.training = mode
  for child in module._modules.values():
    module_train(child, mode)
  return module


def module_eval(module: Any) -> Any:
  """Disable training mode recursively.

  Args:
    module: Root ``Module`` subtree.

  Returns:
    ``module`` after ``train(False)``.
  """
  return module_train(module, mode=False)


def module_apply(module: Any, fn: Any) -> Any:
  """Apply ``fn`` post-order over children then ``module``.

  Args:
    module: Root ``Module`` subtree.
    fn: Callable invoked once per submodule.

  Returns:
    ``module`` after visiting the full subtree.
  """
  for child in module._modules.values():
    module_apply(child, fn)
  fn(module)
  return module
