"""Buffer registration, checkpoint state_dict, and forward hooks for Module."""

from autopilot.core.graph import RemovableHandle
from dataclasses import dataclass
from typing import Any
import json


@dataclass
class IncompatibleKeys:
  """Result of ``Module.load_state_dict`` with mismatch details.

  Attributes:
    missing_keys: Keys present in the module's ``state_dict`` but absent from
      the provided checkpoint.
    unexpected_keys: Keys present in the checkpoint but absent from the
      module's ``state_dict``.
  """

  missing_keys: list[str]
  unexpected_keys: list[str]


def register_forward_pre_hook(module: Any, fn: Any) -> RemovableHandle:
  """Register a callback run before ``forward`` inside ``Module.__call__``.

  Args:
    module: Module instance.
    fn: Callable ``(module, args, kwargs) -> None | tuple``; if it returns a
      ``(new_args, new_kwargs)`` pair, those replace the inputs forwarded to
      ``forward``.

  Returns:
    ``RemovableHandle`` for later removal from ``_forward_pre_hooks``.
  """
  handle = RemovableHandle(module._forward_pre_hooks)
  module._forward_pre_hooks[handle.id] = fn
  return handle


def register_forward_hook(module: Any, fn: Any) -> RemovableHandle:
  """Register a callback run after ``forward`` inside ``Module.__call__``.

  Args:
    module: Module instance.
    fn: Callable ``(module, args, output) -> None | Datum``; if it returns a
      ``Datum``, that replaces the output. The replacement inherits the
      original output's ``grad_fn`` for graph connectivity.

  Returns:
    ``RemovableHandle`` for later removal from ``_forward_hooks``.
  """
  handle = RemovableHandle(module._forward_hooks)
  module._forward_hooks[handle.id] = fn
  return handle


def register_buffer(module: Any, name: str, value: Any, persistent: bool = True) -> None:
  """Register a non-parameter buffer on the module.

  Args:
    module: Module instance.
    name: Attribute name for the buffer.
    value: Buffer value. Must be JSON-serializable for persistent buffers.
    persistent: If ``False``, buffer is excluded from ``state_dict()``.

  Raises:
    ValueError: If ``name`` collides with a reserved Module API name.
  """
  if name in module._RESERVED_NAMES:
    msg = f'cannot use reserved name {name!r} as a buffer'
    raise ValueError(msg)
  module._parameters.pop(name, None)
  module._modules.pop(name, None)
  module._buffers[name] = value
  if persistent:
    module._non_persistent_buffers.discard(name)
  else:
    module._non_persistent_buffers.add(name)
  object.__setattr__(module, name, value)


def state_dict(module: Any) -> dict[str, Any]:
  """Return module state for checkpointing.

  Raises:
    TypeError: If a persistent buffer value is not JSON-serializable.
  """
  state: dict[str, Any] = {}
  for name, param in module._parameters.items():
    state[name] = param.to_dict()
  for name, child in module._modules.items():
    child_state = child.state_dict()
    for key, value in child_state.items():
      state[f'{name}.{key}'] = value
  for name, value in module._buffers.items():
    if name not in module._non_persistent_buffers:
      try:
        json.dumps(value)
      except (TypeError, ValueError) as exc:
        msg = (
          f'Buffer {name!r} value of type {type(value).__name__} is not '
          f'JSON-serializable. Override state_dict()/load_state_dict() to '
          f'handle custom buffer serialization.'
        )
        raise TypeError(msg) from exc
      state[name] = value
  return state


def state_dict_keys(module: Any) -> set[str]:
  """Collect expected ``state_dict`` key names without serialization.

  Returns:
    Key set matching ``state_dict().keys()`` for this subtree.
  """
  keys: set[str] = set()
  keys.update(module._parameters)
  for child_name, child in module._modules.items():
    keys.update(f'{child_name}.{key}' for key in child.state_dict_keys())
  keys.update(name for name in module._buffers if name not in module._non_persistent_buffers)
  return keys


def load_state_dict(
  module: Any, checkpoint: dict[str, Any], strict: bool = True
) -> IncompatibleKeys:
  """Load module state from a flattened checkpoint mapping.

  Args:
    module: Module instance.
    checkpoint: Keys aligned with ``state_dict_keys`` dotted naming.
    strict: When True, require exact parity between expected and incoming keys.

  Returns:
    ``IncompatibleKeys`` with sorted missing and unexpected keys (even when
    loading proceeds under ``strict=False``).

  Raises:
    RuntimeError: When ``strict`` is True and keys mismatch.
  """
  expected_keys = state_dict_keys(module)
  incoming_keys = set(checkpoint.keys())

  missing = sorted(expected_keys - incoming_keys)
  unexpected = sorted(incoming_keys - expected_keys)

  if strict and (missing or unexpected):
    parts: list[str] = []
    if missing:
      parts.append(f'Missing key(s): {missing}')
    if unexpected:
      parts.append(f'Unexpected key(s): {unexpected}')
    msg = (
      f'Error(s) in loading state_dict for {type(module).__name__}. '
      + '. '.join(parts)
      + '. Use strict=False to ignore mismatched keys.'
    )
    raise RuntimeError(msg)

  for name, param in module._parameters.items():
    if name in checkpoint:
      param.load_from_dict(checkpoint[name])

  for name in module._buffers:
    if name not in module._non_persistent_buffers and name in checkpoint:
      module._buffers[name] = checkpoint[name]
      object.__setattr__(module, name, checkpoint[name])

  for name, child in module._modules.items():
    child_state = {}
    prefix = f'{name}.'
    for key, value in checkpoint.items():
      if key.startswith(prefix):
        child_state[key[len(prefix) :]] = value
    if child_state:
      child.load_state_dict(child_state, strict=False)

  return IncompatibleKeys(missing_keys=missing, unexpected_keys=unexpected)
