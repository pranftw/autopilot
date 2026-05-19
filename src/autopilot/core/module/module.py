"""Experiment ``Module`` container API (registration, traversal, checkpoints).

No ``module.log()`` / ``log_dict()`` surface; metrics flow through ``Metric``
objects and Trainer callbacks. Gradient clipping is undefined (non-differentiable).

``IncompatibleKeys`` is defined in ``state.py`` but exported from this module.

Import via::

  from autopilot.core.module.module import Module, IncompatibleKeys
"""

from autopilot.core.graph import RemovableHandle
from autopilot.core.module.module_repr import format_module_repr
from autopilot.core.module.module_runtime import module_call
from autopilot.core.module.module_tree import (
  module_apply as tree_apply,
)
from autopilot.core.module.module_tree import (
  module_children as tree_children,
)
from autopilot.core.module.module_tree import (
  module_eval as tree_eval,
)
from autopilot.core.module.module_tree import (
  module_modules as tree_modules,
)
from autopilot.core.module.module_tree import (
  module_named_children as tree_named_children,
)
from autopilot.core.module.module_tree import (
  module_named_modules as tree_named_modules,
)
from autopilot.core.module.module_tree import (
  module_named_parameters as tree_named_parameters,
)
from autopilot.core.module.module_tree import (
  module_parameters as tree_parameters,
)
from autopilot.core.module.module_tree import (
  module_train as tree_train,
)
from autopilot.core.module.state import (
  IncompatibleKeys,
)
from autopilot.core.module.state import (
  load_state_dict as module_load_state_dict,
)
from autopilot.core.module.state import (
  register_buffer as module_register_buffer,
)
from autopilot.core.module.state import (
  register_forward_hook as module_register_forward_hook,
)
from autopilot.core.module.state import (
  register_forward_pre_hook as module_register_forward_pre_hook,
)
from autopilot.core.module.state import (
  state_dict as module_state_dict,
)
from autopilot.core.module.state import (
  state_dict_keys as module_state_dict_keys,
)
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any


class Module:
  """Base class for experiment modules (nn.Module-shaped registration + traversal).

  Attributes:
    training: Training-mode flag propagated by ``train`` / ``eval``.
    _modules: Registered child modules by attribute name.
    _parameters: Owned ``Parameter`` leaves by attribute name.
    _buffers: Non-optimizable state installed via ``register_buffer``.
    _non_persistent_buffers: Buffer names omitted from checkpoints.
    _forward_pre_hooks: Ordered dict of pre-forward callbacks.
    _forward_hooks: Ordered dict of post-forward callbacks.

  Forward / backward:
    Implement ``forward(*args, **kwargs) -> Datum``. Invoke ``self(...)`` during
    training so hooks + graph recording run; calling ``forward()`` directly skips
    capture. Override ``backward_transform`` when custom gradient shaping is needed.

  Example:
    >>> from autopilot.core.module.module import Module
    >>> from autopilot.core.parameter import ScalarParameter
    >>>
    >>> class MyModule(Module):
    ...   def __init__(self):
    ...     super().__init__()
    ...     self.prompt = ScalarParameter(value='default prompt')
    ...
    ...   def forward(self, data):
    ...     return data
    >>>
    >>> module = MyModule()
    >>> module.prompt.value
    'default prompt'

  See ``AutoPilotModule`` for the Trainer-integrated subclass with step methods.
  """

  def __init__(self) -> None:
    """Create empty parameter/module/buffer registries and hook queues."""
    object.__setattr__(self, '_modules', {})
    object.__setattr__(self, '_parameters', {})
    object.__setattr__(self, '_buffers', {})
    object.__setattr__(self, '_non_persistent_buffers', set())
    object.__setattr__(self, '_forward_pre_hooks', OrderedDict())
    object.__setattr__(self, '_forward_hooks', OrderedDict())
    object.__setattr__(self, 'training', True)

  _RESERVED_NAMES: frozenset[str] = frozenset(
    {
      'eval',
      'train',
      'parameters',
      'named_parameters',
      'modules',
      'named_modules',
      'children',
      'named_children',
      'state_dict',
      'load_state_dict',
      'to_dict',
      'from_dict',
    }
  )

  def __setattr__(self, name: str, value: object) -> None:
    """Auto-register child Modules and Parameters. Like nn.Module.__setattr__.

    Competing-store cleanup: remove name from other internal dicts before adding.
    Buffers are only installed via ``register_buffer``; direct assignment to
    a name that was previously a buffer clears the buffer entry.

    Raises:
      AttributeError: When fields are assigned before ``Module.__init__`` runs.
      ValueError: If ``name`` is reserved for a ``Module``/``Parameter`` child
        (eval, train, parameters, named_parameters, modules, named_modules,
        children, named_children, state_dict, load_state_dict, to_dict, from_dict).
    """
    params = self.__dict__.get('_parameters')
    if params is None:
      msg = 'cannot assign before Module.__init__() call'
      raise AttributeError(msg)
    modules = self.__dict__.get('_modules')
    if modules is None:
      msg = 'cannot assign before Module.__init__() call'
      raise AttributeError(msg)

    if isinstance(value, (Module, Parameter)) and name in self._RESERVED_NAMES:
      msg = f'cannot use reserved name {name!r} as a child module or parameter'
      raise ValueError(msg)

    buffers = self.__dict__.get('_buffers', {})
    non_persistent = self.__dict__.get('_non_persistent_buffers', set())

    if isinstance(value, Parameter):
      modules.pop(name, None)
      buffers.pop(name, None)
      non_persistent.discard(name)
      params[name] = value
    elif isinstance(value, Module):
      params.pop(name, None)
      buffers.pop(name, None)
      non_persistent.discard(name)
      modules[name] = value
    else:
      params.pop(name, None)
      modules.pop(name, None)
      buffers.pop(name, None)
      non_persistent.discard(name)

    object.__setattr__(self, name, value)

  def __getattr__(self, name: str) -> Any:
    """Fallback lookup in _parameters, _modules, _buffers. Like nn.Module.__getattr__.

    Returns:
      Registered ``Parameter``, child ``Module``, or buffer value.

    Raises:
      AttributeError: When ``name`` is absent from all registries.
    """
    parameters_dict = self.__dict__.get('_parameters')
    if parameters_dict is not None and name in parameters_dict:
      return parameters_dict[name]
    modules_dict = self.__dict__.get('_modules')
    if modules_dict is not None and name in modules_dict:
      return modules_dict[name]
    buffers_dict = self.__dict__.get('_buffers')
    if buffers_dict is not None and name in buffers_dict:
      return buffers_dict[name]
    msg = f"'{type(self).__name__}' object has no attribute '{name}'"
    raise AttributeError(msg)

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    """Primary computation. Like nn.Module.forward().

    Override in subclasses. Signature is flexible -- takes runtime data only.
    """
    raise NotImplementedError

  def backward_transform(self, ctx: Any, grad_output: Any) -> Any:
    """Per-module custom backward. Return None to use default passthrough.

    When non-None: if single Gradient, broadcast to all next_functions.
    If tuple matching ctx.n_next_functions, use directly.

    Returns:
      ``None`` for passthrough, or gradients matching operator fan-out rules.
    """
    return None

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Forward pre-hooks, graph-captured ``forward``, then post-hooks.

    Note:
      Always call ``module(datum)`` (or ``self(datum)``), not
      ``module.forward(datum)``, when graph recording is required. Calling
      ``forward()`` directly bypasses the ``ModuleCallOperator`` graph node.

    Returns:
      Forward output, optionally replaced by post-hooks.
    """
    return module_call(self, *args, **kwargs)

  def register_forward_pre_hook(self, fn: Any) -> RemovableHandle:
    """Register a callback run before ``forward`` inside ``__call__``.

    Args:
      fn: Callable ``(module, args, kwargs) -> None | tuple``; if it returns a
        ``(new_args, new_kwargs)`` pair, those replace the inputs forwarded to
        ``forward``.

    Returns:
      ``RemovableHandle`` for later removal from ``_forward_pre_hooks``.
    """
    return module_register_forward_pre_hook(self, fn)

  def register_forward_hook(self, fn: Any) -> RemovableHandle:
    """Register a callback run after ``forward`` inside ``__call__``.

    Args:
      fn: Callable ``(module, args, output) -> None | Datum``; if it returns a
        ``Datum``, that replaces the output. The replacement inherits the
        original output's ``grad_fn`` for graph connectivity.

    Returns:
      ``RemovableHandle`` for later removal from ``_forward_hooks``.
    """
    return module_register_forward_hook(self, fn)

  # buffer registration

  def register_buffer(self, name: str, value: Any, persistent: bool = True) -> None:
    """Register a non-parameter buffer on this module. Like nn.Module.register_buffer.

    Buffers participate in ``state_dict`` (when ``persistent=True``) but are
    excluded from ``parameters()``. Re-registering the same name replaces the
    previous buffer entry.

    Args:
      name: Attribute name for the buffer.
      value: Buffer value. Must be JSON-serializable (dict, list, str, float,
        int, bool, None) for persistent buffers; non-serializable values raise
        ``TypeError`` at ``state_dict()`` time.
      persistent: If ``False``, buffer is excluded from ``state_dict()``.
    """
    module_register_buffer(self, name, value, persistent=persistent)

  # zero grad

  def zero_grad(self) -> None:
    """Clear ``grad`` / ``grad_accumulator`` on every parameter (prefer optimizer hooks)."""
    for param in self.parameters():
      param.grad = None
      param.grad_accumulator = None

  # freeze / unfreeze

  def requires_grad_(self, requires_grad: bool = True) -> 'Module':
    """Set ``requires_grad`` on all parameters recursively. Like nn.Module.requires_grad_.

    Args:
      requires_grad: Whether all parameters should require gradient.

    Returns:
      ``self`` for chaining.
    """
    for param in self.parameters():
      param.requires_grad = requires_grad
    return self

  # tree traversal

  def children(self) -> Iterator['Module']:
    """Yield immediate child modules."""
    yield from tree_children(self)

  def named_children(self) -> Iterator[tuple[str, 'Module']]:
    """Yield ``(name, child)`` pairs for direct children."""
    yield from tree_named_children(self)

  def modules(self) -> Iterator['Module']:
    """Yield ``self`` then descendants depth-first."""
    yield from tree_modules(self)

  def named_modules(self, prefix: str | None = None) -> Iterator[tuple[str, 'Module']]:
    """Yield ``(dotted_name, module)`` for ``self`` and descendants."""
    yield from tree_named_modules(self, prefix)

  def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    """Yield owned parameters (optionally recurse into children)."""
    yield from tree_parameters(self, recurse=recurse)

  def named_parameters(
    self, prefix: str | None = None, recurse: bool = True
  ) -> Iterator[tuple[str, Parameter]]:
    """Yield ``(dotted_name, parameter)`` pairs across this subtree."""
    yield from tree_named_parameters(self, prefix, recurse=recurse)

  # train/eval mode

  def train(self, mode: bool = True) -> 'Module':
    """Recursively set ``training``.

    Returns:
      ``self``.
    """
    return tree_train(self, mode)

  def eval(self) -> 'Module':
    """Disable training mode recursively.

    Returns:
      ``self``.
    """
    return tree_eval(self)

  # apply

  def apply(self, fn: Any) -> 'Module':
    """Post-order ``fn`` on children then ``self``.

    Returns:
      ``self``.
    """
    return tree_apply(self, fn)

  # state dict

  def state_dict(self) -> dict[str, Any]:
    """Checkpoint dict (parameters, subtree state, persistent buffers).

    Returns:
      Nested-compatible flat mapping suitable for JSON checkpoints.
    """
    return module_state_dict(self)

  def state_dict_keys(self) -> set[str]:
    """Expected ``state_dict`` keys without serializing buffers.

    Returns:
      Key set matching ``state_dict().keys()`` for this subtree.
    """
    return module_state_dict_keys(self)

  def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> IncompatibleKeys:
    """Hydrate parameters/buffers/children from a flattened checkpoint mapping.

    Args:
      state_dict: Keys aligned with ``state_dict_keys``.
      strict: Require exact key parity when True.

    Returns:
      Missing/unexpected key lists (sorted).
    """
    return module_load_state_dict(self, state_dict, strict=strict)

  def extra_repr(self) -> str:
    """Hook for ``__repr__`` suffix text.

    Returns:
      Extra parenthetical fragment or empty string.
    """
    return ''

  def __repr__(self) -> str:
    """Pretty-print subtree via ``format_module_repr``.

    Returns:
      Multi-line repr string.
    """
    return format_module_repr(self)
