"""Parameter base class and ScalarParameter. Like nn.Parameter.

Parameter is a Datum that the optimizer is allowed to modify.
Assigned as Module attributes, auto-registered by Module.__setattr__
into _parameters. module.parameters() collects all.

Built-in subclasses:
  - PathParameter (ai/parameter.py): filesystem-scoped parameters.
  - ScalarParameter: JSON-compatible scalar, text, or small structured values.

Datum.id: every Datum (including Parameter) gets an auto-generated,
internal, immutable id (12-char hex from uuid4). Used by CollationResult
to key gradients to parameters.

Misuse kwargs (value, data, text, content, prompt) are rejected early with
guided errors directing users to PathParameter, ScalarParameter, or subclassing.
Subclasses that legitimately accept a banned kwarg (e.g. ScalarParameter accepts
``value=``) are detected via signature inspection and exempted automatically.
"""

from autopilot.core.errors import StoreError
from autopilot.core.gradient import Gradient
from autopilot.core.snapshot import ParameterSchemaEntry
from autopilot.core.types import Datum, _hydrate_datum_base
from dataclasses import dataclass, field
from typing import Any
import functools
import inspect
import json

COMMON_WRONG_KWARGS = frozenset({'value', 'data', 'text', 'content', 'prompt'})


@dataclass
class Parameter(Datum):
  """Declared mutable scope for the optimizer.

  Like nn.Parameter IS-A Tensor, Parameter IS-A Datum.
  requires_grad controls whether the optimizer targets this parameter.
  grad holds a Gradient instance (or None) after Graph.backward()
  (AccumulateGrad materializes it during backward traversal).

  Two versioning protocols (separate concerns):
    snapshot() / restore()           -- content versioning via Store
    state_dict() / load_state_dict() -- checkpoint serialization

  Public extension methods (subclasses override for domain behavior):
    render() -> str         -- describe for prompt inclusion (AgentOptimizer reads this)
    snapshot() -> dict      -- capture managed content as {key: text} pairs
    restore(content: dict)  -- restore from snapshot (inverse of snapshot)

  Backward hooks (inherited from Datum, extended here):
    register_hook(hook) -> RemovableHandle  -- observation-only hook on incoming gradient
    dispatch_backward_hooks(gradient)       -- invoked by AccumulateGrad before grad assignment

  Built-in subclasses:
    PathParameter (ai/parameter.py) -- filesystem-scoped parameters.
    ScalarParameter (this module) -- JSON-compatible scalar/text/config values.

  Misuse detection: subclass constructors reject kwargs from
  ``COMMON_WRONG_KWARGS`` (value, data, text, content, prompt) with a
  tripartite message naming the mistake, explaining the invariant, and
  listing concrete recovery paths.

  Works with ``Gradient`` for backward updates and attaches to ``Module`` via
  attribute assignment.

  Example:
    For scalar state, use :class:`ScalarParameter` (implements :meth:`snapshot` /
    :meth:`restore` for the store protocol):

    >>> from autopilot.core.parameter import ScalarParameter
    >>>
    >>> theta = ScalarParameter(value=0.25)
    >>> captured = theta.snapshot()
    >>> theta.restore({'value.json': '0.75'})
    >>> theta.value
    0.75
  """

  requires_grad: bool = True
  grad: Gradient | None = field(default=None, repr=False)
  grad_accumulator: Any = field(init=False, repr=False, default=None)

  def __init_subclass__(cls, **kwargs: Any) -> None:
    """Wrap subclass ``__init__`` to reject common misuse kwargs early.

    Kwargs that the subclass ``__init__`` explicitly declares as named
    parameters are exempted (e.g. ``ScalarParameter`` accepts ``value=``).
    """
    super().__init_subclass__(**kwargs)
    original_init = cls.__init__

    rejected = _banned_kwargs_for(original_init)

    @functools.wraps(original_init)
    def _guarded_init(self: Any, *args: Any, **kw: Any) -> None:
      wrong = rejected & kw.keys()
      if wrong:
        raise TypeError(_build_kwarg_guard_message(type(self).__name__, wrong))
      original_init(self, *args, **kw)

    cls.__init__ = _guarded_init  # type: ignore[method-assign, ty:invalid-assignment]

  def __post_init__(self) -> None:
    """Initialize Datum identity after field assignment."""
    super().__post_init__()

  def dispatch_backward_hooks(self, gradient: Any) -> None:
    """Invoke all backward hooks registered on this parameter.

    Called by ``AccumulateGrad.__call__`` before gradient accumulation into
    ``param.grad``. Hooks receive the incoming gradient object; return
    values are discarded (observation-only).

    Args:
      gradient: The incoming gradient about to be accumulated.
    """
    for hook in self._backward_hooks.values():
      hook(gradient)

  def render(self) -> str:
    """Describe this parameter for prompt inclusion.

    Subclasses override to provide domain-specific descriptions.
    Default returns empty string.

    Returns:
      Prompt-facing description text.
    """
    return ''

  def snapshot(self) -> dict[str, str]:
    """Capture this parameter's managed content for versioning.

    Subclasses override to export their content. Keys and values are
    domain-specific: file-based params use relative paths as keys and
    file content as values; prompt params use descriptive keys.
    Default returns empty dict (parameter has no external content).

    Returns:
      Path or key to textual content mapping for store versioning.
    """
    return {}

  def restore(self, content: dict[str, str]) -> None:
    """Restore this parameter's managed content from a version snapshot.

    Inverse of snapshot(). Subclasses override to restore their content.
    Default is a no-op.
    """

  def schema_entry(self) -> ParameterSchemaEntry:
    """Return schema metadata for this parameter.

    Base implementation provides the concrete class name with no source
    or pattern. Subclasses (e.g. PathParameter) override to include
    filesystem provenance.

    Returns:
      Schema entry with ``name=''`` (caller fills the registration name),
      ``type_name`` from the concrete class, and ``source``/``pattern`` as None.
    """
    return ParameterSchemaEntry(name='', type_name=type(self).__name__)

  def load_from_dict(self, data: dict[str, Any]) -> None:
    """Apply serialized state into this live parameter instance.

    Used by ``Module.load_state_dict`` to restore checkpoint state into
    existing parameters without replacing the object. Base implementation
    restores ``requires_grad``; subclasses override to handle additional
    fields (e.g. PathParameter restores file payloads).

    Args:
      data: Dict produced by ``to_dict()`` / ``state_dict()``.
    """
    if 'requires_grad' in data:
      self.requires_grad = data['requires_grad']

  def to_dict(self) -> dict[str, Any]:
    """Serialize parameter fields; gradient state is omitted.

    Returns:
      Dict including ``requires_grad`` and Datum payload without ``grad``.
    """
    payload = super().to_dict()
    payload['requires_grad'] = self.requires_grad
    return payload

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'Parameter':
    """Deserialize a Parameter from a ``to_dict`` payload.

    Args:
      data: Mapping from ``Parameter.to_dict`` including nested Datum items.

    Returns:
      Rehydrated Parameter with preserved id when present.
    """
    data = dict(data)
    requires_grad = data.pop('requires_grad', True)
    instance = _hydrate_datum_base(
      cls,
      data,
      hydrate_child=Datum.from_dict,
      pop_type=False,
    )
    instance.requires_grad = requires_grad
    return instance


SNAPSHOT_VALUE_KEY = 'value.json'

_NAMED_PARAM_KINDS = frozenset(
  {
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
  }
)


def _banned_kwargs_for(init_fn: Any) -> frozenset[str]:
  """Compute the subset of ``COMMON_WRONG_KWARGS`` to reject for an init.

  Named parameters explicitly declared by the init are exempted so that
  subclasses like ``ScalarParameter`` can accept ``value=``.

  Args:
    init_fn: The ``__init__`` method to inspect.

  Returns:
    Frozen set of kwarg names to reject.
  """
  try:
    sig = inspect.signature(init_fn)
    explicit = frozenset(name for name, p in sig.parameters.items() if p.kind in _NAMED_PARAM_KINDS)
  except (ValueError, TypeError):
    explicit = frozenset()
  return COMMON_WRONG_KWARGS - explicit


def _build_kwarg_guard_message(cls_name: str, wrong: frozenset[str] | set[str]) -> str:
  """Build the tripartite error message for banned kwargs.

  Args:
    cls_name: Name of the class being constructed.
    wrong: Set of banned kwarg names that were passed.

  Returns:
    Error message with mistake, invariant, and recovery paths.
  """
  names = ', '.join(sorted(wrong))
  return (
    f'{cls_name} does not accept {names!r} as keyword argument(s). '
    f'Parameter is a Datum subclass with items: list[Datum]. '
    f'For file-based parameters, use PathParameter(source=Path("path/to/file")). '
    f'For scalar values, use ScalarParameter(value=...). '
    f'For custom data, subclass Parameter with domain-specific fields.'
  )


class ScalarParameter(Parameter):
  """Parameter holding a JSON-compatible scalar, text, or small structured value.

  For lightweight optimizable state that does not live on disk as files.
  ``PathParameter`` covers filesystem scope; ``ScalarParameter`` covers
  structured in-memory values (strings, numbers, booleans, lists, dicts).

  Storage uses ``_value`` (private) exposed via the ``value`` property.
  Not a dataclass field to avoid init-ordering friction with ``Parameter``.

  Two versioning layers (matching ``Parameter`` contract):

  - **Store**: ``snapshot()`` serializes ``snapshot_value()`` as JSON text
    under a single key (``value.json``). ``restore()`` parses it back.
  - **Checkpoint**: ``to_dict()`` / ``load_from_dict()`` include the scalar
    alongside ``requires_grad``.

  Example::

    class MyModule(Module):
      def __init__(self):
        super().__init__()
        self.temperature = ScalarParameter(value=0.7)
        self.system_prompt = ScalarParameter(value='You are a helpful assistant.')
  """

  def __init__(self, value: Any = None, **kwargs: Any) -> None:
    """Create a scalar parameter.

    Args:
      value: JSON-serializable value (str, int, float, bool, None,
        or nested lists/dicts of the same).
      **kwargs: Forwarded to :class:`Parameter` (requires_grad, items, etc.).
    """
    super().__init__(**kwargs)
    self._value = value

  @property
  def value(self) -> Any:
    """Return the current scalar value."""
    return self._value

  @value.setter
  def value(self, new_value: Any) -> None:
    """Set the scalar value."""
    self._value = new_value

  def snapshot_value(self) -> Any:
    """Return the JSON-serializable representation for store versioning.

    Override in subclasses for custom serialization (e.g. validation,
    transformation before persistence).

    Returns:
      The current value as-is (must be JSON-serializable).
    """
    return self._value

  def restore_value(self, data: Any) -> None:
    """Assign a deserialized value from store restore.

    Override in subclasses for custom deserialization (e.g. validation,
    type coercion after loading).

    Args:
      data: JSON-decoded value from the store blob.
    """
    self._value = data

  def snapshot(self) -> dict[str, str]:
    """Serialize the scalar value as a JSON text blob for store versioning.

    Returns:
      Single-entry dict mapping ``'value.json'`` to the JSON string.

    Raises:
      StoreError: When the value is not JSON-serializable.
    """
    try:
      text = json.dumps(self.snapshot_value(), ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as exc:
      msg = (
        f'ScalarParameter snapshot failed: value {self._value!r} '
        f'(type {type(self._value).__name__}) is not JSON-serializable. '
        f'Use JSON-compatible types (str, int, float, bool, None, list, dict). '
        f'Original error: {exc}'
      )
      raise StoreError(msg) from exc
    return {SNAPSHOT_VALUE_KEY: text}

  def restore(self, content: dict[str, str]) -> None:
    """Restore the scalar value from a store snapshot.

    Empty dict is a valid no-op (parameter absent from snapshot).
    Non-empty dict missing the expected key indicates a corrupt or
    mismatched manifest.

    Args:
      content: Dict from ``snapshot()`` with ``'value.json'`` key.

    Raises:
      StoreError: If content is non-empty but missing the value key.
      StoreError: When the stored text is not valid JSON.
    """
    if not content:
      return
    if SNAPSHOT_VALUE_KEY not in content:
      msg = (
        f'ScalarParameter.restore() received non-empty content without '
        f'{SNAPSHOT_VALUE_KEY!r} key. Keys present: {sorted(content.keys())!r}. '
        f'This suggests a corrupt or mismatched snapshot manifest.'
      )
      raise StoreError(msg)
    text = content[SNAPSHOT_VALUE_KEY]
    try:
      data = json.loads(text)
    except json.JSONDecodeError as exc:
      msg = (
        f'ScalarParameter restore failed: stored text is not valid JSON. '
        f'Raw content: {text!r}. '
        f'Check that the store blob was not corrupted. '
        f'Original error: {exc}'
      )
      raise StoreError(msg) from exc
    self.restore_value(data)

  def schema_entry(self) -> ParameterSchemaEntry:
    """Return schema metadata with type ``ScalarParameter``.

    Returns:
      Schema entry with ``source=None`` and ``pattern=None``.
    """
    return ParameterSchemaEntry(name='', type_name='ScalarParameter')

  def render(self) -> str:
    """Describe the scalar value for prompt inclusion.

    Returns:
      ``repr()`` of the current value.
    """
    return repr(self._value)

  def to_dict(self) -> dict[str, Any]:
    """Serialize parameter fields including the scalar value.

    Returns:
      Dict with ``value`` alongside standard Parameter fields.
    """
    payload = super().to_dict()
    payload['value'] = self._value
    return payload

  def load_from_dict(self, data: dict[str, Any]) -> None:
    """Apply serialized state into this live parameter.

    Args:
      data: Dict from ``to_dict()`` containing ``value`` and
        ``requires_grad`` keys.
    """
    super().load_from_dict(data)
    if 'value' in data:
      self._value = data['value']

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'ScalarParameter':
    """Deserialize a ScalarParameter from a ``to_dict`` payload.

    Args:
      data: Mapping from ``ScalarParameter.to_dict``.

    Returns:
      Rehydrated ``ScalarParameter`` with preserved id when present.
    """
    data = dict(data)
    requires_grad = data.pop('requires_grad', True)
    value = data.pop('value', None)
    stored_id = data.pop('id', None)
    data.pop('type', None)
    data.pop('items', None)
    instance = cls(value=value, requires_grad=requires_grad)
    if stored_id:
      instance._id = stored_id
    return instance


_original_parameter_init = Parameter.__init__


@functools.wraps(_original_parameter_init)
def _guarded_parameter_init(self: Any, *args: Any, **kw: Any) -> None:
  wrong = COMMON_WRONG_KWARGS & kw.keys()
  if wrong:
    raise TypeError(_build_kwarg_guard_message(type(self).__name__, wrong))
  _original_parameter_init(self, *args, **kw)


Parameter.__init__ = _guarded_parameter_init  # type: ignore[method-assign, ty:invalid-assignment]
