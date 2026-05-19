"""Foundational types used across all AutoPilot modules.

Datum is the minimal autograd container (AutoPilot's Tensor). It holds an
ordered ``list[Datum]`` of child data via ``items`` (not a dict-like mapping),
an auto-generated identity (``id``), and an optional ``grad_fn`` link for
backward propagation through the computation graph.

EvalDatum extends Datum with evaluation-specific fields (split, epoch, metrics,
success, error_message, feedback, metadata) for use as the standard metrics /
evaluation payload.

GateResult is the pass/fail/warn/skip enum for policy gates.

Subclassing pattern: extend Datum for domain-specific payloads (e.g. EvalDatum
for evaluation, Parameter for optimizable values, Gradient for feedback signals).
"""

from autopilot.core.graph import RemovableHandle, get_current_graph
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any
from uuid import uuid4
import copy

SHORT_ID_HEX_LEN = 12


def _hydrate_datum_base(
  cls: type[Any],
  data: dict[str, Any],
  *,
  hydrate_child: Callable[[dict[str, Any]], Any] | None = None,
  pop_type: bool = True,
) -> Any:
  """Shared from_dict boilerplate for Datum-family classes.

  Extracts ``id`` and optionally ``type`` from the serialized dict, hydrates
  child items via ``hydrate_child`` when provided, and constructs an instance
  of ``cls`` with only the keys that match its dataclass fields. Restores
  ``_id`` when the input dict carried an ``id`` key.

  Per-class usage:

  - ``Datum``: ``hydrate_child=cls.from_dict, pop_type=True``.
  - ``EvalDatum``: ``hydrate_child=_datum_from_dict, pop_type=True``.
  - ``Parameter``: ``hydrate_child=Datum.from_dict, pop_type=False``;
    caller pops ``requires_grad`` before this helper and assigns it after.
  - ``NumericGradient``: ``hydrate_child=Datum.from_dict, pop_type=True``;
    caller pops ``value`` before this helper and assigns it after.
  - ``TextGradient``: full delegate --
    ``hydrate_child=Datum.from_dict, pop_type=True``; ``text``,
    ``attribution``, and ``severity`` are dataclass fields that flow
    through field-filtered kwargs.

  Args:
    cls: Target dataclass type (``Datum``, ``EvalDatum``, ``Parameter``,
      ``NumericGradient``, or ``TextGradient``).
    data: Serialized dict (shallow-copied internally to avoid mutation).
    hydrate_child: Callable to reconstruct each child item dict. ``Datum``
      passes ``cls.from_dict`` (recursive same-class); ``EvalDatum`` passes
      ``_datum_from_dict`` (type-dispatched); ``Parameter``,
      ``NumericGradient``, and ``TextGradient`` pass ``Datum.from_dict``.
      When ``None``, the helper does not populate ``items``; callers are
      responsible for nested hydration.
    pop_type: When ``True`` (default), remove ``'type'`` from the working
      dict before construction. ``Parameter.from_dict`` passes ``False``
      so the ``type`` key flows into filtered kwargs (and is silently
      ignored by the constructor since it is not a field).

  Returns:
    Hydrated instance of ``cls``.
  """
  data = dict(data)
  stored_id = data.pop('id', None)
  if pop_type:
    data.pop('type', None)
  if hydrate_child is not None:
    items_raw = data.pop('items', [])
    items = [hydrate_child(dict(item)) for item in items_raw]
  else:
    items = data.pop('items', [])
  names = {f.name for f in fields(cls)}
  instance = cls(**{k: v for k, v in data.items() if k in names}, items=items)
  if stored_id:
    instance._id = stored_id
  return instance


class GateResult(StrEnum):
  """Outcome of a gate evaluation: pass, fail, warn, or skip.

  Attributes:
    PASSED: Gate condition satisfied.
    FAIL: Gate condition not satisfied.
    WARN: Advisory failure (e.g. optional gate in QualityFirstPolicy).
    SKIP: Extension point for user-defined gates. No built-in gate produces
      SKIP. Use for gates that want to abstain from the decision (e.g.,
      insufficient data).
  """

  PASSED = 'pass'
  FAIL = 'fail'
  WARN = 'warn'
  SKIP = 'skip'


@dataclass
class Datum:
  """Minimal autograd container. AutoPilot's Tensor.

  Attributes:
    items: Ordered ``list[Datum]`` forming nested batch structure and graph
      inputs. Each element is a child ``Datum`` instance. This is strictly
      a list -- **not** a dict-shaped "payload" field. There is no
      ``Datum.payload`` attribute. For domain-specific data, use ``items``
      lists or subclass-specific attributes (e.g. ``EvalDatum.metrics``).
    _id: 12-character hex identifier (exposed via ``.id`` property).
    grad_fn: Link to the OperatorNode that produced this datum (None if leaf).

  Methods:
    clone() -- deep copy (clears grad_fn and hooks).
    detach() -- deep copy with grad_fn cleared (same as clone in Phase B).
    backward(gradient) -- trigger backward pass through the computation graph.
    register_hook(hook) -- observation-only backward hook; receives the seed
      gradient before graph traversal. Return values are ignored. Returns a
      ``RemovableHandle`` for later detachment.
    to_dict() / from_dict() -- serialization (grad_fn is never serialized).
  """

  items: list['Datum'] = field(default_factory=list)
  _id: str = field(init=False, repr=False, compare=False, default='')
  grad_fn: Any | None = field(init=False, repr=False, compare=False, default=None)
  _backward_hooks: OrderedDict[int, Callable[[Any], Any]] = field(
    init=False, repr=False, compare=False, default_factory=OrderedDict
  )

  def __post_init__(self) -> None:
    """Assign a fresh id and clear ``grad_fn`` after field initialization."""
    object.__setattr__(self, '_id', uuid4().hex[:SHORT_ID_HEX_LEN])
    object.__setattr__(self, 'grad_fn', None)

  def __deepcopy__(self, memo: dict) -> 'Datum':
    """Deep-copy datum fields while clearing autograd links and hooks.

    Returns:
      Independent copy suitable for a new forward without graph linkage.
    """
    cls = type(self)
    result = object.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
      if k in {'grad_fn', 'grad_accumulator', 'grad'}:
        object.__setattr__(result, k, None)
      elif k == '_backward_hooks':
        object.__setattr__(result, k, OrderedDict())
      else:
        object.__setattr__(result, k, copy.deepcopy(v, memo))
    return result

  def clone(self) -> 'Datum':
    """Return a deep copy of this datum (including ``items``).

    Returns:
      New ``Datum`` tree preserving ids with deep copy semantics.
    """
    return copy.deepcopy(self)

  def detach(self) -> 'Datum':
    """Deep copy with ``grad_fn`` cleared.

    Returns:
      Copy detached from the computation graph.
    """
    result = copy.deepcopy(self)
    result.grad_fn = None
    return result

  def register_hook(self, hook: Callable[[Any], Any]) -> RemovableHandle:
    """Register an observation-only backward hook on this datum.

    The hook receives the incoming gradient at the hook site (the seed
    gradient passed to ``backward()``). Return values are **ignored** --
    hooks cannot mutate or replace gradients (observation-only contract,
    intentional divergence from PyTorch tensor hooks).

    Hooks fire in FIFO registration order. Exceptions propagate (fail-fast).

    Args:
      hook: Callable receiving one positional argument (the gradient).

    Returns:
      ``RemovableHandle`` whose ``remove()`` detaches the hook.
    """
    handle = RemovableHandle(self._backward_hooks)
    self._backward_hooks[handle.id] = hook
    return handle

  def backward(self, gradient: Any) -> None:
    """Run backward from this datum through ``grad_fn``.

    Hooks registered via ``register_hook`` are called with the seed gradient
    before graph traversal begins. Return values are discarded.

    Raises:
      RuntimeError: When ``grad_fn`` is ``None``.
    """
    if self.grad_fn is None:
      msg = f'cannot backward through a datum without grad_fn (id={self.id!r})'
      raise RuntimeError(
        msg,
      )
    for hook in self._backward_hooks.values():
      hook(gradient)
    get_current_graph().backward(self.grad_fn, gradient)

  @property
  def id(self) -> str:
    """Return the stable hex identifier for this datum."""
    return self._id

  def to_dict(self) -> dict[str, Any]:
    """Serialize type, id, and child items (never includes ``grad_fn``).

    Returns:
      JSON-friendly dict describing this datum subtree.
    """
    return {
      'type': f'{type(self).__module__}.{type(self).__qualname__}',
      'id': self._id,
      'items': [item.to_dict() for item in self.items],
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'Datum':
    """Hydrate a ``Datum`` (or subclass) from serialized dict form.

    Returns:
      Reconstructed instance with optional restored ``id``.
    """
    return _hydrate_datum_base(cls, data, hydrate_child=cls.from_dict, pop_type=True)


def _datum_from_dict(data: dict[str, Any]) -> 'Datum':
  type_str = data.get('type')
  if type_str is not None and type_str.endswith('.EvalDatum'):
    return EvalDatum.from_dict(data)
  return Datum.from_dict(data)


@dataclass
class EvalDatum(Datum):
  """Standard evaluation / metrics container.

  Use EvalDatum when success, split, epoch, feedback, metrics, or metadata
  fields are required. This is the payload type for evaluation pipelines,
  judge losses, and metric tracking.

  See ``Datum`` for the base autograd container and ``items`` structure.

  Attributes:
    split: Split label (e.g. train/val/test) or None.
    epoch: Training epoch index associated with this sample, or None.
    metrics: Metric name to value mapping for this evaluation.
    success: Whether the evaluation succeeded.
    error_message: Failure description when success is false, else None.
    feedback: Optional free-text or structured feedback string.
    metadata: Arbitrary key-value metadata for this evaluation.

  Example:
    >>> from autopilot.core.types import EvalDatum
    >>>
    >>> sample = EvalDatum(
    ...   split='val',
    ...   epoch=0,
    ...   metrics={'accuracy': 0.91},
    ...   success=True,
    ...   feedback='pass',
    ... )
    >>> sample.metrics['accuracy']
    0.91
  """

  split: str | None = None
  epoch: int | None = None
  metrics: dict[str, Any] = field(default_factory=dict)
  success: bool = True
  error_message: str | None = None
  feedback: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __bool__(self) -> bool:
    """Return whether evaluation ``success`` is true."""
    return self.success

  def to_dict(self) -> dict[str, Any]:
    """Extend ``Datum.to_dict`` with eval fields.

    Returns:
      Dict including split, epoch, metrics, and related eval metadata.
    """
    payload = super().to_dict()
    payload['split'] = self.split
    payload['epoch'] = self.epoch
    payload['metrics'] = self.metrics
    payload['success'] = self.success
    payload['error_message'] = self.error_message
    payload['feedback'] = self.feedback
    payload['metadata'] = self.metadata
    return payload

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'EvalDatum':
    """Deserialize evaluation payload including nested datum items.

    Returns:
      ``EvalDatum`` with child items resolved via ``_datum_from_dict``.
    """
    return _hydrate_datum_base(cls, data, hydrate_child=_datum_from_dict, pop_type=True)
