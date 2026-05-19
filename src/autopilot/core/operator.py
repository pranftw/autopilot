"""Autograd engine primitives: Operator, Context, OperatorNode, AccumulateGrad.

Context -- forward-pass scratch space: saved tensors/state for backward,
  needs_input_grad flags, and arbitrary user metadata via __getattr__/__setattr__
  (non-private names only).
Operator -- stateless forward/backward hooks plus apply(), which creates
  the context, validates outputs, and (when gradients are enabled) records
  OperatorNode instances on Datum outputs.
OperatorNode -- one node per operator application; __call__ dispatches to
  Operator.backward.
AccumulateGrad(OperatorNode) -- leaf node for Parameter; accumulates
  incoming gradients into Parameter.grad using Gradient.accumulate,
  respecting requires_grad.

Wiring helpers (defined here, not in graph.py, to keep graph.py as a leaf):
  flatten_args(args, kwargs) -- flat list of all positional and keyword values.
  collect_input_nodes(args, kwargs, graph) -- walks flattened args,
    deduplicates by id(arg), creates/retrieves AccumulateGrad for Parameter
    inputs, returns (node, output_nr) tuples for graph wiring.

Operator.apply() integration with Graph is finalized alongside graph.py
changes in sub-plan 02. Top-level imports from graph.py (leaf module, no cycle).
"""

from autopilot.core.graph import get_current_graph, is_grad_enabled
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from typing import Any, cast
import copy


class Context:
  """Forward-pass scratch space for Operator backward.

  Attributes:
    _saved: Tuple of values saved via ``save_for_backward`` for use in backward.
    _metadata: Dict of user-defined metadata (non-private attribute names route here).
    _needs_input_grad: Tuple of booleans indicating which inputs require gradients.

  Provides save_for_backward() for state, needs_input_grad flags,
  and arbitrary user metadata via __getattr__/__setattr__
  (private names starting with '_' use real instance attrs;
  all other names route through _metadata dict).
  """

  def __init__(self) -> None:
    """Initialize empty saved tensors, metadata, and grad-required flags."""
    self._saved: tuple = ()
    self._metadata: dict[str, Any] = {}
    self._needs_input_grad: tuple[bool, ...] = ()

  def save_for_backward(self, *args: Any) -> None:
    """Save values for use in ``backward``.

    Args:
      *args: Arbitrary values retrievable later via ``ctx.saved``.
    """
    self._saved = args

  @property
  def saved(self) -> tuple:
    """Tuple of values stored by ``save_for_backward``."""
    return self._saved

  @property
  def needs_input_grad(self) -> tuple[bool, ...]:
    """Per-input flags populated by ``Operator.apply`` before ``forward``."""
    return self._needs_input_grad

  def set_needs_input_grad(self, flags: tuple[bool, ...]) -> None:
    """Set flattened-input gradient flags populated before ``forward``."""
    self._needs_input_grad = flags

  def __setattr__(self, name: str, value: Any) -> None:
    """Route public names to ``_metadata``; keep private attrs on the instance."""
    if name.startswith('_'):
      object.__setattr__(self, name, value)
    else:
      self._metadata[name] = value

  def __getattr__(self, name: str) -> Any:
    """Load arbitrary user metadata from ``_metadata``.

    Returns:
      The stored metadata value.

    Raises:
      AttributeError: When ``name`` is absent from ``_metadata``.
    """
    md = self.__dict__.get('_metadata')
    if md is not None and name in md:
      return md[name]
    msg = f'Context has no attribute {name!r}'
    raise AttributeError(msg)

  def __deepcopy__(self, memo: dict) -> 'Context':
    """Deep-copy saved tensors and metadata for checkpoint-like clones.

    Returns:
      Independent ``Context`` with copied ``_saved`` and ``_metadata``.
    """
    result = Context()
    memo[id(self)] = result
    result._saved = copy.deepcopy(self._saved, memo)
    result._metadata = copy.deepcopy(self._metadata, memo)
    result._needs_input_grad = self._needs_input_grad
    return result


class OperatorNode:
  """One node per operator application in the computation graph.

  Attributes:
    _operator_cls: The ``Operator`` subclass whose ``backward`` is dispatched.
    _ctx: ``Context`` holding saved state from the forward pass.
    _next_functions: Tuple of ``(OperatorNode | None, output_nr)`` edges to
      upstream nodes, matching ``collect_input_nodes`` output.
    sequence_nr: Monotonic sequence number for topological ordering.

  ``__call__`` dispatches to ``Operator.backward`` with the saved context.
  """

  def __init__(
    self,
    operator_cls: 'type[Operator] | type',
    ctx: Context,
    next_functions: tuple[tuple['OperatorNode | None', int], ...] = (),
    sequence_nr: int = 0,
  ) -> None:
    """Bind operator class, backward context, graph edges, and ordering id.

    Args:
      operator_cls: Concrete ``Operator`` type whose ``backward`` runs here.
      ctx: Forward scratch space for this application.
      next_functions: Upstream edges as ``(node, output_nr)`` pairs.
      sequence_nr: Monotonic id used by the graph heap ordering.
    """
    self._operator_cls: type[Operator] = cast('type[Operator]', operator_cls)
    self._ctx = ctx
    self._next_functions = next_functions
    self.sequence_nr = sequence_nr

  @property
  def next_functions(self) -> tuple[tuple['OperatorNode | None', int], ...]:
    """Upstream graph edges consumed during backward."""
    return self._next_functions

  def name(self) -> str:
    """Return operator class name.

    When ``operator_cls`` is ``type(None)`` (as in ``AccumulateGrad``),
    returns ``'NoneType'``; subclasses override for clarity.
    """
    return self._operator_cls.__name__

  def __call__(self, *grads: Any) -> tuple:
    """Dispatch backward through the operator class.

    Args:
      *grads: Incoming gradient(s) from downstream nodes.

    Returns:
      Tuple of gradients to propagate to each upstream ``next_functions`` entry.
    """
    output = self._operator_cls.backward(self._ctx, *grads)
    if not isinstance(output, tuple):
      output = (output,)
    return output

  def __repr__(self) -> str:
    """Return operator name and sequence number."""
    return f'{self.name()}(seq={self.sequence_nr})'


class AccumulateGrad(OperatorNode):
  """Leaf node for Parameter. Accumulates incoming gradients into Parameter.grad.

  Frozen parameters (requires_grad=False) ignore accumulation even when
  __call__ is invoked. Returns empty tuple (no further backward outputs).
  """

  def __init__(self, parameter: Parameter, sequence_nr: int) -> None:
    """Wrap ``parameter`` as a leaf that accumulates gradients into ``.grad``."""
    super().__init__(
      operator_cls=type(None),
      ctx=Context(),
      next_functions=(),
      sequence_nr=sequence_nr,
    )
    self._parameter = parameter

  def name(self) -> str:
    """Return the fixed label ``AccumulateGrad``."""
    return 'AccumulateGrad'

  def __call__(self, *grads: Any) -> tuple:
    """Accumulate ``grads[0]`` into ``parameter.grad`` when required.

    Dispatches backward hooks on the parameter before accumulation so hooks
    observe the incoming gradient prior to any mutation of ``param.grad``.

    Returns:
      Empty tuple (no further upstream gradients).
    """
    if grads and grads[0] is not None and self._parameter.requires_grad:
      self._parameter.dispatch_backward_hooks(grads[0])
      if self._parameter.grad is None:
        self._parameter.grad = grads[0]
      else:
        self._parameter.grad = self._parameter.grad.accumulate(grads[0])
    return ()

  @classmethod
  def get_or_create(cls, param: Parameter, graph: Any) -> 'AccumulateGrad':
    """Get cached AccumulateGrad for param, or create and register one.

    Returns:
      Existing or newly created ``AccumulateGrad`` node registered on ``graph``.
    """
    acc = param.grad_accumulator
    if acc is not None:
      return acc
    acc = cls(param, graph.next_sequence_nr())
    param.grad_accumulator = acc
    graph.add_node(acc)
    return acc


class Operator:
  """Stateless forward/backward hooks with apply() graph recording.

  Subclass and override ``forward()`` and ``backward()`` as ``@staticmethod``.
  Call via ``Operator.apply()`` (not ``forward()`` directly) to record the graph.

  Attributes:
    None at class level; subclasses use ``@staticmethod forward/backward``.
  """

  @staticmethod
  def forward(ctx: Context, *args: Any, **kwargs: Any) -> Any:
    """Compute forward pass. Override in subclasses.

    Args:
      ctx: ``Context`` for saving state needed during backward.
      *args: Positional inputs.
      **kwargs: Keyword inputs.

    Raises:
      NotImplementedError: If not overridden.
    """
    raise NotImplementedError

  @staticmethod
  def backward(ctx: Context, *grads: Any) -> Any:
    """Compute backward pass. Override in subclasses.

    Args:
      ctx: ``Context`` with saved state from forward.
      *grads: Incoming gradients.

    Raises:
      NotImplementedError: If not overridden.
    """
    raise NotImplementedError

  @classmethod
  def apply(cls, *args: Any, **kwargs: Any) -> Any:
    """Run ``forward`` under a fresh ``Context`` and record ``OperatorNode``s.

    Args:
      *args: Positional inputs forwarded to ``forward``.
      **kwargs: Keyword inputs forwarded to ``forward``.

    Returns:
      The value returned by ``forward``; if gradients are enabled and the value
      is a ``Datum``, its ``grad_fn`` is set to the recorded ``OperatorNode``.

    Raises:
      TypeError: If ``forward`` returns a tuple or list (single value required).
    """
    ctx = Context()

    flat = flatten_args(args, kwargs)
    needs: list[bool] = []
    for val in flat:
      if isinstance(val, Parameter):
        needs.append(val.requires_grad)
      elif isinstance(val, Datum):
        needs.append(val.grad_fn is not None)
    ctx.set_needs_input_grad(tuple(needs))

    output = cls.forward(ctx, *args, **kwargs)

    if isinstance(output, (tuple, list)):
      msg = 'Operator.forward() must return a single value, not a tuple or list'
      raise TypeError(msg)

    if is_grad_enabled() and isinstance(output, Datum):
      graph = get_current_graph()
      prev_nodes = collect_input_nodes(args, kwargs, graph)
      seq = graph.next_sequence_nr()
      node = OperatorNode(
        cls,
        ctx,
        next_functions=tuple(prev_nodes),
        sequence_nr=seq,
      )
      graph.add_node(node)
      output.grad_fn = node

    return output


def flatten_args(args: tuple, kwargs: dict) -> list[Any]:
  """Return a flat list of all positional and keyword argument values.

  Returns:
    Flattened sequence preserving leaf objects from nested lists/tuples/dicts.
  """
  result: list[Any] = []
  for arg in args:
    if isinstance(arg, (list, tuple)):
      result.extend(flatten_args(tuple(arg), {}))
    elif isinstance(arg, dict):
      result.extend(flatten_args((), arg))
    else:
      result.append(arg)
  for v in kwargs.values():
    if isinstance(v, (list, tuple)):
      result.extend(flatten_args(tuple(v), {}))
    elif isinstance(v, dict):
      result.extend(flatten_args((), v))
    else:
      result.append(v)
  return result


def collect_input_nodes(
  args: tuple,
  kwargs: dict,
  graph: Any,
) -> list[tuple[OperatorNode | None, int]]:
  """Walk flattened args, deduplicate by id, return (node, output_nr) tuples.

  Returns:
    Edge list wiring upstream ``OperatorNode`` instances (or ``AccumulateGrad``).
  """
  nodes: list[tuple[OperatorNode | None, int]] = []
  seen: set[int] = set()
  for arg in flatten_args(args, kwargs):
    if not isinstance(arg, Datum):
      continue
    aid = id(arg)
    if aid in seen:
      continue
    seen.add(aid)
    if isinstance(arg, Parameter):
      acc = AccumulateGrad.get_or_create(arg, graph)
      nodes.append((acc, 0))
    elif arg.grad_fn is not None:
      nodes.append((arg.grad_fn, 0))
  return nodes
