"""Builtin graph operators, Sequential container, and functional API.

Operators:
  MergeOperator     -- concatenate items from multiple Datum inputs.
  SelectOperator    -- select a single item from a Datum by index (data-first).
  CloneOperator     -- graph-aware deep copy with gradient flow.
  IdentityOperator  -- pass-through (output is clone of input).
  DetachOperator    -- forward data, block gradient.
  BroadcastOperator -- replicate a Datum n times as child items.
  ReduceOperator    -- alias for MergeOperator (aggregation context).
  ScaleGradOperator -- scale NumericGradient in backward.
  TransformGradOperator -- apply arbitrary callable to gradient in backward.
  AttributionOperator   -- attach module-level attribution label; default backward
                           is passthrough (subclasses override for LLM attribution).

Container:
  Sequential -- Module that chains submodules in registration order.

Functional API (lowercase mirrors of Operator names):
  merge, select, clone, identity, detach, broadcast, reduce,
  scale_grad, transform_grad, attribution.

Argument order convention (data-first):
  select(datum, index) -- datum first, index second.
  broadcast(datum, n) -- datum first, count second.
  merge(d1, d2, ...) -- all datum operands.
  Multi-datum selection: use merge(d0, d1, ...) then select(merged, index).

BUG-002 compliance: all pass-through operators that must not alias the input
Datum use datum.clone() (not Datum(...) construction) to preserve subclass types.

BUG-033: batch size / arity for many operators is driven by ``len(datum.items)``.
empty ``items`` changes fan-out and merge semantics. authors should validate
non-empty bundles when operators expect per-item fan-out (e.g. BroadcastOperator
fans out by cloning, MergeOperator concatenates items lists).
"""

from autopilot.core.gradient import NumericGradient
from autopilot.core.module.module import Module
from autopilot.core.operator import Context, Operator
from autopilot.core.types import Datum
from collections.abc import Callable
from typing import Any
import copy


class MergeOperator(Operator):
  """Concatenate items from multiple Datum inputs into one Datum.

  Forward:
    Receives *datums, saves input count.  Returns a new Datum whose items
    list is the concatenation of all input items lists.

  Backward:
    Broadcasts grad_output to every input (tuple of length n).

  Contract:
    Merge is symmetric in inputs, so each branch must see the full downstream
    gradient—there is no privileged operand to absorb a single grad.
  """

  @staticmethod
  def forward(ctx: Context, *datums: Datum) -> Datum:
    """Concatenate all input ``items`` into a single ``Datum``.

    Returns:
      New ``Datum`` whose ``items`` list is the concatenation of inputs.

    Raises:
      TypeError: When no datums are provided.
    """
    if not datums:
      msg = 'MergeOperator requires at least one Datum input'
      raise TypeError(msg)
    ctx.save_for_backward(len(datums))
    all_items = []
    for d in datums:
      all_items.extend(d.items)
    return Datum(items=all_items)

  @staticmethod
  def backward(ctx: Context, *grads: Any) -> tuple[Any, ...]:
    """Broadcast ``grad_output`` to every merge input.

    Returns:
      Tuple of identical gradients, one per merge input.
    """
    n = ctx.saved[0]
    grad_output = grads[0] if grads else None
    return (grad_output,) * n


class SelectOperator(Operator):
  """Select a single item from a Datum by index.

  Data-first convention: select(datum, index), matching PyTorch patterns.

  Forward:
    Receives (datum, index), saves index.  Returns a new Datum containing
    only the item at datum.items[index].

  Backward:
    Passes grad_output through to the single datum input.

  Contract:
    Keeps the gradient path aligned with forward arity (one datum input) so
    we never imply a phantom grad slot for ``index``.
  """

  @staticmethod
  def forward(ctx: Context, datum: Datum, index: int) -> Datum:
    """Return a Datum containing ``datum.items[index]``.

    Args:
      ctx: Operator context for saving forward state.
      datum: The source Datum whose items to select from.
      index: Zero-based index into ``datum.items``.

    Returns:
      New Datum wrapping the selected item.

    Raises:
      TypeError: When ``datum`` is not a Datum (old argument order detected)
        or ``index`` is not an int.
      IndexError: When ``index`` is out of range or ``datum.items`` is empty.
    """
    if not isinstance(datum, Datum):
      msg = (
        f'select() expects a Datum as first argument, got {type(datum).__name__}. '
        f'Argument order changed: use select(datum, index), not select(index, datum).'
      )
      raise TypeError(msg)
    if not isinstance(index, int):
      msg = (
        f'select() expects an int as second argument, got {type(index).__name__}. '
        f'Usage: select(datum, index).'
      )
      raise TypeError(msg)
    if not datum.items:
      msg = 'select() called on a Datum with no items (items is empty).'
      raise IndexError(msg)
    if index < 0 or index >= len(datum.items):
      msg = f'select() index {index} out of range for Datum with {len(datum.items)} items.'
      raise IndexError(msg)
    ctx.save_for_backward(index)
    selected = datum.items[index]
    if isinstance(selected, Datum):
      return selected.clone()
    return Datum(items=[selected])

  @staticmethod
  def backward(ctx: Context, *grads: Any) -> tuple[Any, ...]:
    """Pass gradient through to the datum input.

    Returns:
      One-tuple containing ``grad_output`` for the single datum input.
    """
    grad_output = grads[0] if grads else None
    return (grad_output,)


class CloneOperator(Operator):
  """Graph-aware deep copy with gradient flow.

  Forward:
    Deep-copies the input Datum (preserves subclass type).  Saves nothing
    beyond the empty marker tuple.

  Backward:
    Passes grad_output through unchanged (identity gradient).

  Note: Datum.clone() always returns copy.deepcopy(self) with grad_fn=None.
  Use CloneOperator.apply(datum) when you need cloning with graph participation.

  Contract:
    Forward breaks object sharing for side-effect safety; backward still wires
    the new subgraph root to the original parent's gradient path.
  """

  @staticmethod
  def forward(ctx: Context, datum: Datum) -> Datum:
    """Deep-copy the datum for graph-connected cloning.

    Returns:
      Independent ``Datum`` subtree without an attached ``grad_fn``.
    """
    ctx.save_for_backward(())
    return copy.deepcopy(datum)

  @staticmethod
  def backward(ctx: Context, *grads: Any) -> tuple[Any, ...]:
    """Pass the incoming gradient through unchanged.

    Returns:
      One-tuple containing ``grad_output`` for the single input.
    """
    return (grads[0] if grads else None,)


class IdentityOperator(Operator):
  """Pass-through operator: output is a clone of the input.

  Forward:
    Returns datum.clone() (new instance, subclass-safe, no aliasing).

  Backward:
    Passes grad_output through unchanged.

  Contract:
    clone() avoids accidental in-place sharing across modules while behaving
    like an identity map for gradient purposes.
  """

  @staticmethod
  def forward(_ctx: Context, datum: Datum) -> Datum:
    """Return ``datum.clone()`` for a fresh instance without aliasing.

    Returns:
      Cloned ``Datum`` participating in autograd when enabled.
    """
    return datum.clone()

  @staticmethod
  def backward(ctx: Context, *grads: Any) -> tuple[Any, ...]:
    """Identity gradient on the single input.

    Returns:
      One-tuple containing ``grad_output`` for the cloned forward path.
    """
    return (grads[0] if grads else None,)


class DetachOperator(Operator):
  """Forward data, block gradient flow.

  Forward:
    Returns datum.detach() -- deep copy with grad_fn cleared.

  Backward:
    Returns None for the input, severing the gradient path.

  apply() is overridden to skip graph recording so the output has
  grad_fn=None (true detach semantics, matching torch.Tensor.detach()).

  Contract:
    Bypassing ``Operator.apply`` matches PyTorch detach: values flow, edges do not.
  """

  @staticmethod
  def forward(_ctx: Context, datum: Datum) -> Datum:
    """Return ``datum.detach()`` to clear ``grad_fn`` on the copy.

    Returns:
      Detached ``Datum`` clone.
    """
    return datum.detach()

  @staticmethod
  def backward(ctx: Context, *grads: Any) -> tuple[None, ...]:
    """Sever the backward path for detached values.

    Returns:
      Tuple of ``None`` to stop gradients at the detach boundary.
    """
    return (None,)

  @classmethod
  def apply(cls, *args: Any, **kwargs: Any) -> Datum:
    """Run forward without graph recording (true detach semantics).

    Returns:
      Forward output with ``grad_fn`` unset.
    """
    ctx = Context()
    return cls.forward(ctx, *args, **kwargs)


class BroadcastOperator(Operator):
  """Replicate a Datum n times as child items.

  Forward:
    Receives (datum, n).  Saves n.  Returns a new Datum whose items
    list contains n clones of the input.

  Backward:
    Returns (grad_output, None).  None for the scalar n input (non-Datum).

  Contract:
    ``n`` is a Python int, not a ``Datum``, so backward returns ``None`` in that
    position to mirror forward's non-graph argument.
  """

  @staticmethod
  def forward(ctx: Context, datum: Datum, n: int) -> Datum:
    """Replicate ``datum`` ``n`` times under ``items``.

    Returns:
      New ``Datum`` listing ``n`` clones.
    """
    ctx.save_for_backward(n)
    return Datum(items=[datum.clone() for _ in range(n)])

  @staticmethod
  def backward(ctx: Context, *grads: Any) -> tuple[Any, None]:
    """Route the output gradient back to ``datum``; ``n`` is non-Datum.

    Args:
      ctx: Forward context storing ``n`` via ``save_for_backward``.
      *grads: Downstream gradients; ``grads[0]`` is the output gradient.

    Returns:
      ``(gradient_for_datum, None)`` aligned with ``(datum, n)`` forward inputs.
    """
    grad_output = grads[0] if grads else None
    return (grad_output, None)


class ReduceOperator(MergeOperator):
  """Alias for MergeOperator for aggregation / functional contexts.

  Forward/Backward: identical to MergeOperator (concatenate items,
  broadcast gradient).

  Contract:
    Name-only alias for call sites that read clearer as ``reduce``; semantics
    stay merge so we do not fork gradient rules.
  """


class ScaleGradOperator(Operator):
  """Scale NumericGradient in backward; forward is a clone pass-through.

  Forward:
    Returns datum.clone() (subclass-safe).  Saves the scale factor.

  Backward:
    If grad_output is NumericGradient, returns a new NumericGradient
    with value multiplied by scale.  Otherwise passes through unchanged.

  Contract:
    Only the numeric lane is scaled so text/semantic gradients are not mangled
    unless explicitly numeric.
  """

  @staticmethod
  def forward(ctx: Context, datum: Datum, scale: float = 1.0) -> Datum:
    """Clone ``datum`` and save ``scale`` for backward scaling.

    Returns:
      Cloned ``Datum`` identical in the forward pass.
    """
    ctx.save_for_backward(scale)
    return datum.clone()

  @staticmethod
  def backward(ctx: Context, *grads: Any) -> tuple[Any, ...]:
    """Scale ``NumericGradient`` values; otherwise return passthrough.

    Returns:
      Tuple containing the scaled or untouched gradient for the datum input.
    """
    grad_output = grads[0] if grads else None
    scale = ctx.saved[0]
    if isinstance(grad_output, NumericGradient):
      return (NumericGradient(value=grad_output.value * scale),)
    return (grad_output,)


class TransformGradOperator(Operator):
  """Apply arbitrary callable to gradient in backward.

  Forward:
    Returns datum.clone() (subclass-safe).  Saves the transform function.

  Backward:
    If transform_fn is not None, applies it to grad_output and returns
    the result.  Otherwise passes through unchanged.

  Contract:
    Encapsulates host-specific post-processing (clipping, logging, etc.) without
    subclassing ``Operator`` for each transform.
  """

  @staticmethod
  def forward(
    ctx: Context,
    datum: Datum,
    transform_fn: Callable[..., Any] | None = None,
  ) -> Datum:
    """Clone ``datum`` and stash ``transform_fn`` for backward.

    Returns:
      Cloned ``Datum`` matching forward identity semantics.
    """
    ctx.save_for_backward(transform_fn)
    return datum.clone()

  @staticmethod
  def backward(ctx: Context, *grads: Any) -> tuple[Any, ...]:
    """Apply ``transform_fn`` when present; otherwise identity.

    Returns:
      Tuple containing the transformed or passthrough gradient.
    """
    grad_output = grads[0] if grads else None
    transform_fn = ctx.saved[0]
    if transform_fn is not None:
      return (transform_fn(grad_output),)
    return (grad_output,)


class AttributionOperator(Operator):
  """Attach module-level attribution label; default backward is passthrough.

  Forward:
    Returns datum.clone() (subclass-safe).  Saves module_name for potential
    use by subclasses in backward.

  Backward:
    Default: passes grad_output through unchanged.  Subclasses override
    for LLM-based or custom attribution logic.

  Contract:
    Labels are metadata by default—subclasses opt into real attribution work.
  """

  @staticmethod
  def forward(ctx: Context, datum: Datum, module_name: str | None = None) -> Datum:
    """Clone ``datum`` and record ``module_name`` for subclass attribution hooks.

    Returns:
      Cloned ``Datum`` for passthrough forward behavior.
    """
    ctx.save_for_backward(module_name)
    return datum.clone()

  @staticmethod
  def backward(ctx: Context, *grads: Any) -> tuple[Any, ...]:
    """Default passthrough gradient.

    Returns:
      Tuple containing ``grad_output`` unchanged.
    """
    return (grads[0] if grads else None,)


class Sequential(Module):
  """Transparent chain of submodules. Like nn.Sequential.

  Ordering follows registration: module_0, module_1, ...
  The first module receives *args, **kwargs; subsequent modules
  receive only the prior output. No direct Parameter children --
  inner submodule graph owns learnable edges (container transparency).

  Empty construction is invalid and raises ``ValueError``. At least one
  module is required so that ``forward`` has a well-defined entry point.

  Contract:
    No autograd operators here—ordering is the only policy; gradients follow
    the child ``Module.__call__`` edges.

  DOC-AUTOGRAD-8A: naive ``Parameter`` copying (e.g. via ``copy.deepcopy``)
  can break the identity expected by store snapshot/restore and optimizer
  zero_grad. prefer framework ``snapshot()``/``restore()`` or documented
  patterns when duplicating parameter-bearing submodules.
  """

  def __init__(self, *modules: Module) -> None:
    """Register ordered child modules as ``module_0``, ``module_1``, ...

    Raises:
      ValueError: If no modules are provided.
    """
    if not modules:
      msg = 'Sequential requires at least one module'
      raise ValueError(msg)
    super().__init__()
    for i, m in enumerate(modules):
      self._modules[f'module_{i}'] = m

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    """Run children in registration order, threading outputs forward.

    Returns:
      Final child output ``Datum``.
    """
    carry = self._modules['module_0'](*args, **kwargs)
    for i in range(1, len(self._modules)):
      carry = self._modules[f'module_{i}'](carry)
    return carry


def merge(*datums: Datum) -> Datum:
  """Merge multiple Datum inputs by concatenating their items.

  Returns:
    Output of ``MergeOperator.apply``.
  """
  return MergeOperator.apply(*datums)


def select(datum: Datum, index: int) -> Datum:
  """Select item at index from datum.

  Data-first convention: select(datum, index).

  Note:
    Argument order is datum-first: ``select(datum, index)``, not
    ``select(index, datum)``. Passing a non-``Datum`` as the first argument
    raises ``TypeError``.

  Args:
    datum: Source Datum to select from.
    index: Zero-based index into ``datum.items``.

  Returns:
    Output of ``SelectOperator.apply``.
  """
  return SelectOperator.apply(datum, index)


def clone(datum: Datum) -> Datum:
  """Graph-aware deep copy with gradient flow.

  Returns:
    Output of ``CloneOperator.apply``.
  """
  return CloneOperator.apply(datum)


def identity(datum: Datum) -> Datum:
  """Pass-through: returns a clone of the input with graph recording.

  Returns:
    Output of ``IdentityOperator.apply``.
  """
  return IdentityOperator.apply(datum)


def detach(datum: Datum) -> Datum:
  """Forward data, block gradient flow.

  Returns:
    Output of ``DetachOperator.apply`` without ``grad_fn``.
  """
  return DetachOperator.apply(datum)


def broadcast(datum: Datum, n: int) -> Datum:
  """Replicate a Datum n times as child items.

  Returns:
    Output of ``BroadcastOperator.apply``.
  """
  return BroadcastOperator.apply(datum, n)


def reduce(*datums: Datum) -> Datum:
  """Reduce (merge) multiple Datum inputs by concatenating their items.

  Returns:
    Output of ``ReduceOperator.apply``.
  """
  return ReduceOperator.apply(*datums)


def scale_grad(datum: Datum, scale: float) -> Datum:
  """Scale NumericGradient in backward; forward is identity.

  Returns:
    Output of ``ScaleGradOperator.apply``.
  """
  return ScaleGradOperator.apply(datum, scale)


def transform_grad(datum: Datum, fn: Callable[..., Any]) -> Datum:
  """Apply arbitrary callable to gradient in backward; forward is identity.

  Returns:
    Output of ``TransformGradOperator.apply``.
  """
  return TransformGradOperator.apply(datum, fn)


def attribution(datum: Datum, module_name: str | None = None) -> Datum:
  """Attach module-level attribution label; default backward is passthrough.

  Returns:
    Output of ``AttributionOperator.apply``.
  """
  return AttributionOperator.apply(datum, module_name)
