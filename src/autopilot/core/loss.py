"""Loss base class. Extends Module like nn.CrossEntropyLoss extends nn.Module.

Loss is a Module subclass, so assigning it as an attribute on a parent Module
auto-registers it into _modules. Trainer.fit() discovers the first Loss via
module.modules() walk.

Graph-based backward: Loss accumulates feedback in forward(), produces a seed
Gradient via compute_seed_gradient(), and injects that seed into the computation
graph via get_current_graph().backward(). The graph distributes attribution to
parameters through AccumulateGrad leaf nodes. Loss never sets param.grad directly.

Loss is the gradient source, not a graph node. forward() returns None so
ModuleCallOperator naturally skips grad_fn assignment on the loss output.
"""

from autopilot.core.graph import get_current_graph
from autopilot.core.module.module import Module
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from typing import Any


class Loss(Module):
  """Base loss. Extends Module so it auto-registers as a child module.

  Three-phase contract per accumulation window:
    forward(data, targets)      -- accumulate per-item feedback from one batch
    compute_seed_gradient()     -- produce the seed Gradient from accumulated feedback
    backward()                  -- seed -> graph.backward() distributes to parameters
    reset()                     -- clear internal accumulation state

  Loss is NOT a graph node: forward() returns None, so ModuleCallOperator
  skips grad_fn assignment. Loss is the gradient source -- it produces the seed
  and the graph distributes attribution to parameters via AccumulateGrad.

  Optional _loss_parameters scoping:
    Pass parameters= to __init__ to restrict which Parameters receive gradients.
    When empty, the Loss applies to all parameters discovered by the optimizer.
    Entries in _loss_parameters are NOT registered in Module._parameters -- they
    are foreign references owned by other modules.

  Built-in subclass: JudgeLoss (ai/loss.py) wraps a Judge + GradientCollator.

  Example:
    >>> from autopilot.core.loss import Loss
    >>> from autopilot.core.gradient import NumericGradient
    >>> from autopilot.core.types import Datum
    >>>
    >>> class ConstantLoss(Loss):
    ...   def forward(self, data, targets=None):
    ...     super().forward(data, targets)
    ...
    ...   def compute_seed_gradient(self):
    ...     return NumericGradient(value=1.0)
    >>>
    >>> loss = ConstantLoss()
    >>> loss.forward(Datum(items=[]))
    >>> loss.compute_seed_gradient().render()
    'gradient: 1.0'

  Warning:
    Do not call ``loss.forward()`` or ``loss.backward()`` inside
    ``training_step``. The ``Trainer`` drives the ``Loss`` lifecycle in
    ``_process_batch`` (forward, backward, reset).

  See ``Gradient`` for backward propagation and ``Datum`` for typical forward
  inputs.
  """

  def __init__(self, parameters: list[Parameter] | None = None) -> None:
    """Create loss with optional explicit parameter scope.

    Args:
      parameters: Restricts gradient targets; empty means all optimizer parameters.
    """
    super().__init__()
    self._loss_parameters = list(parameters) if parameters else []
    self._accumulated: list[dict[str, Any]] = []
    self._last_data: Datum | None = None

  def forward(self, data: Datum, targets: Any | None = None) -> None:  # type: ignore[ty:invalid-method-override]  # intentional divergence: Loss.forward() returns None (CLAUDE.md)
    """Accumulate feedback from one batch into internal state.

    Args:
      data: ``Datum`` produced by a module's forward pass. Carries ``grad_fn``
        from the upstream ``ModuleCallOperator`` for backward graph traversal.
      targets: Optional ground-truth or reference data for the batch. The base
        implementation stores it alongside ``data`` but does not use it;
        subclasses (e.g. ``JudgeLoss``) consume it for scoring.

    Returns None -- ``Loss`` is NOT a graph node.
    """
    self._accumulated.append({'data': data, 'targets': targets})
    self._last_data = data

  def compute_seed_gradient(self) -> Any:
    """Produce the seed ``Gradient`` from accumulated feedback.

    Called by ``backward()`` after all batches in the accumulation window
    have been processed via ``forward()``. Subclasses must override.

    Returns:
      A ``Gradient`` instance (e.g. ``TextGradient``, ``NumericGradient``)
      suitable for ``graph.backward(root, seed)``.

    Raises:
      NotImplementedError: On the base class.
    """
    raise NotImplementedError

  def backward(self) -> None:
    """Three-step: collate -> seed -> graph.backward().

    1. compute_seed_gradient() produces the seed Gradient
    2. Find graph root: self._last_data.grad_fn
    3. graph.backward(root, seed_gradient)

    Raises:
      RuntimeError: When forward was never called (no accumulation, no data).
      RuntimeError: When forward() was overridden without calling
        ``super().forward(data, targets)`` -- accumulation occurred but
        ``_last_data`` was never set. Message includes batch count and
        recovery guidance.
      RuntimeError: When ``data`` lacks ``grad_fn`` (not produced by a graph
        operator).
    """
    if not self._accumulated and self._last_data is None:
      msg = 'Loss.backward() called without prior forward()'
      raise RuntimeError(msg)
    if self._last_data is None:
      msg = (
        f'Loss._last_data is None after forward() was called '
        f'({len(self._accumulated)} batch(es) accumulated). '
        f'If you overrode forward(), you must call super().forward(data, targets) '
        f'to initialize internal tracking state. '
        f'Alternatively, override compute_seed_gradient() for custom loss logic.'
      )
      raise RuntimeError(msg)
    if self._last_data.grad_fn is None:
      msg = 'cannot backward: data has no grad_fn'
      raise RuntimeError(msg)

    seed = self.compute_seed_gradient()
    get_current_graph().backward(self._last_data.grad_fn, seed)

  def reset(self) -> None:
    """Clear accumulated batch feedback and last forward datum reference."""
    self._accumulated = []
    self._last_data = None

  @property
  def gradients(self) -> Any:
    """Loss does not expose a collated gradients property (placeholder hook).

    Returns:
      Always None in the base implementation.
    """
    return None
