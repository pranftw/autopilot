"""Lifecycle enum for the experiment system.

Status tracks experiment lifecycle with convenience properties.
Following Lightning's TrainerStatus pattern: enum is internal state,
lifecycle methods on Experiment are the primary API.

Do not add a Direction enum (none exists in the codebase). Use
higher_is_better: bool | None on the Metric class (torchmetrics pattern).

Zero imports from autopilot -- safe bottom of dependency graph,
alongside core/types.py.
"""

from enum import StrEnum


class Status(StrEnum):
  """Experiment lifecycle state. Internal -- prefer lifecycle methods over direct assignment."""

  pending = 'pending'
  running = 'running'
  completed = 'completed'
  failed = 'failed'
  cancelled = 'cancelled'
  invalidated = 'invalidated'

  @property
  def is_terminal(self) -> bool:
    """Whether this status represents a finished experiment lifecycle state.

    Returns:
      True for completed, failed, cancelled, or invalidated.
    """
    return self in {Status.completed, Status.failed, Status.cancelled, Status.invalidated}

  @property
  def is_active(self) -> bool:
    """Whether training is currently in progress for this status.

    Returns:
      True only for running.
    """
    return self == Status.running
