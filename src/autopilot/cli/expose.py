"""--expose mechanism for CLI command audit trail.

ExposeRecord and ExposeCollector live here in cli/ (not core/).
Core has no knowledge of CLI-specific types.
"""

from autopilot.core.serialization import DictMixin
from autopilot.tracking.io import utc_now_iso
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class ExposeRecord(DictMixin):
  """Single CLI command execution record."""

  command: str
  description: str | None = None
  exit_code: int = 0
  duration_s: float = 0.0
  stderr: str | None = None
  timestamp: str = field(default_factory=utc_now_iso)


class ExposeCollector:
  """Collects ExposeRecord entries during a CLI command execution."""

  def __init__(self) -> None:
    """Create an empty record list."""
    self._records: list[ExposeRecord] = []

  def add(
    self,
    description: str,
    command: str,
    exit_code: int = 0,
    duration_s: float = 0.0,
    stderr: str | None = None,
  ) -> None:
    """Append one expose record for a completed subprocess or step.

    Args:
      description: Human-readable label for the step.
      command: Command string that was run.
      exit_code: Process exit code (0 for success).
      duration_s: Elapsed wall time in seconds.
      stderr: Optional captured stderr text.
    """
    self._records.append(
      ExposeRecord(
        description=description,
        command=command,
        exit_code=exit_code,
        duration_s=duration_s,
        stderr=stderr,
      )
    )

  def to_list(self) -> list[dict[str, Any]]:
    """Serialize all records to JSON-friendly dicts.

    Returns:
      One serialized mapping per collected ``ExposeRecord``.
    """
    return [r.to_dict() for r in self._records]

  def __len__(self) -> int:
    """Return the number of collected records."""
    return len(self._records)


def inject_expose(result_dict: dict[str, Any], collector: ExposeCollector) -> dict[str, Any]:
  """Add _commands array to result when collector is non-empty.

  Args:
    result_dict: Mutable result mapping to augment.
    collector: Source of expose records (may be empty).

  Returns:
    The same ``result_dict`` reference, possibly with ``_commands`` set.
  """
  if collector:
    result_dict['_commands'] = collector.to_list()
  return result_dict


@contextmanager
def expose_command(
  collector: ExposeCollector,
  description: str,
  command: str | None = None,
) -> Generator[dict[str, Any], None, None]:
  """Context manager that times a command and auto-records it.

  Args:
    collector: Target collector for the timing record.
    description: Short label stored on the record.
    command: Command string recorded after the block (must be non-None in ``finally``).

  Yields:
    Mutable state dict with ``exit_code`` and ``stderr`` updated on exceptions.
  """
  state: dict[str, Any] = {'exit_code': 0, 'stderr': ''}
  start = time.monotonic()
  try:
    yield state
  except Exception as exc:
    state['exit_code'] = 1
    state['stderr'] = str(exc)
    raise
  finally:
    duration = time.monotonic() - start
    assert command is not None, 'expose_command requires a non-None command name'
    collector.add(
      description=description,
      command=command,
      exit_code=state['exit_code'],
      duration_s=round(duration, 3),
      stderr=state.get('stderr'),
    )
