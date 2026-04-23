"""--expose mechanism for CLI command audit trail.

ExposeRecord and ExposeCollector live here in cli/ (not core/).
Core has no knowledge of CLI-specific types.
"""

from autopilot.core.serialization import DictMixin
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator
import time


@dataclass
class ExposeRecord(DictMixin):
  """Single CLI command execution record."""

  command: str
  description: str | None = None
  exit_code: int = 0
  duration_s: float = 0.0
  stderr: str | None = None
  timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExposeCollector:
  """Collects ExposeRecord entries during a CLI command execution."""

  def __init__(self) -> None:
    self._records: list[ExposeRecord] = []

  def add(
    self,
    description: str,
    command: str,
    exit_code: int = 0,
    duration_s: float = 0.0,
    stderr: str | None = None,
  ) -> None:
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
    return [r.to_dict() for r in self._records]

  def __len__(self) -> int:
    return len(self._records)


def inject_expose(result_dict: dict[str, Any], collector: ExposeCollector) -> dict[str, Any]:
  """Add _commands array to result when collector is non-empty."""
  if len(collector) > 0:
    result_dict['_commands'] = collector.to_list()
  return result_dict


@contextmanager
def expose_command(
  collector: ExposeCollector,
  description: str,
  command: str | None = None,
) -> Generator[dict[str, Any], None, None]:
  """Context manager that times a command and auto-records it."""
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
    collector.add(
      description=description,
      command=command,
      exit_code=state['exit_code'],
      duration_s=round(duration, 3),
      stderr=state.get('stderr'),
    )
