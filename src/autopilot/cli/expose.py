"""--expose mechanism for CLI command audit trail.

ExposeRecord and ExposeCollector live here in cli/ (not core/).
Core has no knowledge of CLI-specific types.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generator
import time


@dataclass
class ExposeRecord:
  """Single CLI command execution record."""

  description: str = ''
  command: str = ''
  exit_code: int = 0
  duration_s: float = 0.0
  stderr: str = ''
  timestamp: str = ''

  def __post_init__(self) -> None:
    if not self.timestamp:
      self.timestamp = datetime.now(timezone.utc).isoformat()

  def to_dict(self) -> dict[str, Any]:
    return {
      'description': self.description,
      'command': self.command,
      'exit_code': self.exit_code,
      'duration_s': self.duration_s,
      'stderr': self.stderr,
      'timestamp': self.timestamp,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'ExposeRecord':
    return cls(
      description=data.get('description', ''),
      command=data.get('command', ''),
      exit_code=data.get('exit_code', 0),
      duration_s=data.get('duration_s', 0.0),
      stderr=data.get('stderr', ''),
      timestamp=data.get('timestamp', ''),
    )


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
    stderr: str = '',
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
  command: str = '',
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
      stderr=state.get('stderr', ''),
    )
