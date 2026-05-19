"""Command history logging with optional argument redaction."""

from autopilot.core.artifacts.experiment import CommandsArtifact
from autopilot.core.models import CommandRecord
from autopilot.tracking.io import utc_now_iso
from pathlib import Path
import re

DEFAULT_REDACT_PATTERNS: list[str] = [
  'token',
  'secret',
  'password',
  'key',
  'auth',
  'cookie',
]


def redact_args(args: list[str], patterns: list[str]) -> list[str]:
  """Redact argv tokens matching any case-insensitive substring pattern.

  Args:
    args: Original argv fragments.
    patterns: Substrings that trigger redaction.

  Returns:
    New list with sensitive tokens replaced by ``[REDACTED]``.
  """
  if not patterns:
    return list(args)
  result: list[str] = []
  for arg in args:
    redacted = False
    for pattern in patterns:
      if re.search(re.escape(pattern), arg, flags=re.IGNORECASE):
        redacted = True
        break
    result.append('[REDACTED]' if redacted else arg)
  return result


def create_command_record(
  command: str,
  args: list[str],
  redact_patterns: list[str] | None = None,
) -> CommandRecord:
  """Create a command audit row with optional redaction patterns.

  Args:
    command: Logical command name.
    args: Argument list as invoked.
    redact_patterns: Optional override patterns (defaults to built-in secrets list).

  Returns:
    ``CommandRecord`` with timestamps and redacted argv mirror.
  """
  patterns = list(DEFAULT_REDACT_PATTERNS) if redact_patterns is None else list(redact_patterns)
  return CommandRecord(
    timestamp=utc_now_iso(),
    command=command,
    args=list(args),
    redacted_args=redact_args(args, patterns),
  )


def log_command(path: Path, record: CommandRecord) -> None:
  """Append ``record`` to the commands artifact for an experiment path.

  Args:
    path: Experiment directory receiving command history.
    record: Row to append.
  """
  CommandsArtifact().append_record(record.to_dict(), path)
