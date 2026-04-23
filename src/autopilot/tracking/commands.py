"""Command history logging with optional argument redaction."""

from autopilot.core.artifacts.experiment import CommandsArtifact
from autopilot.core.models import CommandRecord
from datetime import datetime, timezone
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
  patterns = list(DEFAULT_REDACT_PATTERNS) if redact_patterns is None else list(redact_patterns)
  ts = datetime.now(timezone.utc).isoformat()
  return CommandRecord(
    timestamp=ts,
    command=command,
    args=list(args),
    redacted_args=redact_args(args, patterns),
  )


def log_command(experiment_dir: Path, record: CommandRecord) -> None:
  CommandsArtifact().append_record(record.to_dict(), experiment_dir)
