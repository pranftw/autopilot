"""Command history logging with optional argument redaction."""

from autopilot.core.errors import TrackingError
from autopilot.core.models import CommandRecord
from datetime import datetime, timezone
from pathlib import Path
import json
import re

DEFAULT_REDACT_PATTERNS: list[str] = [
  'token',
  'secret',
  'password',
  'key',
  'auth',
  'cookie',
]


def _commands_path(experiment_dir: Path) -> Path:
  return experiment_dir / 'commands.json'


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


def load_commands(experiment_dir: Path) -> list[CommandRecord]:
  path = _commands_path(experiment_dir)
  if not path.is_file():
    return []
  try:
    raw = path.read_text(encoding='utf-8')
  except OSError as exc:
    raise TrackingError(f'failed to read commands at {path}: {exc}') from exc
  try:
    data = json.loads(raw)
  except json.JSONDecodeError as exc:
    raise TrackingError(f'invalid commands JSON at {path}: {exc}') from exc
  if not isinstance(data, list):
    raise TrackingError(f'commands file must be a JSON array at {path}')
  records: list[CommandRecord] = []
  for idx, item in enumerate(data):
    if not isinstance(item, dict):
      raise TrackingError(
        f'commands[{idx}] at {path} must be an object',
      )
    try:
      records.append(CommandRecord.from_dict(item))
    except (TypeError, ValueError, KeyError) as exc:
      raise TrackingError(
        f'invalid command record at index {idx} in {path}: {exc}',
      ) from exc
  return records


def log_command(experiment_dir: Path, record: CommandRecord) -> None:
  path = _commands_path(experiment_dir)
  existing = load_commands(experiment_dir)
  existing.append(record)
  payload = [r.to_dict() for r in existing]
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + '.tmp')
  try:
    text = json.dumps(payload, indent=2, sort_keys=False)
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)
  except OSError as exc:
    raise TrackingError(f'failed to write commands at {path}: {exc}') from exc
  except (TypeError, ValueError) as exc:
    if tmp.exists():
      tmp.unlink(missing_ok=True)
    raise TrackingError(f'commands are not JSON-serializable: {exc}') from exc
