"""Canonical I/O primitives for JSON and JSONL operations.

All JSON/JSONL persistence in the framework delegates to these four functions.
No other module should implement atomic writes or JSONL append/read logic.
New code that needs JSON/JSONL I/O must use these primitives.

Functions:
  atomic_write_json(path, payload)     -- tmp-write + rename, raises TrackingError
  append_jsonl(path, record)           -- append one JSON line, creates parent dirs
  read_jsonl(path, strict=True)        -- read all records, [] if missing
  read_json(path)                      -- read JSON file, None if missing

JSON format: 2-space indent, sort_keys=False, UTF-8 encoding.
"""

from autopilot.core.errors import TrackingError
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


def atomic_write_json(path: Path, payload: list | dict) -> None:
  """Atomically write a JSON file via tmp-write + rename."""
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + '.tmp')
  try:
    text = json.dumps(payload, indent=2, sort_keys=False)
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)
  except OSError as exc:
    raise TrackingError(f'failed to write JSON at {path}: {exc}') from exc
  except (TypeError, ValueError) as exc:
    if tmp.exists():
      tmp.unlink(missing_ok=True)
    raise TrackingError(f'payload is not JSON-serializable: {exc}') from exc


def append_jsonl(path: Path, record: dict) -> None:
  """Append a single JSON record as one line to a JSONL file."""
  path.parent.mkdir(parents=True, exist_ok=True)
  line = json.dumps(record, sort_keys=False) + '\n'
  try:
    with path.open('a', encoding='utf-8') as fh:
      fh.write(line)
  except OSError as exc:
    raise TrackingError(f'failed to append JSONL to {path}: {exc}') from exc


def read_jsonl(path: Path, strict: bool = True) -> list[dict]:
  """Read all records from a JSONL file.

  Returns empty list if file is missing.
  When strict=True, raises TrackingError on malformed lines.
  When strict=False, skips corrupt lines with a warning.
  """
  if not path.is_file():
    return []
  try:
    raw = path.read_text(encoding='utf-8')
  except OSError as exc:
    raise TrackingError(f'failed to read JSONL at {path}: {exc}') from exc
  records: list[dict] = []
  for line_no, line in enumerate(raw.splitlines(), start=1):
    stripped = line.strip()
    if not stripped:
      continue
    try:
      data = json.loads(stripped)
    except json.JSONDecodeError as exc:
      if strict:
        raise TrackingError(
          f'invalid JSON on line {line_no} of {path}: {exc}',
        ) from exc
      logger.warning('skipping corrupt line %d in %s: %s', line_no, path, exc)
      continue
    if not isinstance(data, dict):
      if strict:
        raise TrackingError(f'line {line_no} of {path} must be a JSON object')
      logger.warning('skipping non-object on line %d in %s', line_no, path)
      continue
    records.append(data)
  return records


def read_json(path: Path) -> dict | list | None:
  """Read a JSON file. Returns None if file is missing."""
  if not path.is_file():
    return None
  try:
    raw = path.read_text(encoding='utf-8')
    return json.loads(raw)
  except OSError as exc:
    raise TrackingError(f'failed to read JSON at {path}: {exc}') from exc
  except json.JSONDecodeError as exc:
    raise TrackingError(f'invalid JSON at {path}: {exc}') from exc
