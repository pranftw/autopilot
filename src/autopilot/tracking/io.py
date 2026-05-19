"""Canonical I/O primitives for JSON and JSONL operations.

All JSON/JSONL persistence in the framework delegates to these functions.
No other module should implement atomic writes or JSONL append/read logic.
New code that needs JSON/JSONL I/O must use these primitives.

Functions:
  atomic_write_json(path, payload)     -- tmp-write + rename, raises TrackingError
  append_jsonl(path, record)           -- append one JSON line, creates parent dirs
  read_jsonl(path, strict=True)        -- read all records, [] if missing
  read_json(path)                      -- read JSON file, None if missing
  utc_now_iso()                        -- current UTC time as ISO 8601 string
  parse_timestamp(value)               -- parse ISO 8601 string to aware datetime
  read_json_dict(path, label)          -- read JSON file requiring a dict root
  iter_jsonl_lines(path)               -- yield stripped non-empty text lines from a file
  exclusive_create(path)               -- create empty file, fail if path exists

JSON format: 2-space indent, sort_keys=False, UTF-8 encoding.
"""

from autopilot.core.errors import TrackingError
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import contextlib
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

BINARY_SNIFF_BYTES = 8192


def _silent_unlink(path: Path) -> None:
  with contextlib.suppress(OSError):
    path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: list | dict) -> None:
  """Atomically write JSON to path. Uses unique temp files per thread/process.

  Follows symlinks: when ``path`` is a symlink, the resolved target file is
  atomically replaced so the symlink itself is preserved. Parent directories
  of the resolved path are created if missing.

  Safe for concurrent writers to the SAME path (last writer wins). NOT safe for
  read-modify-write patterns -- use external locking for those.

  Raises:
    TrackingError: On OS/serialization failures while writing or replacing.
  """
  resolved = path.resolve()
  suffix = f'.{os.getpid()}.{threading.get_ident()}.tmp'
  tmp = resolved.with_suffix(resolved.suffix + suffix)
  try:
    resolved.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False)
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(resolved)
  except OSError as exc:
    _silent_unlink(tmp)
    msg = f'failed to write JSON at {path}: {exc}'
    raise TrackingError(msg) from exc
  except (TypeError, ValueError) as exc:
    _silent_unlink(tmp)
    msg = f'payload is not JSON-serializable: {exc}'
    raise TrackingError(msg) from exc


def append_jsonl(path: Path, record: dict) -> None:
  """Append a JSON record as a single line.

  Each append is a single write call. Line-level atomicity is NOT guaranteed
  cross-process for records exceeding the platform pipe buffer (typically
  4096 bytes on Linux, ``PIPE_BUF``). When multiple writers concurrently
  append records larger than this threshold, partial lines may interleave,
  producing corrupt JSONL. Current autopilot record sizes are well within
  this limit. Callers needing stronger guarantees for large payloads should
  serialize writers externally or adopt a different storage strategy.

  For small records (under ``PIPE_BUF``), appends are effectively atomic on
  POSIX because the kernel completes a single ``write(2)`` call without
  interleaving.

  Raises:
    TrackingError: When the append cannot be written.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  line = json.dumps(record, sort_keys=False) + '\n'
  try:
    with path.open('a', encoding='utf-8') as fh:
      fh.write(line)
  except OSError as exc:
    msg = f'failed to append JSONL to {path}: {exc}'
    raise TrackingError(msg) from exc


def read_jsonl(path: Path, strict: bool = True) -> list[dict]:
  """Read all records from a JSONL file.

  When strict=False, skips corrupt lines with a warning.

  Returns:
    Parsed dict records, or an empty list when the file is missing.

  Raises:
    TrackingError: On read errors, or malformed lines when ``strict`` is true.
  """
  if not path.is_file():
    return []
  try:
    raw = path.read_text(encoding='utf-8')
  except (OSError, UnicodeDecodeError) as exc:
    msg = f'failed to read JSONL at {path}: {exc}'
    raise TrackingError(msg) from exc
  records: list[dict] = []
  for line_no, line in enumerate(raw.splitlines(), start=1):
    stripped = line.strip()
    if not stripped:
      continue
    try:
      data = json.loads(stripped)
    except json.JSONDecodeError as exc:
      if strict:
        msg = f'invalid JSON on line {line_no} of {path}: {exc}'
        raise TrackingError(
          msg,
        ) from exc
      logger.warning('skipping corrupt line %d in %s: %s', line_no, path, exc)
      continue
    if not isinstance(data, dict):
      if strict:
        msg = f'line {line_no} of {path} must be a JSON object'
        raise TrackingError(msg)
      logger.warning('skipping non-object on line %d in %s', line_no, path)
      continue
    records.append(data)
  return records


def read_json(path: Path) -> dict | list | None:
  """Read a JSON file.

  Returns:
    Parsed JSON object/array, or ``None`` when the file is missing or empty.

  Raises:
    TrackingError: On read errors or invalid JSON.
  """
  if not path.exists():
    return None
  try:
    raw = path.read_text(encoding='utf-8')
  except (OSError, UnicodeDecodeError) as exc:
    msg = f'failed to read JSON at {path}: {exc}'
    raise TrackingError(msg) from exc
  raw = raw.lstrip('\ufeff')
  if not raw.strip():
    return None
  try:
    return json.loads(raw)
  except json.JSONDecodeError as exc:
    msg = f'invalid JSON at {path}: {exc}'
    raise TrackingError(msg) from exc


def utc_now_iso() -> str:
  """Return current UTC time as an ISO 8601 string.

  Returns:
    ISO 8601 timestamp string suitable for audit fields and JSON.
  """
  return datetime.now(UTC).isoformat()


def parse_timestamp(value: str) -> datetime:
  """Parse an ISO 8601 timestamp string to an aware datetime in UTC.

  Naive strings (no tzinfo) are treated as UTC. Invalid strings propagate
  ``ValueError`` from :meth:`datetime.fromisoformat`.

  Args:
    value: ISO 8601 timestamp string.

  Returns:
    Timezone-aware datetime.
  """
  dt = datetime.fromisoformat(value)
  if dt.tzinfo is None:
    dt = dt.replace(tzinfo=UTC)
  return dt


def read_json_dict(path: Path, label: str) -> dict[str, Any]:
  """Read JSON from ``path`` and require a JSON object (dict) root.

  Args:
    path: JSON file path.
    label: Short name for error messages (e.g. ``'refs.json'``, ``'checkpoint'``).

  Returns:
    Parsed object as a dict.

  Raises:
    TrackingError: If the file cannot be read, JSON is invalid, payload is
      missing/empty (see :func:`read_json`), or the root value is not a dict.
  """
  data = read_json(path)
  if not isinstance(data, dict):
    msg = f'{label} must contain a JSON object at {path}, got {type(data).__name__}'
    raise TrackingError(msg)
  return data


def iter_jsonl_lines(path: Path) -> Iterator[str]:
  """Yield stripped non-empty text lines from a file.

  Streaming primitive for JSONL or plain-text files. No JSON parsing is
  performed; callers that need parsed objects should use :func:`read_jsonl`.
  When ``path`` does not exist (or is not a regular file), yields zero lines
  (matching :func:`read_jsonl` missing-file semantics).

  Args:
    path: File to read line-by-line.

  Yields:
    Non-empty stripped lines from the file.

  Raises:
    TrackingError: On OS or encoding errors during read.
  """
  if not path.is_file():
    return
  try:
    with path.open(encoding='utf-8') as fh:
      for line in fh:
        stripped = line.strip()
        if stripped:
          yield stripped
  except (OSError, UnicodeDecodeError) as exc:
    msg = f'failed to read lines from {path}: {exc}'
    raise TrackingError(msg) from exc


def exclusive_create(path: Path) -> None:
  """Create an empty file, failing if the path already exists.

  Uses exclusive-open semantics (``'x'`` mode) so that exactly one writer
  succeeds when multiple processes race. Parent directories are created
  automatically.

  Args:
    path: File path to create.

  Raises:
    FileExistsError: When ``path`` already exists (propagated directly).
    TrackingError: On other OS-level failures (e.g. permission denied).
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  try:
    with path.open('x', encoding='utf-8'):
      pass
  except FileExistsError:
    raise
  except OSError as exc:
    msg = f'failed to create {path}: {exc}'
    raise TrackingError(msg) from exc
