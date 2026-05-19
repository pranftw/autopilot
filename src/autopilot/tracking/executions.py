"""Unified execution tracking for all AutoPilot CLI commands.

Append-only JSONL log of dispatch-level runs: command path, argv, timing,
exit code, optional captured stdout/stderr, project/experiment context,
and optional ``context`` reason/provenance string for the dispatch.
I/O uses ``tracking/io.py`` (``append_jsonl`` / ``read_jsonl``).
Serialization uses ``DictMixin`` (``to_dict`` / ``from_dict``).

``TeeWriter`` duplicates stream writes to a buffer while preserving the
original tty behavior for ``isatty()``. ``capture_output()`` installs
tee writers on ``sys.stdout`` and ``sys.stderr`` and always restores the
original streams in ``finally`` (safe for normal exceptions and
``BaseException`` subclasses such as ``SystemExit``).
"""

from autopilot.core.serialization import DictMixin
from autopilot.tracking.io import append_jsonl, read_jsonl, utc_now_iso
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any
import argparse
import contextlib
import io
import sys


class TeeWriter:
  """Wrap a text stream and duplicate every write to a capture buffer.

  Contract: ``write`` forwards to ``original`` then ``buffer``; ``flush``
  flushes both; ``isatty`` reflects ``original`` when it supports
  ``isatty()``. ``encoding`` delegates to ``original`` (fallback ``utf-8``).
  ``writelines`` calls ``write`` per line. ``fileno`` delegates to
  ``original``. Used so terminal output stays live while building a
  string record of the same bytes.
  """

  def __init__(self, original: Any, buffer: IO[str]) -> None:
    """Wrap ``original`` and mirror writes into ``buffer``.

    Args:
      original: Underlying text stream (e.g. ``sys.stdout``).
      buffer: Any writable text stream that accumulates duplicated output.
    """
    self.original = original
    self.buffer = buffer

  def write(self, s: str) -> int:
    """Write ``s`` to the original stream and the capture buffer.

    Returns:
      Length of the written string.
    """
    self.original.write(s)
    self.buffer.write(s)
    return len(s)

  def flush(self) -> None:
    """Flush both the original stream and capture buffer when supported."""
    self.original.flush()
    self.buffer.flush()

  def isatty(self) -> bool:
    """Return whether the original stream reports a TTY."""
    return hasattr(self.original, 'isatty') and self.original.isatty()

  @property
  def encoding(self) -> str:
    """Text encoding taken from ``original`` when available."""
    return getattr(self.original, 'encoding', 'utf-8')

  def writelines(self, lines: list[str]) -> None:
    """Write each line via ``write``."""
    for line in lines:
      self.write(line)

  def fileno(self) -> int:
    """Delegate ``fileno`` to the wrapped original stream.

    Returns:
      File descriptor reported by ``original``.
    """
    return self.original.fileno()


@contextlib.contextmanager
def capture_output():
  """Temporarily tee ``sys.stdout`` and ``sys.stderr`` into string buffers.

  Always restores the previous ``sys.stdout`` / ``sys.stderr`` in a ``finally``
  block after the managed block returns or raises (including ``SystemExit``).

  Yields:
    Tuple of ``(stdout_buf, stderr_buf)`` as ``io.StringIO`` instances.
  """
  stdout_buf = io.StringIO()
  stderr_buf = io.StringIO()
  old_out = sys.stdout
  old_err = sys.stderr
  try:
    sys.stdout = TeeWriter(old_out, stdout_buf)
    sys.stderr = TeeWriter(old_err, stderr_buf)
    yield (stdout_buf, stderr_buf)
  finally:
    sys.stdout = old_out
    sys.stderr = old_err


@dataclass
class ExecutionRecord(DictMixin):
  """One row in ``executions.jsonl``.

  Schema:
    - ``timestamp``: ISO 8601 UTC from ``create_execution_record``.
    - ``command``: resolved subcommand path (e.g. ``optimize train``).
    - ``args``: argv list for reproducibility.
    - ``duration_ms``, ``exit_code``: timing and process result.
    - ``stdout`` / ``stderr``: captured text, or ``None`` if unused.
    - ``experiment`` / ``project``: optional CLI context slugs.
    - ``extra``: forward-compatible metadata dict (git SHA, agent ID, etc.).
    - ``context``: optional reason/provenance string for the dispatch.

  Use ``to_dict`` / ``from_dict`` for JSONL persistence via ``log_execution`` /
  ``load_executions``.
  """

  timestamp: str
  command: str
  args: list[str] = field(default_factory=list)
  duration_ms: float = 0.0
  exit_code: int = 0
  stdout: str | None = None
  stderr: str | None = None
  experiment: str | None = None
  project: str | None = None
  extra: dict[str, Any] = field(default_factory=dict)
  context: str | None = None


def create_execution_record(
  command: str,
  args: list[str],
  duration_ms: float,
  exit_code: int,
  *,
  stdout: str | None = None,
  stderr: str | None = None,
  experiment: str | None = None,
  project: str | None = None,
  extra: dict[str, Any] | None = None,
  context: str | None = None,
) -> ExecutionRecord:
  """Build a timestamped execution record for JSONL logging.

  Args:
    command: Resolved command name / path.
    args: Serialized argv fragments.
    duration_ms: Wall-clock duration in milliseconds.
    exit_code: Process exit code.
    stdout: Optional captured stdout text.
    stderr: Optional captured stderr text.
    experiment: Optional experiment slug from CLI context.
    project: Optional project slug from CLI context.
    extra: Optional additional metadata merged into the record.
    context: Optional reason/provenance string for the dispatch.

  Returns:
    Hydrated ``ExecutionRecord`` ready for ``log_execution``.
  """
  return ExecutionRecord(
    timestamp=utc_now_iso(),
    command=command,
    args=list(args),
    duration_ms=duration_ms,
    exit_code=exit_code,
    stdout=stdout,
    stderr=stderr,
    experiment=experiment,
    project=project,
    extra=dict(extra) if extra is not None else {},
    context=context,
  )


def log_execution(path: Path, record: ExecutionRecord) -> None:
  """Append one execution record to the JSONL audit file.

  Args:
    path: Destination ``executions.jsonl`` path.
    record: Record to persist.
  """
  append_jsonl(path, record.to_dict())


def load_executions(path: Path) -> list[ExecutionRecord]:
  """Load all execution records from disk.

  Args:
    path: JSONL file written by ``log_execution``.

  Returns:
    Parsed ``ExecutionRecord`` list (may be empty).
  """
  return [ExecutionRecord.from_dict(d) for d in read_jsonl(path)]


def filter_executions(
  records: list[ExecutionRecord],
  command: str | None = None,
  project: str | None = None,
  experiment: str | None = None,
  context: str | None = None,
  exit_code: int | None = None,
  predicate: Callable[[ExecutionRecord], bool] | None = None,
) -> list[ExecutionRecord]:
  """Filter execution records by simple field/value matches or predicate.

  Args:
    records: Source records to filter.
    command: Optional exact command match.
    project: Optional project slug match.
    experiment: Optional experiment slug match.
    context: Optional exact context string match.
    exit_code: Optional exit code match.
    predicate: Optional callable filter applied last.

  Returns:
    Records matching all supplied filters.
  """
  result = records
  if command is not None:
    result = [r for r in result if r.command == command]
  if project is not None:
    result = [r for r in result if r.project == project]
  if experiment is not None:
    result = [r for r in result if r.experiment == experiment]
  if context is not None:
    result = [r for r in result if r.context == context]
  if exit_code is not None:
    result = [r for r in result if r.exit_code == exit_code]
  if predicate is not None:
    result = [r for r in result if predicate(r)]
  return result


def resolve_command(args: argparse.Namespace) -> str:
  """Resolve the logical command string from a parsed argparse namespace.

  Args:
    args: Namespace potentially containing ``command`` and ``*_action`` fields.

  Returns:
    Space-joined command path, or ``unknown`` when nothing is set.
  """
  parts: list[str] = []
  if hasattr(args, 'command') and args.command:
    parts.append(args.command)
  parts.extend(v for k, v in vars(args).items() if k.endswith('_action') and v)
  combined = ' '.join(parts)
  return combined or 'unknown'
