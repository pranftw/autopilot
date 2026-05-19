"""Shared output formatting for CLI commands.

Unified output handler. JSON mode emits envelope:
{'ok': bool, 'result': ..., 'messages': [...]}.
Error path uses flush_error() to emit
{'ok': false, 'error': str, 'messages': [...]}.
Text mode prints directly to stdout/stderr.
"""

from autopilot.cli.expose import ExposeCollector, inject_expose
from typing import Any
import json
import sys


def format_not_found_error(
  entity_type: str,
  identifier: str,
  suggestion: str,
) -> str:
  """Format a consistent not-found error with remediation.

  Args:
    entity_type: Kind of entity (e.g. 'Experiment', 'Tree').
    identifier: Value that was not found (shown with repr).
    suggestion: Actionable next step for the user.

  Returns:
    Formatted error message string.
  """
  return f'{entity_type} {identifier!r} not found. {suggestion}'


class Output:
  """Unified output handler for text and JSON CLI output.

  JSON mode emits envelope:
  {'ok': bool, 'result': ..., 'messages': [...]}.
  Error path uses flush_error() to emit
  {'ok': false, 'error': str, 'messages': [...]}.
  Text mode prints directly to stdout/stderr.
  """

  def __init__(
    self,
    use_json: bool = False,
    no_color: bool = False,
    expose_collector: ExposeCollector | None = None,
  ) -> None:
    """Create an output handler.

    Args:
      use_json: When true, buffer structured messages and emit JSON envelopes.
      no_color: Reserved for text-mode coloring toggles.
      expose_collector: Optional collector merged into ``result`` payloads.
    """
    self.use_json = use_json
    self.no_color = no_color
    self._expose_collector = expose_collector
    self._json_buffer: list[dict[str, Any]] = []
    self.retry_attempts: int = 0

  def info(self, message: str) -> None:
    """Emit an informational message (buffered in JSON mode)."""
    if self.use_json:
      self._json_buffer.append({'level': 'info', 'message': message})
    else:
      print(message)

  def success(self, message: str) -> None:
    """Emit a success-line in text mode or a buffered success entry in JSON mode."""
    if self.use_json:
      self._json_buffer.append({'level': 'success', 'message': message})
    else:
      print(f'OK: {message}')

  def warn(self, message: str) -> None:
    """Emit a warning to stderr in text mode or buffer in JSON mode."""
    if self.use_json:
      self._json_buffer.append({'level': 'warn', 'message': message})
    else:
      print(f'WARN: {message}', file=sys.stderr)

  def error(self, message: str) -> None:
    """Emit an error to stderr in text mode or buffer in JSON mode."""
    if self.use_json:
      self._json_buffer.append({'level': 'error', 'message': message})
    else:
      print(f'ERROR: {message}', file=sys.stderr)

  def data(self, payload: dict[str, Any]) -> None:
    """Emit structured data. In JSON mode, becomes the result payload."""
    if self.use_json:
      self._json_buffer.append({'type': 'data', 'payload': payload})
    else:
      for key, value in payload.items():
        print(f'  {key}: {value}')

  def result(self, payload: dict[str, Any], ok: bool = True) -> None:
    """Emit a final result. In JSON mode, prints the full envelope.

    Args:
      payload: Structured result fields (may be wrapped with expose metadata).
      ok: Whether the operation is considered successful for the envelope.
    """
    if self._expose_collector:
      payload = inject_expose(dict(payload), self._expose_collector)
    if self.use_json:
      envelope: dict[str, Any] = {
        'ok': ok,
        'result': payload,
        'messages': self._json_buffer,
      }
      if self.retry_attempts > 0:
        envelope['retry_attempts'] = self.retry_attempts
      print(json.dumps(envelope, indent=2))
      self._json_buffer = []
    else:
      status = 'OK' if ok else 'FAILED'
      print(f'\n{status}')
      for key, value in payload.items():
        print(f'  {key}: {value}')

  def table(self, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Print a simple table of rows.

    Args:
      rows: One dict per row keyed by column names.
      columns: Column order and headers.
    """
    if self.use_json:
      self._json_buffer.append({'type': 'table', 'rows': rows})
      return
    if not rows:
      print('  (none)')
      return
    widths = {col: len(col) for col in columns}
    for row in rows:
      for col in columns:
        cell = row.get(col)
        widths[col] = max(widths[col], len(str(cell) if cell is not None else ''))
    header = '  '.join(col.ljust(widths[col]) for col in columns)
    print(header)
    print('  '.join('-' * widths[col] for col in columns))
    for row in rows:
      line = '  '.join(
        (str(row[col]) if row.get(col) is not None else '').ljust(widths[col]) for col in columns
      )
      print(line)

  def flush_error(self, error_message: str, error_code: str | None = None) -> None:
    """Emit a JSON error envelope with buffered messages. Call on error paths.

    Args:
      error_message: Top-level error string for the envelope.
      error_code: Machine-stable error classification. Defaults to
        ``'handler_error'`` when ``None`` so that every JSON error envelope
        carries an ``error_code`` key.
    """
    if not self.use_json:
      return
    resolved_code = error_code if error_code is not None else 'handler_error'
    envelope = {
      'ok': False,
      'error': error_message,
      'error_code': resolved_code,
      'messages': self._json_buffer,
    }
    print(json.dumps(envelope))
    self._json_buffer.clear()

  def clear_messages(self) -> None:
    """Discard buffered JSON messages without emitting.

    Used by retry logic to discard partial output from failed attempts.
    """
    self._json_buffer.clear()

  def flush_json(self) -> None:
    """Flush any buffered JSON messages as a standalone array.

    No-op unless JSON mode is on and the buffer is non-empty.
    """
    if self.use_json and self._json_buffer:
      print(json.dumps(self._json_buffer, indent=2))
      self._json_buffer = []
