"""Shared output formatting for CLI commands.

Supports both human-readable and --json output modes.
"""

from autopilot.cli.expose import ExposeCollector, inject_expose
from typing import Any
import json
import sys


class Output:
  """Unified output handler that respects --json flag."""

  def __init__(
    self,
    use_json: bool = False,
    no_color: bool = False,
    expose_collector: ExposeCollector | None = None,
  ) -> None:
    self.use_json = use_json
    self.no_color = no_color
    self._expose_collector = expose_collector
    self._json_buffer: list[dict[str, Any]] = []

  def info(self, message: str) -> None:
    if self.use_json:
      self._json_buffer.append({'level': 'info', 'message': message})
    else:
      print(message)

  def success(self, message: str) -> None:
    if self.use_json:
      self._json_buffer.append({'level': 'success', 'message': message})
    else:
      print(f'OK: {message}')

  def warn(self, message: str) -> None:
    if self.use_json:
      self._json_buffer.append({'level': 'warn', 'message': message})
    else:
      print(f'WARN: {message}', file=sys.stderr)

  def error(self, message: str) -> None:
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
    """Emit a final result. In JSON mode, prints the full envelope."""
    if self._expose_collector and len(self._expose_collector) > 0:
      payload = inject_expose(dict(payload), self._expose_collector)
    if self.use_json:
      envelope = {
        'ok': ok,
        'result': payload,
        'messages': self._json_buffer,
      }
      print(json.dumps(envelope, indent=2))
      self._json_buffer = []
    else:
      status = 'OK' if ok else 'FAILED'
      print(f'\n{status}')
      for key, value in payload.items():
        print(f'  {key}: {value}')

  def table(self, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Print a simple table of rows."""
    if self.use_json:
      self._json_buffer.append({'type': 'table', 'rows': rows})
      return
    if not rows:
      print('  (none)')
      return
    widths = {col: len(col) for col in columns}
    for row in rows:
      for col in columns:
        widths[col] = max(widths[col], len(str(row.get(col, ''))))
    header = '  '.join(col.ljust(widths[col]) for col in columns)
    print(header)
    print('  '.join('-' * widths[col] for col in columns))
    for row in rows:
      line = '  '.join(str(row.get(col, '')).ljust(widths[col]) for col in columns)
      print(line)

  def flush_json(self) -> None:
    """Flush any buffered JSON messages as a standalone array."""
    if self.use_json and self._json_buffer:
      print(json.dumps(self._json_buffer, indent=2))
      self._json_buffer = []
