"""Atomic JSON helpers shared by FileStore sibling modules."""

from autopilot.core.errors import StoreError, TrackingError
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from typing import Any, Never


def reraise_as_store_error(exc: TrackingError, msg: str) -> Never:
  """Re-raise a TrackingError as StoreError with exception chaining.

  Args:
    exc: The TrackingError to wrap.
    msg: Error message for the StoreError.

  Raises:
    StoreError: Always raised with ``exc`` as ``__cause__``.
  """
  raise StoreError(msg) from exc


def atomic_write_json_safe(path: Path, payload: dict[str, Any]) -> None:
  """Write JSON atomically, converting TrackingError to StoreError.

  Args:
    path: Destination file path.
    payload: JSON-serializable dict to write.
  """
  try:
    atomic_write_json(path, payload)
  except TrackingError as exc:
    reraise_as_store_error(exc, str(exc))
