"""CostTracker callback for per-epoch wall-clock, API calls, and token usage."""

from autopilot.core.callbacks import Callback
from autopilot.core.stage_io import write_experiment_artifact
from autopilot.core.stage_models import CostEntry
from pathlib import Path
from typing import Any
import time


class CostTracker(Callback):
  """Tracks per-epoch wall-clock time and optional API/token usage."""

  def __init__(self, experiment_dir: Path | None = None) -> None:
    self._dir = experiment_dir
    self._entries: list[CostEntry] = []
    self._epoch_start: float = 0.0

  def on_epoch_start(self, trainer: Any, epoch: int) -> None:
    self._epoch_start = time.monotonic()

  def on_epoch_end(self, trainer: Any, epoch: int, result: Any = None) -> None:
    elapsed = time.monotonic() - self._epoch_start if self._epoch_start else 0.0
    metadata: dict[str, Any] = {}
    if result and hasattr(result, 'metrics'):
      metadata = dict(result.metrics) if result.metrics else {}

    entry = CostEntry(
      epoch=epoch,
      wall_clock_s=round(elapsed, 3),
      metadata=metadata,
    )
    self._entries.append(entry)

  def on_loop_end(self, trainer: Any, result: dict[str, Any]) -> None:
    if self._dir:
      write_experiment_artifact(self._dir, 'cost_summary.json', self.total().to_dict())

  def total(self) -> CostEntry:
    total_wall = sum(e.wall_clock_s for e in self._entries)
    total_api = sum(e.api_calls for e in self._entries)
    total_tokens = sum(e.tokens_used for e in self._entries)
    return CostEntry(
      epoch=0,
      wall_clock_s=round(total_wall, 3),
      api_calls=total_api,
      tokens_used=total_tokens,
    )

  def per_epoch(self) -> list[CostEntry]:
    return list(self._entries)

  def state_dict(self) -> dict[str, Any]:
    return {'entries': [e.to_dict() for e in self._entries]}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    entries = state_dict.get('entries', [])
    self._entries = [CostEntry.from_dict(e) for e in entries]
