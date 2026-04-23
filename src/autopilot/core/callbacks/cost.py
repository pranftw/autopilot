"""CostTrackerCallback for per-epoch wall-clock, API calls, and token usage.

Data model: CostEntry.
"""

from autopilot.core.artifacts.experiment import CostArtifact
from autopilot.core.artifacts.owner import ArtifactOwner
from autopilot.core.callbacks.callback import Callback
from autopilot.core.serialization import DictMixin
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time


@dataclass
class CostEntry(DictMixin):
  """Per-epoch cost tracking."""

  epoch: int = 0
  wall_clock_s: float = 0.0
  api_calls: int = 0
  tokens_used: int = 0
  metadata: dict[str, Any] = field(default_factory=dict)


class CostTrackerCallback(ArtifactOwner, Callback):
  """Tracks per-epoch wall-clock time and optional API/token usage.

  Override measure() to add api_calls, tokens_used, or custom metadata.
  """

  def __init__(self, experiment_dir: Path | None = None) -> None:
    self.__init_artifacts__()
    self._dir = experiment_dir
    self._entries: list[CostEntry] = []
    self._epoch_start: float = 0.0
    self.cost_artifact = CostArtifact()

  def measure(self, epoch: int, elapsed: float, result: Any = None) -> CostEntry:
    """Override to add api_calls, tokens_used, or custom metadata."""
    metadata: dict[str, Any] = {}
    if result is not None and hasattr(result, 'metrics'):
      metadata = dict(result.metrics) if result.metrics else {}
    return CostEntry(epoch=epoch, wall_clock_s=round(elapsed, 3), metadata=metadata)

  def on_epoch_start(self, trainer: Any, epoch: int) -> None:
    self._epoch_start = time.monotonic()

  def on_epoch_end(self, trainer: Any, epoch: int, result: Any = None) -> None:
    elapsed = time.monotonic() - self._epoch_start if self._epoch_start else 0.0
    entry = self.measure(epoch, elapsed, result)
    self._entries.append(entry)

  def on_loop_end(self, trainer: Any, result: dict[str, Any]) -> None:
    if self._dir:
      self.cost_artifact.write(self.total().to_dict(), self._dir)

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
