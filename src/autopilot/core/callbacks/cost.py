"""CostTrackerCallback for per-epoch wall-clock, API calls, and token usage.

Data model: CostEntry. Resolves paths via store.resolve_path(experiment.id)
when available, falls back to path constructor argument.

When ``emit_context=True``, emits structured cost attribution context entries
each epoch via ``trainer.emit_context`` with ``_type`` discriminator
``COST_ATTRIBUTION_TYPE`` for machine-filterable audit trails.
"""

from autopilot.core.artifacts.experiment import CostArtifact
from autopilot.core.artifacts.owner import ArtifactOwner
from autopilot.core.callbacks.callback import Callback
from autopilot.core.errors import ConfigError
from autopilot.core.serialization import DictMixin
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time

WALL_CLOCK_DECIMALS = 3
COST_ATTRIBUTION_TYPE = 'cost_attribution'


@dataclass
class CostEntry(DictMixin):
  """Per-epoch cost tracking."""

  epoch: int = 0
  wall_clock_s: float = 0.0
  api_calls: int = 0
  tokens_used: int = 0
  cost_usd: float = 0.0
  metadata: dict[str, Any] = field(default_factory=dict)


class CostTrackerCallback(ArtifactOwner, Callback):
  """Tracks per-epoch wall-clock time and optional API/token usage.

  Override measure() to add api_calls, tokens_used, or custom metadata.
  Resolves paths via store.resolve_path(experiment.id) when available,
  falls back to path constructor argument.

  CostTracker is read-only reporting. For budget enforcement, compose a
  ``BudgetGate`` in the policy layer: it reads ``cost_usd`` from metrics
  and rejects epochs that exceed the budget. ``cumulative_usd`` on this
  callback is the source for that metric.
  """

  def __init__(self, path: Path | None = None, *, emit_context: bool = False) -> None:
    """Create a cost tracker with optional fallback directory.

    Args:
      path: Base directory when store/experiment path resolution is unavailable.
      emit_context: When ``True``, emits a structured context entry each epoch
        via ``trainer.emit_context`` with metadata schema::

          {
            '_type': COST_ATTRIBUTION_TYPE,
            'epoch': int,
            'cost_usd': float,
            'cumulative': float,
          }

        Default ``False`` avoids surprising context log noise.
    """
    self.init_artifacts()
    self._dir = path
    self._entries: list[CostEntry] = []
    self._epoch_start: float = 0.0
    self.cumulative_usd: float = 0.0
    self._emit_context = emit_context
    self.cost_artifact = CostArtifact()

  def measure(self, epoch: int, elapsed: float, result: Any = None) -> CostEntry:
    """Build a ``CostEntry`` for one epoch.

    Override to add api_calls, tokens_used, or custom metadata.

    Args:
      epoch: Epoch index.
      elapsed: Wall-clock seconds for the epoch.
      result: Optional trainer/loop result (metrics copied into metadata).

    Returns:
      A populated ``CostEntry`` for this epoch.
    """
    metadata: dict[str, Any] = {}
    if result is not None and hasattr(result, 'metrics'):
      metadata = dict(result.metrics) if result.metrics else {}
    return CostEntry(
      epoch=epoch, wall_clock_s=round(elapsed, WALL_CLOCK_DECIMALS), metadata=metadata
    )

  def on_epoch_start(self, trainer: Any, module: Any, epoch: int) -> None:
    """Record monotonic timestamp at epoch start."""
    self._epoch_start = time.monotonic()

  def on_epoch_end(self, trainer: Any, module: Any, epoch: int, result: Any = None) -> None:
    """Append a ``CostEntry``, accumulate ``cumulative_usd``, and optionally emit context."""
    elapsed = time.monotonic() - self._epoch_start if self._epoch_start else 0.0
    entry = self.measure(epoch, elapsed, result)
    self._entries.append(entry)
    self.cumulative_usd += entry.cost_usd
    if self._emit_context and trainer.experiment is not None:
      trainer.emit_context(
        f'cost tracked: ${entry.cost_usd:.4f} (cumulative: ${self.cumulative_usd:.4f})',
        source='cost',
        metadata={
          '_type': COST_ATTRIBUTION_TYPE,
          'epoch': epoch,
          'cost_usd': entry.cost_usd,
          'cumulative': self.cumulative_usd,
        },
      )

  def on_loop_end(self, trainer: Any, module: Any, result: dict[str, Any]) -> None:
    """Persist aggregated cost JSON via ``CostArtifact`` when a base dir exists.

    Raises:
      ConfigError: When cost data was recorded but no path is available for persistence.
    """
    base_dir = self._resolve_dir(trainer)
    if base_dir:
      self.cost_artifact.write(self.total().to_dict(), base_dir)
    elif self._entries:
      msg = (
        'CostTrackerCallback recorded cost data but cannot persist:'
        ' no store/experiment path resolution and no fallback path.'
        ' Pass CostTrackerCallback(Path(...)) to set a fallback directory.'
      )
      raise ConfigError(msg)

  def _resolve_dir(self, trainer: Any) -> Path | None:
    experiment = trainer.experiment
    store = trainer.store
    if store is not None and experiment is not None:
      return store.resolve_path(experiment.id)
    return self._dir

  def total(self) -> CostEntry:
    """Sum wall clock, API calls, tokens, and cost across recorded epochs.

    Returns:
      Aggregate ``CostEntry`` with ``epoch`` set to 0.
    """
    total_wall = sum(e.wall_clock_s for e in self._entries)
    total_api = sum(e.api_calls for e in self._entries)
    total_tokens = sum(e.tokens_used for e in self._entries)
    total_cost = sum(e.cost_usd for e in self._entries)
    return CostEntry(
      epoch=0,
      wall_clock_s=round(total_wall, WALL_CLOCK_DECIMALS),
      api_calls=total_api,
      tokens_used=total_tokens,
      cost_usd=total_cost,
    )

  def per_epoch(self) -> list[CostEntry]:
    """Return a copy of all per-epoch entries in order."""
    return list(self._entries)

  def state_dict(self) -> dict[str, Any]:
    """Serialize recorded entries for checkpointing.

    Returns:
      Dict with an ``entries`` list of ``CostEntry`` dicts.
    """
    return {'entries': [e.to_dict() for e in self._entries]}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    """Restore entries from ``state_dict`` produced by ``state_dict``."""
    entries = state_dict['entries']
    self._entries = [CostEntry.from_dict(e) for e in entries]
    self.cumulative_usd = sum(e.cost_usd for e in self._entries)
