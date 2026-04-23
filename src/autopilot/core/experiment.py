"""Experiment lifecycle. Foundational OOP class.

Idempotent __init__: loads existing manifest or creates new.
Composes Logger and Checkpoint -- both injected, never created internally.

Experiment is the base class. PromotionExperiment adds promote/reject semantics
and runtime domain hooks (baseline tracking, metric comparison, rollback signals).
Artifact auto-registers via ArtifactOwner.__setattr__.
"""

from autopilot.core.artifacts.epoch import MetricComparisonArtifact
from autopilot.core.artifacts.experiment import BaselineArtifact, PromotionArtifact
from autopilot.core.artifacts.owner import ArtifactOwner
from autopilot.core.checkpoint import Checkpoint
from autopilot.core.comparison import MetricComparison, compare_metrics
from autopilot.core.errors import ExperimentError
from autopilot.core.logger import Logger
from autopilot.core.models import Manifest
from autopilot.core.store import Store
from autopilot.core.summary import build_experiment_summary
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Experiment(ArtifactOwner):
  """Experiment lifecycle base.

  Concrete base class, not ABC. Provides manifest management,
  epoch advancement, persistence (via injected Checkpoint + Logger),
  lifecycle hooks, and optional Store ownership. Inherits from ArtifactOwner
  for artifact auto-registration via __setattr__.

  Ownership model:
    Experiment owns manifest, optional Store, should_rollback, and best_epoch.
    These do NOT live on Trainer. Trainer takes experiment= kwarg.
    There is no Trainer.store, Trainer.regression_detected, or Trainer._best_epoch.

  Lifecycle: idempotent open (loads existing manifest or creates),
  advance_epoch(), finalize(status).

  Extension points (override for domain logic):
    on_epoch_complete(epoch, train_metrics)       -- after training phase
    on_validation_complete(epoch, val_metrics, ..) -- after validation phase
    on_loop_complete(loop_result)                  -- when loop finishes
    build_summary(loop_result, cost_tracker)       -- build summary payload
    rollback(to_epoch)                             -- rollback store to prior epoch
    decide(decision, reason)                       -- record a decision
    valid_decisions()                              -- allowed decision strings
    on_epoch_advance(epoch)                        -- hook after epoch increment
    on_finalize(status)                            -- hook before finalization
  """

  def __init__(
    self,
    experiment_dir: Path,
    slug: str,
    logger: Logger,
    checkpoint: Checkpoint,
    store: Store | None = None,
    title: str | None = None,
    idea: str | None = None,
    hypothesis: str | None = None,
    hyperparams: dict | None = None,
    metadata: dict | None = None,
  ) -> None:
    self.__init_artifacts__()
    self._dir = Path(experiment_dir)
    self._checkpoint = checkpoint
    self._logger = logger
    self._store = store
    self._should_rollback = False
    self._best_epoch = 0

    loaded = self._checkpoint.load_manifest(self._dir)
    if loaded is not None:
      self._manifest = loaded
    else:
      self._dir.mkdir(parents=True, exist_ok=True)
      self._manifest = Manifest(
        slug=slug,
        title=title or slug,
        idea=idea,
        hypothesis=hypothesis,
        hyperparams=hyperparams or {},
        metadata=metadata or {},
      )
      self._checkpoint.save_manifest(self._dir, self._manifest)
      self._logger.log('experiment_created', f'created {slug!r}')

  @property
  def slug(self) -> str:
    return self._manifest.slug

  @property
  def epoch(self) -> int:
    return self._manifest.current_epoch

  @property
  def manifest(self) -> Manifest:
    return self._manifest

  @property
  def dir(self) -> Path:
    return self._dir

  @property
  def logger(self) -> Logger:
    return self._logger

  @property
  def decision(self) -> str | None:
    return self._manifest.decision

  @property
  def decision_reason(self) -> str | None:
    return self._manifest.decision_reason

  @property
  def is_decided(self) -> bool:
    return self._manifest.is_decided

  @property
  def store(self) -> Store | None:
    return self._store

  @property
  def should_rollback(self) -> bool:
    return self._should_rollback

  @should_rollback.setter
  def should_rollback(self, value: bool) -> None:
    self._should_rollback = value

  @property
  def best_epoch(self) -> int:
    return self._best_epoch

  @best_epoch.setter
  def best_epoch(self, value: int) -> None:
    self._best_epoch = value

  def rollback(self, to_epoch: int) -> None:
    """Rollback store to a prior epoch. Override for custom rollback."""
    if self._store and to_epoch > 0:
      self._store.checkout(to_epoch)

  def on_epoch_complete(self, epoch: int, train_metrics: dict[str, float]) -> None:
    """Called after training phase. Override for domain logic."""

  def on_validation_complete(
    self,
    epoch: int,
    val_metrics: dict[str, float],
    metric_metadata: dict[str, bool] | None = None,
  ) -> None:
    """Called after validation phase. Override for domain logic.
    Subclass sets should_rollback=True here if the experiment should roll back."""

  def on_loop_complete(self, loop_result: dict[str, Any]) -> None:
    """Called when the training loop finishes. Override for summaries/cleanup."""

  def build_summary(self, loop_result: dict[str, Any], cost_tracker: Any = None) -> dict[str, Any]:
    """Build experiment summary. Override for custom summaries."""
    return loop_result

  def valid_decisions(self) -> frozenset[str] | None:
    """Return the set of valid decision strings, or None for unrestricted."""
    return None

  def decide(self, decision: str, reason: str, **kwargs: Any) -> None:
    """Record a decision on this experiment."""
    allowed = self.valid_decisions()
    if allowed is not None and decision not in allowed:
      raise ExperimentError(f'invalid decision {decision!r}, valid: {sorted(allowed)}')
    if self.is_decided:
      raise ExperimentError(f'experiment already decided: {self.decision!r}')
    self._manifest.decision = decision
    self._manifest.decision_reason = reason
    self._checkpoint.save_manifest(self._dir, self._manifest)
    self._logger.log(decision, reason, metadata=dict(kwargs))

  def advance_epoch(self) -> int:
    """Increment epoch. Returns new epoch number."""
    self._manifest.current_epoch += 1
    self._checkpoint.save_manifest(self._dir, self._manifest)
    self.on_epoch_advance(self._manifest.current_epoch)
    return self._manifest.current_epoch

  def on_epoch_advance(self, epoch: int) -> None:
    """Hook called after epoch is incremented. Override for side effects."""

  def finalize(self, status: str) -> None:
    """End-of-run cleanup."""
    self.on_finalize(status)
    self._logger.finalize(status)
    self._checkpoint.save_manifest(self._dir, self._manifest)

  def on_finalize(self, status: str) -> None:
    """Hook called before finalization persists. Override for cleanup."""

  def state_dict(self) -> dict[str, Any]:
    """Return experiment state for checkpointing."""
    return {
      'manifest': self._manifest.to_dict(),
    }

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    """Restore experiment state from checkpoint."""
    self._manifest = Manifest.from_dict(state_dict['manifest'])

  def __repr__(self) -> str:
    decided = f', decision={self.decision!r}' if self.is_decided else ''
    return f'{type(self).__name__}(slug={self.slug!r}, epoch={self.epoch}{decided})'


class PromotionExperiment(Experiment):
  """Promotion-relegation experiment lifecycle.

  Extends Experiment with promote/reject decision semantics and runtime
  regression detection. Compares validation metrics against a persisted
  best baseline using compare_metrics() from core/comparison.py.

  Decision semantics:
    promote(reason)          -- persist promotion.json and decide 'promoted'
    reject(reason)           -- decide 'rejected'
    is_promoted / is_rejected -- convenience properties

  Runtime domain (on_validation_complete):
    Compares val_metrics against best baseline via compare_metrics().
    On regression: sets should_rollback=True, calls on_regression() hook.
    On improvement: updates baseline, sets best_epoch, calls on_improvement().
    Persists metric_comparison.json per epoch via comparison_artifact.

  Constructor params:
    threshold_pct (float)    -- percent threshold for regression detection

  Extension hooks:
    on_regression(epoch, comparison)  -- called on metric regression
    on_improvement(epoch, metrics)    -- called on metric improvement
  """

  def __init__(
    self,
    experiment_dir: Path,
    slug: str,
    logger: Logger,
    checkpoint: Checkpoint,
    store: Store | None = None,
    threshold_pct: float = 0.0,
    **kwargs: Any,
  ) -> None:
    super().__init__(experiment_dir, slug, logger, checkpoint, store=store, **kwargs)
    self._threshold_pct = threshold_pct
    self.best_baseline_artifact = BaselineArtifact()
    self.comparison_artifact = MetricComparisonArtifact()
    self.promotion_artifact = PromotionArtifact()

  def valid_decisions(self) -> frozenset[str]:
    return frozenset({'promoted', 'rejected'})

  def promote(self, reason: str, **kwargs: Any) -> None:
    """Promote this experiment and persist promotion.json."""
    self.promotion_artifact.write(
      {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'decision': 'promoted',
        'reason': reason,
        'reviewer': kwargs.get('reviewer'),
        'metadata': {k: v for k, v in kwargs.items() if k != 'reviewer'},
      },
      self.dir,
    )
    self.decide('promoted', reason, **kwargs)

  def reject(self, reason: str) -> None:
    """Reject this experiment."""
    self.decide('rejected', reason)

  @property
  def is_promoted(self) -> bool:
    return self.decision == 'promoted'

  @property
  def is_rejected(self) -> bool:
    return self.decision == 'rejected'

  def on_validation_complete(
    self,
    epoch: int,
    val_metrics: dict[str, float],
    metric_metadata: dict[str, bool] | None = None,
  ) -> None:
    baseline = self._read_baseline()
    if baseline is None:
      self._write_baseline(epoch, val_metrics)
      return

    comparison = compare_metrics(
      baseline,
      val_metrics,
      threshold_pct=self._threshold_pct,
      metric_metadata=metric_metadata,
    )
    comparison_with_epoch = MetricComparison(
      epoch=epoch,
      per_metric_deltas=comparison.per_metric_deltas,
      regressions=comparison.regressions,
      improvements=comparison.improvements,
    )
    self.comparison_artifact.write(comparison_with_epoch.to_dict(), self.dir, epoch=epoch)

    if comparison.regression_detected:
      self.should_rollback = True
      self.on_regression(epoch, comparison_with_epoch)
    else:
      self._write_baseline(epoch, val_metrics)
      self.best_epoch = epoch
      self.on_improvement(epoch, val_metrics)

  def on_regression(self, epoch: int, comparison: MetricComparison) -> None:
    """Hook called on regression. Override for custom behavior."""

  def on_improvement(self, epoch: int, metrics: dict[str, float]) -> None:
    """Hook called on improvement. Override for custom behavior."""

  def build_summary(self, loop_result: dict[str, Any], cost_tracker: Any = None) -> Any:
    return build_experiment_summary(
      self.dir,
      loop_result,
      cost_tracker=cost_tracker,
    )

  def _read_baseline(self) -> dict[str, float] | None:
    data = self.best_baseline_artifact.read(self.dir)
    if data is None:
      return None
    return data.get('metrics')

  def _write_baseline(self, epoch: int, metrics: dict[str, float]) -> None:
    self.best_baseline_artifact.write(
      {'epoch': epoch, 'metrics': metrics},
      self.dir,
    )
