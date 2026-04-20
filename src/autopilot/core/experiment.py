"""Experiment lifecycle. Foundational OOP class.

Idempotent __init__: loads existing manifest or creates new.
Composes Logger and Checkpoint -- both injected, never created internally.
reason is required on promote() and reject() -- no empty defaults.
"""

from autopilot.core.checkpoint import Checkpoint
from autopilot.core.logger import Logger
from autopilot.core.models import Manifest
from pathlib import Path


class Experiment:
  """Base experiment lifecycle.

  Subclass for domain-specific experiments (AutoPilotExperiment, etc.).
  Same extension pattern as Module -> AutoPilotModule.
  """

  def __init__(
    self,
    experiment_dir: Path,
    slug: str,
    logger: Logger,
    checkpoint: Checkpoint,
    title: str = '',
    idea: str = '',
    hypothesis: str = '',
    hyperparams: dict | None = None,
    metadata: dict | None = None,
  ) -> None:
    self._dir = Path(experiment_dir)
    self._checkpoint = checkpoint
    self._logger = logger

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
  def decision(self) -> str:
    return self._manifest.decision

  @property
  def is_decided(self) -> bool:
    return self._manifest.decision != ''

  @property
  def is_promoted(self) -> bool:
    return self._manifest.decision == 'promoted'

  @property
  def is_rejected(self) -> bool:
    return self._manifest.decision == 'rejected'

  @property
  def manifest(self) -> Manifest:
    return self._manifest

  @property
  def dir(self) -> Path:
    return self._dir

  @property
  def logger(self) -> Logger:
    return self._logger

  def promote(self, reason: str, **kwargs: object) -> None:
    """Promote this experiment. reason is required."""
    self._manifest.decision = 'promoted'
    self._manifest.decision_reason = reason
    self._checkpoint.save_manifest(self._dir, self._manifest)
    self._logger.log('promoted', reason, metadata=dict(kwargs))

  def reject(self, reason: str) -> None:
    """Reject this experiment. reason is required."""
    self._manifest.decision = 'rejected'
    self._manifest.decision_reason = reason
    self._checkpoint.save_manifest(self._dir, self._manifest)
    self._logger.log('rejected', reason)

  def advance_epoch(self) -> int:
    """Increment epoch. Returns new epoch number."""
    self._manifest.current_epoch += 1
    self._checkpoint.save_manifest(self._dir, self._manifest)
    return self._manifest.current_epoch

  def finalize(self, status: str) -> None:
    """End-of-run cleanup. status is required ('success', 'failed', 'interrupted')."""
    self._logger.finalize(status)
    self._checkpoint.save_manifest(self._dir, self._manifest)

  def __repr__(self) -> str:
    decided = f', decision={self.decision!r}' if self.is_decided else ''
    return f'{type(self).__name__}(slug={self.slug!r}, epoch={self.epoch}{decided})'
