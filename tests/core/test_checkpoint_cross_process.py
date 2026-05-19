"""Tests for cross-process checkpoint resume via disk scanning.

Validates the additive disk-scan fallback for 'last'/'best' resume tokens
when CheckpointCallback's in-memory paths are unset (crash recovery, fresh
process). Tests cover disk helpers in isolation, wired resolution with
in-memory primary / disk fallback, and a Trainer integration scenario.
"""

from autopilot.core.callbacks.checkpoint import CheckpointCallback
from autopilot.core.errors import ConfigError
from autopilot.core.experiment import Experiment
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.checkpoint import (
  _resolve_best_from_disk,
  _resolve_last_from_disk,
  _scan_checkpoint_directory,
)
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum
from pathlib import Path
from tests.doubles import NoopEvalModule, NoOpOptimizer
from typing import Any
import json
import pytest


def _write_valid_checkpoint(path: Path, epoch: int, metrics: dict | None = None) -> None:
  """Write a valid checkpoint JSON file with optional metrics."""
  state: dict[str, Any] = {'module': {}, 'optimizer': {}}
  if metrics is not None:
    state['experiment'] = {'metrics': metrics, 'epoch': epoch}
  else:
    state['experiment'] = {'epoch': epoch}
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(state), encoding='utf-8')


class _ScalarMetric(Metric):
  """Simple metric that returns the last updated value."""

  def __init__(self) -> None:
    super().__init__()
    self._value: float = 0.0
    self._updated: bool = False

  def update(self, datum: Any) -> None:
    self._updated = True
    if isinstance(datum, EvalDatum) and datum.metrics:
      self._value = datum.metrics.get('value', 0.0)
    elif isinstance(datum, (int, float)):
      self._value = float(datum)

  def compute(self) -> dict[str, float]:
    return {'accuracy': self._value}

  def reset(self) -> None:
    self._updated = False


class _ImprovingModule(AutoPilotModule):
  """Module with a metric that improves each epoch."""

  def __init__(self) -> None:
    super().__init__()
    self.accuracy = _ScalarMetric()
    self._epoch_value = 0.0

  def forward(self, *args: Any, **kwargs: Any) -> EvalDatum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    self._epoch_value += 0.1
    self.accuracy.update(EvalDatum(success=True, metrics={'value': self._epoch_value}))
    return EvalDatum(success=True)

  def configure_optimizers(self) -> Any:
    return NoOpOptimizer([Parameter()], lr=1.0)


class TestDiskScanHelpers:
  """Tests for _scan_checkpoint_directory in isolation."""

  def test_scan_empty_directory(self, tmp_path: Path) -> None:
    """Empty directory returns empty list."""
    result = _scan_checkpoint_directory(tmp_path)
    assert result == []

  def test_scan_nonexistent_directory(self, tmp_path: Path) -> None:
    """Non-existent directory returns empty list."""
    result = _scan_checkpoint_directory(tmp_path / 'missing')
    assert result == []

  def test_scan_sorts_by_epoch_integer(self, tmp_path: Path) -> None:
    """Files are sorted by parsed epoch integer, not lexicographically."""
    (tmp_path / 'epoch-0002.json').write_text('{}')
    (tmp_path / 'epoch-0000.json').write_text('{}')
    (tmp_path / 'epoch-0001.json').write_text('{}')
    result = _scan_checkpoint_directory(tmp_path)
    assert result == [
      tmp_path / 'epoch-0000.json',
      tmp_path / 'epoch-0001.json',
      tmp_path / 'epoch-0002.json',
    ]

  def test_scan_ignores_non_matching_files(self, tmp_path: Path) -> None:
    """Non-epoch JSON files are not included."""
    (tmp_path / 'epoch-0000.json').write_text('{}')
    (tmp_path / 'other.json').write_text('{}')
    (tmp_path / 'readme.txt').write_text('hi')
    (tmp_path / 'epoch-abc.json').write_text('{}')
    result = _scan_checkpoint_directory(tmp_path)
    assert result == [tmp_path / 'epoch-0000.json']

  def test_scan_five_digit_epoch(self, tmp_path: Path) -> None:
    """Epoch numbers > 9999 are parsed correctly."""
    (tmp_path / 'epoch-10000.json').write_text('{}')
    (tmp_path / 'epoch-0001.json').write_text('{}')
    result = _scan_checkpoint_directory(tmp_path)
    assert result == [
      tmp_path / 'epoch-0001.json',
      tmp_path / 'epoch-10000.json',
    ]


class TestResolveLastFromDisk:
  """Tests for _resolve_last_from_disk."""

  def test_ckpt_last_from_disk(self, tmp_path: Path) -> None:
    """Returns highest-epoch valid checkpoint path."""
    _write_valid_checkpoint(tmp_path / 'epoch-0000.json', epoch=0)
    _write_valid_checkpoint(tmp_path / 'epoch-0001.json', epoch=1)
    _write_valid_checkpoint(tmp_path / 'epoch-0002.json', epoch=2)
    result = _resolve_last_from_disk(tmp_path)
    assert result == tmp_path / 'epoch-0002.json'

  def test_ckpt_last_corrupt_skipped(self, tmp_path: Path) -> None:
    """Corrupt file is skipped; next valid file returned."""
    _write_valid_checkpoint(tmp_path / 'epoch-0000.json', epoch=0)
    (tmp_path / 'epoch-0001.json').write_text('{corrupt', encoding='utf-8')
    _write_valid_checkpoint(tmp_path / 'epoch-0002.json', epoch=2)
    result = _resolve_last_from_disk(tmp_path)
    assert result == tmp_path / 'epoch-0002.json'

  def test_ckpt_last_non_matching_json_ignored(self, tmp_path: Path) -> None:
    """Non-epoch JSON files in the directory are ignored."""
    _write_valid_checkpoint(tmp_path / 'epoch-0001.json', epoch=1)
    (tmp_path / 'other.json').write_text('{"random": true}', encoding='utf-8')
    result = _resolve_last_from_disk(tmp_path)
    assert result == tmp_path / 'epoch-0001.json'

  def test_ckpt_last_all_corrupt_returns_none(self, tmp_path: Path) -> None:
    """All files corrupt -> returns None (caller raises ConfigError)."""
    (tmp_path / 'epoch-0000.json').write_text('{bad', encoding='utf-8')
    (tmp_path / 'epoch-0001.json').write_text('not json at all', encoding='utf-8')
    result = _resolve_last_from_disk(tmp_path)
    assert result is None

  def test_ckpt_last_five_digit_epoch(self, tmp_path: Path) -> None:
    """Epoch-10000.json parses correctly and is the highest."""
    _write_valid_checkpoint(tmp_path / 'epoch-0099.json', epoch=99)
    _write_valid_checkpoint(tmp_path / 'epoch-10000.json', epoch=10000)
    result = _resolve_last_from_disk(tmp_path)
    assert result == tmp_path / 'epoch-10000.json'

  def test_ckpt_last_highest_corrupt_falls_back(self, tmp_path: Path) -> None:
    """When highest epoch is corrupt, falls back to next highest valid."""
    _write_valid_checkpoint(tmp_path / 'epoch-0000.json', epoch=0)
    _write_valid_checkpoint(tmp_path / 'epoch-0001.json', epoch=1)
    (tmp_path / 'epoch-0002.json').write_text('[1,2,3]', encoding='utf-8')
    result = _resolve_last_from_disk(tmp_path)
    assert result == tmp_path / 'epoch-0001.json'


class TestResolveBestFromDisk:
  """Tests for _resolve_best_from_disk."""

  def test_ckpt_best_from_disk(self, tmp_path: Path) -> None:
    """Picks checkpoint with highest monitored metric value."""
    _write_valid_checkpoint(tmp_path / 'epoch-0000.json', epoch=0, metrics={'accuracy': 0.5})
    _write_valid_checkpoint(tmp_path / 'epoch-0001.json', epoch=1, metrics={'accuracy': 0.9})
    _write_valid_checkpoint(tmp_path / 'epoch-0002.json', epoch=2, metrics={'accuracy': 0.7})
    result = _resolve_best_from_disk(tmp_path, 'accuracy')
    assert result == tmp_path / 'epoch-0001.json'

  def test_ckpt_best_tie_picks_later_epoch(self, tmp_path: Path) -> None:
    """When two checkpoints have the same metric value, later epoch wins."""
    _write_valid_checkpoint(tmp_path / 'epoch-0000.json', epoch=0, metrics={'accuracy': 0.8})
    _write_valid_checkpoint(tmp_path / 'epoch-0001.json', epoch=1, metrics={'accuracy': 0.8})
    _write_valid_checkpoint(tmp_path / 'epoch-0002.json', epoch=2, metrics={'accuracy': 0.8})
    result = _resolve_best_from_disk(tmp_path, 'accuracy')
    assert result == tmp_path / 'epoch-0002.json'

  def test_ckpt_best_missing_monitor_key_skips(self, tmp_path: Path) -> None:
    """Checkpoints without the monitor key are skipped."""
    _write_valid_checkpoint(tmp_path / 'epoch-0000.json', epoch=0, metrics={'loss': 0.1})
    _write_valid_checkpoint(tmp_path / 'epoch-0001.json', epoch=1, metrics={'accuracy': 0.9})
    _write_valid_checkpoint(tmp_path / 'epoch-0002.json', epoch=2, metrics={'loss': 0.05})
    result = _resolve_best_from_disk(tmp_path, 'accuracy')
    assert result == tmp_path / 'epoch-0001.json'

  def test_ckpt_best_all_missing_monitor_returns_none(self, tmp_path: Path) -> None:
    """No checkpoints carry the monitor key -> returns None."""
    _write_valid_checkpoint(tmp_path / 'epoch-0000.json', epoch=0, metrics={'loss': 0.1})
    _write_valid_checkpoint(tmp_path / 'epoch-0001.json', epoch=1, metrics={'loss': 0.2})
    result = _resolve_best_from_disk(tmp_path, 'accuracy')
    assert result is None

  def test_ckpt_best_corrupt_skipped(self, tmp_path: Path) -> None:
    """Corrupt files are skipped silently during best resolution."""
    _write_valid_checkpoint(tmp_path / 'epoch-0000.json', epoch=0, metrics={'accuracy': 0.5})
    (tmp_path / 'epoch-0001.json').write_text('{corrupt', encoding='utf-8')
    _write_valid_checkpoint(tmp_path / 'epoch-0002.json', epoch=2, metrics={'accuracy': 0.9})
    result = _resolve_best_from_disk(tmp_path, 'accuracy')
    assert result == tmp_path / 'epoch-0002.json'


class TestWiredResolutionLast:
  """Tests for wired 'last' resolution (in-memory primary, disk fallback)."""

  def test_ckpt_last_empty_dir_raises(self, tmp_path: Path) -> None:
    """Empty dir + no in-memory path -> ConfigError with directory in message."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    cb = CheckpointCallback(directory=ckpt_dir)
    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=0)
    mod = NoopEvalModule()
    with pytest.raises(ConfigError, match=f'no checkpoints found in {ckpt_dir}'):
      trainer.fit(mod, max_epochs=1, ckpt_path='last')

  def test_ckpt_last_all_corrupt_raises_config_error(self, tmp_path: Path) -> None:
    """All epoch files corrupt -> ConfigError."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    (ckpt_dir / 'epoch-0000.json').write_text('{bad', encoding='utf-8')
    (ckpt_dir / 'epoch-0001.json').write_text('not json', encoding='utf-8')
    cb = CheckpointCallback(directory=ckpt_dir)
    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=0)
    mod = NoopEvalModule()
    with pytest.raises(ConfigError, match=f'no checkpoints found in {ckpt_dir}'):
      trainer.fit(mod, max_epochs=1, ckpt_path='last')

  def test_ckpt_last_uses_in_memory_when_set(self, tmp_path: Path) -> None:
    """Warm callback last_checkpoint_path wins even if disk lists different epochs."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    _write_valid_checkpoint(ckpt_dir / 'epoch-0005.json', epoch=5)
    memory_path = ckpt_dir / 'epoch-0002.json'
    _write_valid_checkpoint(memory_path, epoch=2)

    cb = CheckpointCallback(directory=ckpt_dir)
    cb._last_checkpoint_path = memory_path

    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=0)
    from autopilot.core.trainer.checkpoint import resolve_ckpt_path_token

    resolved = resolve_ckpt_path_token(trainer, 'last')
    assert resolved == memory_path

  def test_ckpt_last_falls_back_to_disk_when_memory_none(self, tmp_path: Path) -> None:
    """Unset memory path, valid epoch JSON on disk -> resolves via scan."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    _write_valid_checkpoint(ckpt_dir / 'epoch-0000.json', epoch=0)
    _write_valid_checkpoint(ckpt_dir / 'epoch-0003.json', epoch=3)

    cb = CheckpointCallback(directory=ckpt_dir)
    assert cb.last_checkpoint_path is None

    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=0)
    from autopilot.core.trainer.checkpoint import resolve_ckpt_path_token

    resolved = resolve_ckpt_path_token(trainer, 'last')
    assert resolved == ckpt_dir / 'epoch-0003.json'


class TestWiredResolutionBest:
  """Tests for wired 'best' resolution (in-memory primary, disk fallback)."""

  def test_ckpt_best_no_monitor_raises(self, tmp_path: Path) -> None:
    """Resolve 'best' without monitor configured -> ConfigError."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    cb = CheckpointCallback(directory=ckpt_dir)
    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=0)
    mod = NoopEvalModule()
    with pytest.raises(ConfigError, match='no monitor set'):
      trainer.fit(mod, max_epochs=1, ckpt_path='best')

  def test_ckpt_best_uses_in_memory_when_set(self, tmp_path: Path) -> None:
    """Warm best_checkpoint_path wins without needing disk scan."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    _write_valid_checkpoint(ckpt_dir / 'epoch-0005.json', epoch=5, metrics={'acc': 0.99})
    memory_path = ckpt_dir / 'epoch-0001.json'
    _write_valid_checkpoint(memory_path, epoch=1, metrics={'acc': 0.5})

    cb = CheckpointCallback(directory=ckpt_dir, monitor='acc')
    cb._best_checkpoint_path = memory_path

    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=0)
    from autopilot.core.trainer.checkpoint import resolve_ckpt_path_token

    resolved = resolve_ckpt_path_token(trainer, 'best')
    assert resolved == memory_path

  def test_ckpt_best_falls_back_to_disk_when_memory_none(self, tmp_path: Path) -> None:
    """best_checkpoint_path None, valid checkpoints on disk -> resolves via scan."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    _write_valid_checkpoint(ckpt_dir / 'epoch-0000.json', epoch=0, metrics={'acc': 0.5})
    _write_valid_checkpoint(ckpt_dir / 'epoch-0001.json', epoch=1, metrics={'acc': 0.9})
    _write_valid_checkpoint(ckpt_dir / 'epoch-0002.json', epoch=2, metrics={'acc': 0.7})

    cb = CheckpointCallback(directory=ckpt_dir, monitor='acc')
    assert cb.best_checkpoint_path is None

    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=0)
    from autopilot.core.trainer.checkpoint import resolve_ckpt_path_token

    resolved = resolve_ckpt_path_token(trainer, 'best')
    assert resolved == ckpt_dir / 'epoch-0001.json'


class TestCrossProcessIntegration:
  """Integration: train in one Trainer, resume in fresh Trainer via disk scan."""

  def test_resume_last_cross_process_trainer(self, tmp_path: Path) -> None:
    """Train 3 epochs, new Trainer (callback paths None), ckpt_path='last' resumes."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()

    cb1 = CheckpointCallback(directory=ckpt_dir)
    mod1 = _ImprovingModule()
    exp1 = Experiment(experiment_id='e1')
    trainer1 = Trainer(callbacks=[cb1], experiment=exp1, num_sanity_val_steps=0)
    trainer1.fit(mod1, train_dataloaders=[EvalDatum(success=True)], max_epochs=3)

    assert cb1.last_checkpoint_path == ckpt_dir / 'epoch-0002.json'
    assert (ckpt_dir / 'epoch-0000.json').exists()
    assert (ckpt_dir / 'epoch-0001.json').exists()
    assert (ckpt_dir / 'epoch-0002.json').exists()

    cb2 = CheckpointCallback(directory=ckpt_dir)
    assert cb2.last_checkpoint_path is None

    mod2 = _ImprovingModule()
    exp2 = Experiment(experiment_id='e2')
    trainer2 = Trainer(callbacks=[cb2], experiment=exp2, num_sanity_val_steps=0)
    result = trainer2.fit(
      mod2, train_dataloaders=[EvalDatum(success=True)], max_epochs=5, ckpt_path='last'
    )

    assert exp2.epoch >= 2
    assert result['total_epochs'] == 2
