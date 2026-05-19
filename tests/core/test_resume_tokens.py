"""Tests for resume token semantics: ckpt_path='last' and ckpt_path='best'."""

from autopilot.core.callbacks.checkpoint import CheckpointCallback
from autopilot.core.checkpoint import JSONCheckpointIO
from autopilot.core.errors import ConfigError, TrackingError
from autopilot.core.experiment import Experiment
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum
from pathlib import Path
from tests.doubles import NoopEvalModule, NoOpOptimizer
from typing import Any
import pytest


class _ImprovingModule(AutoPilotModule):
  """Module with a metric that improves each epoch for best-checkpoint testing."""

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


class _TiedMetricModule(AutoPilotModule):
  """Module that produces tied metric values at specific epochs."""

  def __init__(self, values: list[float]) -> None:
    super().__init__()
    self.accuracy = _ScalarMetric()
    self._values = values
    self._call_count = 0

  def forward(self, *args: Any, **kwargs: Any) -> EvalDatum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    idx = min(self._call_count, len(self._values) - 1)
    self.accuracy.update(EvalDatum(success=True, metrics={'value': self._values[idx]}))
    self._call_count += 1
    return EvalDatum(success=True)

  def configure_optimizers(self) -> Any:
    return NoOpOptimizer([Parameter()], lr=1.0)


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


def _make_train_loader() -> list[EvalDatum]:
  """Create a minimal train loader with one item."""
  return [EvalDatum(success=True)]


class TestResumeLastToken:
  """Tests for ckpt_path='last' token resolution."""

  def test_resume_last_token(self, tmp_path: Path) -> None:
    """fit(ckpt_path='last') loads the latest-epoch checkpoint."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    cb = CheckpointCallback(directory=ckpt_dir)
    mod = _ImprovingModule()
    exp = Experiment(experiment_id='e1')
    trainer = Trainer(callbacks=[cb], experiment=exp, num_sanity_val_steps=0)
    trainer.fit(mod, train_dataloaders=_make_train_loader(), max_epochs=3)

    last_path = cb.last_checkpoint_path
    assert last_path is not None
    assert last_path == ckpt_dir / 'epoch-0002.json'

    exp2 = Experiment(experiment_id='e2')
    mod2 = _ImprovingModule()
    trainer2 = Trainer(callbacks=[cb], experiment=exp2, num_sanity_val_steps=0)
    trainer2.fit(mod2, train_dataloaders=_make_train_loader(), max_epochs=5, ckpt_path='last')
    assert exp2.epoch >= 2

  def test_resume_last_no_checkpoints_raises(self, tmp_path: Path) -> None:
    """No CheckpointCallback saves -> ConfigError mentions empty directory."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    cb = CheckpointCallback(directory=ckpt_dir)
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=0)
    with pytest.raises(ConfigError, match='no checkpoints found in'):
      trainer.fit(mod, max_epochs=1, ckpt_path='last')

  def test_resume_token_with_multiple_checkpoints(self, tmp_path: Path) -> None:
    """Five epoch files: 'last' picks epoch 4 (0-based)."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    cb = CheckpointCallback(directory=ckpt_dir)
    mod = _ImprovingModule()
    exp = Experiment(experiment_id='e1')
    trainer = Trainer(callbacks=[cb], experiment=exp, num_sanity_val_steps=0)
    trainer.fit(mod, train_dataloaders=_make_train_loader(), max_epochs=5)

    assert cb.last_checkpoint_path == ckpt_dir / 'epoch-0004.json'
    assert (ckpt_dir / 'epoch-0004.json').exists()


class TestResumeBestToken:
  """Tests for ckpt_path='best' token resolution."""

  def test_resume_best_token(self, tmp_path: Path) -> None:
    """CheckpointCallback(monitor=...) with improving metric: 'best' selects peak."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    cb = CheckpointCallback(directory=ckpt_dir, monitor='accuracy')
    mod = _ImprovingModule()
    exp = Experiment(experiment_id='e1')
    trainer = Trainer(callbacks=[cb], experiment=exp, num_sanity_val_steps=0)
    trainer.fit(mod, train_dataloaders=_make_train_loader(), max_epochs=3)

    assert cb.best_checkpoint_path is not None
    assert cb.best_checkpoint_path == ckpt_dir / 'epoch-0002.json'
    assert cb.best_metric_value is not None
    assert cb.best_metric_value > 0

  def test_resume_best_no_tracking_raises(self, tmp_path: Path) -> None:
    """No monitor configured -> ConfigError instructs to set monitor."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    cb = CheckpointCallback(directory=ckpt_dir)
    mod = _ImprovingModule()
    exp = Experiment(experiment_id='e1')
    trainer = Trainer(callbacks=[cb], experiment=exp, num_sanity_val_steps=0)
    trainer.fit(mod, train_dataloaders=_make_train_loader(), max_epochs=2)

    mod2 = _ImprovingModule()
    exp2 = Experiment(experiment_id='e2')
    trainer2 = Trainer(callbacks=[cb], experiment=exp2, num_sanity_val_steps=0)
    with pytest.raises(ConfigError, match='no monitor set'):
      trainer2.fit(mod2, train_dataloaders=_make_train_loader(), max_epochs=1, ckpt_path='best')

  def test_resume_best_with_tie_epochs(self, tmp_path: Path) -> None:
    """Two epochs with same monitored value -> 'best' resolves to higher epoch index."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    cb = CheckpointCallback(directory=ckpt_dir, monitor='accuracy')
    mod = _TiedMetricModule(values=[0.5, 0.5, 0.5])
    exp = Experiment(experiment_id='e1')
    trainer = Trainer(callbacks=[cb], experiment=exp, num_sanity_val_steps=0)
    trainer.fit(mod, train_dataloaders=_make_train_loader(), max_epochs=3)

    assert cb.best_checkpoint_path == ckpt_dir / 'epoch-0002.json'


class TestResumeExplicitPath:
  """Tests for explicit Path handling (pre-token behavior preserved)."""

  def test_resume_explicit_path_unchanged(self, tmp_path: Path) -> None:
    """ckpt_path=Path(...): existing Path arguments handled identically."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    state = {'module': {}}
    ckpt_path = ckpt_dir / 'manual.json'
    JSONCheckpointIO().save(state, ckpt_path)

    mod = NoopEvalModule()
    trainer = Trainer(dry_run=True, num_sanity_val_steps=0)
    result = trainer.fit(mod, max_epochs=2, ckpt_path=ckpt_path)
    assert result['total_epochs'] == 2

  def test_resume_last_is_clean_break(self, tmp_path: Path) -> None:
    """Path('last') does NOT trigger token resolution; treats as literal file path."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    cb = CheckpointCallback(directory=ckpt_dir)
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=0)
    with pytest.raises((OSError, FileNotFoundError, TrackingError)):
      trainer.fit(mod, max_epochs=1, ckpt_path=Path('last'))


class TestResumeTokenErrors:
  """Tests for error cases in resume token resolution."""

  def test_resume_unknown_token(self) -> None:
    """ckpt_path='invalid' -> ConfigError mentioning 'last' and 'best'."""
    mod = NoopEvalModule()
    trainer = Trainer(num_sanity_val_steps=0)
    with pytest.raises(ConfigError, match="'last'") as exc_info:
      trainer.fit(mod, max_epochs=1, ckpt_path='invalid')
    assert "'best'" in str(exc_info.value)

  def test_resume_multiple_callbacks_raises(self, tmp_path: Path) -> None:
    """Two CheckpointCallback instances -> ConfigError with 'ambiguous'."""
    dir1 = tmp_path / 'c1'
    dir1.mkdir()
    dir2 = tmp_path / 'c2'
    dir2.mkdir()
    cb1 = CheckpointCallback(directory=dir1)
    cb2 = CheckpointCallback(directory=dir2)
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[cb1, cb2], num_sanity_val_steps=0)
    with pytest.raises(ConfigError, match='ambiguous'):
      trainer.fit(mod, max_epochs=1, ckpt_path='last')


class TestResumeTokenTypeAnnotation:
  """Type-level test: fit signature accepts str | Path | None."""

  def test_resume_token_type_annotation(self) -> None:
    """fit() signature accepts str | Path | None for ckpt_path."""
    import inspect

    sig = inspect.signature(Trainer.fit)
    param = sig.parameters['ckpt_path']
    annotation = str(param.annotation)
    assert 'str' in annotation or 'str' in str(param)
    assert param.default is None


class TestCheckpointCallbackState:
  """Tests for CheckpointCallback state_dict/load_state_dict persistence."""

  def test_checkpoint_callback_state_dict_round_trip(self, tmp_path: Path) -> None:
    """Full state survives save/load cycle."""
    cb = CheckpointCallback(directory=tmp_path, monitor='acc')
    cb._last_checkpoint_path = tmp_path / 'epoch-0009.json'
    cb._best_checkpoint_path = tmp_path / 'epoch-0005.json'
    cb._best_metric_value = 0.95

    state = cb.state_dict()
    cb2 = CheckpointCallback(directory=tmp_path, monitor='acc')
    cb2.load_state_dict(state)

    assert cb2._last_checkpoint_path == tmp_path / 'epoch-0009.json'
    assert cb2._best_checkpoint_path == tmp_path / 'epoch-0005.json'
    assert cb2._best_metric_value == 0.95

  def test_checkpoint_callback_state_dict_empty_fresh(self, tmp_path: Path) -> None:
    """Fresh callback returns empty state_dict."""
    cb = CheckpointCallback(directory=tmp_path)
    assert cb.state_dict() == {}

  def test_checkpoint_callback_state_dict_partial(self, tmp_path: Path) -> None:
    """Only last_checkpoint_path set; state has one key."""
    cb = CheckpointCallback(directory=tmp_path)
    cb._last_checkpoint_path = tmp_path / 'epoch-0003.json'

    state = cb.state_dict()
    assert list(state.keys()) == ['last_checkpoint_path']
    assert state['last_checkpoint_path'] == str(tmp_path / 'epoch-0003.json')

  def test_load_state_dict_empty_dict_noop(self, tmp_path: Path) -> None:
    """load_state_dict({}) leaves all fields as None."""
    cb = CheckpointCallback(directory=tmp_path, monitor='acc')
    cb.load_state_dict({})
    assert cb._last_checkpoint_path is None
    assert cb._best_checkpoint_path is None
    assert cb._best_metric_value is None

  def test_resume_best_resolves_after_load_state_dict(self, tmp_path: Path) -> None:
    """Loaded state enables 'best' token resolution on Trainer."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    best_path = ckpt_dir / 'epoch-0005.json'
    JSONCheckpointIO().save({'module': {}}, best_path)

    cb = CheckpointCallback(directory=ckpt_dir, monitor='acc')
    cb.load_state_dict(
      {
        'best_checkpoint_path': str(best_path),
        'best_metric_value': 0.92,
        'last_checkpoint_path': str(best_path),
      }
    )

    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=0)
    resolved = trainer._resolve_best_checkpoint()
    assert resolved == best_path

  def test_trainer_checkpoint_resume_restores_callback_state(self, tmp_path: Path) -> None:
    """End-to-end: Trainer saves and restores CheckpointCallback state."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    cb = CheckpointCallback(directory=ckpt_dir, monitor='accuracy')
    mod = _ImprovingModule()
    exp = Experiment(experiment_id='e1')
    trainer = Trainer(callbacks=[cb], experiment=exp, num_sanity_val_steps=0)
    trainer.fit(mod, train_dataloaders=_make_train_loader(), max_epochs=3)

    last_path = cb.last_checkpoint_path
    assert last_path is not None

    ckpt_state = JSONCheckpointIO().load(last_path)
    saved_cb_state = ckpt_state['callbacks']['CheckpointCallback_0']

    cb2 = CheckpointCallback(directory=ckpt_dir, monitor='accuracy')
    exp2 = Experiment(experiment_id='e2')
    mod2 = _ImprovingModule()
    trainer2 = Trainer(callbacks=[cb2], experiment=exp2, num_sanity_val_steps=0)
    trainer2.fit(mod2, train_dataloaders=_make_train_loader(), max_epochs=3, ckpt_path=last_path)

    assert cb2._best_checkpoint_path == Path(saved_cb_state['best_checkpoint_path'])
    assert cb2._best_metric_value == saved_cb_state['best_metric_value']
    assert cb2._last_checkpoint_path == Path(saved_cb_state['last_checkpoint_path'])
