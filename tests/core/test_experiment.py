"""Tests for Experiment lifecycle class."""

from autopilot.core.checkpoint import Checkpoint, JSONCheckpoint
from autopilot.core.experiment import Experiment
from autopilot.core.logger import JSONLogger, Logger
from autopilot.core.models import Manifest
from pathlib import Path
import pytest


def _make_experiment(tmp_path: Path, slug: str = 'test-1', **kwargs) -> Experiment:
  """Helper: creates an Experiment with default JSON logger/checkpoint."""
  return Experiment(
    tmp_path,
    slug=slug,
    logger=kwargs.pop('logger', JSONLogger(tmp_path)),
    checkpoint=kwargs.pop('checkpoint', JSONCheckpoint()),
    **kwargs,
  )


class TestExperimentIdempotentInit:
  def test_create_new(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path, title='Test One')
    assert exp.slug == 'test-1'
    assert exp.epoch == 0
    assert not exp.is_decided
    assert (tmp_path / 'manifest.json').exists()

  def test_reload_existing(self, tmp_path: Path) -> None:
    exp1 = _make_experiment(tmp_path)
    exp1.advance_epoch()
    exp2 = _make_experiment(tmp_path)
    assert exp2.epoch == 1

  def test_reload_preserves_decision(self, tmp_path: Path) -> None:
    exp1 = _make_experiment(tmp_path)
    exp1.promote('good results')
    exp2 = _make_experiment(tmp_path)
    assert exp2.is_promoted
    assert exp2.decision == 'promoted'


class TestExperimentPromoteReject:
  def test_promote_persists(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.promote('accuracy improved by 5%')
    assert exp.is_promoted
    assert exp.manifest.decision == 'promoted'
    assert exp.manifest.decision_reason == 'accuracy improved by 5%'

  def test_reject_persists(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.reject('regression detected')
    assert exp.is_rejected
    assert exp.manifest.decision == 'rejected'
    assert exp.manifest.decision_reason == 'regression detected'

  def test_promote_without_reason_raises(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    with pytest.raises(TypeError):
      exp.promote()

  def test_reject_without_reason_raises(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    with pytest.raises(TypeError):
      exp.reject()


class TestExperimentAdvanceEpoch:
  def test_increments_and_returns(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert exp.advance_epoch() == 1
    assert exp.advance_epoch() == 2
    assert exp.epoch == 2


class TestExperimentFinalize:
  def test_finalize_saves(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.finalize('success')
    events_path = tmp_path / 'events.jsonl'
    assert events_path.exists()


class TestExperimentProperties:
  def test_is_decided_false_initially(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert not exp.is_decided
    assert not exp.is_promoted
    assert not exp.is_rejected

  def test_dir_property(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert exp.dir == tmp_path

  def test_logger_property(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert isinstance(exp.logger, JSONLogger)

  def test_manifest_property(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path, idea='my idea')
    assert exp.manifest.idea == 'my idea'


class TestExperimentRepr:
  def test_repr_without_decision(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    r = repr(exp)
    assert 'test-1' in r
    assert 'epoch=0' in r

  def test_repr_with_decision(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.promote('good')
    r = repr(exp)
    assert 'promoted' in r


class TestExperimentCustomComponents:
  def test_custom_logger(self, tmp_path: Path) -> None:
    class MyLogger(Logger):
      def __init__(self):
        self.calls: list[str] = []

      def log(self, event_type, message='', metadata=None):
        self.calls.append(event_type)

    logger = MyLogger()
    Experiment(tmp_path / 'exp', slug='test-1', logger=logger, checkpoint=JSONCheckpoint())
    assert 'experiment_created' in logger.calls

  def test_custom_checkpoint(self, tmp_path: Path) -> None:
    class MyCheckpoint(Checkpoint):
      def __init__(self):
        self.saved: list[Manifest] = []

      def save_manifest(self, experiment_dir, manifest):
        self.saved.append(manifest)

      def load_manifest(self, experiment_dir):
        return self.saved[-1] if self.saved else None

      def exists(self, experiment_dir):
        return bool(self.saved)

    cp = MyCheckpoint()
    Experiment(tmp_path / 'exp', slug='test-1', logger=JSONLogger(tmp_path / 'exp'), checkpoint=cp)
    assert len(cp.saved) == 1
    assert cp.saved[0].slug == 'test-1'

  def test_metadata_passthrough(self, tmp_path: Path) -> None:
    exp = _make_experiment(
      tmp_path,
      metadata={'profile': 'default', 'target': 'pipeline-v3'},
    )
    assert exp.manifest.metadata['profile'] == 'default'
    assert exp.manifest.metadata['target'] == 'pipeline-v3'

  def test_logger_and_checkpoint_required(self) -> None:
    with pytest.raises(TypeError):
      Experiment(Path('/tmp/x'), slug='test-1')
