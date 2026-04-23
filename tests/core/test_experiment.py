"""Tests for Experiment lifecycle class."""

from autopilot.core.artifacts.artifact import JSONArtifact, JSONLArtifact
from autopilot.core.checkpoint import Checkpoint, JSONCheckpoint
from autopilot.core.errors import ExperimentError
from autopilot.core.experiment import Experiment, PromotionExperiment
from autopilot.core.logger import JSONLogger, Logger
from autopilot.core.models import Manifest
from pathlib import Path
from unittest.mock import MagicMock
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


def _make_promotion(tmp_path: Path, slug: str = 'test-1', **kwargs) -> PromotionExperiment:
  """Helper: creates a PromotionExperiment."""
  return PromotionExperiment(
    tmp_path,
    slug=slug,
    logger=kwargs.pop('logger', JSONLogger(tmp_path)),
    checkpoint=kwargs.pop('checkpoint', JSONCheckpoint()),
    **kwargs,
  )


class TestExperimentBase:
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

  def test_advance_epoch(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert exp.advance_epoch() == 1
    assert exp.advance_epoch() == 2
    assert exp.epoch == 2

  def test_on_epoch_advance_hook(self, tmp_path: Path) -> None:
    epochs_seen: list[int] = []

    class HookExp(Experiment):
      def on_epoch_advance(self, epoch: int) -> None:
        epochs_seen.append(epoch)

    exp = HookExp(
      tmp_path,
      slug='test-1',
      logger=JSONLogger(tmp_path),
      checkpoint=JSONCheckpoint(),
    )
    exp.advance_epoch()
    exp.advance_epoch()
    assert epochs_seen == [1, 2]

  def test_finalize(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.finalize('success')
    assert (tmp_path / 'events.jsonl').exists()

  def test_on_finalize_hook(self, tmp_path: Path) -> None:
    statuses: list[str] = []

    class HookExp(Experiment):
      def on_finalize(self, status: str) -> None:
        statuses.append(status)

    exp = HookExp(
      tmp_path,
      slug='test-1',
      logger=JSONLogger(tmp_path),
      checkpoint=JSONCheckpoint(),
    )
    exp.finalize('success')
    assert statuses == ['success']

  def test_decision_none_initially(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert exp.decision is None

  def test_is_decided_false_initially(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert not exp.is_decided

  def test_decide_persists(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.decide('accepted', 'looks good')
    assert exp.decision == 'accepted'
    assert exp.decision_reason == 'looks good'

  def test_decide_logs_event(self, tmp_path: Path) -> None:
    class TrackingLogger(Logger):
      def __init__(self):
        self.calls: list[tuple] = []

      def log(self, event_type, message='', metadata=None):
        self.calls.append((event_type, message, metadata))

    logger = TrackingLogger()
    exp = Experiment(
      tmp_path,
      slug='test-1',
      logger=logger,
      checkpoint=JSONCheckpoint(),
    )
    exp.decide('accepted', 'reason here', extra='data')
    assert any(c[0] == 'accepted' for c in logger.calls)

  def test_decide_one_shot(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.decide('accepted', 'first')
    with pytest.raises(ExperimentError, match='already decided'):
      exp.decide('rejected', 'second')

  def test_decide_unrestricted_by_default(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.decide('any_string', 'reason')
    assert exp.decision == 'any_string'

  def test_decide_validates_against_valid_decisions(self, tmp_path: Path) -> None:
    class RestrictedExp(Experiment):
      def valid_decisions(self) -> frozenset[str]:
        return frozenset({'go', 'nogo'})

    exp = RestrictedExp(
      tmp_path,
      slug='test-1',
      logger=JSONLogger(tmp_path),
      checkpoint=JSONCheckpoint(),
    )
    with pytest.raises(ExperimentError, match='invalid decision'):
      exp.decide('invalid', 'reason')

  def test_valid_decisions_default_none(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert exp.valid_decisions() is None

  def test_state_dict_round_trip(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path, idea='my idea')
    exp.advance_epoch()
    state = exp.state_dict()
    exp2 = _make_experiment(tmp_path / 'other', slug='other')
    exp2.load_state_dict(state)
    assert exp2.epoch == 1

  def test_dir_property(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert exp.dir == tmp_path

  def test_logger_property(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert isinstance(exp.logger, JSONLogger)

  def test_manifest_property(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path, idea='my idea')
    assert exp.manifest.idea == 'my idea'

  def test_repr_without_decision(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    r = repr(exp)
    assert 'test-1' in r
    assert 'epoch=0' in r

  def test_repr_with_decision(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.decide('accepted', 'ok')
    r = repr(exp)
    assert 'accepted' in r

  def test_metadata_passthrough(self, tmp_path: Path) -> None:
    exp = _make_experiment(
      tmp_path,
      metadata={'profile': 'default', 'target': 'pipeline-v3'},
    )
    assert exp.manifest.metadata['profile'] == 'default'

  def test_logger_and_checkpoint_required(self) -> None:
    with pytest.raises(TypeError):
      Experiment(Path('/tmp/x'), slug='test-1')


class TestArtifactRegistration:
  def test_setattr_registers_artifact(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    art = JSONArtifact('test.json')
    exp.events = art
    assert exp.artifacts['events'] is art

  def test_setattr_ignores_non_artifacts(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.foo = 'bar'
    assert 'foo' not in exp.artifacts

  def test_artifacts_property(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    art = JSONArtifact('test.json')
    exp.data = art
    artifacts = exp.artifacts
    assert 'data' in artifacts
    assert artifacts is not exp._artifacts

  def test_multiple_artifacts(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.a = JSONArtifact('a.json')
    exp.b = JSONLArtifact('b.jsonl')
    assert len(exp.artifacts) == 2

  def test_artifact_replace(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.a = JSONArtifact('old.json')
    exp.a = JSONArtifact('new.json')
    assert exp.artifacts['a'].filename == 'new.json'


class TestExperimentCustomSubclass:
  def test_custom_valid_decisions(self, tmp_path: Path) -> None:
    class ABTest(Experiment):
      def valid_decisions(self) -> frozenset[str]:
        return frozenset({'winner', 'loser', 'inconclusive'})

    exp = ABTest(
      tmp_path,
      slug='ab-1',
      logger=JSONLogger(tmp_path),
      checkpoint=JSONCheckpoint(),
    )
    exp.decide('winner', 'significant results')
    assert exp.decision == 'winner'

  def test_custom_hooks(self, tmp_path: Path) -> None:
    events: list[str] = []

    class HookExp(Experiment):
      def on_epoch_advance(self, epoch: int) -> None:
        events.append(f'epoch_{epoch}')

      def on_finalize(self, status: str) -> None:
        events.append(f'finalize_{status}')

    exp = HookExp(
      tmp_path,
      slug='test-1',
      logger=JSONLogger(tmp_path),
      checkpoint=JSONCheckpoint(),
    )
    exp.advance_epoch()
    exp.finalize('success')
    assert events == ['epoch_1', 'finalize_success']

  def test_re_decidable_experiment(self, tmp_path: Path) -> None:
    class ReDecidable(Experiment):
      def decide(self, decision: str, reason: str, **kwargs) -> None:
        self._manifest.decision = decision
        self._manifest.decision_reason = reason
        self._checkpoint.save_manifest(self._dir, self._manifest)

    exp = ReDecidable(
      tmp_path,
      slug='test-1',
      logger=JSONLogger(tmp_path),
      checkpoint=JSONCheckpoint(),
    )
    exp.decide('first', 'reason1')
    exp.decide('second', 'reason2')
    assert exp.decision == 'second'

  def test_custom_logger_integration(self, tmp_path: Path) -> None:
    class MyLogger(Logger):
      def __init__(self):
        self.calls: list[str] = []

      def log(self, event_type, message='', metadata=None):
        self.calls.append(event_type)

    logger = MyLogger()
    Experiment(tmp_path / 'exp', slug='test-1', logger=logger, checkpoint=JSONCheckpoint())
    assert 'experiment_created' in logger.calls

  def test_custom_checkpoint_integration(self, tmp_path: Path) -> None:
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


class TestPromotionExperiment:
  def test_valid_decisions_frozenset(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    assert exp.valid_decisions() == frozenset({'promoted', 'rejected'})

  def test_promote_delegates_to_decide(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    exp.promote('accuracy improved by 5%')
    assert exp.decision == 'promoted'
    assert exp.decision_reason == 'accuracy improved by 5%'

  def test_reject_delegates_to_decide(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    exp.reject('regression detected')
    assert exp.decision == 'rejected'
    assert exp.decision_reason == 'regression detected'

  def test_promote_without_reason_raises(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    with pytest.raises(TypeError):
      exp.promote()

  def test_reject_without_reason_raises(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    with pytest.raises(TypeError):
      exp.reject()

  def test_is_promoted(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    exp.promote('good')
    assert exp.is_promoted

  def test_is_rejected(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    exp.reject('bad')
    assert exp.is_rejected

  def test_invalid_decision_raises(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    with pytest.raises(ExperimentError, match='invalid decision'):
      exp.decide('invalid', 'reason')

  def test_double_promote_raises(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    exp.promote('first')
    with pytest.raises(ExperimentError, match='already decided'):
      exp.promote('second')

  def test_reload_preserves_promote(self, tmp_path: Path) -> None:
    exp1 = _make_promotion(tmp_path)
    exp1.promote('good results')
    exp2 = _make_promotion(tmp_path)
    assert exp2.is_promoted
    assert exp2.decision == 'promoted'

  def test_reload_preserves_reject(self, tmp_path: Path) -> None:
    exp1 = _make_promotion(tmp_path)
    exp1.reject('bad results')
    exp2 = _make_promotion(tmp_path)
    assert exp2.is_rejected

  def test_promote_kwargs_in_metadata(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    exp.promote('good', extra='data')
    assert exp.is_promoted


class TestExperimentStoreAndRollback:
  def test_store_on_experiment(self, tmp_path: Path) -> None:
    store = MagicMock()
    exp = _make_experiment(tmp_path, store=store)
    assert exp.store is store

  def test_store_none_by_default(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert exp.store is None

  def test_should_rollback_default_false(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert exp.should_rollback is False

  def test_should_rollback_settable(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.should_rollback = True
    assert exp.should_rollback is True

  def test_best_epoch_default_zero(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert exp.best_epoch == 0

  def test_best_epoch_settable(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.best_epoch = 5
    assert exp.best_epoch == 5

  def test_rollback_calls_store_checkout(self, tmp_path: Path) -> None:
    store = MagicMock()
    exp = _make_experiment(tmp_path, store=store)
    exp.rollback(3)
    store.checkout.assert_called_once_with(3)

  def test_rollback_without_store_noop(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.rollback(3)

  def test_rollback_epoch_zero_noop(self, tmp_path: Path) -> None:
    store = MagicMock()
    exp = _make_experiment(tmp_path, store=store)
    exp.rollback(0)
    store.checkout.assert_not_called()


class TestExperimentLifecycleHooks:
  def test_on_epoch_complete_default_noop(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.on_epoch_complete(1, {'accuracy': 0.8})

  def test_on_validation_complete_default_noop(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.on_validation_complete(1, {'accuracy': 0.8})

  def test_on_loop_complete_default_noop(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    exp.on_loop_complete({'total_epochs': 3})

  def test_build_summary_default_passthrough(self, tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    loop_result = {'epochs': [], 'total_epochs': 0}
    assert exp.build_summary(loop_result) is loop_result


class TestPromotionExperimentDomain:
  def test_on_validation_complete_first_epoch_writes_baseline(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    exp.on_validation_complete(1, {'accuracy': 0.8})
    baseline = exp._read_baseline()
    assert baseline == {'accuracy': 0.8}
    assert exp.should_rollback is False

  def test_validation_complete_detects_regression(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    exp.on_validation_complete(1, {'accuracy': 0.8})
    exp.on_validation_complete(2, {'accuracy': 0.5})
    assert exp.should_rollback is True

  def test_on_validation_complete_updates_baseline_on_improvement(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    exp.on_validation_complete(1, {'accuracy': 0.7})
    exp.on_validation_complete(2, {'accuracy': 0.9})
    baseline = exp._read_baseline()
    assert baseline == {'accuracy': 0.9}
    assert exp.should_rollback is False
    assert exp.best_epoch == 2

  def test_on_regression_hook_called(self, tmp_path: Path) -> None:
    hooks: list[tuple] = []

    class TrackingPromo(PromotionExperiment):
      def on_regression(self, epoch, comparison):
        hooks.append(('regression', epoch))

    exp = TrackingPromo(
      tmp_path,
      slug='test-1',
      logger=JSONLogger(tmp_path),
      checkpoint=JSONCheckpoint(),
    )
    exp.on_validation_complete(1, {'accuracy': 0.8})
    exp.on_validation_complete(2, {'accuracy': 0.5})
    assert hooks == [('regression', 2)]

  def test_on_improvement_hook_called(self, tmp_path: Path) -> None:
    hooks: list[tuple] = []

    class TrackingPromo(PromotionExperiment):
      def on_improvement(self, epoch, metrics):
        hooks.append(('improvement', epoch))

    exp = TrackingPromo(
      tmp_path,
      slug='test-1',
      logger=JSONLogger(tmp_path),
      checkpoint=JSONCheckpoint(),
    )
    exp.on_validation_complete(1, {'accuracy': 0.7})
    exp.on_validation_complete(2, {'accuracy': 0.9})
    assert hooks == [('improvement', 2)]

  def test_comparison_artifact_written_each_epoch(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    exp.on_validation_complete(1, {'accuracy': 0.7})
    exp.on_validation_complete(2, {'accuracy': 0.9})
    data = exp.comparison_artifact.read(tmp_path, epoch=2)
    assert data is not None
    assert data['epoch'] == 2

  def test_threshold_pct_passed_to_compare(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path, threshold_pct=0.5)
    exp.on_validation_complete(1, {'accuracy': 0.8})
    exp.on_validation_complete(2, {'accuracy': 0.79})
    assert exp.should_rollback is False

  def test_decision_semantics_unchanged(self, tmp_path: Path) -> None:
    exp = _make_promotion(tmp_path)
    exp.promote('good results')
    assert exp.is_promoted
    assert exp.decision == 'promoted'

  def test_store_rollback_on_should_rollback(self, tmp_path: Path) -> None:
    store = MagicMock()
    exp = _make_promotion(tmp_path, store=store)
    exp.on_validation_complete(1, {'accuracy': 0.8})
    exp.best_epoch = 1
    exp.should_rollback = True
    exp.rollback(exp.best_epoch)
    store.checkout.assert_called_once_with(1)
