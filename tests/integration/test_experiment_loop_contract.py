"""Integration tests for the Experiment/EpochLoop/Trainer contract.

Verifies that Trainer.fit() works correctly with base Experiment and
AutoPilotExperiment, that hooks fire in the right order, that lifecycle
state transitions are correct, and that edge cases are handled.
"""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.core.callbacks.callback import Callback
from autopilot.core.enums import Status
from autopilot.core.errors import ExperimentError
from autopilot.core.experiment import Experiment
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum, GateResult
from autopilot.data.dataloader import DataLoader
from autopilot.policy.policy import Policy
from tests.doubles import DirectNumericLoss, NoopEvalModule, NoOpOptimizer
from typing import Any, cast
import pytest


class _FullModule(AutoPilotModule):
  def __init__(self):
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.param])
    self.metric = _CountMetric()
    self._opt = NoOpOptimizer([self.param])

  def forward(self, batch):
    return batch

  def training_step(self, batch, batch_idx):
    return batch

  def validation_step(self, batch, batch_idx):
    return batch

  def configure_optimizers(self):
    return self._opt


class _CountMetric(Metric):
  def __init__(self):
    super().__init__()
    self.add_state('_n', 0)

  def update(self, datum):
    self._n += 1

  def compute(self):
    return {'count': float(self._n)}


class _FailingPolicy(Policy):
  """Policy that always fails."""

  def forward(self, result: Result) -> GateResult:
    return GateResult.FAIL


class _PassingPolicy(Policy):
  """Policy that always passes."""

  def forward(self, result: Result) -> GateResult:
    return GateResult.PASSED


def _make_loader(n: int = 4) -> DataLoader:
  return DataLoader([EvalDatum(metadata={'i': i}) for i in range(n)], batch_size=1)


def _make_empty_loader() -> DataLoader:
  return DataLoader([], batch_size=1)


class TestBaseExperimentWithTrainer:
  """Test 1: Trainer.fit() with base Experiment + real dataloader -- no AttributeError."""

  def test_fit_with_base_experiment_no_attribute_error(self):
    experiment = Experiment(experiment_id='test-base')
    trainer = Trainer(experiment=experiment)
    mod = NoopEvalModule()
    result = trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=2)
    assert result['total_epochs'] == 2
    assert experiment.status == Status.completed

  def test_base_experiment_has_store(self):
    e = Experiment(experiment_id='t')
    assert hasattr(e, 'store')
    assert e.store is None

  def test_base_experiment_has_last_accepted_epoch(self):
    e = Experiment(experiment_id='t')
    assert hasattr(e, 'last_accepted_epoch')
    assert e.last_accepted_epoch is None

  def test_base_experiment_has_rollback(self):
    e = Experiment(experiment_id='t')
    assert hasattr(e, 'rollback')
    assert callable(e.rollback)

  def test_base_experiment_has_on_epoch_complete(self):
    e = Experiment(experiment_id='t')
    assert hasattr(e, 'on_epoch_complete')
    e.on_epoch_complete(0, {'x': 1.0})

  def test_base_experiment_has_on_validation_complete(self):
    e = Experiment(experiment_id='t')
    assert hasattr(e, 'on_validation_complete')
    e.on_validation_complete(0, {'x': 1.0}, metric_metadata={'x': True})


class TestAutoPilotExperimentWithValidation:
  """Test 2: Trainer.fit() with AutoPilotExperiment + validation -- no TypeError."""

  def test_fit_with_autopilot_experiment_and_validation(self):
    experiment = AutoPilotExperiment(experiment_id='test-auto')
    mod = _FullModule()
    trainer = Trainer(experiment=experiment)
    result = trainer.fit(
      mod,
      train_dataloaders=_make_loader(),
      val_dataloaders=_make_loader(2),
      max_epochs=2,
    )
    assert result['total_epochs'] == 2
    assert experiment.status == Status.completed

  def test_on_validation_complete_accepts_kwargs(self):
    calls = []

    class TrackedExperiment(AutoPilotExperiment):
      def on_validation_complete(self, epoch, metrics, **kwargs):
        calls.append({'epoch': epoch, 'kwargs': kwargs})

    experiment = TrackedExperiment(experiment_id='test-kwargs')
    mod = _FullModule()
    trainer = Trainer(experiment=experiment)
    trainer.fit(
      mod,
      train_dataloaders=_make_loader(),
      val_dataloaders=_make_loader(2),
      max_epochs=1,
    )
    assert len(calls) == 1
    assert 'metric_metadata' in calls[0]['kwargs']


class TestDryRunWithBaseExperiment:
  """Test 3: Trainer.fit() with dry_run=True + Experiment -- no AttributeError."""

  def test_dry_run_no_attribute_error(self):
    experiment = Experiment(experiment_id='test-dry')
    trainer = Trainer(experiment=experiment, dry_run=True)
    mod = NoopEvalModule()
    result = trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=2)
    assert result['total_epochs'] == 2
    for epoch_result in result['epochs']:
      assert epoch_result['dry_run'] is True

  def test_dry_run_store_component_reflects_trainer_store(self):
    experiment = Experiment(experiment_id='test-dry-store')
    trainer = Trainer(experiment=experiment, dry_run=True)
    mod = NoopEvalModule()
    result = trainer.fit(mod, max_epochs=1)
    assert result['epochs'][0]['components']['store'] is False


class TestPolicyFailWithNullStore:
  """Test 4: Trainer.fit() with policy that fails + experiment.store is None."""

  def test_policy_fail_rollback_noop_no_attribute_error(self):
    experiment = Experiment(experiment_id='test-policy-fail')
    trainer = Trainer(
      experiment=experiment,
      policy=_FailingPolicy(),
    )
    mod = NoopEvalModule()
    result = trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=3)
    assert experiment.status == Status.failed
    assert experiment.error == 'policy gate rejected all epochs'
    assert result['epochs'][0].get('stopped') is True
    assert result['total_epochs'] == 1

  def test_policy_fail_with_store_none_rollback_is_noop(self):
    experiment = Experiment(experiment_id='test-noop-rollback')
    assert experiment.store is None
    experiment.rollback(0)


class TestExperimentEpochAfterFit:
  """Test 5: experiment.epoch is correct after fit() completes."""

  def test_epoch_incremented_per_epoch(self):
    experiment = Experiment(experiment_id='test-epoch')
    trainer = Trainer(experiment=experiment)
    mod = NoopEvalModule()
    trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=3)
    assert experiment.epoch == 2

  def test_epoch_incremented_with_dry_run(self):
    experiment = Experiment(experiment_id='test-epoch-dry')
    trainer = Trainer(experiment=experiment, dry_run=True)
    mod = NoopEvalModule()
    trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=5)
    assert experiment.epoch == 4

  def test_epoch_correct_after_policy_stop(self):
    experiment = Experiment(experiment_id='test-epoch-stop')
    trainer = Trainer(
      experiment=experiment,
      policy=_FailingPolicy(),
    )
    mod = NoopEvalModule()
    trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=5)
    assert experiment.epoch == -1


class TestCompleteReceivesMetrics:
  """Test 6: experiment.complete() receives metrics (not None)."""

  def test_complete_gets_last_epoch_metrics(self):
    received_metrics = {}

    class TrackingExperiment(Experiment):
      def complete(self, metrics=None):
        received_metrics['value'] = metrics
        super().complete(metrics)

    experiment = TrackingExperiment(experiment_id='test-metrics')
    mod = _FullModule()
    trainer = Trainer(experiment=experiment)
    trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=2)
    assert received_metrics['value'] is not None
    assert 'count' in received_metrics['value']

  def test_complete_gets_none_with_zero_epochs(self):
    received_metrics = {}

    class TrackingExperiment(AutoPilotExperiment):
      def complete(self, metrics=None):
        received_metrics['value'] = metrics
        super().complete(metrics)

    experiment = TrackingExperiment(experiment_id='test-metrics-zero')
    mod = NoopEvalModule()
    trainer = Trainer(experiment=experiment)
    trainer.fit(mod, max_epochs=0)
    assert received_metrics['value'] is None


class TestAccumulateGradBatchesValidation:
  """Test 7: accumulate_grad_batches=0 raises ValueError."""

  def test_zero_raises_value_error(self):
    trainer = Trainer(accumulate_grad_batches=0)
    mod = NoopEvalModule()
    with pytest.raises(ValueError, match='accumulate_grad_batches must be >= 1, got 0'):
      trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=1)

  def test_negative_raises_value_error(self):
    trainer = Trainer(accumulate_grad_batches=-1)
    mod = NoopEvalModule()
    with pytest.raises(ValueError, match='accumulate_grad_batches must be >= 1, got -1'):
      trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=1)


class TestHookCallOrder:
  """Test 8: Hook call order: on_epoch_complete before on_validation_complete."""

  def test_epoch_complete_before_validation_complete(self):
    call_order = []

    class OrderTracker(Experiment):
      def on_epoch_complete(self, epoch, metrics, **kwargs):
        call_order.append('epoch_complete')

      def on_validation_complete(self, epoch, metrics, **kwargs):
        call_order.append('validation_complete')

    experiment = OrderTracker(experiment_id='test-order')
    mod = _FullModule()
    trainer = Trainer(experiment=experiment)
    trainer.fit(
      mod,
      train_dataloaders=_make_loader(),
      val_dataloaders=_make_loader(2),
      max_epochs=1,
    )
    ec_idx = call_order.index('epoch_complete')
    vc_idx = call_order.index('validation_complete')
    assert ec_idx < vc_idx


class TestFitTwiceSameExperiment:
  """Test 9: Trainer.fit() twice with same Experiment raises ExperimentError."""

  def test_fit_twice_same_experiment_raises(self):
    experiment = Experiment(experiment_id='test-twice')
    trainer = Trainer(experiment=experiment)
    mod = NoopEvalModule()
    trainer.fit(mod, max_epochs=1)
    assert experiment.status == Status.completed
    with pytest.raises(ExperimentError, match='cannot enter context'):
      trainer.fit(mod, max_epochs=1)


class TestFitTwiceSameTrainerNewExperiment:
  """Test 10: Trainer.fit() twice with same Trainer but new Experiment works."""

  def test_fit_twice_new_experiment_works(self):
    exp1 = Experiment(experiment_id='test-first')
    trainer = Trainer(experiment=exp1)
    mod = NoopEvalModule()
    trainer.fit(mod, max_epochs=1)
    assert exp1.status == Status.completed

    exp2 = Experiment(experiment_id='test-second')
    trainer._experiment = exp2
    result = trainer.fit(mod, max_epochs=1)
    assert exp2.status == Status.completed
    assert result['total_epochs'] == 1


class TestCallbackDispatchStopsOnException:
  """Test 11: Callback dispatch stops on first exception."""

  def test_first_callback_exception_stops_dispatch(self):
    second_called = []

    class Exploding(Callback):
      def on_fit_start(self, trainer, module):
        msg = 'boom'
        raise ValueError(msg)

    class Tracker(Callback):
      def on_fit_start(self, trainer, module):
        second_called.append(True)

    trainer = Trainer(callbacks=[Exploding(), Tracker()])
    mod = NoopEvalModule()
    with pytest.raises(ValueError, match='boom'):
      trainer.fit(mod, max_epochs=1)
    assert second_called == []


class TestEmptyDataloader:
  """Test 12: Trainer.fit() with empty dataloader -- no crash."""

  def test_empty_dataloader_no_crash(self):
    experiment = Experiment(experiment_id='test-empty')
    trainer = Trainer(experiment=experiment)
    mod = NoopEvalModule()
    result = trainer.fit(mod, train_dataloaders=_make_empty_loader(), max_epochs=2)
    assert result['total_epochs'] == 2
    assert experiment.status == Status.completed

  def test_empty_dataloader_metrics_empty(self):
    experiment = Experiment(experiment_id='test-empty-metrics')
    trainer = Trainer(experiment=experiment)
    mod = NoopEvalModule()
    result = trainer.fit(mod, train_dataloaders=_make_empty_loader(), max_epochs=1)
    assert result['epochs'][0]['metrics'] == {}


class TestAdvanceEpochNotDoubleCallsOnEpochComplete:
  """Verify advance_epoch on AutoPilotExperiment does NOT call on_epoch_complete."""

  def test_advance_epoch_no_on_epoch_complete(self):
    calls = []

    class Tracker(AutoPilotExperiment):
      def on_epoch_complete(self, epoch, metrics, **kwargs):
        calls.append(epoch)

    exp = Tracker(experiment_id='test-no-double')
    exp.start()
    exp.advance_epoch({'x': 1.0})
    assert calls == []

  def test_loop_calls_on_epoch_complete_once_per_epoch(self):
    calls = []

    class Tracker(AutoPilotExperiment):
      def on_epoch_complete(self, epoch, metrics, **kwargs):
        calls.append(epoch)

    exp = Tracker(experiment_id='test-single-call')
    mod = _FullModule()
    trainer = Trainer(experiment=exp)
    trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=3)
    assert len(calls) == 3


class TestTrainerStoreProperty:
  """Verify trainer.store is a public property."""

  def test_store_none_by_default(self):
    trainer = Trainer()
    assert trainer.store is None

  def test_store_set_via_constructor(self):
    class FakeStore:
      pass

    fake = FakeStore()
    trainer = Trainer(store=cast(Any, fake))
    assert trainer.store is fake


class TestExperimentLastAcceptedEpochTracking:
  """Verify last_accepted_epoch is set by EpochLoop on passing policy."""

  def test_last_accepted_epoch_set_after_pass(self):
    experiment = Experiment(experiment_id='test-best')
    trainer = Trainer(experiment=experiment, policy=_PassingPolicy())
    mod = NoopEvalModule()
    trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=3)
    assert experiment.last_accepted_epoch == 2

  def test_last_accepted_epoch_not_set_on_fail(self):
    experiment = Experiment(experiment_id='test-best-fail')
    trainer = Trainer(experiment=experiment, policy=_FailingPolicy())
    mod = NoopEvalModule()
    trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=3)
    assert experiment.last_accepted_epoch is None


class _SequencePolicy(Policy):
  """Policy that returns results from a pre-defined sequence."""

  def __init__(self, sequence: list[GateResult]):
    super().__init__()
    self._sequence = list(sequence)
    self._i = 0

  def forward(self, result: Result) -> GateResult:
    out = self._sequence[self._i]
    self._i += 1
    return out


class TestFitWrapsExperimentContextManager:
  """Verify Trainer.fit() uses experiment context manager internally."""

  def test_fit_success_leaves_experiment_completed(self):
    experiment = Experiment(experiment_id='test-cm-success')
    trainer = Trainer(experiment=experiment)
    mod = NoopEvalModule()
    trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=2)
    assert experiment.status == Status.completed

  def test_fit_failure_leaves_experiment_failed(self):
    experiment = Experiment(experiment_id='test-cm-fail')

    class _FailingModule(AutoPilotModule):
      def forward(self, *args, **kwargs):
        return EvalDatum(success=True)

      def training_step(self, batch, batch_idx):
        msg = 'training exploded'
        raise RuntimeError(msg)

      def configure_optimizers(self):
        return None

    trainer = Trainer(experiment=experiment)
    mod = _FailingModule()
    with pytest.raises(RuntimeError, match='training exploded'):
      trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=1)
    assert experiment.status == Status.failed
    assert experiment.error is not None

  def test_fit_without_experiment_still_works(self):
    trainer = Trainer()
    mod = NoopEvalModule()
    result = trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=1)
    assert result['total_epochs'] == 1

  def test_fit_cm_does_not_suppress_exceptions(self):
    experiment = Experiment(experiment_id='test-cm-no-suppress')

    class _BoomCallback(Callback):
      def on_fit_start(self, trainer, module):
        msg = 'callback boom'
        raise ValueError(msg)

    trainer = Trainer(experiment=experiment, callbacks=[_BoomCallback()])
    mod = NoopEvalModule()
    with pytest.raises(ValueError, match='callback boom'):
      trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=1)
    assert experiment.status == Status.failed

  def test_fit_autopilot_experiment_cm_triggers_hooks(self):
    hooks: list[str] = []

    class _HookedExp(AutoPilotExperiment):
      def on_start(self) -> None:
        hooks.append('on_start')

      def on_complete(self) -> None:
        hooks.append('on_complete')

    experiment = _HookedExp(experiment_id='test-cm-hooks')
    trainer = Trainer(experiment=experiment)
    mod = NoopEvalModule()
    trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=1)
    assert 'on_start' in hooks
    assert 'on_complete' in hooks
    assert experiment.status == Status.completed


class TestRollbackEdgeCases:
  """Verify rollback receives experiment.last_accepted_epoch under edge cases."""

  def test_rollback_with_none_when_fail_before_any_accept(self):
    rollback_args: list[int | None] = []

    class SpyExperiment(Experiment):
      def rollback(self, epoch):
        rollback_args.append(epoch)

    experiment = SpyExperiment(experiment_id='test-rollback-none')
    trainer = Trainer(experiment=experiment, policy=_FailingPolicy())
    mod = NoopEvalModule()
    trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=3)
    assert len(rollback_args) == 1
    assert rollback_args[0] is None

  def test_rollback_count_is_one_on_first_fail(self):
    rollback_args: list[int | None] = []

    class SpyExperiment(Experiment):
      def rollback(self, epoch):
        rollback_args.append(epoch)

    experiment = SpyExperiment(experiment_id='test-rollback-count')
    trainer = Trainer(experiment=experiment, policy=_FailingPolicy())
    mod = NoopEvalModule()
    trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=5)
    assert len(rollback_args) == 1
    assert rollback_args[0] is None

  def test_rollback_after_prior_accepts_uses_last_accepted(self):
    rollback_args: list[int | None] = []

    class SpyExperiment(Experiment):
      def rollback(self, epoch):
        rollback_args.append(epoch)

    experiment = SpyExperiment(experiment_id='test-rollback-after-accept')
    policy = _SequencePolicy([GateResult.PASSED, GateResult.PASSED, GateResult.FAIL])
    trainer = Trainer(experiment=experiment, policy=policy)
    mod = NoopEvalModule()
    trainer.fit(mod, train_dataloaders=_make_loader(), max_epochs=5)
    assert len(rollback_args) == 1
    assert rollback_args[0] == 1
    assert experiment.last_accepted_epoch == 1
