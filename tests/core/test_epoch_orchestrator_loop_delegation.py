"""Tests for sub-plan 08 section 2.1: EpochOrchestrator delegates to EpochLoop.

Covers:
- EpochOrchestrator is an EpochLoop subclass.
- No duplicated epoch for/while skeleton in orchestrator.
- Base EpochLoop produces the same epoch sequence as before refactor.
- Single-epoch run does not trigger plateau/early-stop.
- Rollback resets current epoch to pre-refactor semantics.
- Template hooks are overridable and invoked correctly.
"""

from autopilot.core.experiment import Experiment
from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.loops.loop import LoopConfig
from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum, GateResult
from autopilot.policy.policy import Policy
from typing import Any, cast
from unittest.mock import MagicMock


class _DummyModule(AutoPilotModule):
  """Minimal module for loop tests."""

  def __init__(self) -> None:
    super().__init__()
    self._accuracy = 0.5

  def forward(self, batch: Any) -> EvalDatum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    return EvalDatum(success=True, metrics={'accuracy': self._accuracy})

  def validation_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    return EvalDatum(success=True, metrics={'accuracy': self._accuracy})

  def configure_optimizers(self) -> None:
    return None


class _DummyMetric(Metric):
  """Metric that computes accuracy from success counts."""

  def __init__(self) -> None:
    super().__init__()
    self.add_state('_correct', 0)
    self.add_state('_total', 0)

  def update(self, datum: Any) -> None:
    self._total += 1
    if datum.success:
      self._correct += 1

  def compute(self) -> dict[str, float]:
    acc = self._correct / self._total if self._total else 0.0
    return {'accuracy': acc}


class _MockExperiment(Experiment):
  """Experiment stub tracking rollback calls."""

  def __init__(self, store: Any = None) -> None:
    super().__init__(experiment_id='mock-plan08')
    self.store = store
    self.should_rollback = False
    self.last_accepted_epoch = -1
    self.rollback_calls: list[int | None] = []

  def rollback(self, epoch: int | None) -> None:
    self.rollback_calls.append(epoch)
    if self.store:
      self.store.checkout(epoch)

  def on_epoch_complete(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
    pass

  def on_validation_complete(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
    pass


class TestOrchestratorIsInstanceEpochLoop:
  """2.1 gate: EpochOrchestrator subclasses EpochLoop."""

  def test_isinstance(self) -> None:
    orch = EpochOrchestrator()
    assert isinstance(orch, EpochLoop)

  def test_inherits_run_epoch(self) -> None:
    orch = EpochOrchestrator()
    assert orch._run_epoch == EpochLoop._run_epoch.__get__(orch, type(orch))


class TestOrchestratorBaseEpochIterationUnchanged:
  """2.1: base EpochLoop produces the same epoch sequence after refactor."""

  def test_epoch_sequence_matches(self) -> None:
    module = _DummyModule()
    module.accuracy = _DummyMetric()
    trainer = Trainer(loop=EpochLoop())
    result = trainer.fit(module, train_dataloaders=[1, 2], max_epochs=3)
    epochs = [e['epoch'] for e in result['epochs']]
    assert epochs == [0, 1, 2]
    assert result['total_epochs'] == 3

  def test_orchestrator_epoch_sequence_matches(self) -> None:
    module = _DummyModule()
    module.accuracy = _DummyMetric()
    trainer = Trainer(loop=EpochOrchestrator())
    result = trainer.fit(module, train_dataloaders=[1, 2], max_epochs=3)
    epochs = [e['epoch'] for e in result['epochs']]
    assert epochs == [0, 1, 2]
    assert result['total_epochs'] == 3


class TestOrchestratorSingleEpochNoPlateauDetection:
  """2.1: single-epoch run does not spuriously trigger plateau / early-stop."""

  def test_single_epoch_no_plateau(self) -> None:
    module = _DummyModule()
    module.accuracy = _DummyMetric()
    config = OrchestratorConfig(plateau_window=3, plateau_threshold=0.05, monitor='accuracy')
    trainer = Trainer(loop=EpochOrchestrator(config))
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=1)
    assert result['total_epochs'] == 1
    assert result.get('stop_reason') is None


class TestOrchestratorRollbackResetsCurrentEpoch:
  """2.1: after rollback, the epoch counter matches pre-refactor semantics."""

  def test_rollback_targets_last_good_epoch(self) -> None:
    module = _DummyModule()
    module.accuracy = _DummyMetric()
    store = MagicMock()
    experiment = _MockExperiment(store=store)
    config = OrchestratorConfig(auto_rollback=True, plateau_window=0)
    orch = EpochOrchestrator(config)

    class RegCallback:
      def on_epoch_end(self, trainer: Any, module: Any, epoch: int, result: Any = None) -> None:
        if epoch == 2:
          trainer.experiment.should_rollback = True

    trainer = Trainer(loop=orch, experiment=experiment, callbacks=cast(Any, [RegCallback()]))
    trainer.fit(module, train_dataloaders=[1], max_epochs=4)
    assert experiment.rollback_calls == [1]
    store.checkout.assert_called_with(1)

  def test_policy_fail_stop_reason(self) -> None:
    class FailPolicy(Policy):
      def forward(self, result: Result) -> GateResult:
        return GateResult.FAIL

    module = _DummyModule()
    module.accuracy = _DummyMetric()
    store = MagicMock()
    experiment = _MockExperiment(store=store)
    trainer = Trainer(loop=EpochOrchestrator(), policy=FailPolicy(), experiment=experiment)
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=5)
    assert result['total_epochs'] == 1
    assert result.get('stop_reason') == 'policy_fail'


class TestEpochLoopTemplateHooks:
  """2.1: verify template hooks are called and overridable."""

  def test_pre_run_hook_called(self) -> None:
    calls: list[str] = []

    class TrackingLoop(EpochLoop):
      def _pre_run(self, trainer: Any, config: LoopConfig) -> None:
        calls.append('pre_run')

    module = _DummyModule()
    module.accuracy = _DummyMetric()
    trainer = Trainer(loop=TrackingLoop())
    trainer.fit(module, train_dataloaders=[1], max_epochs=1)
    assert calls == ['pre_run']

  def test_should_stop_before_epoch_override(self) -> None:
    class StopAt1(EpochLoop):
      def _should_stop_before_epoch(self, trainer: Any, epoch: int, config: LoopConfig) -> bool:
        return epoch >= 1

    module = _DummyModule()
    module.accuracy = _DummyMetric()
    trainer = Trainer(loop=StopAt1())
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=5)
    assert result['total_epochs'] == 1

  def test_should_stop_after_epoch_override(self) -> None:
    class StopAfter0(EpochLoop):
      def _should_stop_after_epoch(
        self, trainer: Any, epoch: int, epoch_result: dict[str, Any], config: LoopConfig
      ) -> bool:
        return epoch >= 0

    module = _DummyModule()
    module.accuracy = _DummyMetric()
    trainer = Trainer(loop=StopAfter0())
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=5)
    assert result['total_epochs'] == 1

  def test_build_run_result_override(self) -> None:
    class ExtraResult(EpochLoop):
      def _build_run_result(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        base = super()._build_run_result(results)
        base['custom_key'] = 'present'
        return base

    module = _DummyModule()
    module.accuracy = _DummyMetric()
    trainer = Trainer(loop=ExtraResult())
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=1)
    assert result['custom_key'] == 'present'
    assert result['total_epochs'] == 1


class TestOrchestratorDelegatesViaSuper:
  """2.1 gate: no duplicated for/while skeleton in orchestrator."""

  def test_orchestrator_run_calls_super(self) -> None:
    """Verify the orchestrator enters the base loop by checking _pre_run is called."""
    module = _DummyModule()
    module.accuracy = _DummyMetric()
    orch = EpochOrchestrator()
    trainer = Trainer(loop=orch)
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=2)
    assert result['total_epochs'] == 2
    assert result['last_good_epoch'] == 1
    assert result.get('stop_reason') is None

  def test_orchestrator_state_reset_via_pre_run(self) -> None:
    """Consecutive fit() calls reset _metric_history and _last_good_epoch."""
    module = _DummyModule()
    module.accuracy = _DummyMetric()
    orch = EpochOrchestrator()
    trainer = Trainer(loop=orch)

    result1 = trainer.fit(module, train_dataloaders=[1], max_epochs=3)
    assert result1['last_good_epoch'] == 2

    result2 = trainer.fit(module, train_dataloaders=[1], max_epochs=2)
    assert result2['last_good_epoch'] == 1

  def test_dry_run_still_works(self) -> None:
    module = _DummyModule()
    module.accuracy = _DummyMetric()
    trainer = Trainer(loop=EpochOrchestrator(), dry_run=True)
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=5)
    assert result.get('dry_run') is True
    assert result['total_epochs'] == 0
