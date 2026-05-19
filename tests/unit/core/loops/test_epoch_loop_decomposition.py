"""Tests for EpochLoop decomposed helpers."""

from autopilot.core.graph import no_grad
from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.loops.loop import LoopConfig
from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.types import Datum, EvalDatum, GateResult
from tests.doubles import NoopEvalModule
from typing import Any, cast
from unittest.mock import Mock, patch


class _TrainerShim:
  def __init__(self, module=None) -> None:
    self.module = module or NoopEvalModule()
    self.policy = None
    self.store = None
    self.logger = None
    self.scheduler = None
    self.datamodule = None
    self.current_epoch: int = 0
    self._dispatched: list[str] = []
    self._optimizer: object | None = None
    self._cached_grad_summaries: list[dict[str, str]] = []

  def dispatch_callbacks(self, hook_name: str, **kwargs: object) -> list[object]:
    self._dispatched.append(hook_name)
    return []

  def emit_context(
    self,
    reason: str,
    *,
    source: str | None = None,
    metadata: dict | None = None,
  ) -> None:
    """No-op context emission for loop tests."""

  def capture_gradient_summaries(self) -> None:
    """No-op gradient capture for loop tests."""

  def on_epoch_start(self, epoch: int) -> list[object]:
    return []

  def on_epoch_end(self, epoch: int, result: object = None) -> list[object]:
    return []

  def should_stop_at(self, hook_method: object, **kwargs: object) -> bool:
    return False

  def run_eval_phase(
    self,
    module: Any,
    dataloader: Any,
    *,
    step_method: str = 'validation_step',
    hook_prefix: str = 'validation',
    max_batches: int | None = None,
    epoch_arg: int = 0,
  ) -> dict[str, float]:
    """Eval phase for shim: mirrors real Trainer.run_eval_phase dispatch."""
    step_fn = getattr(module, step_method)
    all_metrics = {
      name: m
      for name, m in module.named_modules()
      if isinstance(m, Metric) and not isinstance(m, MetricCollection)
    }
    for m in all_metrics.values():
      m.reset()
    module.eval()
    try:
      self.dispatch_callbacks(f'on_{hook_prefix}_epoch_start', epoch=epoch_arg)
      for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
          break
        self.dispatch_callbacks(
          f'on_{hook_prefix}_batch_start',
          batch=batch,
          batch_idx=batch_idx,
        )
        with no_grad():
          step_output = step_fn(batch, batch_idx)
        for m in all_metrics.values():
          m.update(step_output)
        self.dispatch_callbacks(
          f'on_{hook_prefix}_batch_end',
          batch=batch,
          batch_idx=batch_idx,
        )
      result: dict[str, float] = {}
      for m in all_metrics.values():
        result.update(m.compute())
      self.dispatch_callbacks(f'on_{hook_prefix}_epoch_end', epoch=epoch_arg)
      return result
    finally:
      module.train()


class TestDryRunEpoch:
  def test_returns_dry_run_dict(self) -> None:
    loop = EpochLoop()
    trainer = _TrainerShim()
    config = LoopConfig(max_epochs=5, dry_run=True)
    result = loop._dry_run_epoch(trainer, 2, config)
    assert result['dry_run'] is True
    assert result['epoch'] == 2
    assert result['planned_epochs'] == 5

  def test_advances_experiment_epoch(self) -> None:
    loop = EpochLoop()
    trainer = _TrainerShim()
    experiment = Mock()
    config = LoopConfig(max_epochs=3, dry_run=True, experiment=experiment)
    loop._dry_run_epoch(trainer, 0, config)
    experiment.advance_epoch.assert_called_once()

  def test_noop_when_no_experiment(self) -> None:
    loop = EpochLoop()
    trainer = _TrainerShim()
    config = LoopConfig(max_epochs=3, dry_run=True)
    result = loop._dry_run_epoch(trainer, 0, config)
    assert result['dry_run'] is True

  def test_components_reflect_config(self) -> None:
    loop = EpochLoop()
    trainer = _TrainerShim()
    config = LoopConfig(
      max_epochs=2,
      dry_run=True,
      loss='fake_loss',
      optimizer='fake_opt',
      train_loader=[1],
      val_loader=[2],
      metrics={'m': Mock()},
    )
    result = loop._dry_run_epoch(trainer, 0, config)
    assert result['components']['loss'] is True
    assert result['components']['optimizer'] is True
    assert result['components']['train_loader'] is True
    assert result['components']['val_loader'] is True
    assert result['components']['metrics'] is True

  def test_dispatches_start_and_end_callbacks(self) -> None:
    loop = EpochLoop()
    trainer = _TrainerShim()
    config = LoopConfig(max_epochs=1, dry_run=True)
    loop._dry_run_epoch(trainer, 0, config)
    assert 'on_train_epoch_start' in trainer._dispatched
    assert 'on_train_epoch_end' in trainer._dispatched


class TestRunTrainBatches:
  def test_processes_single_batch(self) -> None:
    loop = EpochLoop()
    trainer = _TrainerShim()
    module = trainer.module
    config = LoopConfig(train_loader=['batch_a'])
    with patch.object(loop, '_process_batch') as mock_pb:
      loop._run_train_batches(trainer, module, config, None, None, {}, 1)
    mock_pb.assert_called_once()
    pos, kw = mock_pb.call_args
    assert pos == (trainer, module, 0, 'batch_a')
    assert kw == {
      'is_last': True,
      'loss_fn': None,
      'optimizer': None,
      'metrics': {},
      'accumulate': 1,
    }

  def test_processes_two_batches(self) -> None:
    loop = EpochLoop()
    trainer = _TrainerShim()
    module = trainer.module
    config = LoopConfig(train_loader=['b1', 'b2'])
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def recorder(*args: Any, **kwargs: Any) -> None:
      calls.append((args, kwargs))

    with patch.object(loop, '_process_batch', side_effect=recorder):
      loop._run_train_batches(trainer, module, config, None, None, {}, 1)
    assert len(calls) == 2
    assert calls[0][1]['is_last'] is False
    assert calls[1][1]['is_last'] is True

  def test_noop_when_no_loader(self) -> None:
    loop = EpochLoop()
    trainer = _TrainerShim()
    module = trainer.module
    config = LoopConfig(train_loader=None)
    with patch.object(loop, '_process_batch') as mock_pb:
      loop._run_train_batches(trainer, module, config, None, None, {}, 1)
    mock_pb.assert_not_called()

  def test_accumulation_pattern(self) -> None:
    loop = EpochLoop()
    trainer = _TrainerShim()
    module = trainer.module
    config = LoopConfig(train_loader=['b1', 'b2', 'b3'])
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def recorder(*args: Any, **kwargs: Any) -> None:
      calls.append((args, kwargs))

    with patch.object(loop, '_process_batch', side_effect=recorder):
      loop._run_train_batches(trainer, module, config, None, None, {}, 2)
    assert len(calls) == 3
    assert calls[0][1]['is_last'] is False
    assert calls[1][1]['is_last'] is False
    assert calls[2][1]['is_last'] is True


class TestRunValidationPass:
  def test_returns_none_when_no_val_loader(self) -> None:
    loop = EpochLoop()
    trainer = _TrainerShim()
    module = trainer.module
    config = LoopConfig(val_loader=None)
    val_metrics = loop._run_validation_pass(trainer, module, 0, config, {})
    assert val_metrics is None

  def test_calls_on_validation_complete(self) -> None:
    loop = EpochLoop()
    trainer = _TrainerShim()
    module = NoopEvalModule()
    module.validation_step = cast(Any, Mock(return_value=EvalDatum(success=True)))
    trainer.module = module
    experiment = Mock()
    config = LoopConfig(
      val_loader=[Datum()],
      experiment=experiment,
      metric_metadata={'m': True},
    )
    loop._run_validation_pass(trainer, module, 0, config, {})
    experiment.on_validation_complete.assert_called_once()
    call_kwargs = experiment.on_validation_complete.call_args[1]
    assert call_kwargs['metric_metadata'] == {'m': True}

  def test_dispatches_validation_callbacks(self) -> None:
    loop = EpochLoop()
    trainer = _TrainerShim()
    module = NoopEvalModule()
    module.validation_step = cast(Any, Mock(return_value=EvalDatum(success=True)))
    trainer.module = module
    config = LoopConfig(val_loader=[Datum()])
    loop._run_validation_pass(trainer, module, 0, config, {})
    assert 'on_validation_epoch_start' in trainer._dispatched
    assert 'on_validation_epoch_end' in trainer._dispatched


class TestEpochLoopPolicyFailTriggersRollback:
  def test_rollback_on_policy_fail(self) -> None:
    loop = EpochLoop()
    module = NoopEvalModule()

    class _PolicyTrainer(_TrainerShim):
      def __init__(self):
        super().__init__(module)
        self.policy = Mock(return_value=GateResult.FAIL)

    trainer = _PolicyTrainer()
    experiment = Mock()
    experiment.last_accepted_epoch = 0
    experiment.should_rollback = False
    experiment.metrics = {}
    config = LoopConfig(
      max_epochs=3,
      train_loader=[Datum()],
      experiment=experiment,
    )
    result = loop._run_epoch(trainer, 1, config)
    experiment.rollback.assert_called_once_with(0)
    assert result['stopped'] is True

  def test_no_rollback_on_policy_pass(self) -> None:
    loop = EpochLoop()
    module = NoopEvalModule()

    class _PolicyTrainer(_TrainerShim):
      def __init__(self):
        super().__init__(module)
        self.policy = Mock(return_value=GateResult.PASSED)

    trainer = _PolicyTrainer()
    experiment = Mock()
    experiment.last_accepted_epoch = None
    experiment.should_rollback = False
    experiment.metrics = {}
    config = LoopConfig(
      max_epochs=3,
      train_loader=[Datum()],
      experiment=experiment,
    )
    result = loop._run_epoch(trainer, 0, config)
    experiment.rollback.assert_not_called()
    assert result.get('stopped') is None or result.get('stopped') is False


class TestRunEpochIntegration:
  def test_full_epoch_with_train_and_val(self) -> None:
    loop = EpochLoop()
    module = NoopEvalModule()
    module.validation_step = cast(Any, Mock(return_value=EvalDatum(success=True)))
    trainer = _TrainerShim(module)
    experiment = Mock()
    experiment.last_accepted_epoch = None
    experiment.should_rollback = False
    config = LoopConfig(
      max_epochs=1,
      train_loader=[Datum()],
      val_loader=[Datum()],
      experiment=experiment,
    )
    result = loop._run_epoch(trainer, 0, config)
    assert result['epoch'] == 0
    assert 'metrics' in result
    experiment.on_epoch_complete.assert_called_once()
    experiment.on_validation_complete.assert_called_once()
    experiment.advance_epoch.assert_called_once()
