"""Tests for Loop class hierarchy."""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.gradient import NumericGradient
from autopilot.core.graph import get_current_graph, is_grad_enabled, no_grad
from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.loops.loop import Loop, LoopConfig
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum, EvalDatum
from tests.doubles import NoopEvalModule, StepCountingOptimizer
from typing import Any, cast
import pytest


class _TrainerShim:
  """Minimal trainer surface for EpochLoop.run without constructing Trainer."""

  def __init__(self) -> None:
    self.module: AutoPilotModule = NoopEvalModule()
    self.policy = None
    self.store = None
    self.logger = None
    self.scheduler = None
    self.datamodule = None
    self.current_epoch: int = 0
    self._optimizer: Any = None
    self._cached_grad_summaries: list[dict[str, str]] = []

  def dispatch_callbacks(self, *args: object, **kwargs: object) -> list[object]:
    return []

  def on_epoch_start(self, epoch: int) -> list[object]:
    return []

  def on_epoch_end(self, epoch: int, result: dict | None = None) -> list[object]:
    return []

  def should_stop_at(self, hook_method: object, **kwargs: object) -> bool:
    return False

  def emit_context(self, reason: str, **kwargs: object) -> None:
    """No-op context emission for shim."""

  def capture_gradient_summaries(self) -> None:
    """No-op gradient capture for shim."""

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
    """Eval phase for shim: discover metrics from module, run step_fn per batch."""
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
      for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
          break
        with no_grad():
          step_output = step_fn(batch, batch_idx)
        for m in all_metrics.values():
          m.update(step_output)
      result: dict[str, float] = {}
      for m in all_metrics.values():
        result.update(m.compute())
      return result
    finally:
      module.train()


class _GraphModule(AutoPilotModule):
  """Module that uses self(batch) to create graph nodes."""

  def __init__(self):
    super().__init__()
    self.p = Parameter(requires_grad=True)

  def forward(self, batch):
    return Datum()

  def training_step(self, batch, batch_idx):
    return self(batch)

  def validation_step(self, batch, batch_idx):
    return self(batch)

  def configure_optimizers(self):
    return None


class _ForwardBypassModule(AutoPilotModule):
  """Module that calls self.forward() directly, bypassing graph recording."""

  def __init__(self):
    super().__init__()
    self.p = Parameter(requires_grad=True)

  def forward(self, batch):
    return Datum()

  def training_step(self, batch, batch_idx):
    return self.forward(batch)

  def configure_optimizers(self):
    return None


class _GraphLoss(Loss):
  """Loss that uses the graph-based backward path."""

  def compute_seed_gradient(self):
    return NumericGradient(value=1.0)


class _DirectGradLoss(Loss):
  """Loss that directly sets param.grad (legacy-style, for tests that don't
  need graph wiring)."""

  def __init__(self, params):
    super().__init__(params)

  def forward(self, data, targets=None):
    super().forward(data, targets)

  def backward(self):
    for p in self._loss_parameters:
      if p.requires_grad:
        p.grad = NumericGradient(value=1.0)

  def compute_seed_gradient(self):
    return NumericGradient(value=1.0)

  def reset(self):
    self._accumulated = []
    self._last_data = None


class _CallbackTrainerShim:
  """Trainer shim that dispatches callbacks like the real Trainer."""

  def __init__(self, module, callbacks=None):
    self.module = module
    self._module = module
    self._callbacks = callbacks or []
    self.policy = None
    self.store = None
    self.logger = None
    self.scheduler = None
    self.current_epoch: int = 0
    self._optimizer: object | None = None
    self._cached_grad_summaries: list[dict[str, str]] = []

  def dispatch_callbacks(self, hook_name, **kwargs):
    results = []
    for cb in self._callbacks:
      method = getattr(cb, hook_name, None)
      if method and callable(method):
        result = method(trainer=self, module=self._module, **kwargs)
        if result is not None:
          results.append(result)
    return results

  def on_epoch_start(self, epoch):
    return []

  def on_epoch_end(self, epoch, result=None):
    return []

  def should_stop_at(self, hook_method, **kwargs):
    return False

  def emit_context(self, reason, **kwargs):
    """No-op context emission for shim."""

  def capture_gradient_summaries(self):
    """No-op gradient capture for shim."""


class TestLoopABC:
  def test_cannot_instantiate(self) -> None:
    with pytest.raises(TypeError):
      Loop()


class TestEpochLoop:
  def _trainer(self) -> _TrainerShim:
    return _TrainerShim()

  def test_runs_epochs(self) -> None:
    trainer = self._trainer()
    loop = EpochLoop()
    config = LoopConfig(max_epochs=3)
    result = loop.run(trainer, config)
    assert result['total_epochs'] == 3
    assert len(result['epochs']) == 3

  def test_early_stop(self) -> None:
    trainer = self._trainer()
    call_count = 0

    def stop_at_2(_hook: object, **kwargs: object) -> bool:
      nonlocal call_count
      call_count += 1
      return call_count >= 2

    trainer.should_stop_at = cast(Any, stop_at_2)
    loop = EpochLoop()
    config = LoopConfig(max_epochs=5)
    result = loop.run(trainer, config)
    assert result['total_epochs'] < 5

  def test_repr(self) -> None:
    loop = EpochLoop()
    assert 'EpochLoop' in repr(loop)

  def test_overridable_run_epoch(self) -> None:
    class Custom(EpochLoop):
      def _run_epoch(self, trainer, epoch, config):
        return {'epoch': epoch, 'custom': True}

    trainer = self._trainer()
    loop = Custom()
    config = LoopConfig(max_epochs=1)
    result = loop.run(trainer, config)
    assert result['epochs'][0]['custom'] is True

  def test_epoch_start_end_hooks_called(self) -> None:
    calls: list[str] = []

    class T(_TrainerShim):
      def on_epoch_end(self, epoch: int, result: dict | None = None) -> list[object]:
        calls.append('end')
        return []

    trainer = T()
    loop = EpochLoop()
    config = LoopConfig(max_epochs=2)
    loop.run(trainer, config)
    assert len(calls) == 2

  def test_zero_epochs(self) -> None:
    trainer = self._trainer()
    loop = EpochLoop()
    config = LoopConfig(max_epochs=0)
    result = loop.run(trainer, config)
    assert result['total_epochs'] == 0
    assert result['epochs'] == []

  def test_result_structure(self) -> None:
    trainer = self._trainer()
    loop = EpochLoop()
    config = LoopConfig(max_epochs=1)
    result = loop.run(trainer, config)
    assert 'epochs' in result
    assert 'total_epochs' in result
    assert result['epochs'][0]['epoch'] == 0


class TestTrainingLoopIntegration:
  """Tests for graph-based training loop integration (plan 09)."""

  def test_training_step_creates_grad_fn(self) -> None:
    """After module.training_step(batch, batch_idx) with graph recording, the returned
    datum has a grad_fn attribute set by ModuleCallOperator."""
    module = _GraphModule()
    batch = Datum()
    data = module.training_step(batch, 0)
    assert isinstance(data, Datum)
    assert data.grad_fn is not None

  def test_loss_forward_then_backward_sets_param_grad(self) -> None:
    """training_step -> loss(data, batch) -> loss.backward() populates
    at least one trainable param.grad."""
    module = _GraphModule()
    loss_fn = _GraphLoss()
    batch = Datum()
    data = module.training_step(batch, 0)
    loss_fn(data, batch)
    loss_fn.backward()
    assert module.p.grad is not None

  def test_optimizer_step_after_backward(self) -> None:
    """After backward, param.grad is not None before optimizer.step(),
    then after optimizer.zero_grad() param.grad is None and
    param.grad_accumulator is None."""
    module = _GraphModule()
    loss_fn = _GraphLoss()
    opt = StepCountingOptimizer(list(module.parameters()))
    batch = Datum()
    data = module.training_step(batch, 0)
    loss_fn(data, batch)
    loss_fn.backward()
    assert module.p.grad is not None
    opt.step()
    assert opt.step_count == 1
    opt.zero_grad()
    assert module.p.grad is None
    assert module.p.grad_accumulator is None

  def test_zero_grad_clears_after_step(self) -> None:
    """After optimizer.zero_grad(), param.grad is cleared AND
    param.grad_accumulator is cleared."""
    module = _GraphModule()
    loss_fn = _GraphLoss()
    opt = StepCountingOptimizer(list(module.parameters()))
    batch = Datum()
    data = module.training_step(batch, 0)
    loss_fn(data, batch)
    loss_fn.backward()
    assert module.p.grad is not None
    assert module.p.grad_accumulator is not None
    opt.step()
    opt.zero_grad()
    assert module.p.grad is None
    assert module.p.grad_accumulator is None

  def test_loss_reset_clears_accumulated(self) -> None:
    """loss_fn.reset() clears internal accumulation state."""
    module = _GraphModule()
    loss_fn = _GraphLoss()
    batch = Datum()
    data = module.training_step(batch, 0)
    loss_fn(data, batch)
    assert len(loss_fn._accumulated) == 1
    assert loss_fn._last_data is not None
    loss_fn.reset()
    assert len(loss_fn._accumulated) == 0
    assert loss_fn._last_data is None

  def test_full_training_step_cycle(self) -> None:
    """Two full training steps: forward -> loss -> backward -> step ->
    zero_grad -> repeat, with no stale graph or grad state."""
    module = _GraphModule()
    loss_fn = _GraphLoss()
    opt = StepCountingOptimizer(list(module.parameters()))

    for _i in range(2):
      batch = Datum()
      data = module.training_step(batch, 0)
      assert data.grad_fn is not None
      loss_fn(data, batch)
      loss_fn.backward()
      assert module.p.grad is not None
      opt.step()
      opt.zero_grad()
      assert module.p.grad is None
      loss_fn.reset()

    assert opt.step_count == 2


class TestValidationNoGrad:
  """Tests for validation under no_grad (plan 09)."""

  def test_validation_under_no_grad(self) -> None:
    """During validation, the graph recording is disabled via no_grad()."""
    grad_enabled_during_val = []

    class _RecordingModule(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.p = Parameter(requires_grad=True)

      def forward(self, batch):
        grad_enabled_during_val.append(is_grad_enabled())
        return Datum()

      def training_step(self, batch, batch_idx):
        return self(batch)

      def validation_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return None

    module = _RecordingModule()
    trainer = _TrainerShim()
    trainer.module = cast(Any, module)
    loop = EpochLoop()
    config = LoopConfig(
      max_epochs=1,
      val_loader=[Datum(), Datum()],
    )
    loop.run(trainer, config)
    assert len(grad_enabled_during_val) >= 2
    assert all(not v for v in grad_enabled_during_val)

  def test_validation_no_grad_fn(self) -> None:
    """Outputs from validation_step have no grad_fn on attached data."""
    val_outputs = []

    class _CaptureModule(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.p = Parameter(requires_grad=True)

      def forward(self, batch):
        return Datum()

      def training_step(self, batch, batch_idx):
        return self(batch)

      def validation_step(self, batch, batch_idx):
        out = self(batch)
        val_outputs.append(out)
        return out

      def configure_optimizers(self):
        return None

    module = _CaptureModule()
    trainer = _TrainerShim()
    trainer.module = cast(Any, module)
    loop = EpochLoop()
    config = LoopConfig(
      max_epochs=1,
      val_loader=[Datum()],
    )
    loop.run(trainer, config)
    assert len(val_outputs) == 1
    assert val_outputs[0].grad_fn is None

  def test_training_then_validation_no_leak(self) -> None:
    """Training creates graph state; after loss.backward() resets the graph,
    validation does not add new nodes (runs under no_grad)."""
    module = _GraphModule()

    class _TrainerWithValShim(_TrainerShim):
      pass

    trainer = _TrainerWithValShim()
    trainer.module = cast(Any, module)
    loop = EpochLoop()

    train_data = [Datum()]
    loss_fn = _GraphLoss()

    config = LoopConfig(
      max_epochs=1,
      train_loader=train_data,
      val_loader=[Datum(), Datum(), Datum()],
      loss=loss_fn,
    )
    loop.run(trainer, config)
    graph = get_current_graph()
    assert len(graph) == 0


class TestTrainingLoopCallbacks:
  """Tests for callback integration in the training loop (plan 09)."""

  def test_on_before_backward_receives_loss_fn(self) -> None:
    """A test callback records loss_fn from on_before_backward."""
    received_loss_fns = []

    class _TrackCb(Callback):
      def on_before_backward(self, trainer, module, loss_fn=None):
        received_loss_fns.append(loss_fn)

    module = _GraphModule()
    loss_fn = _DirectGradLoss(list(module.parameters()))
    module.loss = loss_fn
    cb = _TrackCb()
    trainer = _CallbackTrainerShim(module, callbacks=[cb])
    loop = EpochLoop()
    config = LoopConfig(
      max_epochs=1,
      train_loader=[Datum()],
      loss=loss_fn,
    )
    loop.run(trainer, config)
    assert len(received_loss_fns) == 1
    assert received_loss_fns[0] is loss_fn

  def test_on_after_backward_no_loss_arg(self) -> None:
    """After backward callback is invoked without loss_fn in kwargs."""
    received_kwargs = []

    class _TrackCb(Callback):
      def on_before_backward(self, trainer, module, loss_fn=None):
        pass

      def on_after_backward(self, trainer, module, **kwargs):
        received_kwargs.append(kwargs)

    module = _GraphModule()
    loss_fn = _DirectGradLoss(list(module.parameters()))
    module.loss = loss_fn
    cb = _TrackCb()
    trainer = _CallbackTrainerShim(module, callbacks=[cb])
    loop = EpochLoop()
    config = LoopConfig(
      max_epochs=1,
      train_loader=[Datum()],
      loss=loss_fn,
    )
    loop.run(trainer, config)
    assert len(received_kwargs) == 1
    assert 'loss_fn' not in received_kwargs[0]


class TestEpochLoopEdgeCases:
  """Tests for EpochLoop edge cases: accumulation and empty loaders."""

  def test_optimizer_steps_once_accumulate_two_with_two_batches(self) -> None:
    """two micro-batches, accumulate_grad_batches=2 -> single step at epoch end."""
    module = _GraphModule()
    loss_fn = _GraphLoss()
    opt = StepCountingOptimizer(list(module.parameters()))
    trainer = _CallbackTrainerShim(module)
    loop = EpochLoop()
    batches = [Datum(), Datum()]
    config = LoopConfig(
      max_epochs=1,
      train_loader=batches,
      loss=loss_fn,
      optimizer=opt,
      accumulate_grad_batches=2,
    )
    loop.run(trainer, config)
    assert opt.step_count == 1

  def test_empty_train_loader_produces_empty_metrics(self) -> None:
    module = _GraphModule()
    loss_fn = _GraphLoss()
    trainer = _CallbackTrainerShim(module)
    loop = EpochLoop()
    config = LoopConfig(
      max_epochs=1,
      train_loader=[],
      loss=loss_fn,
      optimizer=StepCountingOptimizer(list(module.parameters())),
    )
    result = loop.run(trainer, config)
    assert result['total_epochs'] == 1
    assert result['epochs'][0]['metrics'] == {}


class TestTrainingLoopErrorPath:
  """Tests for error paths in the training loop (plan 09)."""

  def test_training_step_forward_bypass_raises(self) -> None:
    """If training_step calls self.forward(batch) instead of self(batch),
    then loss.backward() raises RuntimeError (no grad_fn on root)."""
    module = _ForwardBypassModule()
    loss_fn = _GraphLoss()
    batch = Datum()
    data = module.training_step(batch, 0)
    assert data.grad_fn is None
    loss_fn(data, batch)
    with pytest.raises(RuntimeError, match='cannot backward: data has no grad_fn'):
      loss_fn.backward()


class TestEpochLoopLoggerWiring:
  """Tests for logger.log_metrics calls in EpochLoop."""

  def test_epoch_loop_logs_train_metrics_when_logger_set(self) -> None:
    from autopilot.core.metric import Metric
    from unittest.mock import MagicMock, call

    class FixedMetric(Metric):
      def __init__(self):
        super().__init__()
        self._val = 0.25

      def update(self, datum):
        pass

      def compute(self):
        return {'acc': self._val}

    trainer = _TrainerShim()
    mock_logger = MagicMock(spec_set=['log_metrics'])
    trainer.logger = mock_logger

    loop = EpochLoop()
    config = LoopConfig(
      max_epochs=1,
      train_loader=[Datum()],
      metrics={'acc': FixedMetric()},
    )
    loop.run(trainer, config)
    assert mock_logger.log_metrics.call_args_list == [call({'acc': 0.25}, step=0)]

  def test_epoch_loop_logs_val_metrics_when_logger_and_val_loader(self) -> None:
    from autopilot.core.metric import Metric as MetricBase
    from unittest.mock import MagicMock, call

    class TrainMetric(MetricBase):
      def __init__(self):
        super().__init__()
        self._call_count = 0

      def update(self, datum):
        self._call_count += 1

      def compute(self):
        if self._call_count <= 1:
          return {'acc': 0.25}
        return {'acc': 0.75}

    shared_metric = TrainMetric()

    class _ValModule(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.acc = shared_metric

      def forward(self, *args, **kwargs) -> EvalDatum:
        return EvalDatum(success=True)

      def training_step(self, batch, batch_idx):
        return EvalDatum(success=True)

      def validation_step(self, batch, batch_idx):
        return EvalDatum(success=True)

      def configure_optimizers(self):
        return None

    trainer = _TrainerShim()
    val_module = _ValModule()
    trainer.module = val_module
    mock_logger = MagicMock(spec_set=['log_metrics'])
    trainer.logger = mock_logger

    loop = EpochLoop()
    config = LoopConfig(
      max_epochs=1,
      train_loader=[Datum()],
      val_loader=[Datum()],
      metrics={'acc': shared_metric},
    )
    loop.run(trainer, config)
    assert mock_logger.log_metrics.call_args_list == [
      call({'acc': 0.25}, step=0),
      call({'acc': 0.75}, step=0),
    ]

  def test_epoch_loop_no_log_metrics_when_logger_none(self) -> None:
    from autopilot.core.metric import Metric

    class FixedMetric(Metric):
      def __init__(self):
        super().__init__()

      def update(self, datum):
        pass

      def compute(self):
        return {'acc': 0.25}

    trainer = _TrainerShim()
    assert trainer.logger is None

    loop = EpochLoop()
    config = LoopConfig(
      max_epochs=1,
      train_loader=[Datum()],
      metrics={'acc': FixedMetric()},
    )
    result = loop.run(trainer, config)
    assert result['total_epochs'] == 1
    assert result['epochs'][0]['metrics'] == {'acc': 0.25}
