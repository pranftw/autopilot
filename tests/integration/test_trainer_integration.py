"""Integration tests for Trainer with Loss, Optimizer, metrics, and DataModule."""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.module import AutoPilotModule
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from autopilot.core.trainer import Trainer
from autopilot.core.types import Datum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from helpers import NumericGradient


class _TrackingLoss(Loss):
  def __init__(self, params=None):
    super().__init__(params)
    self.forward_calls = 0
    self.backward_calls = 0
    self.reset_calls = 0

  def forward(self, data, targets=None):
    self.forward_calls += 1

  def backward(self):
    self.backward_calls += 1
    for p in self._loss_parameters:
      if p.requires_grad:
        p.grad = NumericGradient(value=1.0)

  def reset(self):
    self.reset_calls += 1


class _TrackingOptimizer(Optimizer):
  def __init__(self, params, lr=1.0):
    super().__init__(params, lr)
    self.step_calls = 0
    self.zero_grad_calls = 0

  def step(self):
    self.step_calls += 1

  def zero_grad(self):
    self.zero_grad_calls += 1
    super().zero_grad()


class _CountMetric(Metric):
  def __init__(self):
    super().__init__()
    self.add_state('_n', 0)

  def update(self, datum):
    self._n += 1

  def compute(self):
    return {'count': float(self._n)}


class _TrainModule(AutoPilotModule):
  def __init__(self):
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = _TrackingLoss([self.param])
    self.accuracy = _CountMetric()
    self._optimizer = _TrackingOptimizer([self.param])
    self._train_steps = 0
    self._val_steps = 0

  def forward(self, batch):
    return batch

  def training_step(self, batch):
    self._train_steps += 1
    return batch

  def validation_step(self, batch):
    self._val_steps += 1
    return batch

  def configure_optimizers(self):
    return self._optimizer


def _batches(n: int) -> DataLoader:
  return DataLoader([Datum(metadata={'i': i}) for i in range(n)], batch_size=1)


class TestFullLoop:
  def test_basic_loop_one_epoch(self):
    mod = _TrainModule()
    trainer = Trainer(accumulate_grad_batches=1)
    trainer.fit(mod, train_dataloaders=_batches(3), max_epochs=1)
    assert mod.loss.forward_calls == 3
    assert mod.loss.backward_calls == 3
    assert mod.loss.reset_calls == 3
    assert mod._optimizer.step_calls == 3
    assert mod._optimizer.zero_grad_calls == 3

  def test_accumulate_grad_batches_2(self):
    mod = _TrainModule()
    trainer = Trainer(accumulate_grad_batches=2)
    trainer.fit(mod, train_dataloaders=_batches(4), max_epochs=1)
    assert mod.loss.forward_calls == 4
    assert mod.loss.backward_calls == 2
    assert mod.loss.reset_calls == 2
    assert mod._optimizer.step_calls == 2
    assert mod._optimizer.zero_grad_calls == 2

  def test_accumulate_grad_batches_4(self):
    mod = _TrainModule()
    trainer = Trainer(accumulate_grad_batches=4)
    trainer.fit(mod, train_dataloaders=_batches(4), max_epochs=1)
    assert mod.loss.forward_calls == 4
    assert mod.loss.backward_calls == 1
    assert mod.loss.reset_calls == 1
    assert mod._optimizer.step_calls == 1
    assert mod._optimizer.zero_grad_calls == 1

  def test_accumulate_3_with_5_batches(self):
    mod = _TrainModule()
    trainer = Trainer(accumulate_grad_batches=3)
    trainer.fit(mod, train_dataloaders=_batches(5), max_epochs=1)
    assert mod.loss.forward_calls == 5
    assert mod.loss.backward_calls == 2
    assert mod._optimizer.step_calls == 2

  def test_metric_counts_batches(self):
    mod = _TrainModule()
    trainer = Trainer(accumulate_grad_batches=1)
    out = trainer.fit(mod, train_dataloaders=_batches(3), max_epochs=1)
    assert mod.accuracy._n == 0
    assert out['epochs'][0]['metrics']['count'] == 3.0

  def test_metric_resets_between_epochs(self):
    mod = _TrainModule()
    trainer = Trainer(accumulate_grad_batches=1)
    out = trainer.fit(mod, train_dataloaders=_batches(2), max_epochs=2)
    assert out['epochs'][0]['metrics']['count'] == 2.0
    assert out['epochs'][1]['metrics']['count'] == 2.0


class _HookRecorder(Callback):
  def __init__(self):
    self.calls: list[str] = []

  def on_fit_start(self, trainer):
    self.calls.append('on_fit_start')

  def on_fit_end(self, trainer):
    self.calls.append('on_fit_end')

  def on_train_epoch_start(self, trainer, epoch: int):
    self.calls.append('on_train_epoch_start')

  def on_train_epoch_end(self, trainer, epoch: int):
    self.calls.append('on_train_epoch_end')

  def on_train_batch_start(self, trainer, batch_idx: int = 0):
    self.calls.append(f'on_train_batch_start:{batch_idx}')

  def on_train_batch_end(self, trainer, batch_idx: int = 0, data=None):
    self.calls.append(f'on_train_batch_end:{batch_idx}')


class TestHookOrder:
  def test_train_hook_subsequence(self):
    rec = _HookRecorder()
    mod = _TrainModule()
    trainer = Trainer(callbacks=[rec], accumulate_grad_batches=1)
    trainer.fit(mod, train_dataloaders=_batches(2), max_epochs=1)
    names = rec.calls
    assert names[0] == 'on_fit_start'
    te = names.index('on_train_epoch_start')
    assert names[te + 1] == 'on_train_batch_start:0'
    assert names[te + 2] == 'on_train_batch_end:0'
    assert names[te + 3] == 'on_train_batch_start:1'
    assert names[te + 4] == 'on_train_batch_end:1'
    assert names.index('on_train_epoch_end') < names.index('on_fit_end')
    assert names[-1] == 'on_fit_end'


class _RecordingDataModule(DataModule):
  def __init__(self, n: int = 2):
    super().__init__()
    self.n = n
    self.prepared = False
    self.setup_fit = False
    self.torn = False

  def prepare_data(self) -> None:
    self.prepared = True

  def setup(self, stage: str) -> None:
    if stage == 'fit':
      self.setup_fit = True

  def train_dataloader(self) -> DataLoader:
    return _batches(self.n)

  def val_dataloader(self) -> DataLoader:
    return _batches(0)

  def teardown(self, stage: str) -> None:
    if stage == 'fit':
      self.torn = True


class TestDataModuleIntegration:
  def test_datamodule_wired_and_lifecycle(self):
    dm = _RecordingDataModule(n=3)
    mod = _TrainModule()
    trainer = Trainer(accumulate_grad_batches=1)
    trainer.fit(mod, datamodule=dm, max_epochs=1)
    assert dm.prepared is True
    assert dm.setup_fit is True
    assert dm.torn is True
    assert mod.loss.forward_calls == 3
