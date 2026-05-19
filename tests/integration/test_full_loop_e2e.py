"""End-to-end Trainer run with all major components."""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.experiment import Experiment
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum, GateResult
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from autopilot.policy.policy import Policy
from tests.doubles import TrackingNumericLoss, TrackingOptimizer
from typing import Any


class _E2EMetric(Metric):
  def __init__(self):
    super().__init__()
    self.add_state('_n', 0)

  def update(self, datum):
    self._n += 1

  def compute(self):
    return {'n': float(self._n)}


class _E2EModule(AutoPilotModule):
  def __init__(self):
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = TrackingNumericLoss([self.param])
    self.train_metric = _E2EMetric()
    self._opt = TrackingOptimizer([self.param])

  def forward(self, batch):
    return batch

  def training_step(self, batch, batch_idx):
    return batch

  def validation_step(self, batch, batch_idx):
    return batch

  def configure_optimizers(self):
    return self._opt


class _E2EPolicy(Policy):
  def __init__(self):
    super().__init__()
    self.calls = 0

  def forward(self, result: Result) -> GateResult:
    self.calls += 1
    return GateResult.PASSED


class _E2EStore:
  def __init__(self):
    self.checkouts: list[int] = []

  def checkout(self, epoch: int) -> None:
    self.checkouts.append(epoch)


class _E2EExperiment(Experiment):
  def __init__(self, store=None):
    super().__init__(experiment_id='e2e-test')
    self.store = store
    self.should_rollback = False
    self.last_accepted_epoch = 0

  def rollback(self, epoch: int | None) -> None:
    if self.store:
      self.store.checkout(epoch)

  def on_epoch_complete(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
    pass

  def on_validation_complete(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
    pass


class _E2ECallback(Callback):
  def __init__(self):
    self.calls: list[str] = []

  def on_fit_start(self, trainer, module):
    self.calls.append('fit_start')

  def on_fit_end(self, trainer, module):
    self.calls.append('fit_end')

  def on_train_epoch_start(self, trainer, module, epoch: int):
    self.calls.append(f'train_ep_start:{epoch}')

  def on_train_epoch_end(self, trainer, module, epoch: int):
    self.calls.append(f'train_ep_end:{epoch}')


class _E2EDataModule(DataModule):
  def train_dataloader(self) -> DataLoader:
    return DataLoader([EvalDatum(metadata={'i': i}) for i in range(4)], batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader([], batch_size=1)


def test_full_stack_two_epochs_accumulate_two():
  mod = _E2EModule()
  pol = _E2EPolicy()
  store = _E2EStore()
  cb = _E2ECallback()
  dm = _E2EDataModule()
  experiment = _E2EExperiment(store=store)
  trainer = Trainer(
    callbacks=[cb],
    policy=pol,
    experiment=experiment,
    accumulate_grad_batches=2,
  )
  out = trainer.fit(mod, datamodule=dm, max_epochs=2)
  assert mod.loss.forward_calls == 8
  assert mod.loss.backward_calls == 4
  assert mod.loss.reset_calls == 4
  assert mod._opt.step_calls == 4
  assert mod._opt.zero_grad_calls == 4
  assert pol.calls == 2
  assert store.checkouts == []
  assert out['total_epochs'] == 2
  assert out['epochs'][0]['metrics']['n'] == 4.0
  assert out['epochs'][1]['metrics']['n'] == 4.0
  assert cb.calls[0] == 'fit_start'
  assert cb.calls[-1] == 'fit_end'
  assert cb.calls.count('train_ep_start:0') == 1
  assert cb.calls.count('train_ep_end:0') == 1
  assert cb.calls.count('train_ep_start:1') == 1
  assert cb.calls.count('train_ep_end:1') == 1
