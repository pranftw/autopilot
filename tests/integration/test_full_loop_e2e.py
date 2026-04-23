"""End-to-end Trainer run with all major components."""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.module import AutoPilotModule
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from autopilot.core.trainer import Trainer
from autopilot.core.types import Datum, GateResult
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from autopilot.policy.policy import Policy
from helpers import NumericGradient


class _E2ELoss(Loss):
  def __init__(self, params):
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


class _E2EOpt(Optimizer):
  def __init__(self, params):
    super().__init__(params)
    self.step_calls = 0
    self.zero_grad_calls = 0

  def step(self):
    self.step_calls += 1

  def zero_grad(self):
    self.zero_grad_calls += 1
    super().zero_grad()


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
    self.loss = _E2ELoss([self.param])
    self.train_metric = _E2EMetric()
    self._opt = _E2EOpt([self.param])

  def forward(self, batch):
    return batch

  def training_step(self, batch):
    return batch

  def validation_step(self, batch):
    return batch

  def configure_optimizers(self):
    return self._opt


class _E2EPolicy(Policy):
  def __init__(self):
    super().__init__()
    self.calls = 0

  def forward(self, result: Result) -> GateResult:
    self.calls += 1
    return GateResult.PASS


class _E2EStore:
  def __init__(self):
    self.checkouts: list[int] = []

  def checkout(self, epoch: int) -> None:
    self.checkouts.append(epoch)


class _E2EExperiment:
  def __init__(self, store=None):
    self.store = store
    self.should_rollback = False
    self.best_epoch = 0

  def rollback(self, to_epoch):
    if self.store:
      self.store.checkout(to_epoch)

  def on_epoch_complete(self, epoch, train_metrics):
    pass

  def on_validation_complete(self, epoch, val_metrics, metric_metadata=None):
    pass

  def on_loop_complete(self, loop_result):
    pass


class _E2ECallback(Callback):
  def __init__(self):
    self.calls: list[str] = []

  def on_fit_start(self, trainer):
    self.calls.append('fit_start')

  def on_fit_end(self, trainer):
    self.calls.append('fit_end')

  def on_train_epoch_start(self, trainer, epoch: int):
    self.calls.append(f'train_ep_start:{epoch}')

  def on_train_epoch_end(self, trainer, epoch: int):
    self.calls.append(f'train_ep_end:{epoch}')


class _E2EDataModule(DataModule):
  def train_dataloader(self) -> DataLoader:
    return DataLoader([Datum(metadata={'i': i}) for i in range(4)], batch_size=1)

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
  assert cb.calls.count('train_ep_start:1') == 1
  assert cb.calls.count('train_ep_end:1') == 1
  assert cb.calls.count('train_ep_start:2') == 1
  assert cb.calls.count('train_ep_end:2') == 1
