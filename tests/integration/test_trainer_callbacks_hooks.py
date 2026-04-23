"""Callback hooks with gradient accumulation."""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.loss import Loss
from autopilot.core.module import AutoPilotModule
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from autopilot.core.trainer import Trainer
from autopilot.core.types import Datum
from autopilot.data.dataloader import DataLoader
from helpers import NumericGradient


class _TLoss(Loss):
  def __init__(self, params):
    super().__init__(params)

  def forward(self, data, targets=None):
    pass

  def backward(self):
    for p in self._loss_parameters:
      if p.requires_grad:
        p.grad = NumericGradient(value=1.0)

  def reset(self):
    pass


class _TOpt(Optimizer):
  def __init__(self, params):
    super().__init__(params)

  def step(self):
    pass


class _Mod(AutoPilotModule):
  def __init__(self):
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = _TLoss([self.param])
    self._opt = _TOpt([self.param])

  def forward(self, batch):
    return batch

  def training_step(self, batch):
    return batch

  def validation_step(self, batch):
    return batch

  def configure_optimizers(self):
    return self._opt


class _Cb(Callback):
  def __init__(self):
    self.calls: list[str] = []

  def on_train_batch_start(self, trainer, batch_idx: int = 0):
    self.calls.append(f'bs:{batch_idx}')

  def on_train_batch_end(self, trainer, batch_idx: int = 0, data=None):
    self.calls.append(f'be:{batch_idx}')

  def on_before_backward(self, trainer):
    self.calls.append('bb')

  def on_after_backward(self, trainer):
    self.calls.append('ab')

  def on_before_optimizer_step(self, trainer):
    self.calls.append('bos')

  def on_before_zero_grad(self, trainer):
    self.calls.append('bzg')


def _dl(n: int) -> DataLoader:
  return DataLoader([Datum() for _ in range(n)], batch_size=1)


class TestTrainerCallbackHooks:
  def test_batch_level_hooks_fire(self):
    cb = _Cb()
    mod = _Mod()
    trainer = Trainer(callbacks=[cb], accumulate_grad_batches=1)
    trainer.fit(mod, train_dataloaders=_dl(3), max_epochs=1)
    assert cb.calls.count('bs:0') == 1
    assert cb.calls.count('be:0') == 1
    assert cb.calls.count('bs:1') == 1
    assert cb.calls.count('be:1') == 1
    assert cb.calls.count('bs:2') == 1
    assert cb.calls.count('be:2') == 1

  def test_backward_hooks_fire_on_step(self):
    cb = _Cb()
    mod = _Mod()
    trainer = Trainer(callbacks=[cb], accumulate_grad_batches=1)
    trainer.fit(mod, train_dataloaders=_dl(2), max_epochs=1)
    assert cb.calls.count('bb') == 2
    assert cb.calls.count('ab') == 2

  def test_optimizer_hooks_fire_on_step(self):
    cb = _Cb()
    mod = _Mod()
    trainer = Trainer(callbacks=[cb], accumulate_grad_batches=1)
    trainer.fit(mod, train_dataloaders=_dl(2), max_epochs=1)
    assert cb.calls.count('bos') == 2
    assert cb.calls.count('bzg') == 2

  def test_hooks_gated_by_accumulation(self):
    cb = _Cb()
    mod = _Mod()
    trainer = Trainer(callbacks=[cb], accumulate_grad_batches=2)
    trainer.fit(mod, train_dataloaders=_dl(3), max_epochs=1)
    assert cb.calls.count('bb') == 2
    assert cb.calls.count('ab') == 2
