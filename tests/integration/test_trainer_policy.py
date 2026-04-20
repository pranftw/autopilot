"""Trainer integration with Policy and Store."""

from autopilot.core.loss import Loss
from autopilot.core.models import Datum, GateResult, Result
from autopilot.core.module import AutoPilotModule
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from autopilot.core.trainer import Trainer
from autopilot.data.dataloader import DataLoader
from autopilot.policy.policy import Policy


class _StubLoss(Loss):
  def __init__(self, params):
    super().__init__(params)

  def forward(self, data, targets=None):
    pass

  def backward(self):
    for p in self._loss_parameters:
      if p.requires_grad:
        p.grad = 'g'

  def reset(self):
    pass


class _StubOpt(Optimizer):
  def __init__(self, params):
    super().__init__(params)

  def step(self):
    pass


class _PolicyModule(AutoPilotModule):
  def __init__(self):
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = _StubLoss([self.param])
    self._opt = _StubOpt([self.param])

  def forward(self, batch):
    return batch

  def training_step(self, batch):
    return batch

  def validation_step(self, batch):
    return batch

  def configure_optimizers(self):
    return self._opt


class _GatePolicy(Policy):
  def __init__(self, sequence: list[GateResult]):
    super().__init__()
    self._sequence = list(sequence)
    self._i = 0
    self.results_seen: list[Result] = []

  def forward(self, result: Result) -> GateResult:
    self.results_seen.append(result)
    out = self._sequence[self._i]
    self._i += 1
    return out


class _MockStore:
  def __init__(self):
    self.checkouts: list[int] = []

  def checkout(self, epoch: int) -> None:
    self.checkouts.append(epoch)


def _loader(n: int = 1) -> DataLoader:
  return DataLoader([Datum() for _ in range(n)], batch_size=1)


class TestTrainerPolicy:
  def test_policy_pass_continues(self):
    mod = _PolicyModule()
    pol = _GatePolicy([GateResult.PASS, GateResult.PASS])
    trainer = Trainer(policy=pol)
    out = trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=2)
    assert out['total_epochs'] == 2
    assert len(pol.results_seen) == 2

  def test_policy_fail_stops(self):
    mod = _PolicyModule()
    pol = _GatePolicy([GateResult.FAIL])
    trainer = Trainer(policy=pol)
    out = trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=5)
    assert out['total_epochs'] == 1
    assert out['epochs'][0].get('stopped') is True

  def test_store_checkout_on_fail(self):
    mod = _PolicyModule()
    pol = _GatePolicy([GateResult.FAIL])
    store = _MockStore()
    trainer = Trainer(policy=pol, store=store)
    trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=5)
    assert store.checkouts == [0]

  def test_policy_warn_continues(self):
    mod = _PolicyModule()
    pol = _GatePolicy([GateResult.WARN, GateResult.WARN, GateResult.WARN])
    trainer = Trainer(policy=pol)
    out = trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=3)
    assert out['total_epochs'] == 3
    assert len(pol.results_seen) == 3

  def test_best_epoch_tracking(self):
    mod = _PolicyModule()
    pol = _GatePolicy([GateResult.PASS, GateResult.PASS, GateResult.PASS])
    trainer = Trainer(policy=pol)
    trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=3)
    assert trainer._best_epoch - 1 == 2

  def test_no_policy_no_error(self):
    mod = _PolicyModule()
    trainer = Trainer()
    out = trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=2)
    assert out['total_epochs'] == 2

  def test_no_store_on_fail(self):
    mod = _PolicyModule()
    pol = _GatePolicy([GateResult.FAIL])
    trainer = Trainer(policy=pol, store=None)
    out = trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=3)
    assert out['total_epochs'] == 1
    assert out['epochs'][0].get('stopped') is True
