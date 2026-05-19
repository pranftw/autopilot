"""Trainer integration with Policy and Store."""

from autopilot.core.experiment import Experiment
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, GateResult
from autopilot.data.dataloader import DataLoader
from autopilot.policy.policy import Policy
from tests.doubles import DirectNumericLoss, NoOpOptimizer
from typing import Any


class _PolicyModule(AutoPilotModule):
  def __init__(self):
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.param])
    self._opt = NoOpOptimizer([self.param])

  def forward(self, batch):
    return batch

  def training_step(self, batch, batch_idx):
    return batch

  def validation_step(self, batch, batch_idx):
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


class _MockExperiment(Experiment):
  def __init__(self, store=None):
    super().__init__(experiment_id='mock-exp')
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


def _loader(n: int = 1) -> DataLoader:
  return DataLoader([Datum() for _ in range(n)], batch_size=1)


class TestTrainerPolicy:
  def test_policy_pass_continues(self):
    mod = _PolicyModule()
    pol = _GatePolicy([GateResult.PASSED, GateResult.PASSED])
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
    experiment = _MockExperiment(store=store)
    trainer = Trainer(policy=pol, experiment=experiment)
    trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=5)
    assert store.checkouts == [0]

  def test_policy_warn_continues(self):
    mod = _PolicyModule()
    pol = _GatePolicy([GateResult.WARN, GateResult.WARN, GateResult.WARN])
    trainer = Trainer(policy=pol)
    out = trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=3)
    assert out['total_epochs'] == 3
    assert len(pol.results_seen) == 3

  def test_last_accepted_epoch_tracking(self):
    mod = _PolicyModule()
    pol = _GatePolicy([GateResult.PASSED, GateResult.PASSED, GateResult.PASSED])
    experiment = _MockExperiment()
    trainer = Trainer(policy=pol, experiment=experiment)
    trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=3)
    assert experiment.last_accepted_epoch == 2

  def test_no_policy_no_error(self):
    mod = _PolicyModule()
    trainer = Trainer()
    out = trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=2)
    assert out['total_epochs'] == 2

  def test_no_store_on_fail(self):
    mod = _PolicyModule()
    pol = _GatePolicy([GateResult.FAIL])
    trainer = Trainer(policy=pol)
    out = trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=3)
    assert out['total_epochs'] == 1
    assert out['epochs'][0].get('stopped') is True
