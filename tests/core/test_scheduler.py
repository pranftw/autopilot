"""Tests for Plan 15: LR Scheduler / Epoch Schedule Hook.

Covers Scheduler base class, LambdaScheduler, state_dict round-trip,
Trainer integration (discovery, stepping, checkpoint resume), and error paths.
"""

from autopilot.core.errors import ConfigError
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from autopilot.core.scheduler import LambdaScheduler, Scheduler
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.dataset import Dataset
from tests.doubles import DirectNumericLoss, NoOpOptimizer
from typing import Any
import pytest

# -- helpers ------------------------------------------------------------------


class _RisingScheduler(Scheduler):
  """Custom subclass returning rising LR list for testing."""

  def get_lr(self) -> list[float]:
    return [base * (1.0 + 0.1 * self.last_epoch) for base in self.base_lrs]


class _TinyDataset(Dataset[EvalDatum]):
  """Minimal dataset for trainer integration tests."""

  def __getitem__(self, index: int) -> EvalDatum:
    return EvalDatum(success=True)

  def __len__(self) -> int:
    return 3


class _SchedulerModule(AutoPilotModule):
  """Module that returns dict with optimizer and scheduler from configure_optimizers."""

  def __init__(self, optimizer: Optimizer, scheduler: Scheduler | None = None) -> None:
    super().__init__()
    self._opt = optimizer
    self._sched = scheduler
    self.loss = DirectNumericLoss()

  def forward(self, batch: Any) -> Datum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> Datum:
    return self(batch)

  def configure_optimizers(self) -> dict[str, Any] | Optimizer:
    if self._sched is not None:
      return {'optimizer': self._opt, 'scheduler': self._sched}
    return self._opt


# -- Scheduler base class tests -----------------------------------------------


class TestSchedulerStep:
  """Test Scheduler.step updates LR in param_groups."""

  def test_scheduler_step_updates_lr(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=0.5)
    scheduler = _RisingScheduler(optimizer)

    scheduler.step(0)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.5 * 1.0)

    scheduler.step(1)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.5 * 1.1)

    scheduler.step(5)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.5 * 1.5)

  def test_scheduler_no_epoch_arg_increments(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=1.0)
    scheduler = _RisingScheduler(optimizer)

    assert scheduler.last_epoch == -1

    scheduler.step()
    assert scheduler.last_epoch == 0
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1.0)

    scheduler.step()
    assert scheduler.last_epoch == 1
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1.1)

    scheduler.step()
    assert scheduler.last_epoch == 2
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1.2)

  def test_abstract_get_lr_raises_not_implemented(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=1.0)
    scheduler = Scheduler(optimizer)

    with pytest.raises(NotImplementedError):
      scheduler.step(0)

  def test_base_lrs_frozen_at_construction(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=0.8)
    scheduler = _RisingScheduler(optimizer)

    scheduler.step(2)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.8 * 1.2)
    assert scheduler.base_lrs == [0.8]


# -- LambdaScheduler tests ----------------------------------------------------


class TestLambdaScheduler:
  """Test LambdaScheduler with user-supplied lr_lambda."""

  def test_lambda_scheduler_scales_all_groups(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    optimizer = NoOpOptimizer([{'params': [p1], 'lr': 0.1}, {'params': [p2], 'lr': 0.01}])
    scheduler = LambdaScheduler(optimizer, lr_lambda=lambda epoch: 0.5**epoch)

    scheduler.step(0)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.1 * 1.0)
    assert optimizer.param_groups[1]['lr'] == pytest.approx(0.01 * 1.0)

    scheduler.step(1)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.1 * 0.5)
    assert optimizer.param_groups[1]['lr'] == pytest.approx(0.01 * 0.5)

    scheduler.step(3)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.1 * 0.125)
    assert optimizer.param_groups[1]['lr'] == pytest.approx(0.01 * 0.125)

  def test_lambda_scheduler_uses_base_lrs_not_current(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=1.0)
    scheduler = LambdaScheduler(optimizer, lr_lambda=lambda epoch: 0.9**epoch)

    scheduler.step(1)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.9)

    scheduler.step(2)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.81)
    assert scheduler.base_lrs == [1.0]

  def test_lambda_exception_propagation(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=1.0)

    def bad_lambda(epoch: int) -> float:
      msg = 'intentional failure'
      raise ValueError(msg)

    scheduler = LambdaScheduler(optimizer, lr_lambda=bad_lambda)
    with pytest.raises(ValueError, match='intentional failure'):
      scheduler.step(0)


# -- State dict round-trip tests -----------------------------------------------


class TestSchedulerStateDict:
  """Test state_dict / load_state_dict round-trip."""

  def test_scheduler_state_dict_round_trip(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=0.5)
    scheduler = LambdaScheduler(optimizer, lr_lambda=lambda e: 0.9**e)

    scheduler.step(3)
    state = scheduler.state_dict()

    assert state == {'last_epoch': 3, 'base_lrs': [0.5]}

    new_scheduler = LambdaScheduler(optimizer, lr_lambda=lambda e: 0.9**e)
    assert new_scheduler.last_epoch == -1

    new_scheduler.load_state_dict(state)
    assert new_scheduler.last_epoch == 3
    assert new_scheduler.base_lrs == [0.5]

  def test_scheduler_state_dict_multiple_groups(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    optimizer = NoOpOptimizer([{'params': [p1], 'lr': 0.1}, {'params': [p2], 'lr': 0.01}])
    scheduler = _RisingScheduler(optimizer)

    scheduler.step(5)
    state = scheduler.state_dict()

    assert state['last_epoch'] == 5
    assert state['base_lrs'] == [0.1, 0.01]

    new_scheduler = _RisingScheduler(optimizer)
    new_scheduler.load_state_dict(state)
    assert new_scheduler.last_epoch == 5
    assert new_scheduler.base_lrs == [0.1, 0.01]

  def test_scheduler_survives_checkpoint_resume(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=1.0)
    scheduler = LambdaScheduler(optimizer, lr_lambda=lambda e: 0.95**e)

    scheduler.step(4)
    state = scheduler.state_dict()

    optimizer2 = NoOpOptimizer([param], lr=2.0)
    scheduler2 = LambdaScheduler(optimizer2, lr_lambda=lambda e: 0.95**e)
    scheduler2.load_state_dict(state)

    assert scheduler2.last_epoch == 4
    assert scheduler2.base_lrs == [1.0]

    scheduler2.step(5)
    expected = 1.0 * 0.95**5
    assert optimizer2.param_groups[0]['lr'] == pytest.approx(expected)


# -- Param groups tests --------------------------------------------------------


class TestSchedulerWithParamGroups:
  """Test scheduler output length matches number of param_groups."""

  def test_scheduler_with_param_groups(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    p3 = Parameter()
    optimizer = NoOpOptimizer(
      [
        {'params': [p1], 'lr': 0.1},
        {'params': [p2], 'lr': 0.01},
        {'params': [p3], 'lr': 0.001},
      ]
    )
    scheduler = LambdaScheduler(optimizer, lr_lambda=lambda epoch: 2.0)

    scheduler.step(0)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.2)
    assert optimizer.param_groups[1]['lr'] == pytest.approx(0.02)
    assert optimizer.param_groups[2]['lr'] == pytest.approx(0.002)

  def test_base_lrs_snapshot_matches_groups(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    optimizer = NoOpOptimizer([{'params': [p1], 'lr': 0.5}, {'params': [p2], 'lr': 0.05}])
    scheduler = LambdaScheduler(optimizer, lr_lambda=lambda e: 1.0)

    assert scheduler.base_lrs == [0.5, 0.05]
    assert len(scheduler.base_lrs) == len(optimizer.param_groups)


# -- Trainer integration tests -------------------------------------------------


class TestTrainerSchedulerIntegration:
  """Test Trainer discovery and stepping of scheduler."""

  def test_trainer_calls_scheduler_step(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=1.0)
    scheduler = LambdaScheduler(optimizer, lr_lambda=lambda e: 0.9**e)

    module = _SchedulerModule(optimizer, scheduler)
    loader = DataLoader(_TinyDataset(), batch_size=3)
    trainer = Trainer()

    trainer.fit(module, train_dataloaders=loader, max_epochs=3)

    assert scheduler.last_epoch == 2
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.9**2)

  def test_trainer_scheduler_none_when_bare_optimizer(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=1.0)

    module = _SchedulerModule(optimizer, scheduler=None)
    loader = DataLoader(_TinyDataset(), batch_size=3)
    trainer = Trainer()

    trainer.fit(module, train_dataloaders=loader, max_epochs=1)

    assert trainer.scheduler is None

  def test_trainer_scheduler_property_accessible(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=1.0)
    scheduler = LambdaScheduler(optimizer, lr_lambda=lambda e: 1.0)

    module = _SchedulerModule(optimizer, scheduler)
    loader = DataLoader(_TinyDataset(), batch_size=3)
    trainer = Trainer()

    trainer.fit(module, train_dataloaders=loader, max_epochs=1)

    assert trainer.scheduler is scheduler

  def test_configure_optimizers_scheduler_optimizer_mismatch_raises(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    optimizer1 = NoOpOptimizer([p1], lr=1.0)
    optimizer2 = NoOpOptimizer([p2], lr=0.5)
    scheduler = LambdaScheduler(optimizer2, lr_lambda=lambda e: 1.0)

    class _MismatchModule(AutoPilotModule):
      def forward(self, batch):
        return EvalDatum(success=True)

      def training_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return {'optimizer': optimizer1, 'scheduler': scheduler}

    module = _MismatchModule()
    loader = DataLoader(_TinyDataset(), batch_size=3)
    trainer = Trainer()

    with pytest.raises(ConfigError, match=r'scheduler\.optimizer is not the same object'):
      trainer.fit(module, train_dataloaders=loader, max_epochs=1)


# -- Checkpoint integration tests ----------------------------------------------


class TestSchedulerCheckpoint:
  """Test scheduler state in checkpoint save/restore."""

  def test_scheduler_in_checkpoint_state(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=1.0)
    scheduler = LambdaScheduler(optimizer, lr_lambda=lambda e: 0.9**e)

    module = _SchedulerModule(optimizer, scheduler)
    loader = DataLoader(_TinyDataset(), batch_size=3)
    trainer = Trainer()

    trainer.fit(module, train_dataloaders=loader, max_epochs=2)

    state = trainer._build_checkpoint_state()

    assert 'scheduler' in state
    assert state['scheduler']['last_epoch'] == 1
    assert state['scheduler']['base_lrs'] == [1.0]

  def test_scheduler_restored_from_checkpoint(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=1.0)
    scheduler = LambdaScheduler(optimizer, lr_lambda=lambda e: 0.9**e)

    module = _SchedulerModule(optimizer, scheduler)
    loader = DataLoader(_TinyDataset(), batch_size=3)
    trainer = Trainer()

    trainer.fit(module, train_dataloaders=loader, max_epochs=2)

    state = trainer._build_checkpoint_state()

    param2 = Parameter()
    optimizer2 = NoOpOptimizer([param2], lr=1.0)
    scheduler2 = LambdaScheduler(optimizer2, lr_lambda=lambda e: 0.9**e)

    module2 = _SchedulerModule(optimizer2, scheduler2)
    trainer2 = Trainer()
    trainer2._module = module2
    trainer2._optimizer = optimizer2
    trainer2._scheduler = scheduler2

    trainer2._restore_from_checkpoint(state, module2)

    assert scheduler2.last_epoch == 1
    assert scheduler2.base_lrs == [1.0]

  def test_no_scheduler_key_in_checkpoint_when_none(self) -> None:
    param = Parameter()
    optimizer = NoOpOptimizer([param], lr=1.0)

    module = _SchedulerModule(optimizer, scheduler=None)
    loader = DataLoader(_TinyDataset(), batch_size=3)
    trainer = Trainer()

    trainer.fit(module, train_dataloaders=loader, max_epochs=1)

    state = trainer._build_checkpoint_state()
    assert 'scheduler' not in state
