"""EpochLoop stepping and edge cases."""

from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.trainer.trainer import Trainer
from autopilot.data.dataloader import DataLoader
from tests.doubles import NoopEvalModule


class TestEpochLoopShouldStep:
  def test_should_step_every_batch(self):
    loop = EpochLoop()
    for i in range(3):
      assert loop._should_step(i, is_last_batch=(i == 2), accumulate=1) is True

  def test_should_step_every_2(self):
    loop = EpochLoop()
    acc = 2
    assert loop._should_step(0, is_last_batch=False, accumulate=acc) is False
    assert loop._should_step(1, is_last_batch=False, accumulate=acc) is True
    assert loop._should_step(2, is_last_batch=False, accumulate=acc) is False
    assert loop._should_step(3, is_last_batch=True, accumulate=acc) is True

  def test_should_step_last_batch_always(self):
    loop = EpochLoop()
    acc = 100
    assert loop._should_step(0, is_last_batch=False, accumulate=acc) is False
    assert loop._should_step(1, is_last_batch=False, accumulate=acc) is False
    assert loop._should_step(2, is_last_batch=True, accumulate=acc) is True


class TestEpochLoopTrainerIntegration:
  def test_zero_epochs_no_run(self):
    mod = NoopEvalModule()
    trainer = Trainer()
    out = trainer.fit(mod, max_epochs=0)
    assert out['total_epochs'] == 0
    assert out['epochs'] == []

  def test_empty_dataloader(self):
    mod = NoopEvalModule()
    trainer = Trainer()
    out = trainer.fit(mod, train_dataloaders=DataLoader([], batch_size=1), max_epochs=1)
    assert out['total_epochs'] == 1
    assert out['epochs'][0]['epoch'] == 0
