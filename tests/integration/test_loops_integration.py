"""EpochLoop stepping and edge cases."""

from autopilot.core.loops import EpochLoop
from autopilot.core.models import Datum
from autopilot.core.module import Module
from autopilot.core.trainer import Trainer
from autopilot.data.dataloader import DataLoader


class _StubModule(Module):
  def forward(self, *args, **kwargs):
    return Datum(success=True)


class TestEpochLoopShouldStep:
  def test_should_step_every_batch(self):
    loop = EpochLoop()
    for i in range(3):
      assert loop._should_step(i, i == 2, 1) is True

  def test_should_step_every_2(self):
    loop = EpochLoop()
    acc = 2
    assert loop._should_step(0, False, acc) is False
    assert loop._should_step(1, False, acc) is True
    assert loop._should_step(2, False, acc) is False
    assert loop._should_step(3, True, acc) is True

  def test_should_step_last_batch_always(self):
    loop = EpochLoop()
    acc = 100
    assert loop._should_step(0, False, acc) is False
    assert loop._should_step(1, False, acc) is False
    assert loop._should_step(2, True, acc) is True


class TestEpochLoopTrainerIntegration:
  def test_zero_epochs_no_run(self):
    mod = _StubModule()
    trainer = Trainer()
    out = trainer.fit(mod, max_epochs=0)
    assert out['total_epochs'] == 0
    assert out['epochs'] == []

  def test_empty_dataloader(self):
    mod = _StubModule()
    trainer = Trainer()
    out = trainer.fit(mod, train_dataloaders=DataLoader([], batch_size=1), max_epochs=1)
    assert out['total_epochs'] == 1
    assert out['epochs'][0]['epoch'] == 1
