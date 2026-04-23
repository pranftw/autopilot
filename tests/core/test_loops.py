"""Tests for Loop class hierarchy."""

from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.loops.loop import Loop, LoopConfig
from autopilot.core.module import Module
from autopilot.core.types import Datum
import pytest


class _StubModule(Module):
  def forward(self, *args, **kwargs) -> Datum:
    return Datum(success=True)


class _TrainerShim:
  """Minimal trainer surface for EpochLoop.run without constructing Trainer."""

  def __init__(self) -> None:
    self.module = _StubModule()
    self.policy = None

  def _dispatch(self, *args: object, **kwargs: object) -> list[object]:
    return []

  def on_epoch_start(self, epoch: int) -> list[object]:
    return []

  def on_epoch_end(self, epoch: int, result: dict | None = None) -> list[object]:
    return []

  def should_stop_at(self, hook_method: object, **kwargs: object) -> bool:
    return False


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

    trainer.should_stop_at = stop_at_2
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
    assert result['epochs'][0]['epoch'] == 1
