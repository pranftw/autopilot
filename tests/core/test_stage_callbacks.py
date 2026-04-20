"""Tests for stage callbacks."""

from autopilot.core.memory import FileMemory
from autopilot.core.models import Datum, Result
from autopilot.core.optimizer import Optimizer
from autopilot.core.regression import read_best_baseline, write_best_baseline
from autopilot.core.stage_callbacks import (
  EpochRecorderCallback,
  JudgeValidationCallback,
  MemoryCallback,
  RegressionCallback,
)
from autopilot.core.trainer import Trainer
from unittest.mock import MagicMock


class TestEpochRecorderCallback:
  def test_accumulates_batch_data(self, tmp_path):
    cb = EpochRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    for i in range(5):
      cb.on_train_batch_end(trainer=trainer, batch_idx=i, data=Datum(success=True, item_id=str(i)))
    assert len(cb._batch_data) == 5

  def test_writes_data_jsonl(self, tmp_path):
    cb = EpochRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    for i in range(3):
      cb.on_train_batch_end(trainer=trainer, batch_idx=i, data=Datum(success=True))
    cb.on_train_epoch_end(trainer=trainer, epoch=1)
    data_file = tmp_path / 'epoch_1' / 'data.jsonl'
    assert data_file.exists()
    lines = data_file.read_text().strip().splitlines()
    assert len(lines) == 3

  def test_writes_epoch_metrics(self, tmp_path):
    cb = EpochRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    cb.on_train_batch_end(trainer=trainer, batch_idx=0, data=Datum(success=True))
    cb.on_train_batch_end(trainer=trainer, batch_idx=1, data=Datum(success=False))
    cb.on_train_epoch_end(trainer=trainer, epoch=1)
    import json

    metrics_file = tmp_path / 'epoch_1' / 'epoch_metrics.json'
    data = json.loads(metrics_file.read_text())
    assert data['total'] == 2
    assert data['passed'] == 1
    assert data['failed'] == 1
    assert data['accuracy'] == 0.5

  def test_writes_delta_metrics(self, tmp_path):
    cb = EpochRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    cb.on_train_batch_end(trainer=trainer, data=Datum(success=True))
    cb.on_train_epoch_end(trainer=trainer, epoch=1)
    cb.on_train_epoch_start(trainer=trainer, epoch=2)
    cb.on_train_batch_end(trainer=trainer, data=Datum(success=True))
    cb.on_train_epoch_end(trainer=trainer, epoch=2)
    delta_file = tmp_path / 'epoch_2' / 'delta_metrics.json'
    assert delta_file.exists()

  def test_delta_first_epoch(self, tmp_path):
    cb = EpochRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    cb.on_train_batch_end(trainer=trainer, data=Datum(success=True))
    cb.on_train_epoch_end(trainer=trainer, epoch=1)
    delta_file = tmp_path / 'epoch_1' / 'delta_metrics.json'
    assert not delta_file.exists()

  def test_resets_between_epochs(self, tmp_path):
    cb = EpochRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    cb.on_train_batch_end(trainer=trainer, data=Datum(success=True))
    cb.on_train_epoch_end(trainer=trainer, epoch=1)
    cb.on_train_epoch_start(trainer=trainer, epoch=2)
    assert cb._batch_data == []

  def test_empty_epoch(self, tmp_path):
    cb = EpochRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    cb.on_train_epoch_end(trainer=trainer, epoch=1)
    data_file = tmp_path / 'epoch_1' / 'data.jsonl'
    assert not data_file.exists() or data_file.read_text().strip() == ''

  def test_epoch_dir_created(self, tmp_path):
    cb = EpochRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=3)
    cb.on_train_batch_end(trainer=trainer, data=Datum(success=True))
    cb.on_train_epoch_end(trainer=trainer, epoch=3)
    assert (tmp_path / 'epoch_3').is_dir()

  def test_state_dict_round_trip(self, tmp_path):
    cb = EpochRecorderCallback(tmp_path)
    cb._prev_metrics = {'accuracy': 0.75}
    state = cb.state_dict()
    cb2 = EpochRecorderCallback(tmp_path)
    cb2.load_state_dict(state)
    assert cb2._prev_metrics == {'accuracy': 0.75}


class TestJudgeValidationCallback:
  def test_fires_after_backward(self, tmp_path):
    cb = JudgeValidationCallback(tmp_path)
    trainer = MagicMock()
    cb.on_after_backward(trainer=trainer)

  def test_state_dict_empty(self, tmp_path):
    cb = JudgeValidationCallback(tmp_path)
    assert cb.state_dict() == {}


class TestRegressionCallback:
  def test_improvement_no_signal(self, tmp_path):
    write_best_baseline(tmp_path, epoch=1, metrics={'accuracy': 0.5})
    cb = RegressionCallback(tmp_path)
    trainer = MagicMock()
    trainer.regression_detected = False
    trainer._last_val_metrics = {'accuracy': 0.8}
    cb.on_validation_epoch_end(trainer=trainer, epoch=2)
    assert not trainer.regression_detected

  def test_regression_signals_trainer(self, tmp_path):
    write_best_baseline(tmp_path, epoch=1, metrics={'accuracy': 0.8})
    cb = RegressionCallback(tmp_path)
    trainer = MagicMock()
    trainer.regression_detected = False
    trainer._last_val_metrics = {'accuracy': 0.5}
    cb.on_validation_epoch_end(trainer=trainer, epoch=2)
    assert trainer.regression_detected

  def test_does_not_call_store_checkout(self, tmp_path):
    write_best_baseline(tmp_path, epoch=1, metrics={'accuracy': 0.8})
    cb = RegressionCallback(tmp_path)
    trainer = MagicMock()
    trainer.regression_detected = False
    trainer._last_val_metrics = {'accuracy': 0.5}
    trainer._store = MagicMock()
    cb.on_validation_epoch_end(trainer=trainer, epoch=2)
    trainer._store.checkout.assert_not_called()

  def test_updates_best_baseline(self, tmp_path):
    write_best_baseline(tmp_path, epoch=1, metrics={'accuracy': 0.5})
    cb = RegressionCallback(tmp_path)
    trainer = MagicMock()
    trainer.regression_detected = False
    trainer._last_val_metrics = {'accuracy': 0.8}
    cb.on_validation_epoch_end(trainer=trainer, epoch=2)
    from autopilot.core.regression import read_best_baseline

    assert read_best_baseline(tmp_path) == {'accuracy': 0.8}

  def test_regression_preserves_best(self, tmp_path):
    write_best_baseline(tmp_path, epoch=1, metrics={'accuracy': 0.8})
    cb = RegressionCallback(tmp_path)
    trainer = MagicMock()
    trainer.regression_detected = False
    trainer._last_val_metrics = {'accuracy': 0.5}
    cb.on_validation_epoch_end(trainer=trainer, epoch=2)
    from autopilot.core.regression import read_best_baseline

    assert read_best_baseline(tmp_path) == {'accuracy': 0.8}

  def test_writes_regression_analysis(self, tmp_path):
    write_best_baseline(tmp_path, epoch=1, metrics={'accuracy': 0.8})
    cb = RegressionCallback(tmp_path)
    trainer = MagicMock()
    trainer.regression_detected = False
    trainer._last_val_metrics = {'accuracy': 0.5}
    cb.on_validation_epoch_end(trainer=trainer, epoch=2)
    assert (tmp_path / 'epoch_2' / 'regression_analysis.json').exists()

  def test_custom_analysis_filename(self, tmp_path):
    write_best_baseline(tmp_path, epoch=1, metrics={'accuracy': 0.8})
    cb = RegressionCallback(tmp_path, analysis_filename='custom.json')
    trainer = MagicMock()
    trainer.regression_detected = False
    trainer._last_val_metrics = {'accuracy': 0.5}
    cb.on_validation_epoch_end(trainer=trainer, epoch=2)
    assert (tmp_path / 'epoch_2' / 'custom.json').exists()

  def test_first_epoch_no_baseline(self, tmp_path):
    cb = RegressionCallback(tmp_path)
    trainer = MagicMock()
    trainer.regression_detected = False
    trainer._last_val_metrics = {'accuracy': 0.8}
    cb.on_validation_epoch_end(trainer=trainer, epoch=1)
    assert not trainer.regression_detected
    assert (tmp_path / 'best_baseline.json').exists()
    baseline = read_best_baseline(tmp_path)
    assert baseline == {'accuracy': 0.8}

  def test_state_dict_round_trip(self, tmp_path):
    cb = RegressionCallback(tmp_path)
    assert cb.state_dict() == {}


class TestMemoryCallback:
  def test_records_structured_learnings(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    trainer = MagicMock()
    result = Result(metrics={'accuracy': 0.8}, passed=True)
    cb.on_epoch_end(trainer=trainer, epoch=1, result=result)
    records = memory.recall()
    assert len(records) == 1
    assert records[0].epoch == 1
    assert records[0].outcome == 'worked'
    assert records[0].metrics == {'accuracy': 0.8}
    assert records[0].category == 'epoch_result'

  def test_default_category(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    trainer = MagicMock()
    cb.on_epoch_end(trainer=trainer, epoch=1, result=Result(passed=True))
    records = memory.recall()
    assert records[0].category == 'epoch_result'

  def test_custom_category(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory, default_category='custom')
    trainer = MagicMock()
    cb.on_epoch_end(trainer=trainer, epoch=1, result=Result(passed=True))
    records = memory.recall()
    assert records[0].category == 'custom'

  def test_recorded_entry_has_metrics(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    trainer = MagicMock()
    result = Result(metrics={'accuracy': 0.9, 'f1': 0.85}, passed=True)
    cb.on_epoch_end(trainer=trainer, epoch=1, result=result)
    records = memory.recall()
    assert records[0].metrics['accuracy'] == 0.9
    assert records[0].metrics['f1'] == 0.85

  def test_populates_blocked_via_method(self, tmp_path):
    memory = FileMemory(tmp_path)
    memory.block_strategy('bad_approach')
    cb = MemoryCallback(memory)
    opt = Optimizer(parameters=[], lr=1.0)
    trainer = MagicMock()
    trainer.optimizer = opt
    cb.on_before_optimizer_step(trainer=trainer)
    assert opt.is_strategy_blocked('bad_approach')

  def test_no_blocklist(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    opt = Optimizer(parameters=[], lr=1.0)
    trainer = MagicMock()
    trainer.optimizer = opt
    cb.on_before_optimizer_step(trainer=trainer)
    assert opt.blocked_strategies == frozenset()

  def test_accesses_optimizer_via_trainer_property(self, tmp_path):
    memory = FileMemory(tmp_path)
    memory.block_strategy('x')
    cb = MemoryCallback(memory)
    opt = Optimizer(parameters=[], lr=1.0)
    trainer = Trainer()
    trainer._optimizer = opt
    cb.on_before_optimizer_step(trainer=trainer)
    assert opt.is_strategy_blocked('x')

  def test_state_dict_delegates_to_memory(self, tmp_path):
    memory = FileMemory(tmp_path)
    memory.learn(epoch=1, outcome='worked')
    cb = MemoryCallback(memory)
    state = cb.state_dict()
    assert 'records' in state
    assert len(state['records']) == 1

  def test_memory_file_written_structured(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    trainer = MagicMock()
    result = Result(metrics={'accuracy': 0.7}, passed=False)
    cb.on_epoch_end(trainer=trainer, epoch=1, result=result)
    import json

    lines = (tmp_path / 'knowledge_base.jsonl').read_text().strip().splitlines()
    data = json.loads(lines[0])
    assert data['epoch'] == 1
    assert data['outcome'] == 'failed'
    assert data['metrics'] == {'accuracy': 0.7}
    assert data['category'] == 'epoch_result'

  def test_load_state_dict_round_trip(self, tmp_path):
    memory = FileMemory(tmp_path)
    memory.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.8})
    memory.learn(epoch=2, outcome='failed', metrics={'accuracy': 0.6})
    cb = MemoryCallback(memory)
    state = cb.state_dict()

    memory2 = FileMemory(tmp_path / 'other')
    cb2 = MemoryCallback(memory2)
    cb2.load_state_dict(state)
    state2 = cb2.state_dict()
    assert len(state2['records']) == 2


class TestTrainerProperties:
  def test_trainer_optimizer_none_before_fit(self):
    trainer = Trainer()
    assert trainer.optimizer is None

  def test_trainer_regression_detected_default_false(self):
    trainer = Trainer()
    assert trainer.regression_detected is False
