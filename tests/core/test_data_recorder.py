"""Tests for DataRecorderCallback."""

from autopilot.core.callbacks.data_recorder import DataRecorderCallback
from autopilot.core.types import Datum
from unittest.mock import MagicMock


class TestDataRecorderCallback:
  def test_accumulates_batch_data(self, tmp_path):
    cb = DataRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    for i in range(5):
      cb.on_train_batch_end(trainer=trainer, batch_idx=i, data=Datum(success=True))
    assert len(cb._batch_data) == 5

  def test_writes_data_jsonl(self, tmp_path):
    cb = DataRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    for i in range(3):
      cb.on_train_batch_end(trainer=trainer, batch_idx=i, data=Datum(success=True))
    cb.on_train_epoch_end(trainer=trainer, epoch=1)
    data_file = tmp_path / 'epoch_1' / 'data.jsonl'
    assert data_file.exists()
    lines = data_file.read_text().strip().splitlines()
    assert len(lines) == 3

  def test_no_epoch_metrics_produced(self, tmp_path):
    cb = DataRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    cb.on_train_batch_end(trainer=trainer, batch_idx=0, data=Datum(success=True))
    cb.on_train_epoch_end(trainer=trainer, epoch=1)
    assert not (tmp_path / 'epoch_1' / 'epoch_metrics.json').exists()
    assert not (tmp_path / 'epoch_1' / 'delta_metrics.json').exists()

  def test_resets_between_epochs(self, tmp_path):
    cb = DataRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    cb.on_train_batch_end(trainer=trainer, data=Datum(success=True))
    cb.on_train_epoch_end(trainer=trainer, epoch=1)
    cb.on_train_epoch_start(trainer=trainer, epoch=2)
    assert cb._batch_data == []

  def test_empty_epoch(self, tmp_path):
    cb = DataRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    cb.on_train_epoch_end(trainer=trainer, epoch=1)
    data_file = tmp_path / 'epoch_1' / 'data.jsonl'
    assert not data_file.exists() or data_file.read_text().strip() == ''

  def test_epoch_dir_created(self, tmp_path):
    cb = DataRecorderCallback(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=3)
    cb.on_train_batch_end(trainer=trainer, data=Datum(success=True))
    cb.on_train_epoch_end(trainer=trainer, epoch=3)
    assert (tmp_path / 'epoch_3').is_dir()

  def test_serialize_item_datum(self, tmp_path):
    cb = DataRecorderCallback(tmp_path)
    result = cb.serialize_item(Datum(success=True))
    assert isinstance(result, dict)
    assert result['success'] is True

  def test_serialize_item_dict(self, tmp_path):
    cb = DataRecorderCallback(tmp_path)
    result = cb.serialize_item({'key': 'value'})
    assert result == {'key': 'value'}

  def test_serialize_item_none_for_unsupported(self, tmp_path):
    cb = DataRecorderCallback(tmp_path)
    assert cb.serialize_item(42) is None

  def test_serialize_item_override(self, tmp_path):
    class Custom(DataRecorderCallback):
      def serialize_item(self, data):
        return {'custom': True}

    cb = Custom(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    cb.on_train_batch_end(trainer=trainer, data='anything')
    cb.on_train_epoch_end(trainer=trainer, epoch=1)
    data_file = tmp_path / 'epoch_1' / 'data.jsonl'
    assert data_file.exists()

  def test_serialize_item_returns_none_skips(self, tmp_path):
    class SkipAll(DataRecorderCallback):
      def serialize_item(self, data):
        return None

    cb = SkipAll(tmp_path)
    trainer = MagicMock()
    cb.on_train_epoch_start(trainer=trainer, epoch=1)
    cb.on_train_batch_end(trainer=trainer, data=Datum(success=True))
    assert len(cb._batch_data) == 0

  def test_artifact_registration(self, tmp_path):
    cb = DataRecorderCallback(tmp_path)
    assert 'data_artifact' in cb.artifacts

  def test_state_dict_empty(self, tmp_path):
    cb = DataRecorderCallback(tmp_path)
    assert cb.state_dict() == {}
