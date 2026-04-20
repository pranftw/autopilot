"""Tests for DiagnoseCallback."""

from autopilot.core.stage_callbacks import DiagnoseCallback
from autopilot.core.stage_io import read_epoch_artifact, read_epoch_artifact_lines
from autopilot.tracking.io import append_jsonl
from unittest.mock import MagicMock


class TestDiagnoseCallback:
  def test_writes_heatmap_and_diagnoses(self, tmp_path):
    epoch_dir = tmp_path / 'epoch_1'
    epoch_dir.mkdir()
    data_path = epoch_dir / 'data.jsonl'

    items = [
      {'item_id': 'item_1', 'success': True, 'metadata': {'category': 'syntax'}},
      {
        'item_id': 'item_2',
        'success': False,
        'error_message': 'parse error',
        'metadata': {'category': 'syntax'},
      },
      {
        'item_id': 'item_3',
        'success': False,
        'error_message': 'timeout',
        'metadata': {'category': 'network'},
      },
      {'item_id': 'item_1', 'success': True, 'metadata': {'category': 'syntax'}},
    ]
    for item in items:
      append_jsonl(data_path, item)

    cb = DiagnoseCallback(tmp_path)
    cb.on_train_epoch_end(trainer=MagicMock(), epoch=1)

    heatmap = read_epoch_artifact(tmp_path, 1, 'node_heatmap.json')
    assert heatmap is not None
    assert heatmap['item_1']['total'] == 2
    assert heatmap['item_1']['failed'] == 0
    assert heatmap['item_2']['failed'] == 1
    assert heatmap['item_3']['error_rate'] == 1.0

    diagnoses = read_epoch_artifact_lines(tmp_path, 1, 'trace_diagnoses.jsonl')
    assert len(diagnoses) == 2
    categories = {d['category'] for d in diagnoses}
    assert 'syntax' in categories
    assert 'network' in categories

  def test_empty_data_no_artifacts(self, tmp_path):
    cb = DiagnoseCallback(tmp_path)
    cb.on_train_epoch_end(trainer=MagicMock(), epoch=1)

    heatmap = read_epoch_artifact(tmp_path, 1, 'node_heatmap.json')
    assert heatmap is None

  def test_all_success_no_diagnoses(self, tmp_path):
    epoch_dir = tmp_path / 'epoch_1'
    epoch_dir.mkdir()
    data_path = epoch_dir / 'data.jsonl'
    append_jsonl(data_path, {'item_id': 'x', 'success': True, 'metadata': {}})

    cb = DiagnoseCallback(tmp_path)
    cb.on_train_epoch_end(trainer=MagicMock(), epoch=1)

    heatmap = read_epoch_artifact(tmp_path, 1, 'node_heatmap.json')
    assert heatmap['x']['failed'] == 0

    diagnoses = read_epoch_artifact_lines(tmp_path, 1, 'trace_diagnoses.jsonl')
    assert len(diagnoses) == 0

  def test_state_dict_empty(self, tmp_path):
    cb = DiagnoseCallback(tmp_path)
    assert cb.state_dict() == {}
