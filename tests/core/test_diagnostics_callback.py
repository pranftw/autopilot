"""Tests for DiagnosticsCallback."""

from autopilot.core.artifacts.artifact import JSONArtifact, JSONLArtifact
from autopilot.core.callbacks.diagnostics import DiagnosticsCallback
from autopilot.core.diagnostics import Diagnostics
from autopilot.tracking.io import append_jsonl
from unittest.mock import MagicMock

_heatmap = JSONArtifact('node_heatmap.json', scope='epoch')
_diagnoses = JSONLArtifact('trace_diagnoses.jsonl', scope='epoch')


class TestDiagnosticsCallback:
  def test_writes_heatmap_and_diagnoses(self, tmp_path):
    epoch_dir = tmp_path / 'epoch_1'
    epoch_dir.mkdir()
    data_path = epoch_dir / 'data.jsonl'

    items = [
      {'id': 'item_1', 'success': True, 'metadata': {'category': 'syntax'}},
      {
        'id': 'item_2',
        'success': False,
        'error_message': 'parse error',
        'metadata': {'category': 'syntax'},
      },
      {
        'id': 'item_3',
        'success': False,
        'error_message': 'timeout',
        'metadata': {'category': 'network'},
      },
      {'id': 'item_1', 'success': True, 'metadata': {'category': 'syntax'}},
    ]
    for item in items:
      append_jsonl(data_path, item)

    cb = DiagnosticsCallback(Diagnostics(tmp_path))
    cb.on_train_epoch_end(trainer=MagicMock(), epoch=1)

    heatmap = _heatmap.read_raw(tmp_path, epoch=1)
    assert heatmap is not None
    assert heatmap['item_1']['total'] == 2
    assert heatmap['item_1']['failed'] == 0
    assert heatmap['item_2']['failed'] == 1
    assert heatmap['item_3']['error_rate'] == 1.0

    diagnoses = _diagnoses.read_raw(tmp_path, epoch=1)
    assert len(diagnoses) == 2
    categories = {d['category'] for d in diagnoses}
    assert 'syntax' in categories
    assert 'network' in categories

  def test_empty_data_no_artifacts(self, tmp_path):
    cb = DiagnosticsCallback(Diagnostics(tmp_path))
    cb.on_train_epoch_end(trainer=MagicMock(), epoch=1)

    heatmap = _heatmap.read_raw(tmp_path, epoch=1)
    assert heatmap is None

  def test_all_success_no_diagnoses(self, tmp_path):
    epoch_dir = tmp_path / 'epoch_1'
    epoch_dir.mkdir()
    data_path = epoch_dir / 'data.jsonl'
    append_jsonl(data_path, {'id': 'x', 'success': True, 'metadata': {}})

    cb = DiagnosticsCallback(Diagnostics(tmp_path))
    cb.on_train_epoch_end(trainer=MagicMock(), epoch=1)

    heatmap = _heatmap.read_raw(tmp_path, epoch=1)
    assert heatmap['x']['failed'] == 0

    diagnoses = _diagnoses.read_raw(tmp_path, epoch=1)
    assert len(diagnoses) == 0

  def test_state_dict_empty(self, tmp_path):
    cb = DiagnosticsCallback(Diagnostics(tmp_path))
    assert cb.state_dict() == {}

  def test_diagnostics_property(self, tmp_path):
    diag = Diagnostics(tmp_path)
    cb = DiagnosticsCallback(diag)
    assert cb.diagnostics is diag
