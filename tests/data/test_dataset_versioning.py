"""Integration tests for dataset versioning across Experiment and DataModule."""

from autopilot.ai.fingerprint import DatasetFingerprint, compute_fingerprint
from autopilot.core.experiment import Experiment
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from pathlib import Path
from typing import Any


class TestExperimentDatasetMeta:
  """Experiment.dataset_meta persistence round-trip."""

  def test_dataset_meta_state_dict_round_trip(self) -> None:
    """dataset_meta set with nested keys persists via state_dict / load_state_dict."""
    exp = Experiment('exp-1', hypothesis='fingerprint test')
    fp = DatasetFingerprint(
      paths=['/data/train.jsonl', '/data/val.jsonl'],
      hashes=['aaa111', 'bbb222'],
      bundle_hash='ccc333',
      timestamp='2026-05-04T10:00:00+00:00',
    )
    exp.dataset_meta = fp.to_dict()

    state = exp.state_dict()
    assert 'dataset_meta' in state
    assert state['dataset_meta']['paths'] == ['/data/train.jsonl', '/data/val.jsonl']
    assert state['dataset_meta']['bundle_hash'] == 'ccc333'

    exp2 = Experiment('placeholder')
    exp2.load_state_dict(state)
    assert exp2.dataset_meta == exp.dataset_meta
    assert exp2.dataset_meta['hashes'] == ['aaa111', 'bbb222']

  def test_dataset_meta_default_empty(self) -> None:
    """Default dataset_meta is an empty dict and round-trips cleanly."""
    exp = Experiment('exp-2')
    assert exp.dataset_meta == {}
    state = exp.state_dict()
    assert state['dataset_meta'] == {}
    exp2 = Experiment('placeholder')
    exp2.load_state_dict(state)
    assert exp2.dataset_meta == {}

  def test_dataset_meta_backward_compat_missing_key(self) -> None:
    """load_state_dict without 'dataset_meta' key defaults to empty dict."""
    exp = Experiment('exp-3')
    state = exp.state_dict()
    del state['dataset_meta']
    exp2 = Experiment('placeholder')
    exp2.load_state_dict(state)
    assert exp2.dataset_meta == {}


class TestDataModuleFingerprintIntegration:
  """DataModule state_dict includes fingerprint when subclass sets it."""

  def test_fingerprint_in_state_dict(self, tmp_path: Path) -> None:
    """DataModule.state_dict() includes fingerprint when set."""
    data_file = tmp_path / 'corpus.jsonl'
    data_file.write_text('{"text": "hello"}\n{"text": "world"}\n', encoding='utf-8')

    class _FingerprintedDM(DataModule):
      def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.dataset_fingerprint = compute_fingerprint([data_path])

      def train_dataloader(self) -> DataLoader:
        return DataLoader([{'text': 'hello'}], batch_size=1)

    dm = _FingerprintedDM(data_file)
    state = dm.state_dict()
    assert 'dataset_fingerprint' in state
    assert len(state['dataset_fingerprint']['hashes']) == 1
    assert state['dataset_fingerprint']['bundle_hash'] is not None

    dm2 = DataModule()
    dm2.load_state_dict(state)
    assert dm2.dataset_fingerprint is not None
    assert dm.dataset_fingerprint is not None
    assert dm2.dataset_fingerprint.hashes == dm.dataset_fingerprint.hashes
    assert dm2.dataset_fingerprint.bundle_hash == dm.dataset_fingerprint.bundle_hash

  def test_no_fingerprint_state_dict_empty(self) -> None:
    """DataModule without fingerprint has empty state_dict."""
    dm = DataModule()
    assert dm.state_dict() == {}

  def test_load_state_dict_without_fingerprint_clears(self) -> None:
    """Loading state without fingerprint key clears any existing fingerprint."""
    dm = DataModule()
    dm.dataset_fingerprint = DatasetFingerprint(paths=['p'], hashes=['h'])
    dm.load_state_dict({})
    assert dm.dataset_fingerprint is None

  def test_subclass_state_dict_with_super(self, tmp_path: Path) -> None:
    """Subclass calling super().state_dict() preserves fingerprint alongside custom state."""

    class _CustomDM(DataModule):
      def __init__(self) -> None:
        self.counter = 0

      def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result['counter'] = self.counter
        return result

      def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.counter = state.get('counter', 0)

    dm = _CustomDM()
    dm.counter = 5
    dm.dataset_fingerprint = DatasetFingerprint(
      paths=['/data/train.jsonl'],
      hashes=['abc123'],
      bundle_hash='def456',
    )
    state = dm.state_dict()
    assert state['counter'] == 5
    assert 'dataset_fingerprint' in state

    dm2 = _CustomDM()
    dm2.load_state_dict(state)
    assert dm2.counter == 5
    assert dm2.dataset_fingerprint is not None
    assert dm2.dataset_fingerprint.bundle_hash == 'def456'
