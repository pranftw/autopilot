"""Tests for fingerprint drift tri-state null semantics (Plan 12).

Covers:
  - Direct ``_detect_fingerprint_drift`` helper with tri-state return.
  - CLI JSON integration for ``dataset_fingerprint_drift`` null/true values.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.experiment.compare import _detect_fingerprint_drift
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
import pytest

# -- fixtures shared with test_fingerprint_binding.py --


@pytest.fixture
def fp_workspace(tmp_path: Path) -> Path:
  """Workspace root with store dir created."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def fp_forest(fp_workspace: Path) -> FileForest:
  """Forest with an active tree named 'main'."""
  config = AutoPilotConfig(workspace=fp_workspace)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  forest.create_tree('main')
  forest.switch('main')
  forest.save()
  return forest


FINGERPRINT_A = {
  'paths': ['/data.jsonl'],
  'hashes': ['aaa111'],
  'bundle_hash': 'a' * 64,
  'timestamp': 't1',
}

FINGERPRINT_B = {
  'paths': ['/data.jsonl'],
  'hashes': ['bbb222'],
  'bundle_hash': 'b' * 64,
  'timestamp': 't2',
}


def _make_experiment(
  eid: str,
  fingerprint: dict | None = None,
  metrics: dict | None = None,
) -> Experiment:
  """Build a completed experiment with optional fingerprint."""
  exp = Experiment(experiment_id=eid, hypothesis=f'{eid} hypothesis')
  exp.start()
  exp.complete(metrics=metrics or {'accuracy': 0.9})
  if fingerprint is not None:
    exp.dataset_meta['dataset_fingerprint'] = fingerprint
  return exp


def _seed_experiments_with_fingerprints(
  fp_forest: FileForest,
  fp_a: dict | None,
  fp_b: dict | None,
) -> tuple[str, str]:
  """Seed two experiments with optional fingerprint data."""
  tree = fp_forest.active
  assert tree is not None

  exp_a = _make_experiment('cmp-a', fingerprint=fp_a)
  exp_b = _make_experiment('cmp-b', fingerprint=fp_b, metrics={'accuracy': 0.85})

  tree.add(Node(experiment=exp_a))
  tree.add(Node(experiment=exp_b))
  fp_forest.save()

  return 'cmp-a', 'cmp-b'


# -- 4.1 Direct drift helper tests --


def test_drift_both_present_no_drift():
  """Two experiments with identical fingerprints return False."""
  exp_a = _make_experiment('a', fingerprint=FINGERPRINT_A)
  exp_b = _make_experiment('b', fingerprint=FINGERPRINT_A)

  result = _detect_fingerprint_drift(exp_a, exp_b)
  assert result is False


def test_drift_both_present_drift_detected():
  """Two experiments with different fingerprints return True."""
  exp_a = _make_experiment('a', fingerprint=FINGERPRINT_A)
  exp_b = _make_experiment('b', fingerprint=FINGERPRINT_B)

  result = _detect_fingerprint_drift(exp_a, exp_b)
  assert result is True


def test_drift_one_missing_returns_null():
  """One side missing fingerprint returns None."""
  exp_a = _make_experiment('a', fingerprint=FINGERPRINT_A)
  exp_b = _make_experiment('b', fingerprint=None)

  result = _detect_fingerprint_drift(exp_a, exp_b)
  assert result is None

  result_reversed = _detect_fingerprint_drift(exp_b, exp_a)
  assert result_reversed is None


def test_drift_both_missing_returns_null():
  """Neither side has fingerprint; returns None."""
  exp_a = _make_experiment('a', fingerprint=None)
  exp_b = _make_experiment('b', fingerprint=None)

  result = _detect_fingerprint_drift(exp_a, exp_b)
  assert result is None


def test_drift_empty_dict_returns_null():
  """Empty dict fingerprint treated as missing; returns None."""
  exp_a = _make_experiment('a', fingerprint=FINGERPRINT_A)
  exp_b = _make_experiment('b')
  exp_b.dataset_meta['dataset_fingerprint'] = {}

  result = _detect_fingerprint_drift(exp_a, exp_b)
  assert result is None

  result_reversed = _detect_fingerprint_drift(exp_b, exp_a)
  assert result_reversed is None


# -- 4.2 CLI JSON integration --


def test_compare_json_drift_null(fp_workspace, fp_forest):
  """Compare JSON includes dataset_fingerprint_drift as None (null) when missing."""
  _seed_experiments_with_fingerprints(fp_forest, FINGERPRINT_A, None)

  output = run_cli_no_context(
    fp_workspace,
    ['experiment', 'compare', 'cmp-a', 'cmp-b'],
  )
  result = output['result']
  assert result['dataset_fingerprint_drift'] is None


def test_compare_json_drift_true(fp_workspace, fp_forest):
  """Compare JSON includes dataset_fingerprint_drift as True when differing."""
  _seed_experiments_with_fingerprints(fp_forest, FINGERPRINT_A, FINGERPRINT_B)

  output = run_cli_no_context(
    fp_workspace,
    ['experiment', 'compare', 'cmp-a', 'cmp-b'],
  )
  result = output['result']
  assert result['dataset_fingerprint_drift'] is True
