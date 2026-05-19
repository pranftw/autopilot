"""CLI tests for spec_version tracking (Plan 21).

Covers:
  - experiment add --spec-version sets the field
  - experiment add without --spec-version leaves None
  - experiment add --spec-version with whitespace-only rejected
  - experiment compare warns on mismatch, no warn on match/None
  - experiment compare JSON includes spec_version section
  - query --spec-version filters correctly
  - query JSON rows include spec_version key
  - experiment status JSON includes spec_version
  - experiment show JSON includes spec_version
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context, seed_tree_with_experiments
from unittest.mock import patch
import contextlib
import io
import pytest


@pytest.fixture(autouse=True)
def _patch_store_checkout():
  """Patch FileStore.checkout for tests that don't create snapshots."""
  with patch('autopilot.ai.store.file_store.FileStore.checkout'):
    yield


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def ws_with_tree(ws: Path) -> Path:
  """Workspace with an active empty tree."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  forest.create_tree('main')
  forest.switch('main')
  forest.save()
  return ws


def _run_cli_capturing_stderr(ws: Path, argv: list[str]) -> tuple[str, str]:
  """Run a CLI command capturing both stdout and stderr text.

  Returns:
    Tuple of (stdout_text, stderr_text).
  """
  parser = build_parser()
  full_argv = [*list(argv), '--workspace', str(ws), '--context', 'test']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)

  out_buf = io.StringIO()
  err_buf = io.StringIO()
  with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
    parsed.handler(ctx, parsed)

  return out_buf.getvalue(), err_buf.getvalue()


def _seed_versioned_experiments(ws: Path) -> Path:
  """Create experiments with different spec_version values."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_a = Experiment(experiment_id='exp-v1', hypothesis='first version')
  exp_a.spec_version = 'v1'
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.8})
  tree.add(Node(experiment=exp_a))

  exp_b = Experiment(experiment_id='exp-v2', hypothesis='second version')
  exp_b.spec_version = 'v2'
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.9})
  tree.add(Node(experiment=exp_b, parent=tree.get('exp-v1')))

  exp_c = Experiment(experiment_id='exp-none', hypothesis='no version')
  exp_c.start()
  exp_c.complete(metrics={'accuracy': 0.7})
  tree.add(Node(experiment=exp_c, parent=tree.get('exp-v2')))

  exp_d = Experiment(experiment_id='exp-v1-dup', hypothesis='also v1')
  exp_d.spec_version = 'v1'
  exp_d.start()
  exp_d.complete(metrics={'accuracy': 0.85})
  tree.add(Node(experiment=exp_d, parent=tree.get('exp-none')))

  forest.save()
  return ws


class TestExperimentAddSpecVersion:
  def test_sets_spec_version(self, ws_with_tree: Path) -> None:
    result = run_cli(
      ws_with_tree,
      ['experiment', 'add', '--id', 'sv-exp', '--spec-version', 'v1.0'],
    )
    assert result['result']['ok'] is True
    assert result['result']['spec_version'] == 'v1.0'

  def test_omitted_spec_version_is_null(self, ws_with_tree: Path) -> None:
    result = run_cli(
      ws_with_tree,
      ['experiment', 'add', '--id', 'no-sv'],
    )
    assert result['result']['ok'] is True
    assert result['result']['spec_version'] is None

  def test_whitespace_only_spec_version_rejected(self, ws_with_tree: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli(
        ws_with_tree,
        ['experiment', 'add', '--id', 'bad-sv', '--spec-version', '   '],
      )

  def test_spec_version_persists_in_forest(self, ws_with_tree: Path) -> None:
    run_cli(
      ws_with_tree,
      ['experiment', 'add', '--id', 'persist-sv', '--spec-version', '2024-01'],
    )
    result = run_cli_no_context(
      ws_with_tree,
      ['experiment', 'status', 'persist-sv'],
    )
    assert result['result']['spec_version'] == '2024-01'


class TestExperimentCompareSpecVersion:
  def test_warns_on_mismatch(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    _stdout, stderr = _run_cli_capturing_stderr(ws, ['experiment', 'compare', 'exp-v1', 'exp-v2'])
    assert 'spec_version mismatch' in stderr

  def test_no_warn_when_versions_match(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    _stdout, stderr = _run_cli_capturing_stderr(
      ws, ['experiment', 'compare', 'exp-v1', 'exp-v1-dup']
    )
    assert 'spec_version mismatch' not in stderr

  def test_no_warn_when_either_is_none(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    _stdout, stderr = _run_cli_capturing_stderr(ws, ['experiment', 'compare', 'exp-v1', 'exp-none'])
    assert 'spec_version mismatch' not in stderr

  def test_no_warn_when_both_none(self, ws: Path) -> None:
    """Two experiments with spec_version=None produce no warning."""
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [
        {'id': 'a', 'status': 'completed', 'metrics': {'x': 1}},
        {'id': 'b', 'status': 'completed', 'metrics': {'x': 2}, 'parent': 'a'},
      ],
    )
    _stdout, stderr = _run_cli_capturing_stderr(ws, ['experiment', 'compare', 'a', 'b'])
    assert 'spec_version mismatch' not in stderr

  def test_json_includes_spec_version_section(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    result = run_cli_no_context(
      ws,
      ['experiment', 'compare', 'exp-v1', 'exp-v2'],
    )
    sv = result['result']['spec_version']
    assert sv['baseline'] == 'v1'
    assert sv['candidate'] == 'v2'

  def test_json_spec_version_null_when_absent(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    result = run_cli_no_context(
      ws,
      ['experiment', 'compare', 'exp-v1', 'exp-none'],
    )
    sv = result['result']['spec_version']
    assert sv['baseline'] == 'v1'
    assert sv['candidate'] is None


class TestQuerySpecVersionFilter:
  def test_filters_matching_version(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    result = run_cli_no_context(ws, ['query', '--spec-version', 'v1'])
    experiments = result['result']['experiments']
    ids = {e['id'] for e in experiments}
    assert 'exp-v1' in ids
    assert 'exp-v1-dup' in ids
    assert 'exp-v2' not in ids
    assert 'exp-none' not in ids

  def test_no_match_returns_empty(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    result = run_cli_no_context(ws, ['query', '--spec-version', 'v999'])
    assert result['result']['count'] == 0

  def test_none_experiments_never_match(self, ws: Path) -> None:
    """Experiments with spec_version=None don't match any filter value."""
    ws = _seed_versioned_experiments(ws)
    result = run_cli_no_context(ws, ['query', '--spec-version', 'None'])
    ids = {e['id'] for e in result['result']['experiments']}
    assert 'exp-none' not in ids


class TestQueryJsonIncludesSpecVersion:
  def test_render_all_includes_key(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    result = run_cli_no_context(ws, ['query'])
    for exp in result['result']['experiments']:
      assert 'spec_version' in exp

  def test_render_all_values_correct(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    result = run_cli_no_context(ws, ['query'])
    by_id = {e['id']: e for e in result['result']['experiments']}
    assert by_id['exp-v1']['spec_version'] == 'v1'
    assert by_id['exp-v2']['spec_version'] == 'v2'
    assert by_id['exp-none']['spec_version'] is None

  def test_render_best_includes_key(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    result = run_cli_no_context(ws, ['query', '--best', 'accuracy'])
    assert 'spec_version' in result['result']['best']


class TestExperimentStatusSpecVersion:
  def test_status_includes_spec_version(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    result = run_cli_no_context(ws, ['experiment', 'status', 'exp-v1'])
    assert result['result']['spec_version'] == 'v1'

  def test_status_null_when_no_version(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    result = run_cli_no_context(ws, ['experiment', 'status', 'exp-none'])
    assert result['result']['spec_version'] is None


class TestExperimentShowSpecVersion:
  def test_show_includes_spec_version(self, ws: Path) -> None:
    ws = _seed_versioned_experiments(ws)
    result = run_cli_no_context(ws, ['experiment', 'show', 'exp-v2'])
    assert result['result']['spec_version'] == 'v2'
