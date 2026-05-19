"""CLI error path tests.

Covers:
- Commands with missing required args -> exit 2 (argparse) or exit 1 (runtime)
- Commands outside workspace (no .autopilot/ dir) -> error and exit 1
- Nonexistent experiment IDs -> exit 1 with 'not found'
- Nonexistent experiment with --json -> JSON error envelope
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.main import AutoPilotCLI
from autopilot.core.config import AutoPilotConfig
from io import StringIO
from pathlib import Path
from tests.cli.conftest import seed_tree_with_experiments
from unittest.mock import patch
import json
import pytest


def _run_full_cli(argv: list[str]) -> tuple[int, str, str]:
  """Run the full AutoPilotCLI with captured stdout/stderr.

  Returns (exit_code, stdout, stderr).
  """
  cli = AutoPilotCLI()
  out = StringIO()
  err = StringIO()
  exit_code = 0
  with patch('sys.stdout', out), patch('sys.stderr', err):
    try:
      cli(argv=argv)
    except SystemExit as e:
      exit_code = int(e.code) if e.code is not None else 0
  return exit_code, out.getvalue(), err.getvalue()


# -- Fixtures --


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def ws_with_experiments(ws: Path) -> Path:
  """Workspace with tree and completed experiments."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {
        'id': 'exp-a',
        'hypothesis': 'baseline',
        'status': 'completed',
        'metrics': {'accuracy': 0.72},
      },
      {
        'id': 'exp-b',
        'hypothesis': 'improved',
        'status': 'completed',
        'metrics': {'accuracy': 0.85},
        'parent': 'exp-a',
        'baseline': 'exp-a',
      },
    ],
  )
  return ws


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
  """Empty directory with no .autopilot/ workspace."""
  d = tmp_path / 'empty'
  d.mkdir()
  return d


# -- Missing required args -> exit 2 --


class TestMissingRequiredArgs:
  """Argparse-level failures: missing required positional or option."""

  def test_experiment_add_without_hypothesis_is_valid_args(self, ws: Path) -> None:
    exit_code, _, _ = _run_full_cli(
      [
        'experiment',
        'add',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code != 2

  def test_tree_create_missing_name(self, ws: Path) -> None:
    exit_code, _, _ = _run_full_cli(
      [
        'tree',
        'create',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 2

  def test_checkout_missing_experiment_id(self, ws: Path) -> None:
    exit_code, _, _ = _run_full_cli(
      [
        'checkout',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 2

  def test_stabilize_missing_experiment_id(self, ws: Path) -> None:
    exit_code, _, _ = _run_full_cli(
      [
        'stabilize',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 2

  def test_experiment_status_no_id_no_head(self, ws: Path) -> None:
    """experiment status without ID when no HEAD is set.

    The 'id' arg is nargs='?', so argparse accepts it. Runtime raises
    'no experiment specified and no HEAD set' -> caught by dispatch -> exit 1.
    """
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('main')
    forest.switch('main')
    forest.save()

    exit_code, stdout, stderr = _run_full_cli(
      [
        'experiment',
        'status',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'no experiment specified' in combined or 'HEAD' in combined

  def test_store_snapshot_missing_experiment(self, ws: Path) -> None:
    """store snapshot without --experiment is a runtime error (exit 1)."""
    src = ws / 'src'
    src.mkdir()
    (src / 'test.py').write_text('x')
    exit_code, _, _stderr = _run_full_cli(
      [
        'store',
        'snapshot',
        '--context',
        'test',
        '--workspace',
        str(ws),
        '--source',
        str(src),
      ]
    )
    assert exit_code == 1


# -- Commands outside workspace -> exit 1 --


class TestOutsideWorkspace:
  """Commands run from a directory with no workspace context."""

  def test_experiment_add_no_active_tree(self, empty_dir: Path) -> None:
    """Outside workspace, experiment add fails (no active tree)."""
    exit_code, stdout, stderr = _run_full_cli(
      [
        'experiment',
        'add',
        '--hypothesis',
        'test',
        '--id',
        'x',
        '--context',
        'test',
        '--workspace',
        str(empty_dir),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'no active tree' in combined or 'workspace' in combined.lower()

  def test_experiment_status_no_active_tree(self, empty_dir: Path) -> None:
    exit_code, _stdout, _stderr = _run_full_cli(
      [
        'experiment',
        'status',
        'some-id',
        '--context',
        'test',
        '--workspace',
        str(empty_dir),
      ]
    )
    assert exit_code == 1

  def test_tree_list_empty_from_nowhere(self, empty_dir: Path) -> None:
    """tree list from empty dir returns empty list, not an error."""
    exit_code, stdout, _ = _run_full_cli(
      [
        'tree',
        'list',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(empty_dir),
      ]
    )
    assert exit_code == 0
    envelope = json.loads(stdout)
    assert envelope['ok'] is True
    assert envelope['result']['trees'] == []

  def test_query_no_active_tree(self, empty_dir: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'query',
        '--completed',
        '--context',
        'test',
        '--workspace',
        str(empty_dir),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'no active tree' in combined

  def test_checkout_no_active_tree(self, empty_dir: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'checkout',
        'some-id',
        '--context',
        'test',
        '--workspace',
        str(empty_dir),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'no active tree' in combined

  def test_query_no_active_tree_json(self, empty_dir: Path) -> None:
    """--json on error produces JSON envelope."""
    exit_code, stdout, _ = _run_full_cli(
      [
        'query',
        '--completed',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(empty_dir),
      ]
    )
    assert exit_code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'no active tree' in envelope['error']

  def test_experiment_add_no_tree_json(self, empty_dir: Path) -> None:
    exit_code, stdout, _ = _run_full_cli(
      [
        'experiment',
        'add',
        '--hypothesis',
        'test',
        '--id',
        'x',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(empty_dir),
      ]
    )
    assert exit_code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'no active tree' in envelope['error']


# -- Nonexistent experiment IDs -> exit 1 with 'not found' --


class TestNonexistentExperiment:
  def test_experiment_status_not_found(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'experiment',
        'status',
        'nonexistent-id-999',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'not found' in combined

  def test_checkout_not_found(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'checkout',
        'nonexistent-id-999',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'not found' in combined

  def test_stabilize_not_found(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'stabilize',
        'nonexistent-id-999',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'not found' in combined

  def test_experiment_compare_not_found(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'experiment',
        'compare',
        'exp-a',
        'nonexistent',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'not found' in combined

  def test_tree_switch_not_found(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'tree',
        'switch',
        'nonexistent-tree',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'not found' in combined


# -- Commands with --json on error -> JSON envelope {'ok': false, 'error': '...'} --


class TestJsonOnError:
  def test_experiment_status_not_found_json(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, _ = _run_full_cli(
      [
        'experiment',
        'status',
        'nonexistent-id-999',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert exit_code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'not found' in envelope['error']
    assert 'messages' in envelope

  def test_checkout_not_found_json(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, _ = _run_full_cli(
      [
        'checkout',
        'nonexistent-id-999',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert exit_code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'not found' in envelope['error']

  def test_stabilize_not_found_json(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, _ = _run_full_cli(
      [
        'stabilize',
        'nonexistent-id-999',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert exit_code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'not found' in envelope['error']

  def test_tree_switch_not_found_json(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, _ = _run_full_cli(
      [
        'tree',
        'switch',
        'nonexistent-tree',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert exit_code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'not found' in envelope['error']

  def test_experiment_compare_not_found_json(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, _ = _run_full_cli(
      [
        'experiment',
        'compare',
        'exp-a',
        'nonexistent',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert exit_code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'not found' in envelope['error']

  def test_no_active_tree_json(self, empty_dir: Path) -> None:
    """Error envelope for no active tree."""
    exit_code, stdout, _ = _run_full_cli(
      [
        'checkout',
        'any-id',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(empty_dir),
      ]
    )
    assert exit_code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'no active tree' in envelope['error']

  def test_experiment_status_no_head_json(self, ws: Path) -> None:
    """experiment status with no ID and no HEAD, with --json."""
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('main')
    forest.switch('main')
    forest.save()

    exit_code, stdout, _ = _run_full_cli(
      [
        'experiment',
        'status',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'HEAD' in envelope['error'] or 'no experiment' in envelope['error']


# -- Stabilize-specific error paths --


class TestStabilizeErrors:
  def test_stabilize_non_completed_exits_1(self, ws: Path) -> None:
    """Stabilize a non-completed experiment -> exit 1."""
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [
        {'id': 'running-exp', 'hypothesis': 'test', 'status': 'running'},
      ],
    )

    exit_code, stdout, stderr = _run_full_cli(
      [
        'stabilize',
        'running-exp',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'not completed' in combined

  def test_stabilize_non_completed_json(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [
        {'id': 'failed-exp', 'hypothesis': 'test', 'status': 'failed', 'error': 'OOM'},
      ],
    )

    exit_code, stdout, _ = _run_full_cli(
      [
        'stabilize',
        'failed-exp',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False


# -- Edge cases --


class TestEdgeCases:
  def test_duplicate_tree_create(self, ws: Path) -> None:
    """Creating a tree with duplicate name -> exit 1."""
    exit_code, _, _ = _run_full_cli(
      [
        'tree',
        'create',
        'dupe',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 0

    exit_code, _, stderr = _run_full_cli(
      [
        'tree',
        'create',
        'dupe',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 1
    assert 'already exists' in stderr

  def test_duplicate_experiment_id(self, ws: Path) -> None:
    """Adding experiment with duplicate ID -> exit 1."""
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('main')
    forest.switch('main')
    forest.save()

    exit_code, _, _ = _run_full_cli(
      [
        'experiment',
        'add',
        '--hypothesis',
        'first',
        '--id',
        'dup-exp',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 0

    exit_code, _, stderr = _run_full_cli(
      [
        'experiment',
        'add',
        '--hypothesis',
        'second',
        '--id',
        'dup-exp',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 1
    assert 'duplicate' in stderr.lower()

  def test_parent_not_found(self, ws: Path) -> None:
    """experiment add with nonexistent parent -> exit 1."""
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('main')
    forest.switch('main')
    forest.save()

    exit_code, _, stderr = _run_full_cli(
      [
        'experiment',
        'add',
        '--hypothesis',
        'child',
        '--id',
        'child-exp',
        '--parent',
        'nonexistent-parent',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 1
    assert 'not found' in stderr
