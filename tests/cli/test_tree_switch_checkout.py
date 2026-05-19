"""Tests for tree switch auto-checkout behavior (Plan 21 clean break).

Default ``tree switch`` auto-checkouts HEAD experiment tip from the store.
``--no-checkout`` opts out and emits the disk-state advisory.
Empty branches (latest_epoch < 0) are treated as benign no-ops (exit 0).
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.main import AutoPilotCLI
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from io import StringIO
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_text
from unittest.mock import patch
import pytest


def _run_full_cli(argv: list[str]) -> tuple[int, str, str]:
  """Run the full AutoPilotCLI with captured stdout/stderr."""
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


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  """Workspace root directory."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def ws_two_trees(ws: Path) -> Path:
  """Workspace with two trees ('alpha' and 'beta'), beta active.

  Each tree has one completed experiment with HEAD set.
  No store snapshots -- branch refs exist only when snapshots are taken.
  """
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  exp_a = Experiment(experiment_id='exp-alpha', hypothesis='alpha test')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9})
  tree_a.add(Node(experiment=exp_a))
  tree_a.head = 'exp-alpha'

  tree_b = forest.create_tree('beta')
  exp_b = Experiment(experiment_id='exp-beta', hypothesis='beta test')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.8})
  tree_b.add(Node(experiment=exp_b))
  tree_b.head = 'exp-beta'

  forest.switch('beta')
  forest.save()
  return ws


@pytest.fixture
def ws_with_snapshot(ws: Path) -> Path:
  """Workspace with two trees; 'alpha' HEAD has a store snapshot at epoch 0.

  Creates a PathParameter pointing at a source directory with a test file,
  takes a snapshot for 'exp-alpha', and switches active to 'beta'.
  """
  src_dir = ws / 'src'
  src_dir.mkdir()
  (src_dir / 'main.py').write_text('print("alpha")')

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  param = PathParameter(source=str(src_dir), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})

  store.snapshot('exp-alpha', 0)

  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  exp_a = Experiment(experiment_id='exp-alpha', hypothesis='alpha test')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9})
  tree_a.add(Node(experiment=exp_a))
  tree_a.head = 'exp-alpha'

  tree_b = forest.create_tree('beta')
  exp_b = Experiment(experiment_id='exp-beta', hypothesis='beta test')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.8})
  tree_b.add(Node(experiment=exp_b))
  tree_b.head = 'exp-beta'

  forest.switch('beta')
  forest.save()
  return ws


class TestTreeSwitchAutoCheckout:
  """Default tree switch triggers store checkout (Plan 21 default)."""

  def test_tree_switch_auto_checkouts_by_default(self, ws_with_snapshot: Path) -> None:
    """Default tree switch triggers store.checkout exactly once."""
    with patch('autopilot.ai.store.file_store.FileStore.checkout') as mock_checkout:
      result = run_cli(ws_with_snapshot, ['tree', 'switch', 'alpha'])

    mock_checkout.assert_called_once_with('exp-alpha', 0, context='test')
    r = result['result']
    assert r['ok'] is True
    assert r['active'] == 'alpha'
    assert r['checkout'] is True
    assert r['experiment_id'] == 'exp-alpha'
    assert r['epoch'] == 0

  def test_tree_switch_default_success_no_advisory(self, ws_with_snapshot: Path) -> None:
    """Default tree switch does not emit DISK_STATE_ADVISORY."""
    with patch('autopilot.ai.store.file_store.FileStore.checkout'):
      text = run_cli_text(ws_with_snapshot, ['tree', 'switch', 'alpha'])
    assert 'does not sync working tree files' not in text
    assert 'checked out experiment' in text

  def test_tree_switch_default_json_no_advisory(self, ws_with_snapshot: Path) -> None:
    """Default tree switch JSON messages array has no disk-state advisory."""
    with patch('autopilot.ai.store.file_store.FileStore.checkout'):
      result = run_cli(ws_with_snapshot, ['tree', 'switch', 'alpha'])
    messages = result.get('messages', [])
    for msg in messages:
      text = msg['message'] if isinstance(msg, dict) else str(msg)
      assert 'does not sync' not in text

  def test_tree_switch_checkout_no_head_fails(self, ws: Path) -> None:
    """When target tree has no HEAD, exit non-zero with guidance."""
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    forest.create_tree('empty-tree')
    forest.create_tree('start-tree')
    forest.switch('start-tree')
    forest.save()

    exit_code, stdout, stderr = _run_full_cli(
      [
        'tree',
        'switch',
        'empty-tree',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'no HEAD experiment' in combined

  def test_tree_switch_checkout_store_error(self, ws_two_trees: Path) -> None:
    """When store.checkout raises, error includes actionable guidance."""
    with (
      patch(
        'autopilot.ai.store.file_store.FileStore.load_refs',
        return_value={
          'branches': {'exp-alpha': {'latest_epoch': 0, 'parent_id': None, 'parent_epoch': None}}
        },
      ),
      patch(
        'autopilot.cli.commands.tree.register_parameters_from_latest_manifest',
      ),
      patch(
        'autopilot.ai.store.file_store.FileStore.checkout',
        side_effect=StoreError('snapshot not found'),
      ),
    ):
      exit_code, stdout, stderr = _run_full_cli(
        [
          'tree',
          'switch',
          'alpha',
          '--context',
          'test',
          '--workspace',
          str(ws_two_trees),
        ]
      )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'checkout failed' in combined
    assert 'snapshot not found' in combined


class TestTreeSwitchNoCheckout:
  """--no-checkout skips checkout and emits advisory."""

  def test_tree_switch_no_checkout_skips(self, ws_two_trees: Path) -> None:
    """--no-checkout skips checkout call path and emits DISK_STATE_ADVISORY."""
    text = run_cli_text(ws_two_trees, ['tree', 'switch', 'alpha', '--no-checkout'])
    assert 'does not sync working tree files' in text

  def test_tree_switch_no_checkout_json_includes_advisory(self, ws_two_trees: Path) -> None:
    """--json envelope messages array contains disk-state advisory."""
    result = run_cli(ws_two_trees, ['tree', 'switch', 'alpha', '--no-checkout'])
    r = result['result']
    assert r['ok'] is True
    assert r['active'] == 'alpha'
    assert 'checkout' not in r
    messages = result.get('messages', [])
    assert any('does not sync' in msg['message'] for msg in messages)

  def test_no_checkout_with_bind_fails(self, ws_two_trees: Path) -> None:
    """--bind combined with --no-checkout is an error."""
    exit_code, stdout, stderr = _run_full_cli(
      [
        'tree',
        'switch',
        'alpha',
        '--no-checkout',
        '--bind',
        '--context',
        'test',
        '--workspace',
        str(ws_two_trees),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert '--bind requires checkout' in combined


class TestTreeSwitchCheckoutNoSnapshot:
  """Empty branch (latest_epoch < 0) is a benign no-op."""

  def test_tree_switch_checkout_no_snapshot_no_op(self, ws: Path) -> None:
    """Branch with latest_epoch=-1: exit 0, no store.checkout, advisory present."""
    src_dir = ws / 'src'
    src_dir.mkdir()
    (src_dir / 'main.py').write_text('v1')

    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    param = PathParameter(source=str(src_dir), pattern='*')
    store = FileStore(config)
    store.register_parameters({'source': param})

    store.snapshot('exp-reset', 0)
    store.reset_branch('exp-reset')

    forest = FileForest(store)
    tree_a = forest.create_tree('target')
    exp = Experiment(experiment_id='exp-reset', hypothesis='test')
    exp.start()
    exp.complete(metrics={'m': 1})
    tree_a.add(Node(experiment=exp))
    tree_a.head = 'exp-reset'

    forest.create_tree('other')
    forest.switch('other')
    forest.save()

    with patch('autopilot.ai.store.file_store.FileStore.checkout') as mock_checkout:
      exit_code, stdout, stderr = _run_full_cli(
        [
          'tree',
          'switch',
          'target',
          '--context',
          'test',
          '--workspace',
          str(ws),
          '--json',
        ]
      )

    assert exit_code == 0
    mock_checkout.assert_not_called()
    combined = stdout + stderr
    assert 'skipped checkout: no snapshots on branch' in combined

  def test_no_snapshot_no_store_branch(self, ws_two_trees: Path) -> None:
    """When HEAD experiment has no store branch at all, exits non-zero."""
    exit_code, stdout, stderr = _run_full_cli(
      [
        'tree',
        'switch',
        'alpha',
        '--context',
        'test',
        '--workspace',
        str(ws_two_trees),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'no store branch' in combined or 'store create' in combined
