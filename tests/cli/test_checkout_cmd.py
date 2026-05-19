"""Tests for checkout CLI command."""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.main import AutoPilotCLI
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from io import StringIO
from pathlib import Path
from tests.cli.conftest import run_cli, seed_tree_with_experiments
from unittest.mock import patch
import json
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
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def ws_with_experiments(ws: Path) -> Path:
  """Workspace with a tree and experiments."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {
        'id': 'baseline',
        'hypothesis': 'default',
        'status': 'completed',
        'metrics': {'accuracy': 0.72},
      },
      {
        'id': 'improved',
        'hypothesis': 'better',
        'status': 'completed',
        'metrics': {'accuracy': 0.85},
        'parent': 'baseline',
        'baseline': 'baseline',
      },
    ],
  )
  return ws


class TestCheckout:
  def test_sets_tree_head(self, ws_with_experiments: Path) -> None:
    with patch('autopilot.ai.store.file_store.FileStore.checkout'):
      result = run_cli(ws_with_experiments, ['checkout', 'baseline'])
    assert result['result']['ok'] is True
    assert result['result']['experiment_id'] == 'baseline'

    config = AutoPilotConfig(workspace=ws_with_experiments)
    store = FileStore(config)
    forest = FileForest(store)
    active = forest.active
    assert active is not None
    assert active.head == 'baseline'

  def test_checkout_different_experiment(self, ws_with_experiments: Path) -> None:
    with patch('autopilot.ai.store.file_store.FileStore.checkout'):
      run_cli(ws_with_experiments, ['checkout', 'baseline'])
      result = run_cli(ws_with_experiments, ['checkout', 'improved'])
    assert result['result']['experiment_id'] == 'improved'

    config = AutoPilotConfig(workspace=ws_with_experiments)
    store = FileStore(config)
    forest = FileForest(store)
    active = forest.active
    assert active is not None
    assert active.head == 'improved'

  def test_nonexistent_id_exits_1(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'checkout',
        'ghost',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'not found' in combined

  def test_no_active_tree_exits_1(self, ws: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'checkout',
        'anything',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'no active tree' in combined

  def test_json_output(self, ws_with_experiments: Path) -> None:
    with patch('autopilot.ai.store.file_store.FileStore.checkout'):
      result = run_cli(ws_with_experiments, ['checkout', 'baseline'])
    assert 'ok' in result
    assert result['result']['ok'] is True
    assert 'status' in result['result']
    assert 'hypothesis' in result['result']

  def test_checkout_returns_experiment_details(self, ws_with_experiments: Path) -> None:
    with patch('autopilot.ai.store.file_store.FileStore.checkout'):
      result = run_cli(ws_with_experiments, ['checkout', 'baseline'])
    assert result['result']['status'] == 'completed'
    assert result['result']['hypothesis'] == 'default'

  def test_checkout_triggers_store_restoration(self, tmp_path: Path) -> None:
    """When store has snapshots, checkout updates forest HEAD."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    src = ws / 'src'
    src.mkdir()
    (src / 'main.py').write_text('version_a')

    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})

    store.snapshot('exp-a', 0)
    (src / 'main.py').write_text('version_b')
    store.snapshot('exp-a', 1)

    exp_a = Experiment(experiment_id='exp-a', hypothesis='test restore')
    exp_a.start()
    exp_a.advance_epoch()
    exp_a.complete(metrics={'accuracy': 0.9})

    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')
    tree.add(Node(experiment=exp_a))
    forest.save()

    with patch('autopilot.ai.store.file_store.FileStore.checkout'):
      run_cli(ws, ['checkout', 'exp-a'])

    config2 = AutoPilotConfig(workspace=ws)
    store2 = FileStore(config2)
    store2.register_parameters({'source': params[0]})
    forest2 = FileForest(store2)
    active2 = forest2.active
    assert active2 is not None
    assert active2.head == 'exp-a'

  def test_store_failure_exits_nonzero(self, ws_with_experiments: Path) -> None:
    """Store checkout failure produces non-zero exit and error message."""
    with patch('autopilot.core.tree.Tree.checkout', side_effect=RuntimeError('disk full')):
      exit_code, stdout, stderr = _run_full_cli(
        [
          'checkout',
          'baseline',
          '--context',
          'test',
          '--workspace',
          str(ws_with_experiments),
        ]
      )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'store checkout failed' in combined
    assert 'disk full' in combined

  def test_store_failure_json_envelope(self, ws_with_experiments: Path) -> None:
    """Store checkout failure with --json produces error envelope."""
    with patch('autopilot.core.tree.Tree.checkout', side_effect=RuntimeError('disk full')):
      exit_code, stdout, _stderr = _run_full_cli(
        [
          'checkout',
          'baseline',
          '--context',
          'test',
          '--json',
          '--workspace',
          str(ws_with_experiments),
        ]
      )
    assert exit_code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'messages' in envelope
    assert isinstance(envelope['messages'], list)
    assert len(envelope['messages']) >= 1

  def test_no_private_head_access(self) -> None:
    """Verify checkout.py does not use tree._head."""
    import autopilot.cli.commands.checkout as mod
    import inspect

    source = inspect.getsource(mod)
    assert '_head' not in source
