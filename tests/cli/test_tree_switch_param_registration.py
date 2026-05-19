"""Tests for tree switch parameter rehydration before checkout (BUG-003).

When ``tree switch`` auto-checkouts, it must call
``register_parameters_from_latest_manifest`` before ``store.checkout``
so that checkout succeeds even when no project module is loaded (no ``-p``).

On rehydration failure, the switch succeeds with an informational advisory
rather than crashing; the user can follow up with an explicit checkout.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli
from unittest.mock import patch
import pytest


@pytest.fixture
def ws_with_manifest(tmp_path: Path) -> Path:
  """Workspace with two trees; 'alpha' HEAD has a snapshot with PathParameter.

  Creates a PathParameter, takes a snapshot for 'exp-alpha' (epoch 0),
  and switches active to 'beta'. A fresh CLI store opened without a module
  will have no registered parameters -- the rehydration step is required.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  src_dir = ws / 'src'
  src_dir.mkdir()
  (src_dir / 'main.py').write_text('print("hello")\n')

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  param = PathParameter(source=str(src_dir), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})

  store.snapshot('exp-alpha', 0)

  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  exp_a = Experiment(experiment_id='exp-alpha', hypothesis='alpha')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9})
  tree_a.add(Node(experiment=exp_a))
  tree_a.head = 'exp-alpha'

  tree_b = forest.create_tree('beta')
  exp_b = Experiment(experiment_id='exp-beta', hypothesis='beta')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.8})
  tree_b.add(Node(experiment=exp_b))
  tree_b.head = 'exp-beta'

  forest.switch('beta')
  forest.save()
  return ws


class TestTreeSwitchRehydratesParameters:
  """tree switch rehydrates parameters from manifest before checkout."""

  def test_tree_switch_rehydrates_parameters_before_checkout(self, ws_with_manifest: Path) -> None:
    """Checkout succeeds on a fresh store with no manually registered params."""
    with patch('autopilot.ai.store.file_store.FileStore.checkout') as mock_checkout:
      result = run_cli(ws_with_manifest, ['tree', 'switch', 'alpha'])

    mock_checkout.assert_called_once_with('exp-alpha', 0, context='test')
    r = result['result']
    assert r['ok'] is True
    assert r['active'] == 'alpha'
    assert r['checkout'] is True
    assert r['experiment_id'] == 'exp-alpha'
    assert r['epoch'] == 0

  def test_tree_switch_skips_checkout_when_manifest_rehydrate_fails(
    self, ws_with_manifest: Path
  ) -> None:
    """Rehydration failure produces advisory; switch still reports success."""
    with patch(
      'autopilot.cli.commands.tree.register_parameters_from_latest_manifest',
      side_effect=StoreError('no parameter schema'),
    ):
      result = run_cli(ws_with_manifest, ['tree', 'switch', 'alpha'])

    r = result['result']
    assert r['ok'] is True
    assert r['active'] == 'alpha'
    assert 'checkout' not in r

    messages = result.get('messages', [])
    advisory_found = any(
      'skipped checkout' in msg['message'] and 'rehydrate' in msg['message'] for msg in messages
    )
    assert advisory_found, f'expected rehydrate advisory in messages: {messages}'

  def test_tree_switch_skips_checkout_on_oserror_rehydrate(self, ws_with_manifest: Path) -> None:
    """OSError during rehydration also produces advisory, not crash."""
    with patch(
      'autopilot.cli.commands.tree.register_parameters_from_latest_manifest',
      side_effect=OSError('permission denied'),
    ):
      result = run_cli(ws_with_manifest, ['tree', 'switch', 'alpha'])

    r = result['result']
    assert r['ok'] is True
    assert r['active'] == 'alpha'
