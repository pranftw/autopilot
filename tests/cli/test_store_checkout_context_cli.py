"""CLI integration tests for store checkout context threading (plan 01, dogfood-v8).

Verifies that CLI commands forward ``--context`` to ``store.checkout()``
so reflog entries carry audit provenance strings.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.proposal import ChangeProposal, record_proposal
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.propose import ProposeCommand
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.tree import Node
from autopilot.tracking.io import read_jsonl
from pathlib import Path
from tests.cli.conftest import make_mock_cli_context, run_cli
from unittest.mock import MagicMock, patch
import pytest


def _make_workspace_with_snapshot(
  tmp_path: Path,
  experiment_id: str = 'exp-x',
  tree_name: str = 'main',
) -> tuple[Path, FileStore]:
  """Create a full workspace with store, forest, tree, and snapshot."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  src = ws / 'src'
  src.mkdir()
  (src / 'main.py').write_text('print("hello")\n')

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  param = PathParameter(source=str(src), pattern='**/*.py')
  store = FileStore(config)
  store.register_parameters({'source': param})

  store.snapshot(experiment_id, 0, context='initial')

  exp = Experiment(experiment_id=experiment_id)
  exp.start()
  exp.complete(metrics={'accuracy': 0.9})
  node = Node(experiment=exp)

  forest = FileForest(store)
  tree = forest.create_tree(tree_name)
  tree.add(node)
  forest.save()

  return ws, store


def _read_reflog(store: FileStore) -> list[dict]:
  """Read all reflog entries from the store."""
  return read_jsonl(store.config.store_path / 'reflog.jsonl', strict=False)


class TestStoreCheckoutCLIContext:
  """CLI store checkout forwards --context to reflog."""

  def test_store_checkout_cli_forwards_context_to_reflog(self, tmp_path: Path) -> None:
    """run_cli injects --context 'test'; reflog checkout entry should have it."""
    ws, store = _make_workspace_with_snapshot(tmp_path)

    result = run_cli(
      ws,
      [
        '--experiment',
        'exp-x',
        '--epoch',
        '0',
        'store',
        'checkout',
        '--source',
        str(ws / 'src'),
      ],
    )

    assert result['ok'] is True

    entries = _read_reflog(store)
    checkout_entries = [e for e in entries if e['operation'] == 'checkout']
    assert len(checkout_entries) == 1
    assert checkout_entries[0]['context'] == 'test'

  def test_store_checkout_dry_run_no_reflog_entry(self, tmp_path: Path) -> None:
    """Dry-run checkout must not append a reflog entry."""
    ws, store = _make_workspace_with_snapshot(tmp_path)
    entries_before = len(_read_reflog(store))

    result = run_cli(
      ws,
      [
        '--experiment',
        'exp-x',
        '--epoch',
        '0',
        '--dry-run',
        'store',
        'checkout',
        '--source',
        str(ws / 'src'),
      ],
    )

    assert result['ok'] is True
    assert result['result']['dry_run'] is True

    entries_after = len(_read_reflog(store))
    assert entries_after == entries_before


class TestTreeSwitchCheckoutContext:
  """tree switch auto-checkout forwards --context."""

  def test_tree_switch_forwards_context(self, tmp_path: Path) -> None:
    """tree switch passes ctx.context to store.checkout."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    src = ws / 'src'
    src.mkdir()
    (src / 'main.py').write_text('print("hello")\n')

    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    param = PathParameter(source=str(src), pattern='*')
    store = FileStore(config)
    store.register_parameters({'source': param})

    store.snapshot('exp-alpha', 0, context='init')

    forest = FileForest(store)

    tree_alpha = forest.create_tree('alpha')
    exp_alpha = Experiment(experiment_id='exp-alpha')
    exp_alpha.start()
    exp_alpha.complete(metrics={'accuracy': 0.9})
    tree_alpha.add(Node(experiment=exp_alpha))
    tree_alpha.head = 'exp-alpha'

    tree_start = forest.create_tree('start')
    exp_start = Experiment(experiment_id='exp-start')
    exp_start.start()
    exp_start.complete(metrics={'accuracy': 0.8})
    tree_start.add(Node(experiment=exp_start))
    tree_start.head = 'exp-start'

    forest.switch('start')
    forest.save()

    with patch('autopilot.ai.store.file_store.FileStore.checkout') as mock_checkout:
      run_cli(ws, ['tree', 'switch', 'alpha'])

    mock_checkout.assert_called_once_with('exp-alpha', 0, context='test')


class TestProposeRevertCheckoutContext:
  """propose revert forwards --context to store.checkout."""

  def test_propose_revert_forwards_context(
    self, tmp_path: Path, capsys: 'pytest.CaptureFixture[str]'
  ) -> None:
    """propose revert passes ctx.context to store.checkout."""
    ctx = make_mock_cli_context(tmp_path, experiment='test-exp', epoch=0)
    ctx.context = 'revert reason'

    exp_dir = tmp_path / 'test-exp'
    exp_dir.mkdir(parents=True, exist_ok=True)
    record_proposal(
      exp_dir,
      ChangeProposal(
        proposal_id='abc',
        hypothesis='test',
        target_node='accuracy',
        change_type='rule_change',
        epoch=1,
        status='proposed',
      ),
    )

    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    (source_dir / 'data.txt').write_text('hello')

    cmd = ProposeCommand()
    args = MagicMock(
      proposal_id='abc',
      source=str(source_dir),
      store=str(tmp_path / '.store'),
      pattern='**/*',
    )

    with patch('autopilot.cli.commands.propose.FileStore') as mock_store_cls:
      mock_instance = MagicMock()
      mock_store_cls.return_value = mock_instance
      cmd.revert(ctx, args)
      mock_instance.checkout.assert_called_once_with('test-exp', 0, context='revert reason')
