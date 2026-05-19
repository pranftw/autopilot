"""BUG-DFV1-007: error message quality tests.

Verifies that CLI error paths include contextual guidance (type names,
offending values, and "what to do next" suggestions) instead of bare
exception passthrough.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.proposal import ChangeProposal, record_proposal
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.checkout import CheckoutCommand
from autopilot.cli.commands.optimize import OptimizeCommand
from autopilot.cli.commands.propose import ProposeCommand
from autopilot.cli.commands.stabilize import StabilizeCommand
from autopilot.cli.commands.tree import TreeRemove
from autopilot.cli.context import CLIContext
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, seed_tree_with_experiments
from unittest.mock import MagicMock, patch
import argparse
import pytest


def _ws_ctx(tmp_path: Path, *, use_json: bool = False) -> MagicMock:
  """Build a minimal mock CLIContext backed by *tmp_path*."""
  config = AutoPilotConfig(workspace=tmp_path)
  ctx = MagicMock()
  ctx.workspace = tmp_path
  ctx.project = None
  ctx.config = config
  ctx.output = Output(use_json=use_json)
  ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
  ctx.context = 'test'
  ctx.wait_timeout_ms = None
  return ctx


class TestTreeRemoveErrorSuggestsTreeList:
  """BUG-DFV1-007: tree remove error should suggest tree list."""

  def test_nonexistent_tree_suggests_tree_list(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path, use_json=False)
    ctx.config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(ctx.config)
    forest = FileForest(store)
    forest.create_tree('existing')
    forest.switch('existing')
    forest.save()

    cmd = TreeRemove()
    args = argparse.Namespace(name='nonexistent')
    with pytest.raises(SystemExit):
      cmd.forward(ctx, args)
    captured = capsys.readouterr()
    assert 'tree list' in captured.err

  def test_via_run_cli(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('main')
    forest.switch('main')
    forest.save()

    with pytest.raises(SystemExit):
      run_cli(ws, ['tree', 'remove', 'nonexistent'])


class TestStabilizeNotFoundSuggestsQuery:
  """BUG-DFV1-007: stabilize not-found should suggest query."""

  def test_missing_experiment_suggests_query(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path, use_json=False)
    ctx.config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(ctx.config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [{'id': 'real-exp', 'hypothesis': 'h', 'status': 'completed', 'metrics': {'x': 1.0}}],
    )

    cmd = StabilizeCommand()
    args = argparse.Namespace(experiment_id='nonexistent', parameter_prefix=None)
    with pytest.raises(SystemExit):
      cmd.forward(ctx, args)
    captured = capsys.readouterr()
    assert 'query' in captured.err
    assert 'not found' in captured.err


class TestOptimizeValuesJsonErrorIncludesHint:
  """BUG-DFV1-007: optimize --values JSON error should suggest expected format."""

  def test_bad_json_includes_format_hint(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path, use_json=False)
    ctx.hyperparams_file = None
    cmd = OptimizeCommand()
    args = MagicMock(values='not json')
    with pytest.raises(SystemExit):
      cmd.set_hparams(ctx, args)
    captured = capsys.readouterr()
    assert 'JSON object' in captured.err
    assert 'JSONDecodeError' in captured.err

  def test_bad_json_includes_type_name(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path, use_json=False)
    ctx.hyperparams_file = None
    cmd = OptimizeCommand()
    args = MagicMock(values='{invalid')
    with pytest.raises(SystemExit):
      cmd.set_hparams(ctx, args)
    captured = capsys.readouterr()
    assert 'JSONDecodeError' in captured.err


class TestCheckoutErrorIncludesGuidance:
  """BUG-DFV1-007: checkout error should suggest store log."""

  def test_checkout_failure_suggests_store_log(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path, use_json=False)
    ctx.config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(ctx.config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='exp-1', hypothesis='test')
    exp.start()
    exp.complete(metrics={'score': 0.5})
    tree.add(Node(experiment=exp))
    forest.switch('main')
    forest.save()

    cmd = CheckoutCommand()
    args = argparse.Namespace(experiment_id='exp-1')
    args.context = 'test'

    with (
      patch.object(tree, 'checkout', side_effect=StoreError('no snapshot')),
      pytest.raises(SystemExit),
    ):
      cmd.forward(ctx, args)
    captured = capsys.readouterr()
    assert 'store log' in captured.err
    assert 'StoreError' in captured.err


class TestProposeRevertErrorIncludesGuidance:
  """BUG-DFV1-007: propose revert error should suggest propose list."""

  def test_revert_failure_suggests_propose_list(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path, use_json=False)
    ctx.epoch = None
    ctx.experiment = 'test-exp'
    exp_dir = tmp_path / 'test-exp'
    exp_dir.mkdir(parents=True, exist_ok=True)
    ctx.experiment_path.return_value = exp_dir

    proposal = ChangeProposal(
      proposal_id='p001',
      hypothesis='test',
      target_node=None,
      change_type='general',
      epoch=1,
      status='proposed',
    )
    record_proposal(exp_dir, proposal)

    cmd = ProposeCommand()
    args = argparse.Namespace(
      proposal_id='p001',
      source=str(tmp_path / 'src'),
      store=None,
      pattern='**/*',
    )
    with (
      patch(
        'autopilot.cli.commands.propose.FileStore.checkout',
        side_effect=FileNotFoundError('snapshot not found'),
      ),
      pytest.raises(SystemExit),
    ):
      cmd.revert(ctx, args)
    captured = capsys.readouterr()
    assert 'propose list' in captured.err
    assert 'FileNotFoundError' in captured.err
