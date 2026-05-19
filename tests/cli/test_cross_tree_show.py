"""Tests for cross-tree experiment show (FR-010, Plan 10).

Verifies that ``experiment show`` resolves experiment ids across all trees
when the experiment is not in the active tree, that the JSON output always
includes a ``tree`` field, and that text mode emits an info line for
cross-tree hits.

After plan 02, cross-tree resolution delegates to ``Forest.find_experiment``.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context, run_cli_text
from unittest.mock import patch
import contextlib
import io
import json
import pytest


@pytest.fixture(autouse=True)
def _patch_store_checkout():
  """Patch FileStore.checkout for tests that don't create snapshots."""
  with patch('autopilot.ai.store.file_store.FileStore.checkout'):
    yield


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  """Workspace root."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


def _seed_cross_tree_workspace(ws: Path) -> FileForest:
  """Create a workspace with two trees, each with one experiment.

  - Tree 'alpha' (active): contains 'exp-alpha' (completed, accuracy=0.9).
  - Tree 'beta': contains 'exp-beta' (completed, accuracy=0.8).

  Returns:
    The saved FileForest instance.
  """
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  exp_a = Experiment(experiment_id='exp-alpha', hypothesis='alpha hypothesis')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9})
  tree_a.add(Node(experiment=exp_a))

  tree_b = forest.create_tree('beta')
  exp_b = Experiment(experiment_id='exp-beta', hypothesis='beta hypothesis')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.8})
  tree_b.add(Node(experiment=exp_b))

  forest.switch('alpha')
  forest.save()
  return forest


def _seed_cross_tree_with_context(ws: Path) -> FileForest:
  """Create workspace with cross-tree experiment that has context log entries.

  - Tree 'main' (active): empty.
  - Tree 'other': contains 'exp-ctx' with 3 context entries.

  Returns:
    The saved FileForest instance.
  """
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  forest.create_tree('main')

  tree_other = forest.create_tree('other')
  exp = Experiment(experiment_id='exp-ctx', hypothesis='context test')
  exp.add_context('started training', source='trainer', epoch=0)
  exp.add_context('policy accepted', source='policy', epoch=0)
  exp.add_context('completed run', source='trainer', epoch=1)
  exp.start()
  exp.complete(metrics={'f1': 0.75})
  tree_other.add(Node(experiment=exp))

  forest.switch('main')
  forest.save()
  return forest


class TestCrossTreeHelper:
  """Tests for Forest.find_experiment cross-tree lookup (post plan-02)."""

  def test_forest_find_experiment_returns_active_tree(self, ws: Path) -> None:
    """Forest.find_experiment returns owning Tree when id is on active tree."""
    forest = _seed_cross_tree_workspace(ws)

    result = forest.find_experiment('exp-alpha')
    assert result is not None
    node, owning_tree = result
    assert node.experiment.id == 'exp-alpha'
    assert owning_tree.name == 'alpha'

  def test_forest_find_experiment_returns_other_tree(self, ws: Path) -> None:
    """Forest.find_experiment returns the non-active Tree for cross-tree hits."""
    forest = _seed_cross_tree_workspace(ws)

    result = forest.find_experiment('exp-beta')
    assert result is not None
    node, owning_tree = result
    assert node.experiment.id == 'exp-beta'
    assert owning_tree.name == 'beta'

  def test_forest_find_experiment_returns_none_for_missing(self, ws: Path) -> None:
    """Forest.find_experiment returns None for missing ids."""
    forest = _seed_cross_tree_workspace(ws)

    assert forest.find_experiment('no-such-exp') is None


class TestExperimentShowActiveTree:
  """Tests for experiment show with active tree experiments."""

  def test_experiment_show_active_tree(self, ws: Path) -> None:
    """Experiment on active tree: result includes correct tree key."""
    _seed_cross_tree_workspace(ws)
    result = run_cli_no_context(ws, ['experiment', 'show', 'exp-alpha'])
    payload = result['result']
    assert payload['id'] == 'exp-alpha'
    assert payload['tree'] == 'alpha'
    assert payload['status'] == 'completed'
    assert payload['metrics'] == {'accuracy': 0.9}

  def test_experiment_show_active_tree_no_cross_tree_info_line(self, ws: Path) -> None:
    """No 'resolved from tree' info line when experiment is on active tree."""
    _seed_cross_tree_workspace(ws)
    text = run_cli_text(ws, ['experiment', 'show', 'exp-alpha'])
    assert 'resolved from tree' not in text


class TestExperimentShowCrossTree:
  """Tests for cross-tree experiment show resolution."""

  def test_experiment_show_cross_tree_finds_node(self, ws: Path) -> None:
    """Experiment only on non-active tree: found and tree key set."""
    _seed_cross_tree_workspace(ws)
    result = run_cli_no_context(ws, ['experiment', 'show', 'exp-beta'])
    payload = result['result']
    assert payload['id'] == 'exp-beta'
    assert payload['tree'] == 'beta'
    assert payload['hypothesis'] == 'beta hypothesis'

  def test_experiment_show_cross_tree_text_info_line(self, ws: Path) -> None:
    """Text mode emits 'resolved from tree' for cross-tree hits."""
    _seed_cross_tree_workspace(ws)
    text = run_cli_text(ws, ['experiment', 'show', 'exp-beta'])
    assert "resolved from tree 'beta'" in text


class TestExperimentShowNotFound:
  """Tests for experiment show with missing experiments."""

  def test_experiment_show_not_found_anywhere(self, ws: Path) -> None:
    """CLI fails with 'not found in any tree' for unknown experiment ids."""
    _seed_cross_tree_workspace(ws)
    parser = build_parser()
    full_argv = ['experiment', 'show', 'no-such-experiment', '--workspace', str(ws), '--json']
    parsed = parser.parse_args(full_argv)
    ctx = build_context(parsed)

    buf = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stdout(buf):
      parsed.handler(ctx, parsed)

    output = buf.getvalue().strip()
    payload = json.loads(output)
    assert payload['ok'] is False
    assert 'not found in any tree' in payload['error']


class TestExperimentShowContextLogCrossTree:
  """Tests for --context-log with cross-tree resolution."""

  def test_experiment_show_context_log_cross_tree(self, ws: Path) -> None:
    """Cross-tree experiment with --context-log returns context entries."""
    _seed_cross_tree_with_context(ws)
    result = run_cli_no_context(ws, ['experiment', 'show', 'exp-ctx', '--context-log'])
    payload = result['result']
    assert payload['tree'] == 'other'
    context_log = payload['context_log']
    assert isinstance(context_log, list)
    assert len(context_log) == 3
    assert context_log[0]['reason'] == 'started training'
    assert context_log[0]['source'] == 'trainer'
    assert context_log[2]['reason'] == 'completed run'


class TestResolveExperimentInForestRemoved:
  """Guards against reintroducing duplicate lookup helpers (plan 02, section 4.3)."""

  def test_resolve_experiment_in_forest_removed(self) -> None:
    """resolve_experiment_in_forest is NOT defined in compare.py after migration."""
    from autopilot.cli.commands.experiment import compare

    assert not hasattr(compare, 'resolve_experiment_in_forest'), (
      'resolve_experiment_in_forest should be removed; use Forest.find_experiment instead'
    )

  def test_find_experiment_cross_tree_removed(self) -> None:
    """find_experiment_cross_tree is NOT defined in compare.py after migration."""
    from autopilot.cli.commands.experiment import compare

    assert not hasattr(compare, 'find_experiment_cross_tree'), (
      'find_experiment_cross_tree should be removed; use Forest.find_experiment instead'
    )

  def test_forest_has_find_experiment_method(self) -> None:
    """Forest.find_experiment is the canonical cross-tree lookup."""
    from autopilot.core.forest import Forest
    import inspect

    assert hasattr(Forest, 'find_experiment')
    sig = inspect.signature(Forest.find_experiment)
    assert 'experiment_id' in sig.parameters
