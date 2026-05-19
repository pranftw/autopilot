"""Tests for ``tree describe`` CLI command.

Verifies lightweight tree metadata output without full node serialization.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.command import CLI
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
import pytest


@pytest.fixture
def describe_workspace(tmp_path: Path) -> Path:
  """Workspace with two trees for describe tests."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree_main = forest.create_tree('main', description='primary tree')
  exp = Experiment(experiment_id='exp-1', hypothesis='test')
  exp.start()
  exp.complete(metrics={'accuracy': 0.9})
  tree_main.add(Node(experiment=exp))
  tree_main.head = 'exp-1'

  forest.create_tree('secondary', description='alt tree')
  forest.switch('main')
  forest.save()
  return ws


class TestTreeDescribeActiveTree:
  """Describe the active tree via default (no name argument)."""

  def test_tree_describe_active_tree(self, describe_workspace: Path) -> None:
    """JSON result includes all expected metadata fields."""
    result = run_cli_no_context(describe_workspace, ['tree', 'describe'])
    inner = result['result']
    assert inner['name'] == 'main'
    assert inner['description'] == 'primary tree'
    assert inner['head'] == 'exp-1'
    assert inner['node_count'] == 1
    assert inner['created_at'] is None


class TestTreeDescribeNamedTree:
  """Describe a non-active tree by explicit name."""

  def test_tree_describe_named_tree(self, describe_workspace: Path) -> None:
    """Describing by name returns that tree's metadata."""
    result = run_cli_no_context(describe_workspace, ['tree', 'describe', 'secondary'])
    inner = result['result']
    assert inner['name'] == 'secondary'
    assert inner['description'] == 'alt tree'
    assert inner['node_count'] == 0


class TestTreeDescribeNonexistentFails:
  """Unknown tree name exits nonzero with error."""

  def test_tree_describe_nonexistent_tree_fails(self, describe_workspace: Path) -> None:
    """Exit nonzero; error mentions the tree name."""
    with pytest.raises(SystemExit):
      run_cli_no_context(describe_workspace, ['tree', 'describe', 'nope'])


class TestTreeDescribeJsonEnvelope:
  """Standard JSON envelope structure."""

  def test_tree_describe_json_envelope(self, describe_workspace: Path) -> None:
    """Success envelope has ok: True and result dict."""
    result = run_cli_no_context(describe_workspace, ['tree', 'describe'])
    assert result['ok'] is True
    assert isinstance(result['result'], dict)


class TestTreeDescribeContextExempt:
  """tree describe is read-only and does not require --context."""

  def test_tree_describe_context_exempt(self) -> None:
    """requires_context returns False for tree describe."""
    cli = CLI()
    assert cli.requires_context('tree describe') is False

  def test_tree_describe_invocation_no_context(self, describe_workspace: Path) -> None:
    """Invoke without --context exits 0."""
    result = run_cli_no_context(describe_workspace, ['tree', 'describe'])
    assert result['ok'] is True
