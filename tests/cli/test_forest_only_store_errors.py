"""Tests for forest-only store error messages (Plan 26 S2.3).

Verifies that store commands operating on forest-only experiments (those
without store branches) produce the canonical ``MSG_FOREST_ONLY_STORE``
guidance pointing users to ``store create``.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import require_store_branch
from autopilot.cli.main import AutoPilotCLI
from autopilot.cli.messages import MSG_FOREST_ONLY_STORE
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import pytest


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  """Workspace root directory."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def forest_only_ws(ws: Path) -> tuple[Path, str]:
  """Workspace with a forest-only experiment (no store branch).

  Returns:
    Tuple of (workspace path, experiment_id).
  """
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree = forest.create_tree('main')
  exp = Experiment(experiment_id='forest-only-exp', hypothesis='test')
  exp.start()
  exp.complete(metrics={'m': 1.0})
  tree.add(Node(experiment=exp))
  tree.head = 'forest-only-exp'
  forest.switch('main')
  forest.save()
  return ws, 'forest-only-exp'


class TestRequireStoreBranch:
  def test_missing_branch_fails_with_guidance(self, forest_only_ws: tuple) -> None:
    """require_store_branch calls ctx.fail with forest-only message."""
    from autopilot.cli.output import Output

    ws, exp_id = forest_only_ws
    config = AutoPilotConfig(workspace=ws)
    store = FileStore(config)

    ctx = CLIContext(
      workspace=ws,
      config=config,
      output=Output(use_json=False),
    )
    with pytest.raises(SystemExit):
      require_store_branch(ctx, store, exp_id)

  def test_message_contains_forest_only(self, forest_only_ws: tuple) -> None:
    """Error message contains 'forest-only' substring."""
    _, exp_id = forest_only_ws
    formatted = MSG_FOREST_ONLY_STORE.format(experiment_id=exp_id)
    assert 'forest-only' in formatted

  def test_message_contains_store_create(self, forest_only_ws: tuple) -> None:
    """Error message contains 'store create' guidance."""
    _, exp_id = forest_only_ws
    formatted = MSG_FOREST_ONLY_STORE.format(experiment_id=exp_id)
    assert 'store create' in formatted

  def test_tree_switch_checkout_forest_only_message(self, forest_only_ws: tuple) -> None:
    """tree switch (auto-checkout) on forest-only experiment emits canonical message."""
    ws, _exp_id = forest_only_ws
    cli = AutoPilotCLI()
    out = StringIO()
    err = StringIO()
    exit_code = 0
    with patch('sys.stdout', out), patch('sys.stderr', err):
      try:
        cli(
          argv=[
            'tree',
            'switch',
            'main',
            '--context',
            'test',
            '--workspace',
            str(ws),
          ]
        )
      except SystemExit as e:
        exit_code = int(e.code) if e.code is not None else 0

    assert exit_code == 1
    combined = out.getvalue() + err.getvalue()
    assert 'forest-only' in combined
    assert 'store create' in combined

  def test_existing_branch_returns_metadata(self, ws: Path) -> None:
    """When the branch exists, returns the branch metadata dict."""
    from autopilot.ai.parameter import PathParameter
    from autopilot.cli.output import Output

    src = ws / 'src'
    src.mkdir()
    (src / 'f.txt').write_text('hello')

    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    param = PathParameter(source=str(src), pattern='*')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('exp-with-branch', 0)

    ctx = CLIContext(
      workspace=ws,
      config=config,
      output=Output(use_json=False),
    )
    branch = require_store_branch(ctx, store, 'exp-with-branch')
    assert branch is not None
    assert branch['latest_epoch'] == 0
