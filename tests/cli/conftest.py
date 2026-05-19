"""Shared fixtures for CLI command tests."""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import CLIContext, build_context
from autopilot.cli.main import build_parser
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import argparse
import contextlib
import io
import json
import pytest


@pytest.fixture
def cli_workspace(tmp_path: Path) -> Path:
  """Workspace root with .autopilot store dir created."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def cli_config(cli_workspace: Path) -> AutoPilotConfig:
  """AutoPilotConfig for the test workspace."""
  return AutoPilotConfig(workspace=cli_workspace)


@pytest.fixture
def cli_store(cli_config: AutoPilotConfig) -> FileStore:
  """FileStore with no parameters (for tree/forest operations)."""
  cli_config.store_path.mkdir(parents=True, exist_ok=True)
  return FileStore(cli_config)


@pytest.fixture
def cli_forest(cli_store: FileStore) -> FileForest:
  """FileForest backed by cli_store."""
  return FileForest(cli_store)


def make_ctx(
  workspace: Path,
  *,
  use_json: bool = False,
  project: str | None = None,
) -> CLIContext:
  """Build a CLIContext for testing."""
  config = AutoPilotConfig(workspace=workspace, project=project)
  return CLIContext(
    workspace=workspace,
    project=project,
    config=config,
    output=Output(use_json=use_json),
  )


def make_mock_cli_context(
  tmp_path: Path,
  *,
  use_json: bool = True,
  experiment: str | None = None,
  epoch: int | None = None,
  **kwargs: Any,
) -> MagicMock:
  """Build a MagicMock CLIContext with standard test defaults.

  Provides the common base shared across CLI command tests: ``output``,
  ``fail``, ``config``, ``workspace``, ``dry_run``, ``context``, and
  ``experiment_path``. Extra keyword arguments are set as attributes.

  Args:
    tmp_path: Workspace root directory.
    use_json: Whether the output should be JSON mode.
    experiment: Optional experiment id to set on the context.
    epoch: Optional epoch number.
    **kwargs: Additional attributes to set on the mock.

  Returns:
    Configured MagicMock for use as a CLIContext.
  """
  config = AutoPilotConfig(workspace=tmp_path)
  ctx = MagicMock(spec=CLIContext)
  ctx.config = config
  ctx.workspace = tmp_path
  ctx.output = Output(use_json=use_json)
  ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
  ctx.dry_run = False
  ctx.context = None
  ctx.wait_timeout_ms = None
  if experiment is not None:
    ctx.experiment = experiment
    exp_dir = tmp_path / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    ctx.experiment_path.return_value = exp_dir
  if epoch is not None:
    ctx.epoch = epoch
  for key, value in kwargs.items():
    setattr(ctx, key, value)
  return ctx


def run_cli_no_context(workspace: Path, argv: list[str]) -> dict[str, Any]:
  """Run autopilot CLI in-process without injecting --context (read-only tests).

  Mirrors :func:`run_cli` but omits the ``--context`` flag so that tests
  exercising read-only / context-exempt commands do not mask enforcement
  bugs.  Use for all read-only command tests after plan 09.

  Args:
    workspace: Workspace root directory.
    argv: CLI argument tokens (e.g. ``['query']``).

  Returns:
    Parsed JSON envelope from captured stdout, or ``{}`` when no output.
  """
  parser = build_parser()
  full_argv = [*list(argv), '--workspace', str(workspace), '--json']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)

  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    parsed.handler(ctx, parsed)

  output = buf.getvalue().strip()
  if output:
    return json.loads(output)
  return {}


def run_cli(workspace: Path, argv: list[str]) -> dict[str, Any]:
  """Run a CLI command with ``--context 'test'`` injected and capture JSON output.

  Always injects ``--context 'test'`` so mutating commands pass context
  enforcement.  For read-only / context-exempt command tests, use
  :func:`run_cli_no_context` instead to avoid masking enforcement bugs.
  """
  parser = build_parser()
  full_argv = [*list(argv), '--workspace', str(workspace), '--json', '--context', 'test']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)

  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    parsed.handler(ctx, parsed)

  output = buf.getvalue().strip()
  if output:
    return json.loads(output)
  return {}


def run_cli_text(workspace: Path, argv: list[str]) -> str:
  """Run a CLI command and capture text output."""
  parser = build_parser()
  full_argv = [*list(argv), '--workspace', str(workspace), '--context', 'test']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)

  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    parsed.handler(ctx, parsed)

  return buf.getvalue()


@pytest.fixture
def multi_tree_forest(cli_store: FileStore) -> FileForest:
  """Forest with two trees ('alpha' and 'beta') for cross-tree CLI tests.

  Each tree contains one completed experiment so queries and comparisons
  have data to work with.

  Returns:
    FileForest with two populated trees; 'alpha' is active.
  """
  forest = FileForest(cli_store)
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


def seed_tree_with_experiments(
  forest: FileForest,
  tree_name: str,
  experiments: list[dict[str, Any]],
) -> None:
  """Seed a tree with experiments for testing.

  Each dict in experiments should have: id, hypothesis, status, metrics (optional),
  parent (optional experiment id).
  """
  tree = forest.create_tree(tree_name)
  forest.switch(tree_name)

  for exp_data in experiments:
    exp = Experiment(experiment_id=exp_data['id'], hypothesis=exp_data.get('hypothesis'))
    status = exp_data.get('status', 'pending')
    metrics = exp_data.get('metrics', {})

    if status in {'running', 'completed', 'failed', 'cancelled'}:
      exp.start()
    if status == 'completed':
      exp.complete(metrics=metrics)
    elif status == 'failed':
      exp.fail(error=exp_data.get('error'))
    elif status == 'cancelled':
      exp.cancel()
    elif status == 'running' and metrics:
      exp.metrics = metrics

    parent_node = None
    parent_id = exp_data.get('parent')
    if parent_id is not None:
      parent_node = tree.get(parent_id)

    baseline_node = None
    baseline_id = exp_data.get('baseline')
    if baseline_id is not None:
      baseline_node = tree.get(baseline_id)

    node = Node(experiment=exp, parent=parent_node, baseline=baseline_node)
    tree.add(node)

  forest.save()


def collect_leaf_commands(
  parser: argparse.ArgumentParser,
  path: tuple[str, ...] = (),
) -> list[str]:
  """Walk the parser tree and collect leaf command paths.

  Args:
    parser: Root or sub-parser to walk.
    path: Accumulated command path tokens.

  Returns:
    Sorted list of space-joined command paths for leaf subparsers.
  """
  leaves: list[str] = []
  has_subparsers = False
  for action in parser._actions:
    if isinstance(action, argparse._SubParsersAction):
      has_subparsers = True
      for name, sub in action.choices.items():
        leaves.extend(collect_leaf_commands(sub, (*path, name)))
  if not has_subparsers and path:
    leaves.append(' '.join(path))
  return sorted(leaves)


def make_cli_workspace(tmp_path: Path) -> tuple[FileStore, Path]:
  """Create a workspace with store and forest for CLI tests.

  Returns:
    Tuple of (store, workspace_path).
  """
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
  forest = FileForest(store)
  forest.save()
  return store, ws
