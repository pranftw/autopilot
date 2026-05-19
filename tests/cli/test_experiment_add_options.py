"""Tests for ExperimentAdd --no-parent and --parent controls (plan 10, P1#13).

Verifies ``--no-parent`` opt-out from HEAD auto-parenting, mutual
exclusion with ``--parent``, and explicit parent selection.
"""

from autopilot.ai.forest import FileForest
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from pathlib import Path
from tests.cli.conftest import run_cli, seed_tree_with_experiments
from typing import Any
import contextlib
import io
import json
import pytest


def _seed_and_set_head(
  forest: FileForest,
  experiments: list[dict[str, Any]],
  tree_name: str = 'main',
  *,
  head: str | None = None,
) -> None:
  """Seed experiments into a forest tree and optionally set HEAD.

  Args:
    forest: Forest to populate (backed by existing store/workspace).
    experiments: Experiment dicts for ``seed_tree_with_experiments``.
    tree_name: Name of the tree to create.
    head: If provided, set the tree HEAD to this experiment id after seeding.
  """
  seed_tree_with_experiments(forest, tree_name, experiments)
  if head is not None:
    tree = forest.active
    assert tree is not None
    tree.head = head
    forest.save()


def _run_cli_expect_fail(workspace: Path, argv: list[str]) -> dict[str, Any]:
  """Run CLI expecting a SystemExit and return the JSON error envelope."""
  parser = build_parser()
  full_argv = [*list(argv), '--workspace', str(workspace), '--json', '--context', 'test']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)

  buf = io.StringIO()
  with pytest.raises(SystemExit), contextlib.redirect_stdout(buf):
    parsed.handler(ctx, parsed)

  output = buf.getvalue().strip()
  if output:
    return json.loads(output)
  return {}


class TestExperimentAddParentControls:
  """--no-parent, --parent, and HEAD default parenting."""

  def test_experiment_add_no_parent(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """With --no-parent, new node has no parent even when HEAD exists."""
    _seed_and_set_head(
      cli_forest,
      [{'id': 'exp-head', 'hypothesis': 'head', 'status': 'completed'}],
      head='exp-head',
    )
    result = run_cli(
      cli_workspace,
      [
        'experiment',
        'add',
        '--hypothesis',
        'orphan',
        '--id',
        'exp-orphan',
        '--no-parent',
      ],
    )
    assert result['result']['parent'] is None

  def test_experiment_add_explicit_parent(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """--parent X sets parent to X even when HEAD differs."""
    _seed_and_set_head(
      cli_forest,
      [
        {'id': 'exp-a', 'hypothesis': 'a', 'status': 'completed'},
        {'id': 'exp-b', 'hypothesis': 'b', 'status': 'completed', 'parent': 'exp-a'},
      ],
      head='exp-b',
    )
    result = run_cli(
      cli_workspace,
      [
        'experiment',
        'add',
        '--hypothesis',
        'child of a',
        '--id',
        'exp-child',
        '--parent',
        'exp-a',
      ],
    )
    assert result['result']['parent'] == 'exp-a'

  def test_experiment_add_default_parent_is_head(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """Without --parent or --no-parent, parent defaults to HEAD."""
    _seed_and_set_head(
      cli_forest,
      [{'id': 'exp-head', 'hypothesis': 'head', 'status': 'completed'}],
      head='exp-head',
    )
    result = run_cli(
      cli_workspace,
      [
        'experiment',
        'add',
        '--hypothesis',
        'auto-parented',
        '--id',
        'exp-new',
      ],
    )
    assert result['result']['parent'] == 'exp-head'

  def test_experiment_add_parent_and_no_parent_conflict(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """Passing both --parent and --no-parent triggers an error with guidance."""
    _seed_and_set_head(
      cli_forest,
      [{'id': 'exp-a', 'hypothesis': 'a', 'status': 'completed'}],
      head='exp-a',
    )
    result = _run_cli_expect_fail(
      cli_workspace,
      [
        'experiment',
        'add',
        '--hypothesis',
        'conflict',
        '--id',
        'exp-conflict',
        '--parent',
        'exp-a',
        '--no-parent',
      ],
    )
    error_msg = result.get('error')
    assert error_msg is not None
    assert 'mutually exclusive' in error_msg
    assert '--no-parent' in error_msg
    assert '--parent' in error_msg

  def test_experiment_add_unknown_parent_error(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """--parent with an unknown id triggers a non-zero exit with actionable message."""
    _seed_and_set_head(
      cli_forest,
      [{'id': 'exp-a', 'hypothesis': 'a', 'status': 'completed'}],
      head='exp-a',
    )
    result = _run_cli_expect_fail(
      cli_workspace,
      [
        'experiment',
        'add',
        '--hypothesis',
        'bad parent',
        '--id',
        'exp-fail',
        '--parent',
        'bogus',
      ],
    )
    error_msg = result.get('error')
    assert error_msg is not None
    assert 'bogus' in error_msg
    assert 'not found' in error_msg.lower()

  def test_experiment_add_no_parent_baseline_unset(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """With --no-parent and no --baseline, baseline remains None."""
    _seed_and_set_head(
      cli_forest,
      [{'id': 'exp-head', 'hypothesis': 'head', 'status': 'completed'}],
      head='exp-head',
    )
    result = run_cli(
      cli_workspace,
      [
        'experiment',
        'add',
        '--hypothesis',
        'orphan',
        '--id',
        'exp-orphan',
        '--no-parent',
      ],
    )
    assert result['result']['parent'] is None
    assert result['result']['baseline'] is None

  def test_experiment_add_no_parent_with_explicit_baseline(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """--no-parent with --baseline sets baseline but no parent."""
    _seed_and_set_head(
      cli_forest,
      [{'id': 'exp-a', 'hypothesis': 'a', 'status': 'completed'}],
      head='exp-a',
    )
    result = run_cli(
      cli_workspace,
      [
        'experiment',
        'add',
        '--hypothesis',
        'orphan with baseline',
        '--id',
        'exp-orphan',
        '--no-parent',
        '--baseline',
        'exp-a',
      ],
    )
    assert result['result']['parent'] is None
    assert result['result']['baseline'] == 'exp-a'

  def test_experiment_add_no_parent_no_head(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """--no-parent on a tree with no HEAD (first experiment) works fine."""
    cli_forest.create_tree('main')
    cli_forest.switch('main')
    cli_forest.save()

    result = run_cli(
      cli_workspace,
      [
        'experiment',
        'add',
        '--hypothesis',
        'first',
        '--id',
        'exp-first',
        '--no-parent',
      ],
    )
    assert result['result']['parent'] is None
