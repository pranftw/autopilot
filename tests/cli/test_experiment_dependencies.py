"""Tests for experiment dependency graph (Plan 14).

Covers:
  - Core: dependencies field on Experiment, state_dict round-trip
  - CLI: --depends-on on experiment add
  - CLI: experiment impact (transitive dependents)
  - CLI: dependencies in experiment show / status
  - CLI: checkout warning when dependents exist
"""

from autopilot.core.errors import ExperimentError
from autopilot.core.experiment import Experiment, validate_dependency_ids
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context, seed_tree_with_experiments
from typing import Any
from unittest.mock import patch
import contextlib
import io
import pytest

# ---------------------------------------------------------------------------
# 2.1 Core: dependencies field
# ---------------------------------------------------------------------------


class TestExperimentDependenciesCore:
  """Core Experiment.dependencies field and serialization."""

  def test_experiment_dependencies_default_empty(self) -> None:
    """New experiment has empty dependencies list."""
    exp = Experiment(experiment_id='e1')
    assert exp.dependencies == []

  def test_experiment_state_dict_round_trip_dependencies(self) -> None:
    """dependencies survives state_dict -> load_state_dict round-trip."""
    exp = Experiment(experiment_id='e1', hypothesis='h')
    exp.dependencies = ['dep-a', 'dep-b']
    state = exp.state_dict()

    assert state['dependencies'] == ['dep-a', 'dep-b']

    exp2 = Experiment(experiment_id='placeholder')
    exp2.load_state_dict(state)
    assert exp2.dependencies == ['dep-a', 'dep-b']

  def test_load_state_dict_missing_dependencies_defaults_empty(self) -> None:
    """Older state dicts without dependencies key get an empty list."""
    exp = Experiment(experiment_id='e1')
    state = exp.state_dict()
    del state['dependencies']

    exp2 = Experiment(experiment_id='placeholder')
    exp2.load_state_dict(state)
    assert exp2.dependencies == []

  def test_state_dict_dependencies_is_copy(self) -> None:
    """state_dict returns a copy of dependencies, not the original list."""
    exp = Experiment(experiment_id='e1')
    exp.dependencies = ['dep-a']
    state = exp.state_dict()
    state['dependencies'].append('injected')
    assert exp.dependencies == ['dep-a']


class TestValidateDependencyIds:
  """Unit tests for validate_dependency_ids."""

  def test_valid_dependencies_returned_sorted(self) -> None:
    deps = validate_dependency_ids(
      ['z-exp', 'a-exp'],
      self_id='new',
      resolve=lambda x: True,
      all_dependencies=dict,
    )
    assert deps == ['a-exp', 'z-exp']

  def test_duplicate_ids_deduped(self) -> None:
    deps = validate_dependency_ids(
      ['a', 'a', 'b'],
      self_id='new',
      resolve=lambda x: True,
      all_dependencies=dict,
    )
    assert deps == ['a', 'b']

  def test_missing_dependency_raises(self) -> None:
    with pytest.raises(ExperimentError, match='not found in any tree'):
      validate_dependency_ids(
        ['ghost'],
        self_id='new',
        resolve=lambda x: None,
        all_dependencies=dict,
      )

  def test_cycle_detected_raises(self) -> None:
    """A -> B exists; adding B -> A would create a cycle."""
    with pytest.raises(ExperimentError, match='cycle'):
      validate_dependency_ids(
        ['a'],
        self_id='b',
        resolve=lambda x: True,
        all_dependencies=lambda: {'a': ['b']},
      )

  def test_no_cycle_for_diamond(self) -> None:
    """Diamond shape A->B, A->C, B->D, C->D is acyclic."""
    deps = validate_dependency_ids(
      ['b', 'c'],
      self_id='a',
      resolve=lambda x: True,
      all_dependencies=lambda: {'b': ['d'], 'c': ['d'], 'd': []},
    )
    assert deps == ['b', 'c']

  def test_self_dependency_cycle(self) -> None:
    """An experiment cannot depend on itself."""
    with pytest.raises(ExperimentError, match='cycle'):
      validate_dependency_ids(
        ['a'],
        self_id='a',
        resolve=lambda x: True,
        all_dependencies=dict,
      )

  def test_transitive_cycle_detected(self) -> None:
    """A -> B -> C exists; adding C -> A would create a cycle."""
    with pytest.raises(ExperimentError, match='cycle'):
      validate_dependency_ids(
        ['a'],
        self_id='c',
        resolve=lambda x: True,
        all_dependencies=lambda: {'a': ['b'], 'b': ['c']},
      )


# ---------------------------------------------------------------------------
# 2.2 CLI: --depends-on on experiment add
# ---------------------------------------------------------------------------


def _run_cli_capture_fail(workspace: Path, argv: list[str]) -> str:
  """Run CLI expecting SystemExit and capture the error output.

  Builds the parser, runs the handler with stdout captured, and returns
  the raw output string. Raises AssertionError if the command does not
  exit with SystemExit.

  Args:
    workspace: Workspace root directory.
    argv: CLI argument tokens.

  Returns:
    Captured stdout text (includes JSON error envelope in --json mode).
  """
  from autopilot.cli.context import build_context
  from autopilot.cli.main import build_parser

  parser = build_parser()
  full_argv = [*list(argv), '--workspace', str(workspace), '--json', '--context', 'test']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)

  buf = io.StringIO()
  with pytest.raises(SystemExit), contextlib.redirect_stdout(buf):
    parsed.handler(ctx, parsed)

  return buf.getvalue()


def _setup_workspace_with_cycle_seed(tmp_path: Path) -> Path:
  """Create a workspace where adding exp-c depending on exp-b creates a cycle.

  Seeds exp-a (completed, no deps) and exp-b (completed, depends on exp-c).
  exp-b's dependency on exp-c is set via the API before exp-c exists. When
  the CLI tries to add exp-c with --depends-on exp-b, the graph becomes
  B->[C] and C->[B], making C reachable from itself (cycle).
  """
  from autopilot.ai.forest import FileForest
  from autopilot.ai.store.file_store import FileStore
  from autopilot.core.config import AutoPilotConfig
  from autopilot.core.node import Node

  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_a = Experiment(experiment_id='exp-a', hypothesis='a')
  exp_a.start()
  exp_a.complete()
  tree.add(Node(experiment=exp_a))

  exp_b = Experiment(experiment_id='exp-b', hypothesis='b')
  exp_b.start()
  exp_b.complete()
  exp_b.dependencies = ['exp-c']
  tree.add(Node(experiment=exp_b, parent=tree.get('exp-a')))

  forest.save()
  return ws


def _setup_workspace_with_experiments(
  tmp_path: Path,
  experiments: list[dict[str, Any]],
  tree_name: str = 'main',
) -> Path:
  """Create a workspace with seeded experiments and return its path."""
  from autopilot.ai.forest import FileForest
  from autopilot.ai.store.file_store import FileStore
  from autopilot.core.config import AutoPilotConfig

  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(forest, tree_name, experiments)
  return ws


class TestExperimentAddDependsOn:
  """CLI experiment add --depends-on."""

  def test_experiment_add_depends_on_records_dependency(self, tmp_path: Path) -> None:
    """add with --depends-on X records the dependency in the forest."""
    ws = _setup_workspace_with_experiments(
      tmp_path,
      [
        {'id': 'exp-a', 'hypothesis': 'base', 'status': 'completed', 'metrics': {'acc': 0.9}},
      ],
    )
    result = run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'child',
        '--depends-on',
        'exp-a',
        '--id',
        'exp-b',
      ],
    )
    assert result['result']['ok'] is True
    assert result['result']['dependencies'] == ['exp-a']

    show = run_cli_no_context(ws, ['experiment', 'show', 'exp-b'])
    assert show['result']['dependencies'] == ['exp-a']

  def test_experiment_add_depends_on_multiple(self, tmp_path: Path) -> None:
    """Multiple --depends-on flags are collected and sorted."""
    ws = _setup_workspace_with_experiments(
      tmp_path,
      [
        {'id': 'exp-a', 'hypothesis': 'a', 'status': 'completed'},
        {'id': 'exp-b', 'hypothesis': 'b', 'status': 'completed', 'parent': 'exp-a'},
      ],
    )
    result = run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'child',
        '--depends-on',
        'exp-b',
        '--depends-on',
        'exp-a',
        '--id',
        'exp-c',
      ],
    )
    assert result['result']['dependencies'] == ['exp-a', 'exp-b']

  def test_experiment_add_depends_on_missing_parent_errors(self, tmp_path: Path) -> None:
    """Unknown dependency id triggers ctx.fail with offending id in message."""
    ws = _setup_workspace_with_experiments(
      tmp_path,
      [
        {'id': 'exp-a', 'hypothesis': 'a', 'status': 'completed'},
      ],
    )
    output = _run_cli_capture_fail(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'child',
        '--depends-on',
        'nonexistent',
        '--id',
        'exp-b',
      ],
    )
    assert 'nonexistent' in output

  def test_experiment_add_depends_on_cycle_rejected(self, tmp_path: Path) -> None:
    """Cycle detection prevents circular dependency chains at add time.

    Seed: exp-a (completed), exp-b (completed, dependencies=['exp-c']).
    exp-b already declares a forward dependency on the not-yet-created exp-c.
    Adding exp-c with --depends-on exp-b creates a cycle: C->B->C.
    """
    ws = _setup_workspace_with_cycle_seed(tmp_path)
    output = _run_cli_capture_fail(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'would close cycle',
        '--depends-on',
        'exp-b',
        '--id',
        'exp-c',
      ],
    )
    assert 'cycle' in output

  def test_experiment_add_without_depends_on(self, tmp_path: Path) -> None:
    """add without --depends-on produces empty dependencies."""
    ws = _setup_workspace_with_experiments(tmp_path, [])
    result = run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'standalone',
        '--id',
        'exp-solo',
      ],
    )
    assert 'dependencies' not in result['result']

    show = run_cli_no_context(ws, ['experiment', 'show', 'exp-solo'])
    assert show['result']['dependencies'] == []


# ---------------------------------------------------------------------------
# 2.3 CLI: experiment impact
# ---------------------------------------------------------------------------


class TestExperimentImpact:
  """CLI experiment impact command."""

  def test_experiment_impact_transitive_a_b_c(self, tmp_path: Path) -> None:
    """Chain A; B depends on A; C depends on B. Impact of A = {B, C}."""
    ws = _setup_workspace_with_experiments(
      tmp_path,
      [
        {'id': 'exp-a', 'hypothesis': 'base', 'status': 'completed'},
      ],
    )
    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'b depends on a',
        '--depends-on',
        'exp-a',
        '--id',
        'exp-b',
      ],
    )
    run_cli(ws, ['experiment', 'complete', 'exp-b'])
    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'c depends on b',
        '--depends-on',
        'exp-b',
        '--id',
        'exp-c',
      ],
    )

    result = run_cli_no_context(ws, ['experiment', 'impact', 'exp-a'])
    assert set(result['result']['dependents']) == {'exp-b', 'exp-c'}
    assert result['result']['direct_dependents'] == ['exp-b']

  def test_experiment_impact_b_only_c(self, tmp_path: Path) -> None:
    """Impact of B in A->B->C chain is {C} only."""
    ws = _setup_workspace_with_experiments(
      tmp_path,
      [
        {'id': 'exp-a', 'hypothesis': 'base', 'status': 'completed'},
      ],
    )
    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'b depends on a',
        '--depends-on',
        'exp-a',
        '--id',
        'exp-b',
      ],
    )
    run_cli(ws, ['experiment', 'complete', 'exp-b'])
    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'c depends on b',
        '--depends-on',
        'exp-b',
        '--id',
        'exp-c',
      ],
    )

    result = run_cli_no_context(ws, ['experiment', 'impact', 'exp-b'])
    assert result['result']['dependents'] == ['exp-c']
    assert result['result']['direct_dependents'] == ['exp-c']

  def test_experiment_impact_leaf_empty(self, tmp_path: Path) -> None:
    """Leaf experiment with no dependents returns empty lists."""
    ws = _setup_workspace_with_experiments(
      tmp_path,
      [
        {'id': 'exp-a', 'hypothesis': 'base', 'status': 'completed'},
      ],
    )
    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'b depends on a',
        '--depends-on',
        'exp-a',
        '--id',
        'exp-b',
      ],
    )

    result = run_cli_no_context(ws, ['experiment', 'impact', 'exp-b'])
    assert result['result']['dependents'] == []
    assert result['result']['direct_dependents'] == []

  def test_experiment_impact_unknown_id_fails(self, tmp_path: Path) -> None:
    """Unknown experiment id triggers ctx.fail."""
    ws = _setup_workspace_with_experiments(
      tmp_path,
      [
        {'id': 'exp-a', 'hypothesis': 'a', 'status': 'completed'},
      ],
    )
    with pytest.raises(SystemExit):
      run_cli_no_context(ws, ['experiment', 'impact', 'ghost'])


# ---------------------------------------------------------------------------
# 2.4 Visibility: show/status include dependencies, checkout warns
# ---------------------------------------------------------------------------


class TestExperimentShowDependencies:
  """experiment show and status include dependencies in output."""

  def test_experiment_show_includes_dependencies(self, tmp_path: Path) -> None:
    """JSON output of experiment show includes dependencies key."""
    ws = _setup_workspace_with_experiments(
      tmp_path,
      [
        {'id': 'exp-a', 'hypothesis': 'base', 'status': 'completed'},
      ],
    )
    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'b depends on a',
        '--depends-on',
        'exp-a',
        '--id',
        'exp-b',
      ],
    )
    result = run_cli_no_context(ws, ['experiment', 'show', 'exp-b'])
    assert 'dependencies' in result['result']
    assert result['result']['dependencies'] == ['exp-a']

  def test_experiment_status_includes_dependencies(self, tmp_path: Path) -> None:
    """JSON output of experiment status includes dependencies key."""
    ws = _setup_workspace_with_experiments(
      tmp_path,
      [
        {'id': 'exp-a', 'hypothesis': 'base', 'status': 'completed'},
      ],
    )
    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'b depends on a',
        '--depends-on',
        'exp-a',
        '--id',
        'exp-b',
      ],
    )
    result = run_cli_no_context(ws, ['experiment', 'status', 'exp-b'])
    assert 'dependencies' in result['result']
    assert result['result']['dependencies'] == ['exp-a']

  def test_experiment_show_empty_dependencies(self, tmp_path: Path) -> None:
    """Experiment without dependencies shows empty list."""
    ws = _setup_workspace_with_experiments(
      tmp_path,
      [
        {'id': 'exp-a', 'hypothesis': 'a', 'status': 'completed'},
      ],
    )
    result = run_cli_no_context(ws, ['experiment', 'show', 'exp-a'])
    assert result['result']['dependencies'] == []


class TestCheckoutDependentWarning:
  """Checkout emits a warning when dependents exist."""

  def test_checkout_warns_when_dependents_exist(self, tmp_path: Path) -> None:
    """Checkout to an experiment that has dependents emits a warn message."""
    ws = _setup_workspace_with_experiments(
      tmp_path,
      [
        {'id': 'exp-a', 'hypothesis': 'base', 'status': 'completed'},
      ],
    )
    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'b depends on a',
        '--depends-on',
        'exp-a',
        '--id',
        'exp-b',
      ],
    )

    with patch('autopilot.core.tree.Tree.checkout'):
      result = run_cli(ws, ['checkout', 'exp-a'])

    messages = result.get('messages', [])
    warn_messages = [m for m in messages if m.get('level') == 'warn']
    assert len(warn_messages) >= 1
    assert 'exp-b' in warn_messages[0]['message']
    assert 'experiment impact' in warn_messages[0]['message']

  def test_checkout_no_warning_when_no_dependents(self, tmp_path: Path) -> None:
    """Checkout to an experiment with no dependents emits no warning."""
    ws = _setup_workspace_with_experiments(
      tmp_path,
      [
        {'id': 'exp-a', 'hypothesis': 'a', 'status': 'completed'},
      ],
    )

    with patch('autopilot.core.tree.Tree.checkout'):
      result = run_cli(ws, ['checkout', 'exp-a'])

    messages = result.get('messages', [])
    warn_messages = [m for m in messages if m.get('level') == 'warn']
    assert len(warn_messages) == 0
