"""Tests for experiment compare & query robustness (plan 05).

Covers:
  - Non-numeric metric handling in compare (BUG-006): strings, bools, lists,
    NaN values produce ``delta=None`` and ``type='non_numeric'``.
  - Missing metric values produce ``type='missing'``.
  - Verdict derived from numeric entries only; all-non-numeric = inconclusive.
  - Query temporal filters: ``--created-after``, ``--created-before``.
  - ``--cancelled`` shorthand for status=cancelled.
  - ``--context-contains`` case-insensitive default, ``--case-sensitive`` opt-in.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.experiment.verdict import (
  build_compare_deltas,
  compute_verdict,
  is_numeric_metric_value,
)
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context, seed_tree_with_experiments
from typing import Any
import pytest

# ---------------------------------------------------------------------------
# 2.1 is_numeric_metric_value
# ---------------------------------------------------------------------------


class TestIsNumericMetricValue:
  """Tests for the numeric metric value predicate."""

  def test_int_is_numeric(self) -> None:
    assert is_numeric_metric_value(42) is True

  def test_float_is_numeric(self) -> None:
    assert is_numeric_metric_value(0.75) is True

  def test_zero_is_numeric(self) -> None:
    assert is_numeric_metric_value(0) is True
    assert is_numeric_metric_value(0.0) is True

  def test_negative_is_numeric(self) -> None:
    assert is_numeric_metric_value(-1) is True
    assert is_numeric_metric_value(-0.5) is True

  def test_bool_is_not_numeric(self) -> None:
    bool_true = True
    bool_false = False
    assert is_numeric_metric_value(bool_true) is False
    assert is_numeric_metric_value(bool_false) is False

  def test_nan_is_not_numeric(self) -> None:
    assert is_numeric_metric_value(float('nan')) is False

  def test_string_is_not_numeric(self) -> None:
    assert is_numeric_metric_value('hello') is False

  def test_none_is_not_numeric(self) -> None:
    assert is_numeric_metric_value(None) is False

  def test_list_is_not_numeric(self) -> None:
    assert is_numeric_metric_value([1, 2, 3]) is False

  def test_dict_is_not_numeric(self) -> None:
    assert is_numeric_metric_value({'a': 1}) is False

  def test_inf_is_numeric(self) -> None:
    assert is_numeric_metric_value(float('inf')) is True
    assert is_numeric_metric_value(float('-inf')) is True


# ---------------------------------------------------------------------------
# 2.1 build_compare_deltas
# ---------------------------------------------------------------------------


class TestBuildCompareDeltas:
  """Tests for the compare deltas builder with non-numeric robustness."""

  def test_compare_string_metrics_returns_null_delta(self) -> None:
    """String metric values produce type='non_numeric' and delta=None."""
    deltas = build_compare_deltas({'model': 'gpt-4'}, {'model': 'claude-3'})
    assert len(deltas) == 1
    entry = deltas[0]
    assert entry['type'] == 'non_numeric'
    assert entry['delta'] is None
    assert entry['baseline'] == 'gpt-4'
    assert entry['candidate'] == 'claude-3'

  def test_compare_bool_metrics_returns_null_delta(self) -> None:
    """Boolean values (despite isinstance(True, int)) produce non_numeric."""
    deltas = build_compare_deltas({'passed': True}, {'passed': False})
    assert len(deltas) == 1
    assert deltas[0]['type'] == 'non_numeric'
    assert deltas[0]['delta'] is None

  def test_compare_none_metric_values(self) -> None:
    """Missing side yields delta=None and type='missing'."""
    deltas = build_compare_deltas({'accuracy': 0.9}, {'loss': 0.1})
    by_metric = {d['metric']: d for d in deltas}
    assert by_metric['accuracy']['type'] == 'missing'
    assert by_metric['accuracy']['delta'] is None
    assert by_metric['accuracy']['baseline'] == 0.9
    assert by_metric['accuracy']['candidate'] is None
    assert by_metric['loss']['type'] == 'missing'
    assert by_metric['loss']['baseline'] is None
    assert by_metric['loss']['candidate'] == 0.1

  def test_compare_mixed_numeric_nonnumeric(self) -> None:
    """Numeric rows have numeric delta; string rows have delta None."""
    metrics_a: dict[str, Any] = {'accuracy': 0.8, 'model': 'v1'}
    metrics_b: dict[str, Any] = {'accuracy': 0.9, 'model': 'v2'}
    deltas = build_compare_deltas(metrics_a, metrics_b)
    by_metric = {d['metric']: d for d in deltas}
    assert by_metric['accuracy']['type'] == 'numeric'
    assert by_metric['accuracy']['delta'] == pytest.approx(0.1)
    assert by_metric['model']['type'] == 'non_numeric'
    assert by_metric['model']['delta'] is None

  def test_compare_list_metrics_returns_null_delta(self) -> None:
    """List metric values produce type='non_numeric'."""
    deltas = build_compare_deltas({'tags': ['a', 'b']}, {'tags': ['c']})
    assert deltas[0]['type'] == 'non_numeric'
    assert deltas[0]['delta'] is None

  def test_compare_nan_treated_non_numeric(self) -> None:
    """float('nan') rows are non-numeric, never voted on."""
    deltas = build_compare_deltas({'accuracy': float('nan')}, {'accuracy': float('nan')})
    assert deltas[0]['type'] == 'non_numeric'
    assert deltas[0]['delta'] is None
    assert compute_verdict(deltas) == 'inconclusive'

  def test_compare_one_side_nan(self) -> None:
    """One side NaN, other numeric -> non_numeric (not a valid subtraction)."""
    deltas = build_compare_deltas({'accuracy': float('nan')}, {'accuracy': 0.9})
    assert deltas[0]['type'] == 'non_numeric'
    assert deltas[0]['delta'] is None

  def test_numeric_deltas_produce_correct_values(self) -> None:
    """Standard numeric comparison still works correctly."""
    deltas = build_compare_deltas(
      {'accuracy': 0.7, 'loss': 0.3},
      {'accuracy': 0.9, 'loss': 0.1},
    )
    by_metric = {d['metric']: d for d in deltas}
    assert by_metric['accuracy']['delta'] == pytest.approx(0.2)
    assert by_metric['loss']['delta'] == pytest.approx(-0.2)
    assert by_metric['accuracy']['type'] == 'numeric'
    assert by_metric['loss']['type'] == 'numeric'

  def test_deltas_are_ordered_by_key(self) -> None:
    """Delta list is sorted by metric name."""
    deltas = build_compare_deltas(
      {'z_metric': 1, 'a_metric': 2},
      {'z_metric': 3, 'a_metric': 4},
    )
    assert deltas[0]['metric'] == 'a_metric'
    assert deltas[1]['metric'] == 'z_metric'

  def test_empty_metrics_produce_empty_deltas(self) -> None:
    """Two empty dicts produce empty delta list."""
    deltas = build_compare_deltas({}, {})
    assert deltas == []
    assert compute_verdict(deltas) == 'inconclusive'


# ---------------------------------------------------------------------------
# 2.1 compute_verdict
# ---------------------------------------------------------------------------


class TestComputeVerdict:
  """Tests for verdict computation from list-shaped deltas."""

  def test_verdict_ignores_nonnumeric(self) -> None:
    """Non-numeric rows are excluded from the verdict vote."""
    deltas: list[dict[str, Any]] = [
      {'metric': 'acc', 'delta': 0.1, 'type': 'numeric'},
      {'metric': 'model', 'delta': None, 'type': 'non_numeric'},
    ]
    assert compute_verdict(deltas) == 'improved'

  def test_all_nonnumeric_verdict_inconclusive(self) -> None:
    """When no entry has a numeric delta, verdict must be 'inconclusive'."""
    deltas: list[dict[str, Any]] = [
      {'metric': 'model', 'delta': None, 'type': 'non_numeric'},
      {'metric': 'tags', 'delta': None, 'type': 'non_numeric'},
    ]
    assert compute_verdict(deltas) == 'inconclusive'

  def test_all_missing_verdict_inconclusive(self) -> None:
    """All missing entries -> inconclusive."""
    deltas: list[dict[str, Any]] = [
      {'metric': 'a', 'delta': None, 'type': 'missing'},
    ]
    assert compute_verdict(deltas) == 'inconclusive'

  def test_improved_verdict(self) -> None:
    deltas: list[dict[str, Any]] = [
      {'metric': 'a', 'delta': 0.1, 'type': 'numeric'},
      {'metric': 'b', 'delta': 0.2, 'type': 'numeric'},
    ]
    assert compute_verdict(deltas) == 'improved'

  def test_regressed_verdict(self) -> None:
    deltas: list[dict[str, Any]] = [
      {'metric': 'a', 'delta': -0.1, 'type': 'numeric'},
      {'metric': 'b', 'delta': -0.2, 'type': 'numeric'},
    ]
    assert compute_verdict(deltas) == 'regressed'

  def test_tied_verdict_inconclusive(self) -> None:
    deltas: list[dict[str, Any]] = [
      {'metric': 'a', 'delta': 0.1, 'type': 'numeric'},
      {'metric': 'b', 'delta': -0.1, 'type': 'numeric'},
    ]
    assert compute_verdict(deltas) == 'inconclusive'

  def test_empty_deltas_inconclusive(self) -> None:
    assert compute_verdict([]) == 'inconclusive'

  def test_zero_delta_neither_improved_nor_regressed(self) -> None:
    deltas: list[dict[str, Any]] = [
      {'metric': 'a', 'delta': 0.0, 'type': 'numeric'},
    ]
    assert compute_verdict(deltas) == 'inconclusive'


# ---------------------------------------------------------------------------
# 2.1 Integration: CLI experiment compare with non-numeric metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def compare_workspace(tmp_path: Path) -> tuple[Path, FileForest]:
  """Workspace with store and forest for compare tests."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  return ws, forest


def test_compare_cli_with_string_metrics(compare_workspace) -> None:
  """CLI compare with string metrics does not raise; returns non_numeric type."""
  ws, forest = compare_workspace
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {'id': 'a', 'hypothesis': 'a', 'status': 'completed', 'metrics': {'model': 'gpt-4'}},
      {'id': 'b', 'hypothesis': 'b', 'status': 'completed', 'metrics': {'model': 'claude'}},
    ],
  )
  result = run_cli(ws, ['experiment', 'compare', 'a', 'b'])
  deltas = result['result']['deltas']
  assert len(deltas) == 1
  assert deltas[0]['type'] == 'non_numeric'
  assert deltas[0]['delta'] is None


def test_compare_cli_mixed_types(compare_workspace) -> None:
  """CLI compare with mixed numeric and non-numeric metrics."""
  ws, forest = compare_workspace
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {
        'id': 'a',
        'hypothesis': 'a',
        'status': 'completed',
        'metrics': {'accuracy': 0.7, 'model': 'v1', 'passed': True},
      },
      {
        'id': 'b',
        'hypothesis': 'b',
        'status': 'completed',
        'metrics': {'accuracy': 0.9, 'model': 'v2', 'passed': False},
      },
    ],
  )
  result = run_cli(ws, ['experiment', 'compare', 'a', 'b'])
  deltas = result['result']['deltas']
  by_metric = {d['metric']: d for d in deltas}
  assert by_metric['accuracy']['type'] == 'numeric'
  assert by_metric['accuracy']['delta'] == pytest.approx(0.2)
  assert by_metric['model']['type'] == 'non_numeric'
  assert by_metric['passed']['type'] == 'non_numeric'
  assert result['result']['verdict'] == 'improved'


def test_compare_cli_all_nonnumeric_inconclusive(compare_workspace) -> None:
  """When all metrics are non-numeric, verdict must be inconclusive."""
  ws, forest = compare_workspace
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {'id': 'a', 'hypothesis': 'a', 'status': 'completed', 'metrics': {'model': 'v1'}},
      {'id': 'b', 'hypothesis': 'b', 'status': 'completed', 'metrics': {'model': 'v2'}},
    ],
  )
  result = run_cli(ws, ['experiment', 'compare', 'a', 'b'])
  assert result['result']['verdict'] == 'inconclusive'


# ---------------------------------------------------------------------------
# 2.2 Query: --created-after / --created-before
# ---------------------------------------------------------------------------


def _make_experiment_with_timestamp(
  id_: str, created_at: str, metrics: dict | None = None
) -> Experiment:
  """Build a completed experiment with a specific created_at timestamp."""
  exp = Experiment(experiment_id=id_, hypothesis=f'{id_} hypothesis')
  exp.created_at = created_at
  exp.start()
  exp.complete(metrics=metrics or {})
  return exp


@pytest.fixture
def temporal_workspace(tmp_path: Path) -> tuple[Path, FileForest]:
  """Workspace with experiments at different timestamps."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  old_exp = _make_experiment_with_timestamp('old', '2025-01-01T00:00:00+00:00')
  mid_exp = _make_experiment_with_timestamp('mid', '2025-06-15T12:00:00+00:00')
  new_exp = _make_experiment_with_timestamp('new', '2026-01-01T00:00:00+00:00')

  tree.add(Node(experiment=old_exp))
  tree.add(Node(experiment=mid_exp))
  tree.add(Node(experiment=new_exp))
  forest.save()
  return ws, forest


def test_query_created_after_filters(temporal_workspace) -> None:
  """--created-after excludes older experiments."""
  ws, _ = temporal_workspace
  result = run_cli_no_context(ws, ['query', '--created-after', '2025-06-01T00:00:00+00:00'])
  experiments = result['result']['experiments']
  ids = {e['id'] for e in experiments}
  assert 'old' not in ids
  assert 'mid' in ids
  assert 'new' in ids


def test_query_created_before_filters(temporal_workspace) -> None:
  """--created-before excludes newer experiments."""
  ws, _ = temporal_workspace
  result = run_cli_no_context(ws, ['query', '--created-before', '2025-07-01T00:00:00+00:00'])
  experiments = result['result']['experiments']
  ids = {e['id'] for e in experiments}
  assert 'old' in ids
  assert 'mid' in ids
  assert 'new' not in ids


def test_query_created_after_timezone_boundary(temporal_workspace) -> None:
  """Timezone-aware ISO string compared correctly via parse_timestamp."""
  ws, _ = temporal_workspace
  result = run_cli_no_context(ws, ['query', '--created-after', '2025-06-15T12:00:00+00:00'])
  experiments = result['result']['experiments']
  ids = {e['id'] for e in experiments}
  assert 'old' not in ids
  assert 'mid' in ids
  assert 'new' in ids


def test_query_created_after_and_before_combined(temporal_workspace) -> None:
  """Both temporal bounds compose (AND)."""
  ws, _ = temporal_workspace
  result = run_cli_no_context(
    ws,
    [
      'query',
      '--created-after',
      '2025-03-01T00:00:00+00:00',
      '--created-before',
      '2025-12-31T00:00:00+00:00',
    ],
  )
  experiments = result['result']['experiments']
  ids = {e['id'] for e in experiments}
  assert ids == {'mid'}


# ---------------------------------------------------------------------------
# 2.2 Query: --cancelled shorthand
# ---------------------------------------------------------------------------


@pytest.fixture
def status_workspace(tmp_path: Path) -> tuple[Path, FileForest]:
  """Workspace with experiments in various statuses."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {'id': 'exp-done', 'hypothesis': 'done', 'status': 'completed', 'metrics': {}},
      {'id': 'exp-cancel', 'hypothesis': 'cancel', 'status': 'cancelled'},
      {'id': 'exp-fail', 'hypothesis': 'fail', 'status': 'failed'},
      {'id': 'exp-run', 'hypothesis': 'run', 'status': 'running'},
    ],
  )
  return ws, forest


def test_query_cancelled_shorthand(status_workspace) -> None:
  """--cancelled yields only status=='cancelled' rows."""
  ws, _ = status_workspace
  result = run_cli_no_context(ws, ['query', '--cancelled'])
  experiments = result['result']['experiments']
  assert len(experiments) == 1
  assert experiments[0]['id'] == 'exp-cancel'
  assert experiments[0]['status'] == 'cancelled'


# ---------------------------------------------------------------------------
# 2.2 Query: --context-contains case insensitivity
# ---------------------------------------------------------------------------


@pytest.fixture
def context_workspace(tmp_path: Path) -> tuple[Path, FileForest]:
  """Workspace with experiments containing context log entries."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_a = Experiment(experiment_id='exp-with-regression', hypothesis='a')
  exp_a.start()
  exp_a.add_context('Found a regression in model performance', source='user')
  exp_a.complete(metrics={'accuracy': 0.5})
  tree.add(Node(experiment=exp_a))

  exp_b = Experiment(experiment_id='exp-without', hypothesis='b')
  exp_b.start()
  exp_b.add_context('everything is fine', source='user')
  exp_b.complete(metrics={'accuracy': 0.9})
  tree.add(Node(experiment=exp_b))

  exp_c = Experiment(experiment_id='exp-empty-context', hypothesis='c')
  exp_c.start()
  exp_c.complete(metrics={'accuracy': 0.7})
  tree.add(Node(experiment=exp_c))

  forest.save()
  return ws, forest


def test_context_contains_case_insensitive_default(context_workspace) -> None:
  """'REGRESSION' finds 'regression' in reason (case-insensitive by default)."""
  ws, _ = context_workspace
  result = run_cli_no_context(ws, ['query', '--context-contains', 'REGRESSION'])
  experiments = result['result']['experiments']
  ids = {e['id'] for e in experiments}
  assert 'exp-with-regression' in ids
  assert 'exp-without' not in ids


def test_context_contains_case_sensitive_flag(context_workspace) -> None:
  """--case-sensitive requires exact case match; uppercase does not match lowercase."""
  ws, _ = context_workspace
  result = run_cli_no_context(ws, ['query', '--context-contains', 'REGRESSION', '--case-sensitive'])
  experiments = result['result']['experiments']
  assert len(experiments) == 0


def test_context_contains_case_sensitive_exact_match(context_workspace) -> None:
  """--case-sensitive matches when case is exact."""
  ws, _ = context_workspace
  result = run_cli_no_context(ws, ['query', '--context-contains', 'regression', '--case-sensitive'])
  experiments = result['result']['experiments']
  ids = {e['id'] for e in experiments}
  assert 'exp-with-regression' in ids


def test_context_contains_null_reason_safe(context_workspace) -> None:
  """Experiments with empty context log do not crash during context search."""
  ws, _ = context_workspace
  result = run_cli_no_context(ws, ['query', '--context-contains', 'anything'])
  assert 'result' in result
