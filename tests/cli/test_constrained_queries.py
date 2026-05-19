"""Tests for cross-tree constrained queries (plan 08).

Covers --metric-between CLI flag, BUG-044 cross-tree deduplication,
QueryBuilder immutable chaining, and constraint composition with
--sort, --best, and --all-trees.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.node import Node
from autopilot.core.query import QueryBuilder
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
from tests.doubles import make_completed_experiment
import contextlib
import io
import json
import pytest


def _seed_constrained_workspace(ws: Path) -> None:
  """Seed a workspace with experiments at varied accuracy levels."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  for exp_id, hyp, metrics in [
    ('exp-low', 'low accuracy', {'accuracy': 0.3, 'loss': 0.9}),
    ('exp-mid', 'mid accuracy', {'accuracy': 0.5, 'loss': 0.5}),
    ('exp-good', 'good accuracy', {'accuracy': 0.7, 'loss': 0.3}),
    ('exp-great', 'great accuracy', {'accuracy': 0.9, 'loss': 0.1}),
  ]:
    tree.add(Node(experiment=make_completed_experiment(exp_id, hyp, metrics)))

  forest.save()


def _seed_cross_tree_dedup_workspace(ws: Path) -> None:
  """Seed workspace where two trees share the same experiment id."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  exp_shared = make_completed_experiment('shared-id', 'alpha version', {'accuracy': 0.8})
  tree_a.add(Node(experiment=exp_shared))

  exp_unique_a = make_completed_experiment('only-alpha', 'alpha unique', {'accuracy': 0.6})
  tree_a.add(Node(experiment=exp_unique_a))

  tree_b = forest.create_tree('beta')
  exp_shared_b = make_completed_experiment('shared-id', 'beta version', {'accuracy': 0.9})
  tree_b.add(Node(experiment=exp_shared_b))

  exp_unique_b = make_completed_experiment('only-beta', 'beta unique', {'accuracy': 0.7})
  tree_b.add(Node(experiment=exp_unique_b))

  forest.switch('alpha')
  forest.save()


@pytest.fixture
def constrained_ws(tmp_path: Path) -> Path:
  """Workspace with four experiments at varied accuracy levels."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_constrained_workspace(ws)
  return ws


@pytest.fixture
def dedup_ws(tmp_path: Path) -> Path:
  """Workspace with duplicate experiment id across two trees."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_cross_tree_dedup_workspace(ws)
  return ws


# ---------------------------------------------------------------------------
# 2.1 --metric-between tests
# ---------------------------------------------------------------------------


class TestMetricBetween:
  """Tests for --metric-between CLI flag."""

  def test_metric_between_cli(self, constrained_ws: Path) -> None:
    """Bounded filter returns only experiments in range."""
    result = run_cli_no_context(constrained_ws, ['query', '--metric-between', 'accuracy:0.4:0.8'])
    assert result['ok'] is True
    exps = result['result']['experiments']
    ids = {e['id'] for e in exps}
    assert ids == {'exp-mid', 'exp-good'}
    for exp in exps:
      assert 0.4 <= exp['metrics']['accuracy'] <= 0.8

  def test_metric_between_invalid_syntax(self, constrained_ws: Path) -> None:
    """Malformed token yields non-zero exit; JSON envelope has ok: false."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as exc_info:
      run_cli_no_context(constrained_ws, ['query', '--metric-between', 'accuracy:0.5'])
    assert exc_info.value.code != 0
    output = buf.getvalue().strip()
    if output:
      envelope = json.loads(output)
      assert envelope['ok'] is False

  def test_metric_between_low_equals_high(self, constrained_ws: Path) -> None:
    """Inclusive range where low == high returns exact-match experiments."""
    result = run_cli_no_context(constrained_ws, ['query', '--metric-between', 'accuracy:0.5:0.5'])
    assert result['ok'] is True
    exps = result['result']['experiments']
    ids = {e['id'] for e in exps}
    assert ids == {'exp-mid'}

  def test_metric_between_empty_metric_name(self, constrained_ws: Path) -> None:
    """Empty metric name in --metric-between exits with error."""
    with pytest.raises(SystemExit) as exc_info:
      run_cli_no_context(constrained_ws, ['query', '--metric-between', ':0.1:0.9'])
    assert exc_info.value.code != 0

  def test_metric_between_non_numeric_bounds(self, constrained_ws: Path) -> None:
    """Non-numeric bounds in --metric-between exits with error."""
    with pytest.raises(SystemExit) as exc_info:
      run_cli_no_context(constrained_ws, ['query', '--metric-between', 'accuracy:abc:def'])
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# 2.2 BUG-044 cross-tree deduplication
# ---------------------------------------------------------------------------


class TestCrossTreeDedup:
  """Tests for BUG-044 deduplication fix."""

  def test_query_cross_tree_dedup(self, dedup_ws: Path) -> None:
    """Duplicate IDs across trees: no ValueError; one row per id."""
    result = run_cli_no_context(dedup_ws, ['query', '--all-trees'])
    assert result['ok'] is True
    exps = result['result']['experiments']
    ids = [e['id'] for e in exps]
    assert len(ids) == len(set(ids)), 'duplicate ids in result'
    assert 'shared-id' in ids
    assert 'only-alpha' in ids
    assert 'only-beta' in ids
    assert len(ids) == 3


# ---------------------------------------------------------------------------
# 2.3 QueryBuilder immutable chaining
# ---------------------------------------------------------------------------


class TestQueryBuilderImmutableChain:
  """Tests for QueryBuilder immutable chaining contract."""

  def test_query_builder_immutable_chain(self) -> None:
    """Two successive calls do not mutate a shared builder instance."""
    exp_a = make_completed_experiment('a', 'h', {'accuracy': 0.9, 'loss': 0.1})
    exp_b = make_completed_experiment('b', 'h', {'accuracy': 0.5, 'loss': 0.5})
    exp_c = make_completed_experiment('c', 'h', {'accuracy': 0.3, 'loss': 0.9})
    nodes = [Node(experiment=e) for e in [exp_a, exp_b, exp_c]]
    lookup = {n.experiment.id: n for n in nodes}

    base = QueryBuilder(nodes, lookup.get)
    filtered_gt = base.metric_gt('accuracy', 0.4)
    filtered_lt = base.metric_lt('loss', 0.3)

    assert base is not filtered_gt
    assert base is not filtered_lt
    assert filtered_gt is not filtered_lt

    assert base.count() == 3
    assert filtered_gt.count() == 2
    assert filtered_lt.count() == 1

    base_ids = {n.experiment.id for n in base.all()}
    assert base_ids == {'a', 'b', 'c'}


# ---------------------------------------------------------------------------
# 2.4 Consolidated CLI tests
# ---------------------------------------------------------------------------


class TestConstrainedQueriesConsolidated:
  """Consolidated constraint tests for composition, sorting, JSON schema, and exit codes."""

  def test_three_constraints_chained(self, constrained_ws: Path) -> None:
    """--metric-gt, --metric-lt, and --metric-between together narrow correctly."""
    result = run_cli_no_context(
      constrained_ws,
      [
        'query',
        '--metric-gt',
        'accuracy:0.2',
        '--metric-lt',
        'loss:0.8',
        '--metric-between',
        'accuracy:0.4:0.8',
      ],
    )
    assert result['ok'] is True
    exps = result['result']['experiments']
    ids = {e['id'] for e in exps}
    assert ids == {'exp-mid', 'exp-good'}

  def test_best_with_constraints_all_trees(self, dedup_ws: Path) -> None:
    """--best + --all-trees + constraints resolve correctly post-dedup."""
    result = run_cli_no_context(
      dedup_ws,
      ['query', '--best', 'accuracy', '--metric-gt', 'accuracy:0.5', '--all-trees'],
    )
    assert result['ok'] is True
    best = result['result']['best']
    assert best is not None
    assert best['metrics']['accuracy'] > 0.5
    assert 'tree' in best

  def test_no_match_after_constraints(self, constrained_ws: Path) -> None:
    """Empty result set, exit 0; JSON has ok: true and empty result array."""
    result = run_cli_no_context(
      constrained_ws,
      ['query', '--metric-between', 'accuracy:10.0:20.0'],
    )
    assert result['ok'] is True
    assert result['result']['experiments'] == []
    assert result['result']['count'] == 0

  def test_constraints_with_sort(self, constrained_ws: Path) -> None:
    """Constrained + --sort ordering preserved among remaining rows."""
    result = run_cli_no_context(
      constrained_ws,
      [
        'query',
        '--metric-between',
        'accuracy:0.4:0.95',
        '--sort',
        'accuracy',
      ],
    )
    assert result['ok'] is True
    exps = result['result']['experiments']
    accuracies = [e['metrics']['accuracy'] for e in exps]
    assert accuracies == sorted(accuracies, reverse=True)

  def test_constraints_json_schema(self, constrained_ws: Path) -> None:
    """--json envelope keys ok, result, stable experiment rows."""
    result = run_cli_no_context(constrained_ws, ['query'])
    assert 'ok' in result
    assert result['ok'] is True
    assert 'result' in result
    assert 'experiments' in result['result']
    assert 'count' in result['result']
    for exp in result['result']['experiments']:
      assert 'id' in exp
      assert 'status' in exp
      assert 'metrics' in exp
      assert 'metrics_trusted' in exp

  def test_constraints_exit_code(self, constrained_ws: Path) -> None:
    """Success path exit 0; user-error path non-zero."""
    result = run_cli_no_context(constrained_ws, ['query'])
    assert result['ok'] is True

    with pytest.raises(SystemExit) as exc_info:
      run_cli_no_context(constrained_ws, ['query', '--metric-between', 'bad'])
    assert exc_info.value.code != 0
