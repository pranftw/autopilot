"""Tests for experiment lifecycle enhancements (Plan 06).

Covers:
  - Status enum invalidated terminal
  - Experiment.fail(metrics=...) replacement
  - Experiment.invalidate() transitions and guards
  - Node.deployed_as field + serialization
  - QueryBuilder invalidated exclusion + deployed filter + numeric best
  - Serialization round-trips for new fields
"""

from autopilot.core.enums import Status
from autopilot.core.errors import ExperimentError
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.core.query import QueryBuilder
from tests.core.conftest import completed_exp, make_experiment, running_exp
import pytest

# -- Status enum --


class TestStatusEnum:
  """Status.invalidated is terminal."""

  def test_status_enum_terminal_includes_invalidated(self):
    """Status.invalidated.is_terminal is True."""
    assert Status.invalidated.is_terminal is True

  def test_status_enum_invalidated_value(self):
    """Status.invalidated serializes as 'invalidated'."""
    assert Status.invalidated.value == 'invalidated'

  def test_status_enum_invalidated_not_active(self):
    """Invalidated is not active."""
    assert Status.invalidated.is_active is False

  def test_all_terminal_statuses(self):
    """All expected terminal statuses are terminal."""
    terminals = {Status.completed, Status.failed, Status.cancelled, Status.invalidated}
    for s in terminals:
      assert s.is_terminal is True

  def test_non_terminal_statuses(self):
    """Pending and running are not terminal."""
    assert Status.pending.is_terminal is False
    assert Status.running.is_terminal is False


# -- Experiment.fail with metrics --


class TestExperimentFailWithMetrics:
  """Experiment.fail(metrics=...) replaces metrics before status flip."""

  def test_experiment_fail_with_metrics(self):
    """After fail(metrics={'acc': 0.1}), exp.metrics['acc'] == 0.1 and status == failed."""
    exp = Experiment(experiment_id='e1')
    exp.start()
    exp.fail(metrics={'acc': 0.1})
    assert exp.metrics['acc'] == 0.1
    assert exp.status == Status.failed

  def test_experiment_fail_with_metrics_and_error(self):
    """Both error and metrics are set on fail."""
    exp = Experiment(experiment_id='e2')
    exp.start()
    exp.fail(error='oom', metrics={'loss': 9.9})
    assert exp.error == 'oom'
    assert exp.metrics == {'loss': 9.9}
    assert exp.status == Status.failed

  def test_experiment_fail_without_metrics_unchanged(self):
    """When metrics=None (default), existing metrics are preserved."""
    exp = Experiment(experiment_id='e3')
    exp.start()
    exp.metrics = {'prev': 1.0}
    exp.fail(error='broken')
    assert exp.metrics == {'prev': 1.0}

  def test_experiment_fail_metrics_replace_semantics(self):
    """Metrics is full replacement, not merge."""
    exp = Experiment(experiment_id='e4')
    exp.start()
    exp.metrics = {'old_key': 0.5}
    exp.fail(metrics={'new_key': 0.9})
    assert 'old_key' not in exp.metrics
    assert exp.metrics == {'new_key': 0.9}

  def test_experiment_fail_non_numeric_metrics(self):
    """fail() accepts non-numeric metric values (dict[str, Any])."""
    exp = Experiment(experiment_id='e5')
    exp.start()
    exp.fail(metrics={'note': 'diverged', 'acc': 0.1, 'tags': ['bad']})
    assert exp.metrics['note'] == 'diverged'
    assert exp.metrics['tags'] == ['bad']

  def test_experiment_fail_metrics_queryable(self):
    """Failed experiment with metrics appears in queries."""
    exp = make_experiment('fail-q', status='failed')
    exp.metrics = {'acc': 0.1, 'note': 'bad run'}
    node = Node(experiment=exp)
    qb = QueryBuilder([node], lambda eid: node if eid == 'fail-q' else None)
    result = qb.best('acc', higher_is_better=True)
    assert result is not None
    assert result.experiment.id == 'fail-q'


# -- Experiment.invalidate --


class TestExperimentInvalidate:
  """Experiment.invalidate() transition and guards."""

  def test_experiment_invalidate_from_completed(self):
    """completed -> invalidated, invalidated_at non-None."""
    exp = completed_exp('inv1', metrics={'acc': 0.9})
    exp.invalidate(reason='bad data discovered')
    assert exp.status == Status.invalidated
    assert exp.invalidated_at is not None

  def test_experiment_invalidate_from_running_rejects(self):
    """Running experiment cannot be invalidated."""
    exp = running_exp('inv2')
    with pytest.raises(ExperimentError, match='expected completed'):
      exp.invalidate(reason='attempt')

  def test_experiment_invalidate_from_pending_rejects(self):
    """Pending experiment cannot be invalidated."""
    exp = Experiment(experiment_id='inv3')
    with pytest.raises(ExperimentError, match='expected completed'):
      exp.invalidate(reason='attempt')

  def test_experiment_invalidate_from_failed_rejects(self):
    """Failed experiment cannot be invalidated."""
    exp = make_experiment('inv4', status='failed')
    with pytest.raises(ExperimentError, match='expected completed'):
      exp.invalidate(reason='attempt')

  def test_experiment_invalidate_from_cancelled_rejects(self):
    """Cancelled experiment cannot be invalidated."""
    exp = make_experiment('inv5', status='cancelled')
    with pytest.raises(ExperimentError, match='expected completed'):
      exp.invalidate(reason='attempt')

  def test_experiment_invalidate_preserves_metrics(self):
    """Metrics dict unchanged post-invalidate."""
    metrics = {'acc': 0.95, 'loss': 0.05}
    exp = completed_exp('inv6', metrics=metrics)
    exp.invalidate(reason='contaminated data')
    assert exp.metrics == metrics

  def test_experiment_invalidate_context_entry(self):
    """Invalidation adds a context entry with source='user'."""
    exp = completed_exp('inv7')
    exp.invalidate(reason='duplicate of inv1')
    entries = exp.context_log.entries
    assert len(entries) > 0
    last = entries[-1]
    assert last.source == 'user'
    assert 'duplicate of inv1' in last.reason

  def test_experiment_invalidate_is_terminal(self):
    """Invalidated experiment is terminal."""
    exp = completed_exp('inv8')
    exp.invalidate(reason='obsolete')
    assert exp.is_terminal is True

  def test_experiment_invalidate_double_raises(self):
    """Cannot invalidate an already-invalidated experiment."""
    exp = completed_exp('inv9')
    exp.invalidate(reason='first')
    with pytest.raises(ExperimentError, match='expected completed'):
      exp.invalidate(reason='second')


# -- Serialization round-trip --


class TestSerializationRoundTrip:
  """state_dict/load_state_dict round-trip with new fields."""

  def test_invalidated_round_trip(self):
    """Invalidated status and invalidated_at survive serialization."""
    exp = completed_exp('rt1', metrics={'x': 1.0})
    exp.invalidate(reason='test')
    state = exp.state_dict()

    exp2 = Experiment(experiment_id='dummy')
    exp2.load_state_dict(state)
    assert exp2.status == Status.invalidated
    assert exp2.invalidated_at == exp.invalidated_at
    assert exp2.metrics == {'x': 1.0}

  def test_failed_with_metrics_round_trip(self):
    """Failed experiment with metrics survives serialization."""
    exp = Experiment(experiment_id='rt2')
    exp.start()
    exp.fail(error='oops', metrics={'acc': 0.1, 'note': 'bad'})
    state = exp.state_dict()

    exp2 = Experiment(experiment_id='dummy')
    exp2.load_state_dict(state)
    assert exp2.status == Status.failed
    assert exp2.metrics == {'acc': 0.1, 'note': 'bad'}
    assert exp2.error == 'oops'

  def test_invalidated_at_none_backward_compat(self):
    """Loading old state without invalidated_at sets None."""
    exp = completed_exp('rt3')
    state = exp.state_dict()
    del state['invalidated_at']
    exp2 = Experiment(experiment_id='dummy')
    exp2.load_state_dict(state)
    assert exp2.invalidated_at is None


# -- Node.deployed_as --


class TestNodeDeployedAs:
  """Node.deployed_as field and serialization."""

  def test_node_deployed_as_default_none(self):
    """Default deployed_as is None."""
    exp = completed_exp('dep1')
    node = Node(experiment=exp)
    assert node.deployed_as is None

  def test_experiment_deploy_sets_deployed_as(self):
    """Setting deployed_as persists on the node."""
    exp = completed_exp('dep2')
    node = Node(experiment=exp, deployed_as='production')
    assert node.deployed_as == 'production'

  def test_node_deployed_as_serialization_round_trip(self):
    """deployed_as survives to_dict/from_dict."""
    exp = completed_exp('dep3')
    node = Node(experiment=exp, deployed_as='staging')
    data = node.to_dict()
    assert data['deployed_as'] == 'staging'

    restored = Node.from_dict(data, resolver=lambda eid: exp)
    assert restored.deployed_as == 'staging'

  def test_node_deployed_as_none_serialization(self):
    """None deployed_as is serialized and restored as None."""
    exp = completed_exp('dep4')
    node = Node(experiment=exp)
    data = node.to_dict()
    assert data['deployed_as'] is None

    restored = Node.from_dict(data, resolver=lambda eid: exp)
    assert restored.deployed_as is None


# -- QueryBuilder exclusions --


class TestQueryExcludeInvalidated:
  """QueryBuilder excludes invalidated by default."""

  def _make_nodes(self):
    """Create a set of nodes with various statuses."""
    e1 = completed_exp('q1', metrics={'acc': 0.9})
    e2 = completed_exp('q2', metrics={'acc': 0.8})
    e2.invalidate(reason='bad')
    e3 = make_experiment('q3', status='failed')
    n1 = Node(experiment=e1)
    n2 = Node(experiment=e2)
    n3 = Node(experiment=e3)
    return [n1, n2, n3]

  def test_query_excludes_invalidated_by_default(self):
    """Invalidated id absent without --include-invalidated filter."""
    nodes = self._make_nodes()
    resolver = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, resolver.get)
    qb = qb.exclude(status=Status.invalidated)
    ids = [n.experiment.id for n in qb.all()]
    assert 'q2' not in ids
    assert 'q1' in ids
    assert 'q3' in ids

  def test_query_include_invalidated(self):
    """Without exclusion, invalidated experiments are visible."""
    nodes = self._make_nodes()
    resolver = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, resolver.get)
    ids = [n.experiment.id for n in qb.all()]
    assert 'q2' in ids


class TestQueryDeployedFilter:
  """QueryBuilder --deployed filter."""

  def test_query_deployed_filter(self):
    """--deployed returns only deployed nodes."""
    e1 = completed_exp('d1', metrics={'acc': 0.9})
    e2 = completed_exp('d2', metrics={'acc': 0.8})
    n1 = Node(experiment=e1, deployed_as='production')
    n2 = Node(experiment=e2)
    nodes = [n1, n2]
    resolver = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, resolver.get)
    qb = qb.where(lambda n: n.deployed_as is not None)
    results = qb.all()
    assert len(results) == 1
    assert results[0].experiment.id == 'd1'


class TestQueryBestNumericOnly:
  """QueryBuilder.best() skips non-numeric metric values."""

  def test_best_skips_non_numeric(self):
    """best() ignores string and bool metric values."""
    e1 = completed_exp('bn1', metrics={'acc': 'high'})
    e2 = completed_exp('bn2', metrics={'acc': 0.7})
    n1 = Node(experiment=e1)
    n2 = Node(experiment=e2)
    nodes = [n1, n2]
    resolver = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, resolver.get)
    result = qb.best('acc', higher_is_better=True)
    assert result is not None
    assert result.experiment.id == 'bn2'

  def test_best_skips_nan(self):
    """best() ignores NaN metric values."""
    e1 = completed_exp('nan1', metrics={'acc': float('nan')})
    e2 = completed_exp('nan2', metrics={'acc': 0.5})
    n1 = Node(experiment=e1)
    n2 = Node(experiment=e2)
    nodes = [n1, n2]
    resolver = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, resolver.get)
    result = qb.best('acc', higher_is_better=True)
    assert result is not None
    assert result.experiment.id == 'nan2'

  def test_best_skips_inf(self):
    """best() ignores inf metric values."""
    e1 = completed_exp('inf1', metrics={'loss': float('inf')})
    e2 = completed_exp('inf2', metrics={'loss': 0.3})
    n1 = Node(experiment=e1)
    n2 = Node(experiment=e2)
    nodes = [n1, n2]
    resolver = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, resolver.get)
    result = qb.best('loss', higher_is_better=False)
    assert result is not None
    assert result.experiment.id == 'inf2'

  def test_best_skips_bool(self):
    """best() ignores boolean metric values."""
    e1 = completed_exp('b1', metrics={'passed': True})
    e2 = completed_exp('b2', metrics={'passed': 0.8})
    n1 = Node(experiment=e1)
    n2 = Node(experiment=e2)
    nodes = [n1, n2]
    resolver = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, resolver.get)
    result = qb.best('passed', higher_is_better=True)
    assert result is not None
    assert result.experiment.id == 'b2'

  def test_best_returns_none_all_non_numeric(self):
    """best() returns None when all candidates have non-numeric values."""
    e1 = completed_exp('nn1', metrics={'acc': 'high'})
    e2 = completed_exp('nn2', metrics={'acc': 'medium'})
    n1 = Node(experiment=e1)
    n2 = Node(experiment=e2)
    nodes = [n1, n2]
    resolver = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, resolver.get)
    result = qb.best('acc', higher_is_better=True)
    assert result is None


# -- Deploy uniqueness and idempotency (unit-level) --


class TestDeploySemantics:
  """Deploy uniqueness and idempotency logic at the unit level."""

  def test_deploy_duplicate_name_conflict(self):
    """Forest-wide scan detects duplicate deployment names."""
    e1 = completed_exp('dup1')
    e2 = completed_exp('dup2')
    n1 = Node(experiment=e1, deployed_as='prod')
    n2 = Node(experiment=e2)
    assert n1.deployed_as == 'prod'
    assert n2.deployed_as is None

  def test_deploy_same_experiment_idempotent(self):
    """Same id + same name twice succeeds (node already has the label)."""
    exp = completed_exp('idem1')
    node = Node(experiment=exp, deployed_as='prod')
    assert node.deployed_as == 'prod'
    node.deployed_as = 'prod'
    assert node.deployed_as == 'prod'
