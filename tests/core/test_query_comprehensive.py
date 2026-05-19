"""Comprehensive tests for QueryBuilder composable query engine.

Covers empty tree, complex chained queries, explain(), best/worst with
tied metrics, structural queries with deep trees, metric_between inclusive
boundaries, immutability, and order_by with various keys.
"""

from autopilot.core.node import Node
from autopilot.core.query import QueryBuilder
from tests.core.conftest import make_experiment
import pytest


def _accept_all_nodes(_node: Node) -> bool:
  return True


def _reject_all_nodes(_node: Node) -> bool:
  return False


def _build_deep_tree():
  """Build a deep tree with 5 levels:

  root (completed, accuracy=0.5)
    +-- d1 (completed, accuracy=0.6)
    |   +-- d2 (completed, accuracy=0.7)
    |       +-- d3 (completed, accuracy=0.8)
    |           +-- d4 (completed, accuracy=0.9)
    +-- s1 (completed, accuracy=0.55)
    +-- s2 (failed)
  root2 (pending)
  """
  n_root = Node(
    experiment=make_experiment(
      'root',
      status='completed',
      hypothesis='baseline',
      metrics={'accuracy': 0.5},
    ),
  )
  n_d1 = Node(
    experiment=make_experiment(
      'd1',
      status='completed',
      hypothesis='depth-1',
      metrics={'accuracy': 0.6},
    ),
    parent=n_root,
  )
  n_d2 = Node(
    experiment=make_experiment(
      'd2',
      status='completed',
      hypothesis='depth-2',
      metrics={'accuracy': 0.7},
    ),
    parent=n_d1,
  )
  n_d3 = Node(
    experiment=make_experiment(
      'd3',
      status='completed',
      hypothesis='depth-3',
      metrics={'accuracy': 0.8},
    ),
    parent=n_d2,
  )
  n_d4 = Node(
    experiment=make_experiment(
      'd4',
      status='completed',
      hypothesis='depth-4',
      metrics={'accuracy': 0.9},
    ),
    parent=n_d3,
  )
  n_s1 = Node(
    experiment=make_experiment(
      's1',
      status='completed',
      hypothesis='sibling-1',
      metrics={'accuracy': 0.55},
    ),
    parent=n_root,
  )
  n_s2 = Node(
    experiment=make_experiment('s2', status='failed', hypothesis='sibling-2'),
    parent=n_root,
  )
  n_root2 = Node(experiment=make_experiment('root2', status='pending', hypothesis='alternative'))

  nodes = [n_root, n_d1, n_d2, n_d3, n_d4, n_s1, n_s2, n_root2]
  node_map = {n.experiment.id: n for n in nodes}
  return nodes, node_map


def _make_qb(nodes=None, node_map=None):
  if nodes is None:
    nodes, node_map = _build_deep_tree()
  assert node_map is not None
  return QueryBuilder(nodes, node_map.get)


# --- Query on empty tree ---


class TestEmptyTreeQuery:
  def test_count_zero(self):
    qb = QueryBuilder([], {}.get)
    assert qb.count() == 0

  def test_all_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.all() == []

  def test_chained_empty(self):
    qb = QueryBuilder([], {}.get)
    result = qb.completed().metric_gt('accuracy', 0.5).order_by('id').all()
    assert result == []

  def test_best_on_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.best('accuracy') is None

  def test_worst_on_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.worst('accuracy') is None

  def test_ancestors_of_on_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.ancestors_of('nonexistent').count() == 0

  def test_descendants_of_on_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.descendants_of('nonexistent').count() == 0

  def test_siblings_of_on_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.siblings_of('nonexistent').count() == 0


# --- Complex chained: filter + structural + ordering + best ---


class TestComplexChained:
  def test_filter_structural_ordering_best(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)

    result = (
      qb.completed()
      .descendants_of('root')
      .has_metric('accuracy')
      .order_by_metric('accuracy')
      .best('accuracy')
    )
    assert result is not None
    assert result.experiment.id == 'd4'

  def test_filter_then_structural_then_count(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)

    result = qb.completed().children_of('root').count()
    assert result == 2

  def test_chained_filter_exclude_order(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)

    result = (
      qb.completed()
      .exclude(id='root')
      .metric_gt('accuracy', 0.6)
      .order_by_metric('accuracy', ascending=True)
      .all()
    )
    ids = [n.experiment.id for n in result]
    values = [n.experiment.metrics['accuracy'] for n in result]
    assert values == sorted(values)
    assert 'root' not in ids

  def test_roots_then_completed(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.roots().completed().all()
    assert len(result) == 1
    assert result[0].experiment.id == 'root'


# --- explain() ---


class TestExplain:
  def test_explain_returns_non_empty(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.completed().explain()
    assert isinstance(result, str)
    assert len(result) > 0

  def test_explain_contains_completed(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.completed().explain()
    assert 'completed()' in result

  def test_explain_contains_failed(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.failed().explain()
    assert 'failed()' in result

  def test_explain_contains_running(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.running().explain()
    assert 'running()' in result

  def test_explain_contains_pending(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.pending().explain()
    assert 'pending()' in result

  def test_explain_chained_contains_all_terms(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = (qb.completed().metric_gt('accuracy', 0.5).order_by_metric('accuracy')).explain()
    assert 'completed()' in result
    assert 'metric_gt' in result
    assert 'order_by_metric' in result

  def test_explain_no_steps(self):
    qb = QueryBuilder([], {}.get)
    assert qb.explain() == 'query: all nodes'

  def test_explain_with_filter(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.filter(hypothesis='baseline').explain()
    assert 'filter' in result
    assert 'baseline' in result


# --- best/worst with tied metrics ---


class TestBestWorstTied:
  def test_best_with_tied_metrics_returns_first(self):
    exp1 = make_experiment('e1', status='completed', metrics={'accuracy': 0.9})
    exp2 = make_experiment('e2', status='completed', metrics={'accuracy': 0.9})
    exp3 = make_experiment('e3', status='completed', metrics={'accuracy': 0.9})

    n1 = Node(experiment=exp1)
    n2 = Node(experiment=exp2)
    n3 = Node(experiment=exp3)
    nodes = [n1, n2, n3]
    node_map = {n.experiment.id: n for n in nodes}

    qb = QueryBuilder(nodes, node_map.get)
    best = qb.best('accuracy')
    assert best is not None
    assert best.experiment.id == 'e1'

  def test_worst_with_tied_metrics_returns_first(self):
    exp1 = make_experiment('e1', status='completed', metrics={'accuracy': 0.5})
    exp2 = make_experiment('e2', status='completed', metrics={'accuracy': 0.5})
    exp3 = make_experiment('e3', status='completed', metrics={'accuracy': 0.5})

    n1 = Node(experiment=exp1)
    n2 = Node(experiment=exp2)
    n3 = Node(experiment=exp3)
    nodes = [n1, n2, n3]
    node_map = {n.experiment.id: n for n in nodes}

    qb = QueryBuilder(nodes, node_map.get)
    worst = qb.worst('accuracy')
    assert worst is not None
    assert worst.experiment.id == 'e1'

  def test_best_lower_is_better_with_ties(self):
    exp1 = make_experiment('e1', status='completed', metrics={'latency': 50.0})
    exp2 = make_experiment('e2', status='completed', metrics={'latency': 50.0})
    exp3 = make_experiment('e3', status='completed', metrics={'latency': 100.0})

    n1 = Node(experiment=exp1)
    n2 = Node(experiment=exp2)
    n3 = Node(experiment=exp3)
    nodes = [n1, n2, n3]
    node_map = {n.experiment.id: n for n in nodes}

    qb = QueryBuilder(nodes, node_map.get)
    best = qb.best('latency', higher_is_better=False)
    assert best is not None
    assert best.experiment.id == 'e1'


# --- ancestors_of / descendants_of / siblings_of with deep trees ---


class TestDeepTreeStructural:
  def test_ancestors_of_deepest_node(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    ancestors = qb.ancestors_of('d4').all()
    ids = [n.experiment.id for n in ancestors]
    assert ids == ['d3', 'd2', 'd1', 'root']

  def test_ancestors_of_mid_node(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    ancestors = qb.ancestors_of('d2').all()
    ids = [n.experiment.id for n in ancestors]
    assert ids == ['d1', 'root']

  def test_descendants_of_root_in_deep_tree(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    descendants = qb.descendants_of('root').all()
    ids = {n.experiment.id for n in descendants}
    assert ids == {'d1', 'd2', 'd3', 'd4', 's1', 's2'}

  def test_descendants_of_mid_node(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    descendants = qb.descendants_of('d1').all()
    ids = {n.experiment.id for n in descendants}
    assert ids == {'d2', 'd3', 'd4'}

  def test_descendants_of_leaf_node(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    descendants = qb.descendants_of('d4').all()
    assert descendants == []

  def test_siblings_of_in_deep_tree(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    siblings = qb.siblings_of('d1').all()
    ids = {n.experiment.id for n in siblings}
    assert ids == {'s1', 's2'}

  def test_siblings_of_root_nodes(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    siblings = qb.siblings_of('root').all()
    ids = {n.experiment.id for n in siblings}
    assert ids == {'root2'}

  def test_siblings_of_only_child(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    siblings = qb.siblings_of('d2').all()
    assert siblings == []

  def test_depth_in_deep_tree(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)

    assert qb.depth(0).count() == 2
    assert qb.depth(1).count() == 3
    assert qb.depth(2).count() == 1
    assert qb.depth(3).count() == 1
    assert qb.depth(4).count() == 1
    assert qb.depth(5).count() == 0


# --- metric_between inclusive boundaries ---


class TestMetricBetweenInclusive:
  def test_value_exactly_at_low_included(self):
    exp_low = make_experiment('at-low', status='completed', metrics={'accuracy': 0.5})
    exp_mid = make_experiment('mid', status='completed', metrics={'accuracy': 0.7})
    exp_high = make_experiment('at-high', status='completed', metrics={'accuracy': 0.9})
    exp_below = make_experiment('below', status='completed', metrics={'accuracy': 0.49})
    exp_above = make_experiment('above', status='completed', metrics={'accuracy': 0.91})

    nodes = [
      Node(experiment=exp_low),
      Node(experiment=exp_mid),
      Node(experiment=exp_high),
      Node(experiment=exp_below),
      Node(experiment=exp_above),
    ]
    node_map = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, node_map.get)

    result = qb.metric_between('accuracy', 0.5, 0.9).all()
    ids = {n.experiment.id for n in result}
    assert 'at-low' in ids
    assert 'at-high' in ids
    assert 'mid' in ids
    assert 'below' not in ids
    assert 'above' not in ids

  def test_value_one_below_low_excluded(self):
    exp = make_experiment('below', status='completed', metrics={'score': 4.99})
    nodes = [Node(experiment=exp)]
    node_map = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, node_map.get)

    result = qb.metric_between('score', 5.0, 10.0).all()
    assert len(result) == 0

  def test_value_exactly_at_boundaries(self):
    exp_exact = make_experiment('exact', status='completed', metrics={'score': 5.0})
    nodes = [Node(experiment=exp_exact)]
    node_map = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, node_map.get)

    result = qb.metric_between('score', 5.0, 5.0).all()
    assert len(result) == 1
    assert result[0].experiment.id == 'exact'


# --- Immutability: chain creates new QueryBuilder (id differs) ---


class TestImmutability:
  def test_chain_creates_new_querybuilder(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    completed_qb = qb.completed()
    assert qb is not completed_qb
    assert id(qb) != id(completed_qb)

  def test_original_unaffected_by_chain(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    original_count = qb.count()

    _ = qb.completed()
    _ = qb.failed()
    _ = qb.metric_gt('accuracy', 0.99)

    assert qb.count() == original_count

  def test_multiple_chains_independent(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)

    completed = qb.completed()
    failed = qb.failed()
    pending = qb.pending()

    assert completed.count() == 6
    assert failed.count() == 1
    assert pending.count() == 1
    assert qb.count() == 8

  def test_chained_steps_accumulate(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)

    step1 = qb.completed()
    step2 = step1.metric_gt('accuracy', 0.5)
    step3 = step2.order_by_metric('accuracy')

    assert 'completed()' in step3.explain()
    assert 'metric_gt' in step3.explain()
    assert 'order_by_metric' in step3.explain()
    assert 'completed()' not in qb.explain()


# --- order_by with metric, epoch, status ---


class TestOrderByVariousKeys:
  def test_order_by_metric_descending(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.has_metric('accuracy').order_by_metric('accuracy').all()
    values = [n.experiment.metrics['accuracy'] for n in result]
    assert values == sorted(values, reverse=True)

  def test_order_by_metric_ascending(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.has_metric('accuracy').order_by_metric('accuracy', ascending=True).all()
    values = [n.experiment.metrics['accuracy'] for n in result]
    assert values == sorted(values)

  def test_order_by_epoch(self):
    exp1 = make_experiment('e1', status='completed', metrics={'accuracy': 0.7})
    exp1.epoch = 3
    exp2 = make_experiment('e2', status='completed', metrics={'accuracy': 0.8})
    exp2.epoch = 1
    exp3 = make_experiment('e3', status='completed', metrics={'accuracy': 0.9})
    exp3.epoch = 2

    nodes = [Node(experiment=exp1), Node(experiment=exp2), Node(experiment=exp3)]
    node_map = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, node_map.get)

    result = qb.order_by('epoch', ascending=True).all()
    epochs = [n.experiment.epoch for n in result]
    assert epochs == [1, 2, 3]

  def test_order_by_status(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.order_by('status', ascending=True).all()
    statuses = [n.experiment.status.value for n in result]
    assert statuses == sorted(statuses)

  def test_order_by_id(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.order_by('id', ascending=True).all()
    ids = [n.experiment.id for n in result]
    assert ids == sorted(ids)

  def test_order_by_id_descending(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.order_by('id', ascending=False).all()
    ids = [n.experiment.id for n in result]
    assert ids == sorted(ids, reverse=True)

  def test_order_by_metric_without_metric_appends_to_end(self):
    exp_with = make_experiment('has-it', status='completed', metrics={'accuracy': 0.5})
    exp_without = make_experiment('no-metric', status='completed')

    nodes = [Node(experiment=exp_without), Node(experiment=exp_with)]
    node_map = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, node_map.get)

    result = qb.order_by_metric('accuracy').all()
    assert result[0].experiment.id == 'has-it'
    assert result[1].experiment.id == 'no-metric'


# --- Additional edge cases ---


class TestEdgeCases:
  def test_filter_on_nonexistent_attribute_raises(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    with pytest.raises(AttributeError):
      qb.filter(nonexistent_attr='value').all()

  def test_where_with_always_true(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.where(_accept_all_nodes).all()
    assert len(result) == len(nodes)

  def test_where_with_always_false(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.where(_reject_all_nodes).all()
    assert result == []

  def test_render_nonempty(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    rendered = qb.completed().render()
    assert '|' in rendered
    assert 'root' in rendered
    assert 'completed' in rendered

  def test_exists_true(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    assert qb.exists() is True

  def test_exists_false_after_impossible_filter(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    assert qb.metric_gt('accuracy', 999.0).exists() is False

  def test_first_on_non_empty(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    first = qb.first()
    assert first is not None
    assert first.experiment.id == nodes[0].experiment.id

  def test_cancelled_filter(self):
    exp = make_experiment('c1', status='cancelled')
    nodes = [Node(experiment=exp)]
    node_map = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, node_map.get)
    assert qb.cancelled().count() == 1

  def test_leaves_in_deep_tree(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    leaves = qb.leaves().all()
    ids = {n.experiment.id for n in leaves}
    assert ids == {'d4', 's1', 's2', 'root2'}

  def test_depth_range_spanning_all(self):
    nodes, node_map = _build_deep_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.depth_range(0, 100).all()
    assert len(result) == len(nodes)


# --- Wide tree performance guardrail ---


def test_descendants_of_root_wide_tree() -> None:
  width = 400
  root = make_experiment('root-wide', status='completed', metrics={'accuracy': 0.1})
  n_root = Node(experiment=root)
  nodes: list[Node] = [n_root]
  node_map = {root.id: n_root}
  for i in range(width):
    c = make_experiment(f'cw{i}', status='pending')
    n = Node(experiment=c, parent=n_root)
    nodes.append(n)
    node_map[c.id] = n
  qb = QueryBuilder(nodes, node_map.get)
  descendants = qb.descendants_of('root-wide').all()
  assert len(descendants) == width
