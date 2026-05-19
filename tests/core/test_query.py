"""Tests for QueryBuilder composable query engine."""

from autopilot.core.enums import Status
from autopilot.core.node import Node
from autopilot.core.query import QueryBuilder
from tests.core.conftest import make_experiment


def _build_tree():
  """Build a small tree:

  root (completed, accuracy=0.7, latency=100)
    +-- child1 (completed, accuracy=0.8, latency=90)
    |   +-- grandchild1 (failed)
    |   +-- grandchild2 (completed, accuracy=0.85, latency=85)
    +-- child2 (running)
    +-- child3 (pending)
  root2 (cancelled)
  """
  n_root = Node(
    experiment=make_experiment(
      'root',
      status='completed',
      hypothesis='baseline',
      metrics={'accuracy': 0.7, 'latency': 100.0},
    ),
  )
  n_child1 = Node(
    experiment=make_experiment(
      'child1',
      status='completed',
      hypothesis='cot prompting',
      metrics={'accuracy': 0.8, 'latency': 90.0},
    ),
    parent=n_root,
  )
  n_gc1 = Node(
    experiment=make_experiment('gc1', status='failed', hypothesis='bad idea'),
    parent=n_child1,
  )
  n_gc2 = Node(
    experiment=make_experiment(
      'gc2',
      status='completed',
      hypothesis='refinement',
      metrics={'accuracy': 0.85, 'latency': 85.0},
    ),
    parent=n_child1,
  )
  n_child2 = Node(
    experiment=make_experiment('child2', status='running', hypothesis='few-shot'),
    parent=n_root,
  )
  n_child3 = Node(
    experiment=make_experiment('child3', status='pending', hypothesis='zero-shot'),
    parent=n_root,
  )
  n_root2 = Node(
    experiment=make_experiment('root2', status='cancelled', hypothesis='abandoned'),
  )

  nodes = [n_root, n_child1, n_gc1, n_gc2, n_child2, n_child3, n_root2]
  node_map = {n.experiment.id: n for n in nodes}
  return nodes, node_map


def _make_qb(nodes=None, node_map=None):
  if nodes is None:
    nodes, node_map = _build_tree()
  assert node_map is not None
  return QueryBuilder(nodes, node_map.get)


# --- filter ---


class TestFilter:
  def test_filter_by_status(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.filter(status=Status.completed).all()
    ids = {n.experiment.id for n in result}
    assert ids == {'root', 'child1', 'gc2'}

  def test_filter_by_single_field(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.filter(hypothesis='baseline').all()
    assert len(result) == 1
    assert result[0].experiment.id == 'root'

  def test_filter_by_multiple_fields(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.filter(status=Status.completed, hypothesis='cot prompting').all()
    assert len(result) == 1
    assert result[0].experiment.id == 'child1'


# --- convenience status methods ---


class TestConvenienceMethods:
  def test_completed(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    ids = {n.experiment.id for n in qb.completed().all()}
    assert ids == {'root', 'child1', 'gc2'}

  def test_failed(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    ids = {n.experiment.id for n in qb.failed().all()}
    assert ids == {'gc1'}

  def test_running(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    ids = {n.experiment.id for n in qb.running().all()}
    assert ids == {'child2'}

  def test_pending(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    ids = {n.experiment.id for n in qb.pending().all()}
    assert ids == {'child3'}

  def test_cancelled(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    ids = {n.experiment.id for n in qb.cancelled().all()}
    assert ids == {'root2'}

  def test_terminal(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    ids = {n.experiment.id for n in qb.terminal().all()}
    assert ids == {'root', 'child1', 'gc1', 'gc2', 'root2'}


# --- where ---


class TestWhere:
  def test_custom_predicate(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.where(lambda n: 'cot' in (n.experiment.hypothesis or '')).all()
    assert len(result) == 1
    assert result[0].experiment.id == 'child1'


# --- exclude ---


class TestExclude:
  def test_exclude_by_status(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.exclude(status=Status.pending).all()
    ids = {n.experiment.id for n in result}
    assert 'child3' not in ids
    assert len(result) == 6


# --- metric predicates ---


class TestMetricPredicates:
  def test_metric_gt(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.metric_gt('accuracy', 0.75).all()
    ids = {n.experiment.id for n in result}
    assert ids == {'child1', 'gc2'}

  def test_metric_lt(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.metric_lt('latency', 91.0).all()
    ids = {n.experiment.id for n in result}
    assert ids == {'child1', 'gc2'}

  def test_metric_between(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.metric_between('accuracy', 0.75, 0.82).all()
    ids = {n.experiment.id for n in result}
    assert ids == {'child1'}

  def test_has_metric(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.has_metric('accuracy').all()
    ids = {n.experiment.id for n in result}
    assert ids == {'root', 'child1', 'gc2'}


# --- structural queries ---


class TestStructural:
  def test_ancestors_of(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.ancestors_of('gc2').all()
    ids = [n.experiment.id for n in result]
    assert ids == ['child1', 'root']

  def test_ancestors_of_root(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.ancestors_of('root').all()
    assert result == []

  def test_ancestors_of_nonexistent(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.ancestors_of('nonexistent').all()
    assert result == []

  def test_descendants_of(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.descendants_of('root').all()
    ids = {n.experiment.id for n in result}
    assert ids == {'child1', 'gc1', 'gc2', 'child2', 'child3'}

  def test_descendants_of_leaf(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.descendants_of('gc2').all()
    assert result == []

  def test_children_of(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.children_of('root').all()
    ids = {n.experiment.id for n in result}
    assert ids == {'child1', 'child2', 'child3'}

  def test_children_of_not_grandchildren(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.children_of('root').all()
    ids = {n.experiment.id for n in result}
    assert 'gc1' not in ids
    assert 'gc2' not in ids

  def test_siblings_of(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.siblings_of('child1').all()
    ids = {n.experiment.id for n in result}
    assert ids == {'child2', 'child3'}

  def test_siblings_of_excludes_self(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.siblings_of('child1').all()
    ids = {n.experiment.id for n in result}
    assert 'child1' not in ids

  def test_siblings_of_root(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.siblings_of('root').all()
    ids = {n.experiment.id for n in result}
    assert ids == {'root2'}

  def test_roots(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.roots().all()
    ids = {n.experiment.id for n in result}
    assert ids == {'root', 'root2'}

  def test_leaves(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.leaves().all()
    ids = {n.experiment.id for n in result}
    assert ids == {'gc1', 'gc2', 'child2', 'child3', 'root2'}

  def test_depth(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.depth(0).all()
    ids = {n.experiment.id for n in result}
    assert ids == {'root', 'root2'}

    result = qb.depth(1).all()
    ids = {n.experiment.id for n in result}
    assert ids == {'child1', 'child2', 'child3'}

    result = qb.depth(2).all()
    ids = {n.experiment.id for n in result}
    assert ids == {'gc1', 'gc2'}

  def test_depth_range(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.depth_range(1, 2).all()
    ids = {n.experiment.id for n in result}
    assert ids == {'child1', 'child2', 'child3', 'gc1', 'gc2'}


# --- ordering ---


class TestOrdering:
  def test_order_by_attribute(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.completed().order_by('id', ascending=True).all()
    ids = [n.experiment.id for n in result]
    assert ids == sorted(ids)

  def test_order_by_descending(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.completed().order_by('id', ascending=False).all()
    ids = [n.experiment.id for n in result]
    assert ids == sorted(ids, reverse=True)

  def test_order_by_metric(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.has_metric('accuracy').order_by_metric('accuracy').all()
    values = [n.experiment.metrics['accuracy'] for n in result]
    assert values == sorted(values, reverse=True)

  def test_order_by_metric_ascending(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.has_metric('accuracy').order_by_metric('accuracy', ascending=True).all()
    values = [n.experiment.metrics['accuracy'] for n in result]
    assert values == sorted(values)


# --- terminal methods ---


class TestTerminals:
  def test_best(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.completed().best('accuracy')
    assert result is not None
    assert result.experiment.id == 'gc2'
    assert result.experiment.metrics['accuracy'] == 0.85

  def test_best_lower_is_better(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.completed().best('latency', higher_is_better=False)
    assert result is not None
    assert result.experiment.id == 'gc2'

  def test_worst(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.completed().worst('accuracy')
    assert result is not None
    assert result.experiment.id == 'root'

  def test_worst_lower_is_better(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.completed().worst('latency', higher_is_better=False)
    assert result is not None
    assert result.experiment.id == 'root'

  def test_best_skips_missing_metric(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.best('accuracy')
    assert result is not None
    assert result.experiment.id == 'gc2'

  def test_best_all_missing_returns_none(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.best('nonexistent_metric')
    assert result is None

  def test_worst_skips_missing_metric(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.worst('accuracy')
    assert result is not None
    assert result.experiment.id == 'root'

  def test_worst_all_missing_returns_none(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.worst('nonexistent_metric')
    assert result is None

  def test_count(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    assert qb.count() == 7
    assert qb.completed().count() == 3

  def test_exists(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    assert qb.exists() is True
    assert qb.completed().exists() is True

  def test_first(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.completed().first()
    assert result is not None
    assert result.experiment.status == Status.completed

  def test_first_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.first() is None


# --- chaining ---


class TestChaining:
  def test_filter_metric_order_first(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.filter(status=Status.completed).metric_gt('accuracy', 0.75).order_by('id').first()
    assert result is not None
    assert result.experiment.status == Status.completed
    assert result.experiment.metrics['accuracy'] > 0.75

  def test_completed_metric_order_first(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.completed().metric_gt('accuracy', 0.75).order_by_metric('accuracy').first()
    assert result is not None
    assert result.experiment.id == 'gc2'

  def test_descendants_failed(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.descendants_of('root').failed().all()
    ids = {n.experiment.id for n in result}
    assert ids == {'gc1'}

  def test_immutable_chain(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    completed = qb.completed()
    failed = qb.failed()
    assert completed.count() == 3
    assert failed.count() == 1
    assert qb.count() == 7


# --- empty tree ---


class TestEmpty:
  def test_all_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.all() == []

  def test_first_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.first() is None

  def test_best_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.best('accuracy') is None

  def test_worst_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.worst('accuracy') is None

  def test_count_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.count() == 0

  def test_exists_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.exists() is False

  def test_render_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.render() == '(no results)'

  def test_explain_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.explain() == 'query: all nodes'

  def test_roots_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.roots().all() == []

  def test_leaves_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.leaves().all() == []

  def test_completed_empty(self):
    qb = QueryBuilder([], {}.get)
    assert qb.completed().count() == 0

  def test_chained_empty(self):
    qb = QueryBuilder([], {}.get)
    result = qb.filter(status=Status.completed).metric_gt('accuracy', 0.5).all()
    assert result == []


# --- large tree ---


class TestLargeTree:
  def test_50_nodes(self):
    root_exp = make_experiment(
      'root',
      status='completed',
      hypothesis='root',
      metrics={'accuracy': 0.5},
    )
    root_node = Node(experiment=root_exp)
    nodes = [root_node]
    node_map = {'root': root_node}

    parent = root_node
    for i in range(50):
      exp = make_experiment(
        f'exp-{i}',
        status='completed',
        hypothesis=f'hypothesis {i}',
        metrics={'accuracy': 0.5 + (i + 1) * 0.01},
      )
      node = Node(experiment=exp, parent=parent)
      nodes.append(node)
      node_map[exp.id] = node
      parent = node

    qb = QueryBuilder(nodes, node_map.get)
    assert qb.count() == 51
    assert qb.completed().count() == 51

    best = qb.best('accuracy')
    assert best is not None
    assert best.experiment.id == 'exp-49'

    leaves = qb.leaves().all()
    assert len(leaves) == 1
    assert leaves[0].experiment.id == 'exp-49'

    ancestors = qb.ancestors_of('exp-49').all()
    assert len(ancestors) == 50

    descendants = qb.descendants_of('root').all()
    assert len(descendants) == 50


# --- explain ---


class TestExplain:
  def test_explain_no_steps(self):
    qb = QueryBuilder([], {}.get)
    assert qb.explain() == 'query: all nodes'

  def test_explain_with_steps(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.completed().metric_gt('accuracy', 0.7).explain()
    assert 'completed()' in result
    assert 'metric_gt' in result


# --- render ---


class TestRender:
  def test_render_nonempty(self):
    nodes, node_map = _build_tree()
    qb = _make_qb(nodes, node_map)
    result = qb.completed().render()
    assert '|' in result
    assert 'root' in result
    assert 'completed' in result

  def test_render_empty(self):
    qb = QueryBuilder([], {}.get)
    result = qb.render()
    assert result == '(no results)'
