"""Tests for Node dataclass -- tree entry for the experiment DAG."""

from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from dataclasses import dataclass
import pytest


def _make_experiment(id_: str, **kwargs) -> Experiment:
  exp = Experiment(experiment_id=id_, **kwargs)
  return exp


class TestNodeCreation:
  def test_create_with_experiment(self) -> None:
    exp = _make_experiment('exp-1')
    node = Node(experiment=exp)
    assert node.experiment is exp

  def test_create_with_parent_and_baseline(self) -> None:
    exp1 = _make_experiment('exp-1')
    exp2 = _make_experiment('exp-2')
    exp3 = _make_experiment('exp-3')
    parent = Node(experiment=exp1)
    baseline = Node(experiment=exp2)
    node = Node(experiment=exp3, parent=parent, baseline=baseline)
    assert node.parent is parent
    assert node.baseline is baseline

  def test_default_parent_none(self) -> None:
    exp = _make_experiment('exp-1')
    node = Node(experiment=exp)
    assert node.parent is None

  def test_default_baseline_none(self) -> None:
    exp = _make_experiment('exp-1')
    node = Node(experiment=exp)
    assert node.baseline is None


class TestNodeIdentity:
  def test_identity_is_experiment_id(self) -> None:
    exp = _make_experiment('my-exp')
    node = Node(experiment=exp)
    assert node.experiment.id == 'my-exp'

  def test_two_nodes_different_identity(self) -> None:
    n1 = Node(experiment=_make_experiment('a'))
    n2 = Node(experiment=_make_experiment('b'))
    assert n1.experiment.id != n2.experiment.id


class TestNodeObjectAccess:
  def test_experiment_hypothesis(self) -> None:
    exp = _make_experiment('e', hypothesis='test it')
    node = Node(experiment=exp)
    assert node.experiment.hypothesis == 'test it'

  def test_parent_experiment_metrics(self) -> None:
    parent_exp = _make_experiment('p')
    parent_exp.start()
    parent_exp.complete(metrics={'accuracy': 0.85})
    parent_node = Node(experiment=parent_exp)

    child_exp = _make_experiment('c')
    child_node = Node(experiment=child_exp, parent=parent_node)
    assert child_node.parent is not None
    assert child_node.parent.experiment.metrics == {'accuracy': 0.85}

  def test_baseline_experiment_id(self) -> None:
    baseline_exp = _make_experiment('baseline')
    baseline_node = Node(experiment=baseline_exp)

    exp = _make_experiment('candidate')
    node = Node(experiment=exp, baseline=baseline_node)
    assert node.baseline is not None
    assert node.baseline.experiment.id == 'baseline'


class TestNodeToDict:
  def test_serializes_experiment_as_id(self) -> None:
    exp = _make_experiment('exp-1')
    node = Node(experiment=exp)
    d = node.to_dict()
    assert d['experiment'] == 'exp-1'

  def test_serializes_parent_as_id(self) -> None:
    parent = Node(experiment=_make_experiment('parent'))
    child = Node(experiment=_make_experiment('child'), parent=parent)
    d = child.to_dict()
    assert d['parent'] == 'parent'

  def test_serializes_baseline_as_id(self) -> None:
    baseline = Node(experiment=_make_experiment('baseline'))
    node = Node(experiment=_make_experiment('exp'), baseline=baseline)
    d = node.to_dict()
    assert d['baseline'] == 'baseline'

  def test_none_parent_serializes_as_none(self) -> None:
    node = Node(experiment=_make_experiment('root'))
    d = node.to_dict()
    assert d['parent'] is None

  def test_none_baseline_serializes_as_none(self) -> None:
    node = Node(experiment=_make_experiment('fresh'))
    d = node.to_dict()
    assert d['baseline'] is None

  def test_includes_type_field(self) -> None:
    node = Node(experiment=_make_experiment('e'))
    d = node.to_dict()
    assert d['type'] == 'Node'


class TestNodeFromDict:
  def test_from_dict_hydrates_ids(self) -> None:
    exp1 = _make_experiment('exp-1')
    exp2 = _make_experiment('exp-2')
    node1 = Node(experiment=exp1)
    node2 = Node(experiment=exp2, parent=node1)

    objects = {'exp-1': node1, 'exp-2': node2}

    def resolver(id_str):
      if id_str in objects:
        obj = objects[id_str]
        if isinstance(obj, Node):
          return obj
        return obj
      msg = f'not found: {id_str}'
      raise KeyError(msg)

    data = node2.to_dict()
    exp_resolver_map = {'exp-1': exp1, 'exp-2': exp2}

    def full_resolver(id_str):
      if id_str in exp_resolver_map:
        return exp_resolver_map[id_str]
      msg = f'not found: {id_str}'
      raise KeyError(msg)

    restored = Node.from_dict(
      data,
      resolver=full_resolver,
    )
    assert restored.experiment.id == 'exp-2'

  def test_from_dict_with_none_fields(self) -> None:
    exp = _make_experiment('root')
    data = {'type': 'Node', 'experiment': 'root', 'parent': None, 'baseline': None}

    restored = Node.from_dict(data, resolver=lambda id_str: exp)
    assert restored.parent is None
    assert restored.baseline is None

  def test_round_trip(self) -> None:
    exp_root = _make_experiment('root', hypothesis='baseline')
    exp_root.start()
    exp_root.complete(metrics={'accuracy': 0.7})
    root_node = Node(experiment=exp_root)

    exp_child = _make_experiment('child', hypothesis='improve')
    child_node = Node(experiment=exp_child, parent=root_node, baseline=root_node)

    d = child_node.to_dict()

    experiments = {'root': exp_root, 'child': exp_child}
    nodes = {'root': root_node}

    def resolver(id_str):
      if id_str in nodes:
        return nodes[id_str]
      return experiments[id_str]

    restored = Node.from_dict(d, resolver=resolver)
    assert restored.experiment.id == 'child'
    assert restored.parent is not None
    assert restored.parent.experiment.id == 'root'
    assert restored.baseline is not None
    assert restored.baseline.experiment.id == 'root'

  def test_from_dict_without_node_types_creates_base(self) -> None:
    exp = _make_experiment('e')
    data = {'type': 'Node', 'experiment': 'e', 'parent': None, 'baseline': None}
    restored = Node.from_dict(data, resolver=lambda x: exp, node_types=None)
    assert type(restored) is Node

  def test_from_dict_with_node_types_resolves_subclass(self) -> None:
    @dataclass
    class TranslationNode(Node):
      source_lang: str = ''
      target_lang: str = ''

    qualname = TranslationNode.__qualname__
    exp = _make_experiment('e')
    data = {
      'type': qualname,
      'experiment': 'e',
      'parent': None,
      'baseline': None,
      'source_lang': 'en',
      'target_lang': 'hi',
    }
    restored = Node.from_dict(
      data,
      resolver=lambda x: exp,
      node_types={qualname: TranslationNode},
    )
    assert type(restored) is TranslationNode
    assert restored.source_lang == 'en'
    assert restored.target_lang == 'hi'

  def test_from_dict_unknown_type_raises(self) -> None:
    exp = _make_experiment('e')
    data = {'type': 'UnknownNode', 'experiment': 'e', 'parent': None, 'baseline': None}
    with pytest.raises(KeyError, match='unknown node type'):
      Node.from_dict(data, resolver=lambda x: exp, node_types={'Node': Node})


class TestNodeSubclass:
  def test_translation_node_creation(self) -> None:
    @dataclass
    class TranslationNode(Node):
      source_lang: str = ''
      target_lang: str = ''

    exp = _make_experiment('t1')
    node = TranslationNode(experiment=exp, source_lang='en', target_lang='fr')
    assert node.source_lang == 'en'
    assert node.target_lang == 'fr'
    assert node.experiment.id == 't1'

  def test_translation_node_to_dict(self) -> None:
    @dataclass
    class TranslationNode(Node):
      source_lang: str = ''
      target_lang: str = ''

    exp = _make_experiment('t1')
    node = TranslationNode(experiment=exp, source_lang='en', target_lang='fr')
    d = node.to_dict()
    assert d['type'] == type(node).__qualname__
    assert d['source_lang'] == 'en'
    assert d['target_lang'] == 'fr'
    assert d['experiment'] == 't1'

  def test_translation_node_round_trip(self) -> None:
    @dataclass
    class TranslationNode(Node):
      source_lang: str = ''
      target_lang: str = ''

    exp = _make_experiment('t1')
    node = TranslationNode(experiment=exp, source_lang='en', target_lang='hi')
    d = node.to_dict()
    qualname = type(node).__qualname__

    restored = Node.from_dict(
      d,
      resolver=lambda x: exp,
      node_types={qualname: TranslationNode},
    )
    assert type(restored) is TranslationNode
    assert restored.source_lang == 'en'
    assert restored.target_lang == 'hi'
    assert restored.experiment.id == 't1'

  def test_subclass_with_parent(self) -> None:
    @dataclass
    class TaggedNode(Node):
      tag: str = ''

    parent_exp = _make_experiment('p')
    parent_node = TaggedNode(experiment=parent_exp, tag='root')

    child_exp = _make_experiment('c')
    child_node = TaggedNode(experiment=child_exp, parent=parent_node, tag='child')

    d = child_node.to_dict()
    assert d['tag'] == 'child'
    assert d['parent'] == 'p'
