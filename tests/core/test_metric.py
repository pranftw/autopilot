"""Tests for torchmetrics-style metric base classes."""

from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.module import Module
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
import pytest


class _SumMetric(Metric):
  higher_is_better = True

  def __init__(self):
    super().__init__()
    self.add_state('_total', 0.0)
    self.add_state('_count', 0)

  def update(self, datum):
    for v in datum.metrics.values():
      self._total += v
      self._count += 1

  def compute(self):
    avg = self._total / self._count if self._count else 0.0
    return {'avg': avg}


class _CountMetric(Metric):
  higher_is_better = True

  def __init__(self):
    super().__init__()
    self.add_state('_n', 0)

  def update(self, datum):
    self._n += 1

  def compute(self):
    return {'count': float(self._n)}


class _ErrorRateMetric(Metric):
  higher_is_better = False

  def __init__(self):
    super().__init__()
    self.add_state('_errors', 0)
    self.add_state('_total', 0)

  def update(self, datum):
    self._total += 1
    if not datum.success:
      self._errors += 1

  def compute(self):
    return {'error_rate': self._errors / self._total if self._total else 0.0}


class _LeafModule(Module):
  def forward(self, *args, **kwargs):
    return Datum(success=True)


class _MetricWithParam(Metric):
  def __init__(self):
    super().__init__()
    self.w = Parameter()

  def update(self, datum):
    pass

  def compute(self):
    return {}


class _StubMetric(Metric):
  def update(self, datum):
    pass

  def compute(self):
    return {}


class TestMetricIsModule:
  def test_metric_is_module_instance(self) -> None:
    assert isinstance(Metric(), Module)

  def test_metric_has_modules_dict(self) -> None:
    m = Metric()
    assert isinstance(m._modules, dict)
    assert isinstance(m._parameters, dict)

  def test_metric_no_metrics_dict(self) -> None:
    m = Metric()
    assert not hasattr(m, '_metrics')


class TestMetricBase:
  def test_update_raises(self) -> None:
    with pytest.raises(NotImplementedError):
      Metric().update(Datum())

  def test_compute_raises(self) -> None:
    with pytest.raises(NotImplementedError):
      Metric().compute()

  def test_name_returns_class_name(self) -> None:
    assert Metric().name() == 'Metric'

  def test_subclass_name(self) -> None:
    assert _SumMetric().name() == '_SumMetric'

  def test_repr(self) -> None:
    assert repr(Metric()) == 'Metric()'

  def test_higher_is_better_default_none(self) -> None:
    assert Metric.higher_is_better is None
    assert Metric().higher_is_better is None

  def test_no_forward_on_metric(self) -> None:
    m = _SumMetric()
    with pytest.raises(NotImplementedError):
      m.forward(Datum())


class TestAddState:
  def test_add_state_value_default(self) -> None:
    m = _SumMetric()
    assert m._total == 0.0
    assert m._count == 0

  def test_add_state_factory_default(self) -> None:
    class ListMetric(Metric):
      def __init__(self):
        super().__init__()
        self.add_state('items', list)

      def update(self, datum):
        self.items.append(datum)

      def compute(self):
        return {'n': float(len(self.items))}

    m = ListMetric()
    assert m.items == []
    assert isinstance(m.items, list)

  def test_add_state_dict_factory(self) -> None:
    class DictMetric(Metric):
      def __init__(self):
        super().__init__()
        self.add_state('acc', dict)

      def update(self, datum):
        pass

      def compute(self):
        return {}

    m = DictMetric()
    assert m.acc == {}
    assert isinstance(m.acc, dict)

  def test_reset_restores_value(self) -> None:
    m = _SumMetric()
    m.update(Datum(metrics={'x': 10.0}))
    assert m._total == 10.0
    m.reset()
    assert m._total == 0.0
    assert m._count == 0

  def test_reset_restores_factory(self) -> None:
    class ListMetric(Metric):
      def __init__(self):
        super().__init__()
        self.add_state('items', list)

      def update(self, datum):
        self.items.append(1)

      def compute(self):
        return {}

    m = ListMetric()
    m.update(Datum())
    assert len(m.items) == 1
    m.reset()
    assert m.items == []

  def test_reset_multiple_states(self) -> None:
    m = _SumMetric()
    m.update(Datum(metrics={'x': 5.0}))
    m.reset()
    assert m._total == 0.0
    assert m._count == 0

  def test_reset_resets_update_count(self) -> None:
    m = _SumMetric()
    m.update(Datum(metrics={'x': 1.0}))
    assert m.update_count == 1
    m.reset()
    assert m.update_count == 0

  def test_add_state_accessible_as_attribute(self) -> None:
    m = _SumMetric()
    assert hasattr(m, '_total')
    assert hasattr(m, '_count')

  def test_add_state_in_defaults_dict(self) -> None:
    m = _SumMetric()
    assert '_total' in m._defaults
    assert '_count' in m._defaults

  def test_no_manual_reset_needed(self) -> None:
    m = _SumMetric()
    m.update(Datum(metrics={'x': 10.0}))
    m.reset()
    assert m.compute() == {'avg': 0.0}


class TestUpdateWrapping:
  def test_update_count_incremented(self) -> None:
    m = _SumMetric()
    m.update(Datum(metrics={'x': 1.0}))
    m.update(Datum(metrics={'x': 2.0}))
    assert m.update_count == 2

  def test_update_count_property(self) -> None:
    m = _CountMetric()
    assert m.update_count == 0
    m.update(Datum())
    assert m.update_count == 1

  def test_update_count_reset_by_reset(self) -> None:
    m = _SumMetric()
    m.update(Datum(metrics={'x': 1.0}))
    m.reset()
    assert m.update_count == 0

  def test_wrapped_update_preserves_behavior(self) -> None:
    m = _SumMetric()
    m.update(Datum(metrics={'x': 10.0}))
    m.update(Datum(metrics={'y': 20.0}))
    assert m.compute() == {'avg': 15.0}


class TestHigherIsBetter:
  def test_default_none(self) -> None:
    assert Metric.higher_is_better is None

  def test_subclass_true(self) -> None:
    assert _SumMetric.higher_is_better is True
    assert _SumMetric().higher_is_better is True

  def test_subclass_false(self) -> None:
    assert _ErrorRateMetric.higher_is_better is False
    assert _ErrorRateMetric().higher_is_better is False

  def test_inheritable(self) -> None:
    class SubSum(_SumMetric):
      pass

    assert SubSum.higher_is_better is True


class TestMetricClone:
  def test_clone_fresh_state(self) -> None:
    m = _SumMetric()
    m.update(Datum(metrics={'x': 5.0}))
    c = m.clone()
    assert c._total == 5.0
    c.reset()
    assert c._total == 0.0
    assert m._total == 5.0

  def test_clone_same_config(self) -> None:
    m = _SumMetric()
    c = m.clone()
    assert type(c) is type(m)

  def test_clone_independent(self) -> None:
    m = _SumMetric()
    c = m.clone()
    c.update(Datum(metrics={'x': 100.0}))
    assert m._total == 0.0


class TestMetricUpdateComputeReset:
  def test_update_compute_cycle(self) -> None:
    m = _SumMetric()
    m.update(Datum(metrics={'a': 1.0}))
    m.update(Datum(metrics={'b': 2.0}))
    m.update(Datum(metrics={'c': 3.0}))
    assert m.compute() == {'avg': 2.0}

  def test_reset_clears(self) -> None:
    m = _SumMetric()
    m.update(Datum(metrics={'x': 5.0}))
    m.reset()
    assert m.compute() == {'avg': 0.0}

  def test_multiple_cycles(self) -> None:
    m = _SumMetric()
    m.update(Datum(metrics={'x': 4.0}))
    assert m.compute() == {'avg': 4.0}
    m.reset()
    m.update(Datum(metrics={'x': 1.0}))
    m.update(Datum(metrics={'x': 3.0}))
    assert m.compute() == {'avg': 2.0}


class TestMetricOnModule:
  def test_metric_registers_in_modules(self) -> None:
    mod = Module()
    m = _SumMetric()
    mod.score = m
    assert mod._modules['score'] is m

  def test_metric_in_children(self) -> None:
    mod = Module()
    m = _SumMetric()
    mod.score = m
    assert m in list(mod.children())

  def test_metric_in_modules_iterator(self) -> None:
    mod = Module()
    m = _SumMetric()
    mod.score = m
    assert m in list(mod.modules())

  def test_metric_in_named_modules(self) -> None:
    mod = Module()
    m = _SumMetric()
    mod.score = m
    named = dict(mod.named_modules())
    assert named['score'] is m

  def test_train_eval_propagation(self) -> None:
    mod = Module()
    m = _SumMetric()
    mod.score = m
    mod.eval()
    assert mod.training is False
    assert m.training is False
    mod.train()
    assert mod.training is True
    assert m.training is True

  def test_apply_visits_metric(self) -> None:
    mod = Module()
    m = _SumMetric()
    mod.score = m
    visited: list[str] = []

    def record(module: Module) -> None:
      visited.append(type(module).__name__)

    mod.apply(record)
    assert visited == ['_SumMetric', 'Module']

  def test_metric_in_state_dict(self) -> None:
    mod = Module()
    mod.m = _MetricWithParam()
    state = mod.state_dict()
    assert 'm.w' in state

    mod_empty = Module()
    mod_empty.m = _StubMetric()
    assert mod_empty.state_dict() == {}

  def test_metric_and_parameter_coexist(self) -> None:
    mod = Module()
    mod.m = _SumMetric()
    mod.p = Parameter()
    assert mod.m is mod._modules['m']
    assert mod.p is mod._parameters['p']


class TestMetricCollection:
  def test_compose_via_add(self) -> None:
    c = _SumMetric() + _CountMetric()
    assert isinstance(c, MetricCollection)

  def test_from_list(self) -> None:
    c = MetricCollection([_SumMetric(), _CountMetric()])
    assert isinstance(c, MetricCollection)

  def test_from_dict(self) -> None:
    c = MetricCollection({'a': _SumMetric(), 'b': _CountMetric()})
    assert isinstance(c, MetricCollection)

  def test_duplicate_names_raise(self) -> None:
    with pytest.raises(ValueError, match='duplicate'):
      MetricCollection([_SumMetric(), _SumMetric()])

  def test_update_delegates(self) -> None:
    c = _SumMetric() + _CountMetric()
    c.update(Datum(metrics={'x': 2.0}))
    assert c.compute() == {'avg': 2.0, 'count': 1.0}

  def test_compute_merges(self) -> None:
    c = _SumMetric() + _CountMetric()
    c.update(Datum(metrics={'x': 2.0}))
    c.update(Datum(metrics={'x': 4.0}))
    assert c.compute() == {'avg': 3.0, 'count': 2.0}

  def test_key_collision_raises(self) -> None:
    class SameKeyMetric(Metric):
      def __init__(self):
        super().__init__()

      def update(self, datum):
        pass

      def compute(self):
        return {'avg': 1.0}

    c = MetricCollection([_SumMetric(), SameKeyMetric()])
    c.update(Datum(metrics={'x': 1.0}))
    with pytest.raises(ValueError, match='collision'):
      c.compute()

  def test_prefix(self) -> None:
    c = MetricCollection([_SumMetric()], prefix='val_')
    c.update(Datum(metrics={'x': 5.0}))
    assert 'val_avg' in c.compute()

  def test_postfix(self) -> None:
    c = MetricCollection([_SumMetric()], postfix='_mean')
    c.update(Datum(metrics={'x': 5.0}))
    assert 'avg_mean' in c.compute()

  def test_prefix_and_postfix(self) -> None:
    c = MetricCollection([_SumMetric()], prefix='val_', postfix='_mean')
    c.update(Datum(metrics={'x': 5.0}))
    assert 'val_avg_mean' in c.compute()

  def test_reset_all(self) -> None:
    c = _SumMetric() + _CountMetric()
    c.update(Datum(metrics={'x': 1.0}))
    c.reset()
    assert c.compute() == {'avg': 0.0, 'count': 0.0}

  def test_children_registered_as_modules(self) -> None:
    c = MetricCollection({'a': _SumMetric(), 'b': _CountMetric()})
    named = dict(c.named_modules())
    assert 'a' in named
    assert 'b' in named

  def test_clone(self) -> None:
    c = MetricCollection([_SumMetric(), _CountMetric()])
    c.update(Datum(metrics={'x': 1.0}))
    c2 = c.clone()
    c2.reset()
    assert c.compute()['count'] == 1.0
    assert c2.compute()['count'] == 0.0

  def test_repr(self) -> None:
    c = MetricCollection({'a': _SumMetric(), 'b': _CountMetric()})
    r = repr(c)
    assert 'MetricCollection' in r
    assert 'a' in r
    assert 'b' in r

  def test_higher_is_better_none_for_collection(self) -> None:
    c = MetricCollection([_SumMetric(), _CountMetric()])
    assert c.higher_is_better is None

  def test_nested_collection(self) -> None:
    inner = MetricCollection([_SumMetric()], prefix='inner_')
    outer = MetricCollection({'group': inner, 'count': _CountMetric()})
    outer.update(Datum(metrics={'x': 5.0}))
    result = outer.compute()
    assert 'inner_avg' in result
    assert 'count' in result


class TestMetricCompetingStore:
  def test_metric_replaces_module(self) -> None:
    mod = Module()
    mod.x = _LeafModule()
    mod.x = _SumMetric()
    assert isinstance(mod._modules['x'], _SumMetric)

  def test_module_replaces_metric(self) -> None:
    mod = Module()
    mod.x = _SumMetric()
    child = _LeafModule()
    mod.x = child
    assert mod._modules['x'] is child

  def test_parameter_replaces_metric(self) -> None:
    mod = Module()
    mod.x = _SumMetric()
    p = Parameter()
    mod.x = p
    assert 'x' in mod._parameters
    assert 'x' not in mod._modules
    assert mod.x is p
