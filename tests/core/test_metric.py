from autopilot.core.metric import CompositeMetric, Metric
from autopilot.core.models import Datum
from autopilot.core.module import Module
from autopilot.core.parameter import Parameter
import pytest


class _SumMetric(Metric):
  def __init__(self):
    super().__init__()
    self._total = 0.0
    self._count = 0

  def update(self, datum):
    for v in datum.metrics.values():
      self._total += v
      self._count += 1

  def compute(self):
    avg = self._total / self._count if self._count else 0.0
    return {'avg': avg}

  def reset(self):
    self._total = 0.0
    self._count = 0


class _CountMetric(Metric):
  def __init__(self):
    super().__init__()
    self._n = 0

  def update(self, datum):
    self._n += 1

  def compute(self):
    return {'count': float(self._n)}

  def reset(self):
    self._n = 0


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

  def reset(self):
    pass


class _StubMetric(Metric):
  def update(self, datum):
    pass

  def compute(self):
    return {}

  def reset(self):
    pass


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

  def test_reset_clears_state(self) -> None:
    m = Metric()
    m._state['x'] = 1
    m.reset()
    assert m._state == {}

  def test_name_returns_class_name(self) -> None:
    assert Metric().name() == 'Metric'

  def test_subclass_name(self) -> None:
    assert _SumMetric().name() == '_SumMetric'

  def test_repr(self) -> None:
    assert repr(Metric()) == 'Metric()'


class TestMetricForward:
  def test_forward_calls_update_and_returns_compute(self) -> None:
    m = _SumMetric()
    d = Datum(metrics={'x': 10.0})
    assert m.forward(d) == {'avg': 10.0}

  def test_forward_accumulates(self) -> None:
    m = _SumMetric()
    m.forward(Datum(metrics={'x': 10.0}))
    out = m.forward(Datum(metrics={'y': 20.0}))
    assert out == {'avg': 15.0}


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


class TestCompositeMetric:
  def test_compose_via_add(self) -> None:
    c = _SumMetric() + _CountMetric()
    assert isinstance(c, CompositeMetric)

  def test_parts_not_metrics(self) -> None:
    c = _SumMetric() + _CountMetric()
    assert hasattr(c, '_parts')
    assert isinstance(c._parts, list)
    assert not hasattr(c, '_metrics')

  def test_children_registered(self) -> None:
    c = _SumMetric() + _CountMetric()
    named = dict(c.named_modules())
    assert 'metric_0' in named
    assert 'metric_1' in named

  def test_update_delegates(self) -> None:
    c = _SumMetric() + _CountMetric()
    c.update(Datum(metrics={'x': 2.0}))
    assert c.compute() == {'avg': 2.0, 'count': 1.0}

  def test_compute_merges(self) -> None:
    c = _SumMetric() + _CountMetric()
    c.update(Datum(metrics={'x': 2.0}))
    c.update(Datum(metrics={'x': 4.0}))
    assert c.compute() == {'avg': 3.0, 'count': 2.0}

  def test_reset_all(self) -> None:
    c = _SumMetric() + _CountMetric()
    c.update(Datum(metrics={'x': 1.0}))
    c.reset()
    assert c.compute() == {'avg': 0.0, 'count': 0.0}

  def test_forward_works(self) -> None:
    c = _SumMetric() + _CountMetric()
    out = c.forward(Datum(metrics={'x': 10.0}))
    assert out == {'avg': 10.0, 'count': 1.0}

  def test_chain_three(self) -> None:
    a = _SumMetric()
    b = _CountMetric()
    c = _SumMetric()
    comp = a + b + c
    assert len(comp._parts) == 2
    assert isinstance(comp._parts[0], CompositeMetric)

  def test_repr(self) -> None:
    c = _SumMetric() + _CountMetric()
    assert repr(c) == 'CompositeMetric([_SumMetric, _CountMetric])'


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
