"""Executable API reference for agents -- 26 contract tests.

Each test pairs a **positive** example (correct API usage) with a **negative**
example (common agent mistake that fails with a specific exception type).  Pin
exception types, not message substrings (unless the message IS the contract).

Part of Dogfood V4 master plan (sub-plan 01).  Agents should read these tests
before inventing APIs -- they document the actual AutoPilot surface.
"""

from autopilot.ai.gradient import TextGradient
from autopilot.ai.store.file_store import FileStore
from autopilot.core.callbacks.checkpoint import CheckpointCallback
from autopilot.core.callbacks.diagnostics import DiagnosticsCallback
from autopilot.core.comparison import Delta, MetricsComparator
from autopilot.core.config import AutoPilotConfig
from autopilot.core.context import ContextEntry, ContextLog
from autopilot.core.diagnostics import Diagnostics
from autopilot.core.gradient import NumericGradient
from autopilot.core.graph import Graph
from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.module.module import Module
from autopilot.core.node import Node
from autopilot.core.ops import broadcast, select
from autopilot.core.parameter import Parameter
from autopilot.core.query import QueryBuilder
from autopilot.core.store.types import ConflictEntry, DiffEntry
from autopilot.core.types import Datum
import pytest

# ---------------------------------------------------------------------------
# 2.1  Datum / graph / ops contracts (tests 1, 6, 17-20)
# ---------------------------------------------------------------------------


class _LeafModule(Module):
  """Minimal leaf module that returns an EvalDatum with grad_fn wiring."""

  def __init__(self) -> None:
    super().__init__()
    self.p = Parameter()

  def forward(self, batch: Datum) -> Datum:
    return Datum()


def test_backward_via_datum_not_standalone() -> None:
  """backward is a Datum method or Graph method, not a standalone function."""
  mod = _LeafModule()
  result = mod(Datum())
  assert result.grad_fn is not None
  result.backward(NumericGradient(value=1.0))

  with pytest.raises(ImportError):
    exec('from autopilot.core.graph import backward')


def test_datum_items_must_be_datum_instances() -> None:
  """Datum.items must contain Datum instances; strings break serialization."""
  good = Datum(items=[Datum(), Datum()])
  assert len(good.to_dict()['items']) == 2

  bad = Datum(items=['x'])  # ty: ignore[invalid-argument-type]
  with pytest.raises(AttributeError):
    bad.to_dict()


def test_select_datum_first_arg() -> None:
  """select(datum, index) -- datum first, then index."""
  d = Datum(items=[Datum(), Datum(items=[Datum(), Datum()])])
  result = select(d, 0)
  assert isinstance(result, Datum)

  with pytest.raises(TypeError):
    select(0, d)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_broadcast_returns_datum_with_items() -> None:
  """broadcast returns a Datum whose .items has length n (not a list)."""
  d = Datum()
  result = broadcast(d, 3)
  assert isinstance(result, Datum)
  assert len(result.items) == 3
  assert not isinstance(result, list)


def test_datum_clone_preserves_id() -> None:
  """clone() preserves the datum's stable id."""
  d = Datum()
  c = d.clone()
  assert c.id == d.id


def test_retain_graph_on_graph_backward_only() -> None:
  """retain_graph is a Graph.backward kwarg, not a Datum.backward kwarg."""
  graph = Graph()
  assert 'retain_graph' in graph.backward.__code__.co_varnames

  d = Datum()
  with pytest.raises(TypeError):
    d.backward(NumericGradient(value=1.0), retain_graph=True)  # ty: ignore[unknown-argument]


# ---------------------------------------------------------------------------
# 2.2  Gradients and metrics (tests 2-5, 25-26)
# ---------------------------------------------------------------------------


def test_numeric_gradient_accumulate_not_add() -> None:
  """NumericGradient uses .accumulate(), not + operator."""
  g1 = NumericGradient(value=2.5)
  g2 = NumericGradient(value=0.5)
  acc = g1.accumulate(g2)
  assert acc.value == 3.0

  with pytest.raises(TypeError):
    g1 + g2  # ty: ignore[unsupported-operator]


def test_text_gradient_text_not_direction() -> None:
  """TextGradient uses text= kwarg (renamed from direction)."""
  tg = TextGradient(text='refine prompt')
  assert tg.text == 'refine prompt'

  with pytest.raises(TypeError, match='renamed'):
    TextGradient(direction='some feedback')


def test_metric_compute_requires_update() -> None:
  """Metric.compute() raises RuntimeError when called without prior update()."""

  class SimpleMetric(Metric):
    def update(self, datum: Datum) -> None:
      pass

    def compute(self) -> dict[str, float]:
      return {'x': 1.0}

  m = SimpleMetric()
  with pytest.raises(RuntimeError):
    m.compute()

  m.update(Datum())
  result = m.compute()
  assert result == {'x': 1.0}


def test_register_buffer_setattr_clears_buffer() -> None:
  """Assigning a plain value to a buffer name clears the buffer entry."""

  class Buffered(Module):
    def __init__(self) -> None:
      super().__init__()
      self.register_buffer('counter', 0)

    def forward(self, *args, **kwargs) -> Datum:
      return Datum()

  m = Buffered()
  assert 'counter' in m._buffers
  assert m.counter == 0

  m.counter = 42
  assert 'counter' not in m._buffers
  assert m.counter == 42


def test_metric_collection_not_dict_like_values() -> None:
  """MetricCollection is not a dict -- use update()/compute(), not .values()."""

  class CountMetric(Metric):
    def __init__(self) -> None:
      super().__init__()
      self.add_state('total', 0)

    def update(self, datum: Datum) -> None:
      self.total += 1

    def compute(self) -> dict[str, float]:
      return {'count': float(self.total)}

  mc = MetricCollection([CountMetric()])
  mc.update(Datum())
  result = mc.compute()
  assert 'count' in result

  assert not hasattr(mc, 'values') or not callable(getattr(mc, 'values', None))


def test_no_data_item_use_datum() -> None:
  """The correct class is Datum, not DataItem."""
  assert Datum is not None

  with pytest.raises(ImportError):
    exec('from autopilot.core.types import DataItem')


# ---------------------------------------------------------------------------
# 2.3  Policy gates and callbacks (tests 7-12)
# ---------------------------------------------------------------------------


def test_range_gate_min_max_not_low_high() -> None:
  """RangeGate uses min_value/max_value, not low/high."""
  from autopilot.policy.gates import RangeGate

  gate = RangeGate('accuracy', min_value=0.5, max_value=1.0)
  assert gate.min_value == 0.5
  assert gate.max_value == 1.0

  with pytest.raises(TypeError):
    RangeGate('accuracy', low=0.5, high=1.0)  # type: ignore[call-arg]


def test_custom_gate_fn_receives_float_not_result() -> None:
  """CustomGate fn receives a float metric value, not a Result object."""
  from autopilot.core.models import Result
  from autopilot.core.types import GateResult
  from autopilot.policy.gates import CustomGate

  gate_good = CustomGate('m', fn=lambda v: v > 0.5)
  result = Result(metrics={'m': 0.8})
  assert gate_good(result) == GateResult.PASSED

  gate_bad = CustomGate('m', fn=lambda r: r.metrics)  # ty: ignore[unresolved-attribute]
  with pytest.raises(AttributeError):
    gate_bad(result)


def test_monotonic_gate_direction_values() -> None:
  """MonotonicGate direction must be non_decreasing or non_increasing."""
  from autopilot.policy.gates import MonotonicGate

  gate = MonotonicGate('m', direction='non_decreasing')
  assert gate._direction == 'non_decreasing'

  with pytest.raises(ValueError, match='non_decreasing'):
    MonotonicGate('m', direction='increasing')


def test_budget_gate_max_usd_not_budget() -> None:
  """BudgetGate uses max_usd=, not budget=."""
  from autopilot.policy.gates import BudgetGate

  gate = BudgetGate(max_usd=5.0)
  assert gate._max_usd == 5.0

  with pytest.raises(TypeError):
    BudgetGate(budget=5.0)  # type: ignore[call-arg]


def test_checkpoint_callback_requires_directory(tmp_path) -> None:
  """CheckpointCallback requires a directory= argument."""
  cb = CheckpointCallback(directory=tmp_path)
  assert cb is not None

  with pytest.raises(TypeError):
    CheckpointCallback()


def test_diagnostics_callback_requires_diagnostics_instance() -> None:
  """DiagnosticsCallback requires a Diagnostics instance."""
  cb = DiagnosticsCallback(Diagnostics())
  assert cb is not None

  with pytest.raises(TypeError):
    DiagnosticsCallback()


# ---------------------------------------------------------------------------
# 2.4  Context, store, comparison, query (tests 13-16, 21-24)
# ---------------------------------------------------------------------------


def test_context_log_append_accepts_entry_and_string() -> None:
  """append() accepts both a reason string and a pre-built ContextEntry.

  V6 plan 02 replaced the TypeError guard with an overload: append(ContextEntry)
  delegates to record() and returns the entry.
  """
  log = ContextLog()

  entry = ContextEntry.create('canonical reason', source='test')
  log.record(entry)
  assert len(log) == 1
  assert log.entries[0].reason == 'canonical reason'

  entry2 = ContextEntry.create('via append')
  result = log.append(entry2)
  assert len(log) == 2
  assert result is entry2


def test_file_store_snapshot_positionals_and_context(tmp_path) -> None:
  """FileStore.snapshot requires experiment_id and epoch positionals."""
  config = AutoPilotConfig(workspace=tmp_path)
  store = FileStore(config)
  store.register_parameters({'p': Parameter()})

  manifest = store.snapshot('exp1', 0, context='initial')
  assert manifest is not None

  with pytest.raises(TypeError):
    store.snapshot(context='missing positionals')  # ty: ignore[missing-argument]


def test_store_load_snapshot_not_load_manifest(tmp_path) -> None:
  """FileStore has load_snapshot, not load_manifest."""
  config = AutoPilotConfig(workspace=tmp_path)
  store = FileStore(config)
  store.register_parameters({'p': Parameter()})

  store.snapshot('exp1', 0)
  manifest = store.load_snapshot('exp1', 0)
  assert manifest is not None

  assert not hasattr(store, 'load_manifest')


def test_diff_entry_path_status_not_key_kind() -> None:
  """DiffEntry uses .path and .status, not .key or .kind."""
  entry = DiffEntry(path='prompt.txt', status='modified')
  assert entry.path == 'prompt.txt'
  assert entry.status == 'modified'

  with pytest.raises(AttributeError):
    _ = entry.key  # ty: ignore[unresolved-attribute]
  with pytest.raises(AttributeError):
    _ = entry.kind  # ty: ignore[unresolved-attribute]


def test_conflict_entry_ancestor_not_base() -> None:
  """ConflictEntry uses .ancestor, not .base."""
  entry = ConflictEntry(key='prompt.txt', ancestor=None, ours=None, theirs=None)
  assert entry.ancestor is None

  with pytest.raises(AttributeError):
    _ = entry.base  # ty: ignore[unresolved-attribute]


def test_metrics_comparator_float_args_not_delta_object() -> None:
  """MetricsComparator.is_significant takes (delta: float, baseline: float)."""

  class SimpleMetric(Metric):
    higher_is_better = True

    def update(self, datum: Datum) -> None:
      pass

    def compute(self) -> dict[str, float]:
      return {'m': 1.0}

  mc = MetricsComparator([SimpleMetric()])
  assert mc.is_significant(0.5, 1.0) is True

  delta_obj = Delta(
    metric='m',
    baseline=1.0,
    candidate=1.5,
    delta=0.5,
    higher_is_better=True,
    significant=True,
  )
  with pytest.raises(TypeError):
    mc.is_significant(delta_obj)  # ty: ignore[invalid-argument-type, missing-argument]


def test_delta_metric_not_metric_name() -> None:
  """Delta field is .metric, not .metric_name."""
  d = Delta(
    metric='accuracy',
    baseline=0.8,
    candidate=0.9,
    delta=0.1,
    higher_is_better=True,
    significant=True,
  )
  assert d.metric == 'accuracy'

  with pytest.raises(AttributeError):
    _ = d.metric_name  # ty: ignore[unresolved-attribute]


def test_query_builder_filter_kwargs_not_where_string() -> None:
  """QueryBuilder.filter(**kwargs) for equality; where(predicate) for callables."""
  from autopilot.core.experiment import Experiment

  exp = Experiment(experiment_id='abc123')
  node = Node(experiment=exp)
  qb = QueryBuilder([node], resolver=lambda eid: node if eid == 'abc123' else None)

  filtered = qb.filter(id='abc123')
  assert len(filtered.all()) == 1

  with pytest.raises(TypeError):
    qb.where(hypothesis='x')  # ty: ignore[unknown-argument, missing-argument]
