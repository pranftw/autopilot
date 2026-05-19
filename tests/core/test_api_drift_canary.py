"""API drift canary tests.

Smoke-import and trivial-construction tests for the representative public
API surface. Each row mirrors documented construction patterns after
Plans 01-16 (e.g. ``FileStore(config)`` then ``register_parameters``, not
legacy constructors).

Goal: import-time and trivial-construction regressions fail fast before
agents read stale docs.
"""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.callbacks.callback import Callback
from autopilot.core.checkpoint import CheckpointIO, JSONCheckpointIO
from autopilot.core.comparison import ComparatorMetric, Delta, MetricsComparator
from autopilot.core.config import AutoPilotConfig
from autopilot.core.context import ContextEntry, ContextLog
from autopilot.core.experiment import Experiment
from autopilot.core.gradient import Gradient, NumericGradient
from autopilot.core.logger import JSONLogger, Logger
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.module.module import IncompatibleKeys, Module
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from autopilot.core.scheduler import LambdaScheduler, Scheduler
from autopilot.core.snapshot import (
  FileEntry,
  ParameterSchema,
  ParameterSchemaEntry,
  SnapshotManifest,
)
from autopilot.core.store.base import Store
from autopilot.core.store.types import (
  ConflictEntry,
  DiffKind,
  MergeAnalysisResult,
  MergeClassification,
  MergeIndex,
  MergeStrategy,
  TagEntry,
)
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum, GateResult
from autopilot.policy.gates import (
  BudgetGate,
  CustomGate,
  Gate,
  MaxGate,
  MinGate,
  MonotonicGate,
  RangeGate,
)
from autopilot.policy.quality_first import QualityFirstPolicy
from tests.doubles import NoOpOptimizer

# --- import smoke tests ---


def test_import_core_types():
  """Core types are importable from their canonical modules."""
  assert Datum is not None
  assert EvalDatum is not None
  assert GateResult is not None


def test_import_parameter_types():
  """Parameter hierarchy is importable."""
  assert Parameter is not None
  assert PathParameter is not None


def test_import_gradient_types():
  """Gradient hierarchy is importable."""
  assert Gradient is not None
  assert NumericGradient is not None


def test_import_module_types():
  """Module hierarchy is importable."""
  assert Module is not None
  assert AutoPilotModule is not None
  assert IncompatibleKeys is not None


def test_import_optimizer_and_scheduler():
  """Optimizer and Scheduler are importable."""
  assert Optimizer is not None
  assert Scheduler is not None
  assert LambdaScheduler is not None


def test_import_trainer():
  """Trainer is importable."""
  assert Trainer is not None


def test_import_store_types():
  """Store hierarchy and merge types are importable."""
  assert Store is not None
  assert FileStore is not None
  assert MergeStrategy is not None
  assert ConflictEntry is not None
  assert MergeAnalysisResult is not None
  assert MergeIndex is not None
  assert MergeClassification is not None
  assert DiffKind is not None
  assert TagEntry is not None


def test_import_snapshot_types():
  """Snapshot data models are importable."""
  assert FileEntry is not None
  assert SnapshotManifest is not None
  assert ParameterSchema is not None
  assert ParameterSchemaEntry is not None


def test_import_policy_types():
  """Policy gates and quality-first policy are importable."""
  assert Gate is not None
  assert MinGate is not None
  assert MaxGate is not None
  assert RangeGate is not None
  assert CustomGate is not None
  assert MonotonicGate is not None
  assert BudgetGate is not None
  assert QualityFirstPolicy is not None


def test_import_context_types():
  """Context system types are importable."""
  assert ContextEntry is not None
  assert ContextLog is not None


def test_import_comparison_types():
  """Comparison types are importable."""
  assert Delta is not None
  assert ComparatorMetric is not None
  assert MetricsComparator is not None


def test_import_callback():
  """Callback is importable."""
  assert Callback is not None


def test_import_checkpoint():
  """Checkpoint types are importable."""
  assert CheckpointIO is not None
  assert JSONCheckpointIO is not None


def test_import_logger():
  """Logger types are importable."""
  assert Logger is not None
  assert JSONLogger is not None


# --- trivial construction tests ---


def test_construct_datum():
  """Datum() with no args produces a valid instance with empty items list."""
  d = Datum()
  assert isinstance(d.items, list)
  assert len(d.items) == 0
  assert d.id


def test_construct_eval_datum():
  """EvalDatum() defaults to success=True and empty metrics."""
  ed = EvalDatum()
  assert ed.success is True
  assert ed.metrics == {}
  assert ed.split is None
  assert ed.epoch is None


def test_construct_parameter():
  """Parameter() with defaults has requires_grad=True and empty items."""
  p = Parameter()
  assert p.requires_grad is True
  assert p.grad is None
  assert isinstance(p.items, list)


def test_construct_numeric_gradient():
  """NumericGradient() defaults to value=0.0."""
  ng = NumericGradient()
  assert ng.value == 0.0


def test_construct_numeric_gradient_with_value():
  """NumericGradient(value=...) stores the value correctly."""
  ng = NumericGradient(value=42.0)
  assert ng.value == 42.0


def test_construct_module():
  """Module() creates a valid empty module."""
  m = Module()
  assert list(m.parameters()) == []
  assert list(m.children()) == []


def test_construct_loss():
  """Loss() creates a loss with empty parameter scope."""
  loss = Loss()
  assert loss._loss_parameters == []


def test_construct_optimizer():
  """Optimizer with param_groups construction pattern."""
  p = Parameter()
  opt = NoOpOptimizer([p], lr=0.5)
  assert len(opt.param_groups) == 1
  assert opt.param_groups[0]['lr'] == 0.5
  assert opt.parameters == [p]


def test_construct_optimizer_with_groups():
  """Optimizer accepts list of group dicts."""
  p1, p2 = Parameter(), Parameter()
  opt = NoOpOptimizer([{'params': [p1], 'lr': 0.1}, {'params': [p2], 'lr': 0.2}])
  assert len(opt.param_groups) == 2
  assert opt.param_groups[0]['lr'] == 0.1
  assert opt.param_groups[1]['lr'] == 0.2


def test_construct_scheduler():
  """LambdaScheduler captures base_lrs from optimizer."""
  p = Parameter()
  opt = NoOpOptimizer([p], lr=0.5)
  sched = LambdaScheduler(opt, lr_lambda=lambda epoch: 0.9**epoch)
  assert sched.base_lrs == [0.5]
  assert sched.last_epoch == -1


def test_construct_trainer():
  """Trainer() with no args creates a valid trainer."""
  t = Trainer()
  assert t.module is None
  assert t.store is None
  assert t.experiment is None


def test_construct_config(tmp_path):
  """AutoPilotConfig(workspace) creates a valid config and is a Config subclass."""
  from autopilot.core.config import Config

  cfg = AutoPilotConfig(workspace=tmp_path)
  assert cfg.workspace == tmp_path
  assert isinstance(cfg, Config)


def test_construct_file_store(tmp_path):
  """FileStore(config) then register_parameters(dict) pattern."""
  cfg = AutoPilotConfig(workspace=tmp_path)
  store = FileStore(cfg)
  p = Parameter()
  store.register_parameters({'my_param': p})


def test_construct_experiment():
  """Experiment(experiment_id) creates a pending experiment."""
  exp = Experiment(experiment_id='test-exp')
  assert exp.id == 'test-exp'
  assert exp.status.value == 'pending'


def test_construct_autopilot_experiment():
  """AutoPilotExperiment construction matches Experiment pattern."""
  exp = AutoPilotExperiment(experiment_id='auto-exp')
  assert exp.id == 'auto-exp'


def test_construct_context_log():
  """ContextLog() creates an empty log."""
  log = ContextLog()
  assert len(log) == 0


def test_construct_context_entry():
  """ContextEntry.create() produces a timestamped entry."""
  entry = ContextEntry.create('test reason', source='test')
  assert entry.reason == 'test reason'
  assert entry.source == 'test'
  assert entry.timestamp


def test_construct_comparator_metric():
  """ComparatorMetric uses metric_name, not name."""
  cm = ComparatorMetric(metric_name='accuracy', higher_is_better=True)
  assert cm.metric_name == 'accuracy'
  assert cm.name() == 'accuracy'
  assert cm.higher_is_better is True


def test_construct_metrics_comparator():
  """MetricsComparator accepts a sequence of ComparatorMetric."""
  cm = ComparatorMetric(metric_name='accuracy', higher_is_better=True)
  mc = MetricsComparator([cm])
  assert mc is not None


def test_construct_parameter_schema():
  """ParameterSchema uses parameters (list), not entries (dict)."""
  entry = ParameterSchemaEntry(name='prompt', type_name='PathParameter')
  schema = ParameterSchema(parameters=[entry])
  assert len(schema.parameters) == 1
  assert schema.parameters[0].name == 'prompt'


def test_construct_snapshot_manifest():
  """SnapshotManifest uses epoch, timestamp, entries dict, optional schema."""
  manifest = SnapshotManifest(epoch=0, timestamp='2026-01-01T00:00:00+00:00')
  assert manifest.epoch == 0
  assert manifest.entries == {}
  assert manifest.schema is None
  assert manifest.context is None


def test_construct_budget_gate():
  """BudgetGate uses max_usd, not generic metric/threshold."""
  bg = BudgetGate(max_usd=50.0)
  assert bg.metric == 'cost_usd'
  assert bg._max_usd == 50.0


def test_construct_monotonic_gate():
  """MonotonicGate uses metric + direction."""
  mg = MonotonicGate('accuracy', direction='non_decreasing')
  assert mg.metric == 'accuracy'


def test_construct_quality_first_policy():
  """QualityFirstPolicy accepts gates list."""
  policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.8)])
  assert policy is not None


def test_construct_incompatible_keys():
  """IncompatibleKeys is a dataclass with missing/unexpected lists."""
  ik = IncompatibleKeys(missing_keys=['a'], unexpected_keys=['b'])
  assert ik.missing_keys == ['a']
  assert ik.unexpected_keys == ['b']


def test_construct_callback():
  """Callback() creates a valid instance with default no-op hooks."""
  cb = Callback()
  assert cb is not None


def test_construct_tag_entry():
  """TagEntry is a frozen dataclass with experiment_id and epoch."""
  tag = TagEntry(name='v1', experiment_id='exp-1', epoch=3, timestamp='now')
  assert tag.name == 'v1'
  assert tag.experiment_id == 'exp-1'
  assert tag.epoch == 3


def test_construct_path_parameter(tmp_path):
  """PathParameter(source, pattern) construction."""
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'file.txt').write_text('hello')
  pp = PathParameter(source=src, pattern='*.txt')
  assert pp is not None


def test_construct_file_forest(tmp_path):
  """FileForest construction from a FileStore instance."""
  cfg = AutoPilotConfig(workspace=tmp_path)
  store = FileStore(cfg)
  ff = FileForest(store=store)
  assert ff is not None


def test_construct_metric():
  """Metric base class is importable and has expected attributes."""
  assert issubclass(Metric, Module)
  assert Metric.higher_is_better is None


def test_construct_metric_collection():
  """MetricCollection is importable."""
  assert MetricCollection is not None


def test_gate_result_enum():
  """GateResult has PASSED, FAIL, WARN, SKIP members."""
  assert GateResult.PASSED.value == 'pass'
  assert GateResult.FAIL.value == 'fail'
  assert GateResult.WARN.value == 'warn'
  assert GateResult.SKIP.value == 'skip'
