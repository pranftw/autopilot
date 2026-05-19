"""Dogfood V8 regression tests - permanent guards for V4-V7 findings.

Covers: MonotonicGate epsilon + Trainer.fit, BudgetGate threshold strings,
QualityFirstPolicy.forward() gates population, checkout reflog context on
rollback, orchestrator plateau context, batch_sampler set_epoch wiring,
trace completeness on clean runs, tree name validation, DataLoader
batch_sampler property, debug commands JSON catalog.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.constraint import ConstraintResult
from autopilot.core.decision import DecisionEntry
from autopilot.core.enums import Status
from autopilot.core.experiment import Experiment
from autopilot.core.loops.epoch import _set_sampler_epoch_for_loader
from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trace import TraceReport, verify_trace_completeness
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.sampler import BatchSampler, WeightedSampler
from autopilot.policy.gates import BudgetGate, Gate, MonotonicGate
from autopilot.policy.quality_first import QualityFirstPolicy
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
from tests.data.conftest import SizedDataset
from tests.doubles import DirectNumericLoss, NoOpOptimizer
import contextlib
import io
import json
import pytest

# ---------------------------------------------------------------------------
# Local test-specific stubs (unique behaviour, not worth tests/doubles.py)
# ---------------------------------------------------------------------------


class _NoisyValMetric(Metric):
  """Returns accuracy sequence: 0.90, 0.85, 0.82 (all within epsilon=0.1)."""

  higher_is_better = True

  def __init__(self) -> None:
    super().__init__()
    self._epoch_idx = 0
    self._values = [0.90, 0.85, 0.82]

  def update(self, datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    idx = min(self._epoch_idx, len(self._values) - 1)
    self._epoch_idx += 1
    return {'accuracy': self._values[idx]}

  def reset(self) -> None:
    super().reset()


class _EpsilonModule(AutoPilotModule):
  """Module with noisy-but-within-epsilon validation metric."""

  def __init__(self) -> None:
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.param])
    self._opt = NoOpOptimizer([self.param])
    self.val_metric = _NoisyValMetric()

  def training_step(self, batch, batch_idx):
    return EvalDatum(success=True)

  def validation_step(self, batch, batch_idx):
    return EvalDatum(success=True)

  def configure_optimizers(self):
    return self._opt


class _FlatAccMetric(Metric):
  """Returns constant accuracy for plateau detection tests."""

  higher_is_better = True

  def update(self, datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    return {'accuracy': 0.5}

  def reset(self) -> None:
    super().reset()


class _FlatModule(AutoPilotModule):
  """Module with flat accuracy metric for plateau detection."""

  def __init__(self) -> None:
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.param])
    self._opt = NoOpOptimizer([self.param])
    self.acc_metric = _FlatAccMetric()

  def training_step(self, batch, batch_idx):
    return EvalDatum(success=True)

  def configure_optimizers(self):
    return self._opt


class _StableAccMetric(Metric):
  """Returns stable accuracy for trace completeness tests."""

  higher_is_better = True

  def update(self, datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    return {'accuracy': 0.95}

  def reset(self) -> None:
    super().reset()


class _TraceModule(AutoPilotModule):
  """Module with stable metric for trace completeness verification."""

  def __init__(self) -> None:
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.param])
    self._opt = NoOpOptimizer([self.param])
    self.acc_metric = _StableAccMetric()

  def training_step(self, batch, batch_idx):
    return EvalDatum(success=True)

  def validation_step(self, batch, batch_idx):
    return EvalDatum(success=True)

  def configure_optimizers(self):
    return self._opt


# ---------------------------------------------------------------------------
# Test 1 — MonotonicGate epsilon E2E through Trainer.fit (REGR-001 / BUG-006)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1)
def test_monotonic_gate_epsilon_trainer_e2e(tmp_path: Path) -> None:
  """MonotonicGate(epsilon=0.1) in Trainer.fit: noisy metric within tolerance passes.

  Gate metric is 'val_accuracy' -- _NoisyValMetric.compute() returns
  {'accuracy': ...} which gets prefixed to 'val_accuracy' by Trainer
  metric merge (both splits present).
  """
  gate = MonotonicGate('val_accuracy', direction='non_decreasing', epsilon=0.1)
  policy = QualityFirstPolicy(gates=[gate])
  exp = Experiment(experiment_id='epsilon-e2e', hypothesis='test')
  exp.start()

  trainer = Trainer(policy=policy, experiment=exp, num_sanity_val_steps=0)
  loader = DataLoader([Datum()], batch_size=1)
  trainer.fit(
    _EpsilonModule(),
    train_dataloaders=loader,
    val_dataloaders=loader,
    max_epochs=3,
  )

  assert exp.status == Status.completed
  assert exp.last_accepted_epoch == 2
  rejected = [e for e in exp.context_log if 'rejected' in e.reason]
  assert rejected == []


# ---------------------------------------------------------------------------
# Test 2 — BudgetGate threshold in ConstraintResult (BUG-002)
# ---------------------------------------------------------------------------


def test_budget_gate_threshold_in_constraint_result() -> None:
  """BudgetGate constraint threshold is human-readable and contains 'USD'."""
  gate = BudgetGate(max_usd=50.0)
  policy = QualityFirstPolicy(gates=[gate])
  result = Result(metrics={'cost_usd': 25.0}, summary='ok')

  policy.forward(result)

  assert len(result.gates) == 1
  constraint = result.gates[0]
  assert isinstance(constraint, ConstraintResult)
  assert constraint.name == 'BudgetGate'
  assert 'USD' in constraint.threshold
  assert 'BudgetGate(' not in constraint.threshold


# ---------------------------------------------------------------------------
# Test 3 — QualityFirstPolicy.forward() gates populated (BUG-001)
# ---------------------------------------------------------------------------


def test_quality_first_policy_forward_gates_populated() -> None:
  """QualityFirstPolicy.forward() populates result.gates with ConstraintResult rows."""
  gates: list[Gate] = [
    MonotonicGate('accuracy', direction='non_decreasing'),
    BudgetGate(max_usd=100.0),
  ]
  policy = QualityFirstPolicy(gates=gates)
  result = Result(metrics={'accuracy': 0.9, 'cost_usd': 10.0}, summary='ok')

  policy.forward(result)

  assert len(result.gates) == 2
  assert all(isinstance(c, ConstraintResult) for c in result.gates)
  assert result.passed is True
  names = {c.name for c in result.gates}
  assert names == {'MonotonicGate', 'BudgetGate'}


# ---------------------------------------------------------------------------
# Test 4 — Checkout reflog has context after rollback (CRITICAL-4)
# ---------------------------------------------------------------------------


def test_checkout_reflog_has_context_after_rollback(tmp_path: Path) -> None:
  """After experiment.rollback(), reflog checkout entry has non-null context."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)

  src_dir = ws / 'src'
  src_dir.mkdir()
  (src_dir / 'main.py').write_text('v0\n', encoding='utf-8')

  param = PathParameter(source=str(src_dir), pattern='**/*.py')
  store = FileStore(config)
  store.register_parameters({'source': param})

  exp_id = 'rollback-exp'
  store.snapshot(exp_id, 0, context='epoch 0')
  (src_dir / 'main.py').write_text('v1\n', encoding='utf-8')
  store.snapshot(exp_id, 1, context='epoch 1')

  exp = Experiment(experiment_id=exp_id, hypothesis='rollback test')
  exp.store = store
  exp.epoch = 1

  exp.rollback(0)

  checkout_entries = [
    entry
    for entry in store.iter_reflog()
    if entry.get('operation') == 'checkout' and entry.get('experiment_id') == exp_id
  ]
  assert len(checkout_entries) >= 1
  last_checkout = checkout_entries[-1]
  assert last_checkout.get('context') is not None
  assert last_checkout['context'] == 'rolled back to epoch 0'


# ---------------------------------------------------------------------------
# Test 5 — Plateau detection emits context (CRITICAL-3)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1)
def test_plateau_detection_emits_context(tmp_path: Path) -> None:
  """EpochOrchestrator plateau stop emits context log entry with source='plateau'."""
  exp = Experiment(experiment_id='plateau-ctx', hypothesis='test')
  exp.start()

  orch_config = OrchestratorConfig(
    plateau_window=3,
    plateau_threshold=0.05,
    monitor='accuracy',
  )
  orch = EpochOrchestrator(config=orch_config)
  trainer = Trainer(loop=orch, experiment=exp)

  trainer.fit(_FlatModule(), train_dataloaders=[1, 2, 3, 4, 5], max_epochs=10)

  assert orch.stop_reason == 'plateau'
  plateau_entries = [e for e in exp.context_log if e.source == 'plateau']
  assert len(plateau_entries) >= 1
  entry = plateau_entries[-1]
  assert entry.metadata.get('_type') == DecisionEntry.PLATEAU_STOP_TYPE
  assert entry.metadata.get('monitor') == 'accuracy'
  assert 'values' in entry.metadata


# ---------------------------------------------------------------------------
# Test 6 — EpochLoop batch_sampler set_epoch wiring (BUG-003)
# ---------------------------------------------------------------------------


def test_epoch_loop_batch_sampler_set_epoch() -> None:
  """BatchSampler(WeightedSampler) inner sampler receives set_epoch via batch_sampler path."""
  ds = SizedDataset(6)
  inner = WeightedSampler(ds, weights=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
  bs = BatchSampler(inner, batch_size=2, drop_last=False)
  loader = DataLoader(ds, batch_sampler=bs)

  assert loader.sampler is None
  assert loader.batch_sampler is bs

  _set_sampler_epoch_for_loader(loader, epoch=7)
  assert inner._epoch == 7

  batches_epoch_7 = list(bs)
  _set_sampler_epoch_for_loader(loader, epoch=3)
  assert inner._epoch == 3
  batches_epoch_3 = list(bs)
  assert batches_epoch_7 != batches_epoch_3


# ---------------------------------------------------------------------------
# Test 7 — Trace completeness on clean run (CRITICAL-1)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1)
def test_trace_completeness_clean_run(tmp_path: Path) -> None:
  """3-epoch Trainer.fit run yields verify_trace_completeness complete=True."""
  exp = Experiment(experiment_id='trace-clean', hypothesis='test')
  exp.start()

  gate = MonotonicGate('val_accuracy', direction='non_decreasing')
  policy = QualityFirstPolicy(gates=[gate])
  trainer = Trainer(policy=policy, experiment=exp, num_sanity_val_steps=0)

  loader = DataLoader([Datum()], batch_size=1)
  trainer.fit(_TraceModule(), train_dataloaders=loader, val_dataloaders=loader, max_epochs=3)

  report = verify_trace_completeness(
    exp.context_log,
    reflog_entries=[],
    epochs_run=3,
    check_cost=False,
  )

  assert isinstance(report, TraceReport)
  assert report.complete is True
  assert report.gaps == []
  policy_dim = next(d for d in report.dimensions if d.name == 'policy_gate')
  assert policy_dim.passed is True


# ---------------------------------------------------------------------------
# Test 8 — Tree create rejects empty name (BUG-004)
# ---------------------------------------------------------------------------


def test_tree_create_rejects_empty_name(tmp_path: Path) -> None:
  """tree create '' exits non-zero with actionable error."""
  from autopilot.cli.context import build_context
  from autopilot.cli.main import build_parser

  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  FileStore(config)

  parser = build_parser()
  full_argv = ['tree', 'create', '', '--workspace', str(ws), '--json', '--context', 'test']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)

  buf = io.StringIO()
  with pytest.raises(SystemExit) as exc_info, contextlib.redirect_stdout(buf):
    parsed.handler(ctx, parsed)

  assert exc_info.value.code != 0
  envelope = json.loads(buf.getvalue())
  assert envelope['ok'] is False
  error_lower = envelope['error'].lower()
  assert 'name' in error_lower or 'empty' in error_lower


# ---------------------------------------------------------------------------
# Test 9 — DataLoader batch_sampler property (BUG-003 property gap)
# ---------------------------------------------------------------------------


def test_dataloader_batch_sampler_property() -> None:
  """DataLoader(batch_sampler=bs).batch_sampler returns the same BatchSampler instance."""
  ds = SizedDataset(8)
  inner = WeightedSampler(ds, weights=[1.0] * len(ds))
  bs = BatchSampler(inner, batch_size=2, drop_last=False)
  loader = DataLoader(ds, batch_sampler=bs)

  assert loader.batch_sampler is bs
  assert loader.sampler is None


# ---------------------------------------------------------------------------
# Test 10 — Command catalog JSON parseable (CRITICAL-2)
# ---------------------------------------------------------------------------


def test_command_catalog_json_parseable(tmp_path: Path) -> None:
  """debug commands --json returns valid structured catalog."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  FileStore(config)

  envelope = run_cli_no_context(ws, ['debug', 'commands'])

  assert envelope['ok'] is True
  result = envelope['result']
  assert isinstance(result, dict)
  commands = result['commands']
  assert isinstance(commands, list)
  assert len(commands) > 50
  names = {cmd['name'] for cmd in commands}
  assert 'experiment add' in names
  assert 'store checkout' in names
  assert 'tree create' in names
  sample = next(cmd for cmd in commands if cmd['name'] == 'experiment add')
  assert 'requires_context' in sample
  assert sample['requires_context'] is True
  assert 'arguments' in sample
  assert isinstance(sample['arguments'], list)
