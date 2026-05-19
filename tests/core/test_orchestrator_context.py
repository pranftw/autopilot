"""Tests for EpochOrchestrator plateau context emission.

Covers:
  - test_plateau_emits_context_entry (4.2.4)
  - test_plateau_context_type_discriminator (4.2.5)
  - test_plateau_context_metadata_fields (4.2.6)
  - test_plateau_context_epoch_matches_stop_epoch (4.2.7)
  - test_plateau_context_recorded_in_experiment_log (4.2.8)
  - test_no_plateau_context_on_normal_completion (4.2.9)
  - test_no_plateau_context_on_policy_fail (4.2.10)
  - test_no_plateau_context_without_experiment (4.2.11)
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.constraint import ConstraintResult
from autopilot.core.context import ContextEntry
from autopilot.core.decision import DecisionEntry
from autopilot.core.experiment import Experiment
from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum, GateResult
from autopilot.policy.policy import Policy
from typing import Any

FLAT_ACCURACY = 0.5
PLATEAU_WINDOW = 3
PLATEAU_THRESHOLD = 0.05


class SpyCallback(Callback):
  """Records on_context_emit calls for assertion."""

  def __init__(self) -> None:
    super().__init__()
    self.entries: list[ContextEntry] = []

  def on_context_emit(self, trainer, module, entry) -> None:
    """Capture emitted context entries."""
    self.entries.append(entry)


class PlateauModule(AutoPilotModule):
  """Module that returns a constant accuracy to trigger plateau detection."""

  def __init__(self, accuracy: float = FLAT_ACCURACY) -> None:
    super().__init__()
    self._acc = accuracy
    self.acc_metric = _AccMetric()

  def forward(self, batch: Any) -> EvalDatum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    return EvalDatum(success=True, metrics={'accuracy': self._acc})

  def validation_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    return EvalDatum(success=True, metrics={'accuracy': self._acc})

  def configure_optimizers(self) -> None:
    return None


class _AccMetric(Metric):
  """Simple accuracy metric that returns the latest value from data."""

  def __init__(self) -> None:
    super().__init__()
    self.add_state('_value', 0.0)
    self.add_state('_count', 0)

  def update(self, datum: Any) -> None:
    self._count += 1
    if hasattr(datum, 'metrics') and datum.metrics:
      self._value = datum.metrics.get('accuracy', self._value)

  def compute(self) -> dict[str, float]:
    return {'accuracy': self._value}


class _FailPolicy(Policy):
  """Policy that always rejects."""

  def forward(self, result: Result) -> GateResult:
    result.gates = [
      ConstraintResult(
        name='MinGate',
        passed=False,
        metric='accuracy',
        value=result.metrics.get('accuracy'),
        threshold='>= 0.9',
      )
    ]
    return GateResult.FAIL


def _make_plateau_config() -> OrchestratorConfig:
  """Build a standard plateau-triggering OrchestratorConfig."""
  return OrchestratorConfig(
    plateau_window=PLATEAU_WINDOW,
    plateau_threshold=PLATEAU_THRESHOLD,
    monitor='accuracy',
  )


def test_plateau_emits_context_entry() -> None:
  """Plateau detection emits exactly one context entry with source='plateau'."""
  exp = Experiment('plateau-ctx')
  spy = SpyCallback()
  module = PlateauModule()
  config = _make_plateau_config()
  trainer = Trainer(
    loop=EpochOrchestrator(config),
    experiment=exp,
    callbacks=[spy],
  )

  trainer.fit(module, train_dataloaders=[1], max_epochs=10)

  plateau_entries = [e for e in spy.entries if e.source == 'plateau']
  assert len(plateau_entries) == 1
  assert 'plateau detected' in plateau_entries[0].reason


def test_plateau_context_type_discriminator() -> None:
  """Plateau context entry has _type==PLATEAU_STOP_TYPE in metadata."""
  exp = Experiment('plateau-type')
  spy = SpyCallback()
  module = PlateauModule()
  config = _make_plateau_config()
  trainer = Trainer(
    loop=EpochOrchestrator(config),
    experiment=exp,
    callbacks=[spy],
  )

  trainer.fit(module, train_dataloaders=[1], max_epochs=10)

  plateau_entries = [e for e in spy.entries if e.source == 'plateau']
  assert len(plateau_entries) == 1
  assert plateau_entries[0].metadata['_type'] == DecisionEntry.PLATEAU_STOP_TYPE
  assert plateau_entries[0].metadata['_type'] == 'plateau_stop'


def test_plateau_context_metadata_fields() -> None:
  """Plateau context metadata includes monitor, window, threshold, and values."""
  exp = Experiment('plateau-fields')
  spy = SpyCallback()
  module = PlateauModule()
  config = _make_plateau_config()
  trainer = Trainer(
    loop=EpochOrchestrator(config),
    experiment=exp,
    callbacks=[spy],
  )

  trainer.fit(module, train_dataloaders=[1], max_epochs=10)

  plateau_entries = [e for e in spy.entries if e.source == 'plateau']
  assert len(plateau_entries) == 1
  meta = plateau_entries[0].metadata
  assert meta['monitor'] == 'accuracy'
  assert meta['plateau_window'] == PLATEAU_WINDOW
  assert meta['plateau_threshold'] == PLATEAU_THRESHOLD
  assert len(meta['values']) == PLATEAU_WINDOW
  for val in meta['values']:
    assert val == FLAT_ACCURACY


def test_plateau_context_epoch_matches_stop_epoch() -> None:
  """Plateau context metadata['epoch'] matches the epoch where plateau was detected."""
  exp = Experiment('plateau-epoch')
  spy = SpyCallback()
  module = PlateauModule()
  config = _make_plateau_config()
  orch = EpochOrchestrator(config)
  trainer = Trainer(
    loop=orch,
    experiment=exp,
    callbacks=[spy],
  )

  result = trainer.fit(module, train_dataloaders=[1], max_epochs=10)
  assert result['stop_reason'] == 'plateau'

  plateau_entries = [e for e in spy.entries if e.source == 'plateau']
  assert len(plateau_entries) == 1
  stop_epoch = plateau_entries[0].metadata['epoch']
  assert stop_epoch == PLATEAU_WINDOW - 1
  assert f'after epoch {stop_epoch}' in plateau_entries[0].reason


def test_plateau_context_recorded_in_experiment_log() -> None:
  """With default ContextLogCallback, plateau entry appears in experiment context log."""
  exp = Experiment('plateau-log')
  module = PlateauModule()
  config = _make_plateau_config()
  trainer = Trainer(
    loop=EpochOrchestrator(config),
    experiment=exp,
  )

  trainer.fit(module, train_dataloaders=[1], max_epochs=10)

  plateau_log = exp.context_log.filter_by_source('plateau')
  assert len(plateau_log) == 1
  assert plateau_log[0].metadata['_type'] == DecisionEntry.PLATEAU_STOP_TYPE


def test_no_plateau_context_on_normal_completion() -> None:
  """When plateau detection is disabled, no plateau context entries are emitted."""
  exp = Experiment('no-plateau')
  spy = SpyCallback()
  module = PlateauModule()
  config = OrchestratorConfig(plateau_window=0)
  trainer = Trainer(
    loop=EpochOrchestrator(config),
    experiment=exp,
    callbacks=[spy],
  )

  trainer.fit(module, train_dataloaders=[1], max_epochs=2)

  plateau_entries = [e for e in spy.entries if e.source == 'plateau']
  assert len(plateau_entries) == 0


def test_no_plateau_context_on_policy_fail() -> None:
  """When policy rejects, stop_reason is 'policy_fail' and no plateau context emitted."""
  exp = Experiment('policy-no-plateau')
  spy = SpyCallback()
  module = PlateauModule()
  orch = EpochOrchestrator()
  trainer = Trainer(
    loop=orch,
    experiment=exp,
    callbacks=[spy],
    policy=_FailPolicy(),
  )

  result = trainer.fit(module, train_dataloaders=[1], max_epochs=5)
  assert result['stop_reason'] == 'policy_fail'

  plateau_entries = [e for e in spy.entries if e.source == 'plateau']
  assert len(plateau_entries) == 0


def test_no_plateau_context_without_experiment() -> None:
  """Plateau emits context even without experiment; SpyCallback still captures it."""
  spy = SpyCallback()
  module = PlateauModule()
  config = _make_plateau_config()
  trainer = Trainer(
    loop=EpochOrchestrator(config),
    callbacks=[spy],
  )

  trainer.fit(module, train_dataloaders=[1], max_epochs=10)

  plateau_entries = [e for e in spy.entries if e.source == 'plateau']
  assert len(plateau_entries) == 1
  assert plateau_entries[0].metadata['_type'] == DecisionEntry.PLATEAU_STOP_TYPE
