"""Trainer E2E tests with real QualityFirstPolicy gate journaling."""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.callbacks.cost import CostTrackerCallback
from autopilot.core.context import ContextEntry
from autopilot.core.decision import DecisionEntry
from autopilot.core.trainer.trainer import Trainer
from autopilot.policy.gates import BudgetGate, MinGate
from autopilot.policy.quality_first import QualityFirstPolicy
from pathlib import Path
from tests.dogfood_regressions.test_trainer_policy_and_cli import (
  _GateExperiment,
  _GateModule,
  _single_batch,
)
import pytest


class _SpyCallback(Callback):
  """Records on_context_emit calls for assertion."""

  def __init__(self) -> None:
    super().__init__()
    self.entries: list[ContextEntry] = []

  def on_context_emit(self, trainer, module, entry) -> None:
    """Capture emitted context entries."""
    self.entries.append(entry)


def _policy_gate_entries(entries: list[ContextEntry]) -> list[ContextEntry]:
  """Filter context entries to policy gate type only."""
  return [
    e
    for e in entries
    if e.metadata is not None and e.metadata.get('_type') == DecisionEntry.POLICY_GATE_TYPE
  ]


class TestTrainerQualityFirstReject:
  """Trainer E2E: QualityFirstPolicy reject path emits structured gate metadata."""

  @pytest.mark.timeout(1)
  def test_trainer_quality_first_reject_emits_gate_metadata(self) -> None:
    """Low accuracy triggers reject; metadata has per-gate ConstraintResult dicts."""
    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.8)])
    module = _GateModule(metric_name='accuracy', metric_value=0.5)
    spy = _SpyCallback()
    exp = _GateExperiment()
    exp.start()

    trainer = Trainer(policy=policy, experiment=exp, callbacks=[spy])
    trainer.fit(module, train_dataloaders=_single_batch(), max_epochs=1)

    gate_entries = _policy_gate_entries(spy.entries)
    assert len(gate_entries) >= 1

    reject_entry = gate_entries[0]
    meta = reject_entry.metadata
    assert meta['_type'] == DecisionEntry.POLICY_GATE_TYPE
    assert len(meta['gates']) == 1
    gate_dict = meta['gates'][0]
    assert gate_dict['threshold'] == '>= 0.8'
    assert gate_dict['passed'] is False
    assert gate_dict['name'] == 'MinGate'
    assert 'metrics' not in meta


class TestTrainerQualityFirstAccept:
  """Trainer E2E: QualityFirstPolicy accept path emits gate metadata with metrics."""

  @pytest.mark.timeout(1)
  def test_trainer_quality_first_accept_emits_gate_metadata(self) -> None:
    """High accuracy passes; accept metadata includes metrics and passed=True."""
    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.8)])
    module = _GateModule(metric_name='accuracy', metric_value=0.9)
    spy = _SpyCallback()
    exp = _GateExperiment()
    exp.start()

    trainer = Trainer(policy=policy, experiment=exp, callbacks=[spy])
    trainer.fit(module, train_dataloaders=_single_batch(), max_epochs=1)

    gate_entries = _policy_gate_entries(spy.entries)
    assert len(gate_entries) >= 1

    accept_entry = gate_entries[0]
    meta = accept_entry.metadata
    assert meta['_type'] == DecisionEntry.POLICY_GATE_TYPE
    assert len(meta['gates']) == 1
    gate_dict = meta['gates'][0]
    assert gate_dict['passed'] is True
    assert 'metrics' in meta


class TestTrainerQualityFirstMultiGate:
  """Trainer E2E: multiple gates in QualityFirstPolicy produce metadata per gate."""

  @pytest.mark.timeout(1)
  def test_trainer_quality_first_multi_gate_all_in_metadata(self, tmp_path: Path) -> None:
    """MinGate + BudgetGate accept path shows 2 gate dicts with correct thresholds."""
    policy = QualityFirstPolicy(
      gates=[
        MinGate('accuracy', 0.8),
        BudgetGate(max_usd=100.0),
      ]
    )
    module = _GateModule(metric_name='accuracy', metric_value=0.9)
    spy = _SpyCallback()
    exp = _GateExperiment()
    exp.start()

    cost_tracker = CostTrackerCallback(tmp_path)
    cost_tracker.cumulative_usd = 50.0
    trainer = Trainer(
      policy=policy,
      experiment=exp,
      callbacks=[spy, cost_tracker],
    )
    trainer.fit(module, train_dataloaders=_single_batch(), max_epochs=1)

    gate_entries = _policy_gate_entries(spy.entries)
    assert len(gate_entries) >= 1

    meta = gate_entries[0].metadata
    gates = meta['gates']
    assert len(gates) == 2
    assert gates[0]['name'] == 'MinGate'
    assert gates[0]['threshold'] == '>= 0.8'
    assert gates[1]['name'] == 'BudgetGate'
    assert gates[1]['threshold'] == '100.0 USD'


class TestTrainerBudgetGateReject:
  """Trainer E2E: BudgetGate reject when accumulated cost exceeds budget."""

  @pytest.mark.timeout(1)
  def test_trainer_budget_gate_reject_via_cost_tracker(self, tmp_path: Path) -> None:
    """BudgetGate fails when pre-accumulated cost exceeds budget.

    _inject_cost_usd reads cumulative_usd before on_epoch_end accumulates
    the current epoch's cost, so we pre-set it above the budget threshold.
    """
    policy = QualityFirstPolicy(gates=[BudgetGate(max_usd=10.0)])
    module = _GateModule(metric_name='accuracy', metric_value=0.9)
    spy = _SpyCallback()
    exp = _GateExperiment()
    exp.start()

    cost_tracker = CostTrackerCallback(tmp_path)
    cost_tracker.cumulative_usd = 15.0
    trainer = Trainer(
      policy=policy,
      experiment=exp,
      callbacks=[spy, cost_tracker],
    )
    trainer.fit(module, train_dataloaders=_single_batch(), max_epochs=1)

    gate_entries = _policy_gate_entries(spy.entries)
    assert len(gate_entries) >= 1

    reject_entry = gate_entries[0]
    meta = reject_entry.metadata
    gates = meta['gates']
    assert len(gates) == 1
    gate_dict = gates[0]
    assert gate_dict['name'] == 'BudgetGate'
    assert gate_dict['threshold'] == '10.0 USD'
    assert gate_dict['passed'] is False
