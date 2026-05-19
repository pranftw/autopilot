"""Tests for DecisionEntry-typed metadata on policy gate context emissions.

Verifies that ``EpochLoop._check_policy_gate`` emits context entries with
``_type = DecisionEntry.POLICY_GATE_TYPE`` and a ``gates`` list of
``ConstraintResult.to_dict()`` payloads. Reject path emits ``_type`` +
``gates`` only; accept path adds ``metrics``.
"""

from autopilot.core.constraint import ConstraintResult
from autopilot.core.decision import DecisionEntry
from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.models import Result
from autopilot.core.types import GateResult
from autopilot.policy.policy import Policy
from typing import Any
from unittest.mock import MagicMock


def _make_trainer(policy: Policy | None = None) -> MagicMock:
  """Build a minimal mock trainer for ``_check_policy_gate``."""
  trainer = MagicMock()
  trainer.policy = policy
  trainer.callbacks = []
  captured: list[dict[str, Any]] = []

  def _capture_emit(reason, *, source, metadata):
    captured.append(
      {
        'reason': reason,
        'source': source,
        'metadata': metadata,
      }
    )

  trainer.emit_context = _capture_emit
  trainer._captured = captured
  return trainer


class _RejectPolicy(Policy):
  """Policy that rejects and populates ``result.gates`` with one failing gate."""

  def forward(self, result: Result) -> GateResult:
    result.gates = [
      ConstraintResult(
        name='MinGate',
        passed=False,
        metric='accuracy',
        value=0.5,
        threshold='>= 0.8',
        message='MinGate failed',
      ),
    ]
    return GateResult.FAIL


class _AcceptPolicy(Policy):
  """Policy that accepts and populates ``result.gates`` with one passing gate."""

  def forward(self, result: Result) -> GateResult:
    result.gates = [
      ConstraintResult(
        name='MinGate',
        passed=True,
        metric='accuracy',
        value=0.9,
        threshold='>= 0.8',
      ),
    ]
    return GateResult.PASSED


class _ThreeGatePolicy(Policy):
  """Policy with three gates: two pass, one fails."""

  def forward(self, result: Result) -> GateResult:
    result.gates = [
      ConstraintResult(
        name='MinGate',
        passed=True,
        metric='accuracy',
        value=0.9,
        threshold='>= 0.8',
      ),
      ConstraintResult(
        name='MaxGate',
        passed=False,
        metric='latency',
        value=500.0,
        threshold='<= 200',
        message='MaxGate failed',
      ),
      ConstraintResult(
        name='RangeGate',
        passed=True,
        metric='f1',
        value=0.85,
        threshold='[0.7, 1.0]',
      ),
    ]
    return GateResult.FAIL


class TestPolicyGateRejectMetadata:
  """Reject-path metadata shape tests."""

  def test_policy_gate_reject_metadata_has_type(self) -> None:
    trainer = _make_trainer(policy=_RejectPolicy())
    loop = EpochLoop()
    loop._check_policy_gate(trainer, epoch=0, metric_values={'accuracy': 0.5}, experiment=None)
    entry = trainer._captured[0]
    assert entry['metadata']['_type'] == DecisionEntry.POLICY_GATE_TYPE

  def test_policy_gate_reject_has_gates_list(self) -> None:
    trainer = _make_trainer(policy=_RejectPolicy())
    loop = EpochLoop()
    loop._check_policy_gate(trainer, epoch=0, metric_values={'accuracy': 0.5}, experiment=None)
    metadata = trainer._captured[0]['metadata']
    assert isinstance(metadata['gates'], list)
    assert len(metadata['gates']) == 1
    assert isinstance(metadata['gates'][0], dict)

  def test_policy_gate_reject_per_gate_detail(self) -> None:
    trainer = _make_trainer(policy=_RejectPolicy())
    loop = EpochLoop()
    loop._check_policy_gate(trainer, epoch=0, metric_values={'accuracy': 0.5}, experiment=None)
    gate_dict = trainer._captured[0]['metadata']['gates'][0]
    assert gate_dict['name'] == 'MinGate'
    assert gate_dict['passed'] is False
    assert gate_dict['metric'] == 'accuracy'
    assert gate_dict['value'] == 0.5
    assert gate_dict['threshold'] == '>= 0.8'
    assert 'message' in gate_dict

  def test_policy_gate_reject_metadata_omits_metrics_and_gate_result(self) -> None:
    trainer = _make_trainer(policy=_RejectPolicy())
    loop = EpochLoop()
    loop._check_policy_gate(trainer, epoch=0, metric_values={'accuracy': 0.5}, experiment=None)
    metadata = trainer._captured[0]['metadata']
    assert set(metadata.keys()) == {'_type', 'gates'}

  def test_constraint_result_value_in_gate_dict(self) -> None:
    trainer = _make_trainer(policy=_RejectPolicy())
    loop = EpochLoop()
    loop._check_policy_gate(trainer, epoch=0, metric_values={'accuracy': 0.5}, experiment=None)
    gate_dict = trainer._captured[0]['metadata']['gates'][0]
    assert isinstance(gate_dict['value'], float)
    assert isinstance(gate_dict['passed'], bool)
    assert isinstance(gate_dict['threshold'], str)


class TestPolicyGateAcceptMetadata:
  """Accept-path metadata shape tests."""

  def test_policy_gate_accept_metadata_has_type(self) -> None:
    trainer = _make_trainer(policy=_AcceptPolicy())
    loop = EpochLoop()
    loop._check_policy_gate(trainer, epoch=0, metric_values={'accuracy': 0.9}, experiment=None)
    entry = trainer._captured[0]
    assert entry['metadata']['_type'] == DecisionEntry.POLICY_GATE_TYPE

  def test_policy_gate_accept_has_metrics_in_metadata(self) -> None:
    trainer = _make_trainer(policy=_AcceptPolicy())
    loop = EpochLoop()
    metrics = {'accuracy': 0.9}
    loop._check_policy_gate(trainer, epoch=0, metric_values=metrics, experiment=None)
    metadata = trainer._captured[0]['metadata']
    assert '_type' in metadata
    assert 'gates' in metadata
    assert 'metrics' in metadata
    assert metadata['metrics']['accuracy'] == 0.9


class TestPolicyGateMultiGate:
  """Multi-gate scenarios."""

  def test_policy_3_gates_1_fails_all_shown(self) -> None:
    trainer = _make_trainer(policy=_ThreeGatePolicy())
    loop = EpochLoop()
    loop._check_policy_gate(
      trainer,
      epoch=0,
      metric_values={'accuracy': 0.9, 'latency': 500.0, 'f1': 0.85},
      experiment=None,
    )
    metadata = trainer._captured[0]['metadata']
    gates = metadata['gates']
    assert len(gates) == 3
    failed = [g for g in gates if not g['passed']]
    passed = [g for g in gates if g['passed']]
    assert len(failed) == 1
    assert len(passed) == 2
    assert failed[0]['name'] == 'MaxGate'
    assert failed[0]['metric'] == 'latency'
