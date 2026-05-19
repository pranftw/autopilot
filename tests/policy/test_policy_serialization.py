"""Policy and gate serialization round-trip tests.

Verifies state_dict() -> from_dict() -> state_dict() identity for all
gate types and QualityFirstPolicy.
"""

from autopilot.policy.gates import (
  BudgetGate,
  MaxGate,
  MinGate,
  MonotonicGate,
  RangeGate,
)
from autopilot.policy.quality_first import QualityFirstPolicy
import pytest


def test_min_gate_round_trip():
  """MinGate state_dict -> from_dict preserves all fields."""
  gate = MinGate('accuracy', threshold=0.8, required=False)
  state = gate.state_dict()
  restored = MinGate.from_dict(state)
  assert restored.metric == 'accuracy'
  assert restored.threshold == 0.8
  assert restored.required is False
  assert state['type'] == 'MinGate'


def test_max_gate_round_trip():
  """MaxGate state_dict -> from_dict preserves threshold."""
  gate = MaxGate('loss', threshold=1.5, required=True)
  state = gate.state_dict()
  restored = MaxGate.from_dict(state)
  assert restored.metric == 'loss'
  assert restored.threshold == 1.5
  assert restored.required is True


def test_range_gate_round_trip():
  """RangeGate state_dict -> from_dict preserves min/max values."""
  gate = RangeGate('score', min_value=0.0, max_value=1.0, required=False)
  state = gate.state_dict()
  restored = RangeGate.from_dict(state)
  assert restored.metric == 'score'
  assert restored.min_value == 0.0
  assert restored.max_value == 1.0
  assert restored.required is False


def test_monotonic_gate_round_trip():
  """MonotonicGate state_dict -> from_dict preserves direction and epsilon."""
  gate = MonotonicGate('val_accuracy', direction='non_increasing', epsilon=0.05)
  state = gate.state_dict()
  restored = MonotonicGate.from_dict(state)
  assert restored.metric == 'val_accuracy'
  assert restored.direction == 'non_increasing'
  assert restored.epsilon == 0.05
  assert restored.required is True


def test_budget_gate_round_trip():
  """BudgetGate state_dict -> from_dict preserves max_usd."""
  gate = BudgetGate(max_usd=50.0, required=False)
  state = gate.state_dict()
  restored = BudgetGate.from_dict(state)
  assert restored.max_usd == 50.0
  assert restored.required is False


def test_quality_first_policy_state_dict():
  """Policy state_dict includes gates and human_review_on_warn."""
  policy = QualityFirstPolicy(
    gates=[MinGate('acc', 0.8), MaxGate('loss', 1.0), BudgetGate(max_usd=25.0)],
    human_review_on_warn=False,
  )
  state = policy.state_dict()
  assert len(state['gates']) == 3
  assert state['human_review_on_warn'] is False
  assert state['gates'][0]['type'] == 'MinGate'
  assert state['gates'][1]['type'] == 'MaxGate'
  assert state['gates'][2]['type'] == 'BudgetGate'


def test_quality_first_policy_load_state_dict():
  """load_state_dict restores gate count and types from serialized state."""
  policy = QualityFirstPolicy(gates=[])
  state = {
    'gates': [
      {'type': 'MinGate', 'metric': 'acc', 'threshold': 0.9, 'required': True},
      {'type': 'RangeGate', 'metric': 'x', 'min_value': 0, 'max_value': 1, 'required': True},
    ],
    'human_review_on_warn': True,
  }
  policy.load_state_dict(state)
  assert len(policy.gates) == 2
  assert type(policy.gates[0]).__name__ == 'MinGate'
  assert type(policy.gates[1]).__name__ == 'RangeGate'
  assert policy.human_review_on_warn is True


def test_policy_round_trip_full():
  """state_dict -> load_state_dict -> state_dict yields identical dicts."""
  original = QualityFirstPolicy(
    gates=[
      MinGate('acc', 0.8),
      MonotonicGate('val_loss', direction='non_increasing', epsilon=0.1),
      BudgetGate(max_usd=100.0),
    ],
    human_review_on_warn=True,
  )
  state1 = original.state_dict()
  restored = QualityFirstPolicy(gates=[])
  restored.load_state_dict(state1)
  state2 = restored.state_dict()
  assert state1 == state2


def test_unknown_gate_type_raises():
  """Unrecognized gate type in load_state_dict raises KeyError."""
  policy = QualityFirstPolicy(gates=[])
  state = {
    'gates': [{'type': 'UnknownGate', 'metric': 'x'}],
    'human_review_on_warn': False,
  }
  with pytest.raises(KeyError):
    policy.load_state_dict(state)


def test_load_state_dict_overwrites_existing_gates():
  """load_state_dict replaces gates, not appends."""
  policy = QualityFirstPolicy(
    gates=[MinGate('a', 0.5), MaxGate('b', 1.0)],
  )
  assert len(policy.gates) == 2
  state = {
    'gates': [{'type': 'BudgetGate', 'max_usd': 10.0, 'required': True}],
    'human_review_on_warn': False,
  }
  policy.load_state_dict(state)
  assert len(policy.gates) == 1
  assert type(policy.gates[0]).__name__ == 'BudgetGate'
