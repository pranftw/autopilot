"""Tests for Optimizer per-parameter state checkpoint round-trip (Plan 25).

Proves that subclasses stashing data in ``self.state`` survive
``state_dict`` / ``load_state_dict`` round-trips with Parameter.id-keyed
checkpoint serialization.
"""

from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from tests.doubles import StateTrackingOptimizer


def test_optimizer_state_round_trip():
  """StateTrackingOptimizer writes plan25_marker; round-trip restores it."""
  p1 = Parameter()
  p2 = Parameter()
  opt = StateTrackingOptimizer([p1, p2])

  opt.step()
  assert opt.state[id(p1)]['plan25_marker'] == 1
  assert opt.state[id(p2)]['plan25_marker'] == 1

  opt.step()
  assert opt.state[id(p1)]['plan25_marker'] == 2

  saved = opt.state_dict()
  assert p1.id in saved['state']
  assert saved['state'][p1.id]['plan25_marker'] == 2
  assert saved['state'][p2.id]['plan25_marker'] == 2

  fresh_p1 = Parameter()
  fresh_p2 = Parameter()
  fresh_p1._id = p1.id
  fresh_p2._id = p2.id

  opt2 = StateTrackingOptimizer([fresh_p1, fresh_p2])
  opt2.load_state_dict(saved)

  assert opt2.state[id(fresh_p1)]['plan25_marker'] == 2
  assert opt2.state[id(fresh_p2)]['plan25_marker'] == 2


def test_optimizer_state_empty_when_no_step():
  """State dict has empty state when step() was never called."""
  p = Parameter()
  opt = StateTrackingOptimizer([p])
  saved = opt.state_dict()
  assert saved['state'] == {}


def test_optimizer_state_round_trip_preserves_other_entries():
  """Additional entries in per-parameter state survive round-trip."""

  class CustomOptimizer(Optimizer):
    def step(self):
      for param in self.parameters:
        entry = self.state.setdefault(id(param), {})
        entry['momentum'] = entry.get('momentum', 0.0) + 0.1
        entry['step_count'] = entry.get('step_count', 0) + 1

  p = Parameter()
  opt = CustomOptimizer([p])
  opt.step()
  opt.step()

  saved = opt.state_dict()
  assert saved['state'][p.id]['momentum'] == 0.2
  assert saved['state'][p.id]['step_count'] == 2

  fresh_p = Parameter()
  fresh_p._id = p.id
  opt2 = CustomOptimizer([fresh_p])
  opt2.load_state_dict(saved)

  assert opt2.state[id(fresh_p)]['momentum'] == 0.2
  assert opt2.state[id(fresh_p)]['step_count'] == 2


def test_optimizer_state_round_trip_multi_group():
  """State round-trips correctly across multiple parameter groups."""
  p1 = Parameter()
  p2 = Parameter()
  opt = StateTrackingOptimizer(
    [
      {'params': [p1], 'lr': 0.1},
      {'params': [p2], 'lr': 0.01},
    ]
  )
  opt.step()

  saved = opt.state_dict()
  assert saved['state'][p1.id]['plan25_marker'] == 1
  assert saved['state'][p2.id]['plan25_marker'] == 1

  fresh_p1 = Parameter()
  fresh_p2 = Parameter()
  fresh_p1._id = p1.id
  fresh_p2._id = p2.id

  opt2 = StateTrackingOptimizer(
    [
      {'params': [fresh_p1], 'lr': 0.1},
      {'params': [fresh_p2], 'lr': 0.01},
    ]
  )
  opt2.load_state_dict(saved)

  assert opt2.state[id(fresh_p1)]['plan25_marker'] == 1
  assert opt2.state[id(fresh_p2)]['plan25_marker'] == 1
  assert opt2.param_groups[0]['lr'] == 0.1
  assert opt2.param_groups[1]['lr'] == 0.01


def test_no_parallel_shadow_dicts_pattern():
  """Subclasses must use self.state, not parallel shadow dicts."""
  p = Parameter()
  opt = StateTrackingOptimizer([p])
  opt.step()

  assert id(p) in opt.state
  assert 'plan25_marker' in opt.state[id(p)]
  assert not hasattr(opt, '_custom_state')
