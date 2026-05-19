"""Tests for Optimizer param_groups (sub-plan 09).

Covers: single-group auto-wrap, multi-group construction, per-group lr,
add_param_group, flattened parameters property, state_dict round-trip,
zero_grad across groups, step sees groups, no _parameters attribute,
per-parameter state keyed by Parameter.id in checkpoint, add_param_group
missing 'params' raises TypeError, empty params no-op, duplicate param
across groups, load_state_dict wrong group count.
"""

from autopilot.core.gradient import NumericGradient
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
import pytest


class TestSingleGroupAutoWrap:
  """Optimizer([p1, p2]) has one group with both params and default lr."""

  def test_single_group_created(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt = Optimizer([p1, p2])
    assert len(opt.param_groups) == 1

  def test_group_contains_params(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt = Optimizer([p1, p2])
    assert opt.param_groups[0]['params'] == [p1, p2]

  def test_group_has_default_lr(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt = Optimizer([p1, p2])
    assert opt.param_groups[0]['lr'] == 1.0


class TestMultiGroupConstruction:
  """Optimizer([{'params': [p1], 'lr': 0.1}, {'params': [p2], 'lr': 0.2}])."""

  def test_two_groups(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt = Optimizer([{'params': [p1], 'lr': 0.1}, {'params': [p2], 'lr': 0.2}])
    assert len(opt.param_groups) == 2

  def test_group_params(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt = Optimizer([{'params': [p1], 'lr': 0.1}, {'params': [p2], 'lr': 0.2}])
    assert opt.param_groups[0]['params'] == [p1]
    assert opt.param_groups[1]['params'] == [p2]

  def test_group_lr_values(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt = Optimizer([{'params': [p1], 'lr': 0.1}, {'params': [p2], 'lr': 0.2}])
    assert opt.param_groups[0]['lr'] == 0.1
    assert opt.param_groups[1]['lr'] == 0.2


class TestPerGroupLr:
  """Per-group lr differs after construction."""

  def test_different_lr_per_group(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt = Optimizer([{'params': [p1], 'lr': 0.1}, {'params': [p2], 'lr': 0.2}])
    assert opt.param_groups[0]['lr'] != opt.param_groups[1]['lr']

  def test_defaults_filled_for_missing_keys(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt = Optimizer(
      [{'params': [p1], 'lr': 0.1}, {'params': [p2]}],
      lr=0.5,
    )
    assert opt.param_groups[0]['lr'] == 0.1
    assert opt.param_groups[1]['lr'] == 0.5


class TestAddParamGroup:
  """add_param_group appends and fills defaults."""

  def test_add_third_group(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    p3 = Parameter()
    opt = Optimizer([{'params': [p1], 'lr': 0.1}, {'params': [p2], 'lr': 0.2}])
    opt.add_param_group({'params': [p3], 'lr': 0.5})
    assert len(opt.param_groups) == 3
    assert opt.param_groups[2]['lr'] == 0.5

  def test_defaults_seeded(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt = Optimizer([p1], lr=0.3, momentum=0.9)
    opt.add_param_group({'params': [p2]})
    assert opt.param_groups[1]['lr'] == 0.3
    assert opt.param_groups[1]['momentum'] == 0.9

  def test_missing_params_key_raises_typeerror(self) -> None:
    opt = Optimizer([])
    with pytest.raises(TypeError, match='params'):
      opt.add_param_group({'lr': 0.1})

  def test_params_not_list_raises_typeerror(self) -> None:
    opt = Optimizer([])
    with pytest.raises(TypeError, match='list'):
      opt.add_param_group({'params': Parameter()})


class TestParametersPropertyFlattened:
  """Flattened view across all groups."""

  def test_three_groups_disjoint(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    p3 = Parameter()
    opt = Optimizer(
      [
        {'params': [p1]},
        {'params': [p2]},
        {'params': [p3]},
      ]
    )
    result = opt.parameters
    assert len(result) == 3
    assert result[0] is p1
    assert result[1] is p2
    assert result[2] is p3

  def test_single_group_order_preserved(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    p3 = Parameter()
    opt = Optimizer([p1, p2, p3])
    assert opt.parameters == [p1, p2, p3]

  def test_returns_new_list(self) -> None:
    p = Parameter()
    opt = Optimizer([p])
    a = opt.parameters
    b = opt.parameters
    assert a == b
    assert a is not b


class TestStateDictRoundTripWithGroups:
  """Save -> construct fresh -> load -> groups match."""

  def test_lr_values_preserved(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt = Optimizer([{'params': [p1], 'lr': 0.1}, {'params': [p2], 'lr': 0.2}])
    opt.block_strategy('x')
    state = opt.state_dict()

    opt2 = Optimizer([{'params': [p1]}, {'params': [p2]}])
    opt2.load_state_dict(state)
    assert opt2.param_groups[0]['lr'] == 0.1
    assert opt2.param_groups[1]['lr'] == 0.2
    assert opt2.blocked_strategies == frozenset({'x'})

  def test_defaults_preserved(self) -> None:
    p = Parameter()
    opt = Optimizer([p], lr=0.42, weight_decay=0.01)
    state = opt.state_dict()

    opt2 = Optimizer([p])
    opt2.load_state_dict(state)
    assert opt2.defaults['lr'] == 0.42
    assert opt2.defaults['weight_decay'] == 0.01


class TestZeroGradAcrossGroups:
  """zero_grad clears grad/grad_accumulator across all groups."""

  def test_clears_all_groups(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    p1.grad = NumericGradient(value=1.0)
    p2.grad = NumericGradient(value=2.0)
    p1.grad_accumulator = object()
    p2.grad_accumulator = object()

    opt = Optimizer([{'params': [p1]}, {'params': [p2]}])
    opt.zero_grad()

    assert p1.grad is None
    assert p2.grad is None
    assert p1.grad_accumulator is None
    assert p2.grad_accumulator is None


class TestStepSeesGroups:
  """Minimal subclass step() can access param_groups."""

  def test_subclass_step_sees_groups(self) -> None:
    p1 = Parameter()
    p2 = Parameter()

    class _GroupAwareOpt(Optimizer):
      def step(self) -> None:
        self.seen_groups = len(self.param_groups)

    opt = _GroupAwareOpt([{'params': [p1]}, {'params': [p2]}])
    opt.step()
    assert opt.seen_groups == 2


class TestNoUnderscoreParameters:
  """Optimizer has no _parameters attribute."""

  def test_no_underscore_parameters(self) -> None:
    p = Parameter()
    opt = Optimizer([p])
    assert not hasattr(opt, '_parameters')


class TestStateKeyedByParamIdInCheckpoint:
  """Runtime state uses id(param); checkpoint uses Parameter.id."""

  def test_state_in_checkpoint(self) -> None:
    p = Parameter()
    opt = Optimizer([p])
    opt.state[id(p)] = {'momentum': 0.9, 'step': 42}
    state = opt.state_dict()
    assert p.id in state['state']
    assert state['state'][p.id] == {'momentum': 0.9, 'step': 42}

  def test_state_restored_by_position(self) -> None:
    p = Parameter()
    opt = Optimizer([p])
    opt.state[id(p)] = {'momentum': 0.9}
    state = opt.state_dict()

    opt2 = Optimizer([p])
    opt2.load_state_dict(state)
    assert id(p) in opt2.state
    assert opt2.state[id(p)] == {'momentum': 0.9}


class TestEmptyParams:
  """Empty params list yields empty param_groups."""

  def test_empty_param_groups(self) -> None:
    opt = Optimizer([])
    assert opt.param_groups == []
    assert opt.parameters == []

  def test_zero_grad_noop(self) -> None:
    opt = Optimizer([])
    opt.zero_grad()

  def test_state_dict_empty(self) -> None:
    opt = Optimizer([])
    state = opt.state_dict()
    assert state['param_groups'] == []
    assert state['state'] == {}


class TestDuplicateParamAcrossGroups:
  """Same param in multiple groups appears in flattened list multiple times."""

  def test_duplicate_counted(self) -> None:
    p = Parameter()
    opt = Optimizer([{'params': [p]}, {'params': [p]}])
    assert len(opt.parameters) == 2
    assert opt.parameters == [p, p]


class TestLoadStateDictWrongGroupCount:
  """Mismatched group counts raise ValueError."""

  def test_too_few_groups(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt1 = Optimizer([{'params': [p1]}, {'params': [p2]}])
    state = opt1.state_dict()

    opt2 = Optimizer([p1])
    with pytest.raises(ValueError, match='param group'):
      opt2.load_state_dict(state)

  def test_too_many_groups(self) -> None:
    p = Parameter()
    opt1 = Optimizer([p])
    state = opt1.state_dict()

    opt2 = Optimizer([{'params': [p]}, {'params': [p]}])
    with pytest.raises(ValueError, match='param group'):
      opt2.load_state_dict(state)


class TestLoadStateDictWrongParamCount:
  """Mismatched parameter count within a group raises ValueError."""

  def test_wrong_param_count_in_group(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt1 = Optimizer([p1, p2])
    state = opt1.state_dict()

    opt2 = Optimizer([p1])
    with pytest.raises(ValueError, match='parameter'):
      opt2.load_state_dict(state)


class TestLoadStateDictPositionBased:
  """State is rebound by position, matching PyTorch semantics."""

  def test_different_ids_loads_by_position(self) -> None:
    p1 = Parameter()
    opt1 = Optimizer([p1])
    opt1.state[id(p1)] = {'momentum': 0.9}
    state = opt1.state_dict()

    p2 = Parameter()
    opt2 = Optimizer([p2])
    opt2.load_state_dict(state)
    assert id(p2) in opt2.state
    assert opt2.state[id(p2)] == {'momentum': 0.9}


class TestCustomDefaults:
  """Extra keyword defaults propagate to groups."""

  def test_extra_defaults_in_groups(self) -> None:
    p = Parameter()
    opt = Optimizer([p], lr=0.01, weight_decay=1e-4, betas=(0.9, 0.999))
    assert opt.defaults == {'lr': 0.01, 'weight_decay': 1e-4, 'betas': (0.9, 0.999)}
    assert opt.param_groups[0]['weight_decay'] == 1e-4
    assert opt.param_groups[0]['betas'] == (0.9, 0.999)


class TestParamGroupsSerializationFormat:
  """state_dict param_groups replaces Parameter objects with id strings."""

  def test_params_are_id_strings(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    opt = Optimizer([p1, p2])
    state = opt.state_dict()
    saved_params = state['param_groups'][0]['params']
    assert saved_params == [p1.id, p2.id]
    assert all(isinstance(pid, str) for pid in saved_params)


class TestStateDictContainsAllSections:
  """state_dict returns all required sections."""

  def test_all_sections_present(self) -> None:
    p = Parameter()
    opt = Optimizer([p])
    state = opt.state_dict()
    assert 'param_groups' in state
    assert 'defaults' in state
    assert 'blocked_strategies' in state
    assert 'state' in state
