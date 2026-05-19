"""Tests for Module enhancements: buffers, freeze/unfreeze, strict load_state_dict, zero_grad."""

from autopilot.core.gradient import NumericGradient
from autopilot.core.module.module import IncompatibleKeys, Module
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from tests.doubles import NoOpOptimizer
import pytest


class _SimpleModule(Module):
  def forward(self, *args, **kwargs):
    return Datum()


# -- register_buffer tests --


class TestRegisterBuffer:
  def test_register_buffer_in_state_dict(self) -> None:
    mod = Module()
    mod.register_buffer('x', {'k': 1})
    state = mod.state_dict()
    assert 'x' in state
    assert state['x'] == {'k': 1}

  def test_register_buffer_not_in_parameters(self) -> None:
    mod = Module()
    mod.register_buffer('buf', [1, 2, 3])
    assert list(mod.parameters()) == []

  def test_non_persistent_buffer_excluded(self) -> None:
    mod = Module()
    mod.register_buffer('ephemeral', 42, persistent=False)
    state = mod.state_dict()
    assert 'ephemeral' not in state

  def test_buffer_accessible_as_attribute(self) -> None:
    mod = Module()
    value = {'key': 'val'}
    mod.register_buffer('x', value)
    assert mod.x is value

  def test_register_buffer_replace(self) -> None:
    """Re-registering same name replaces buffer entry."""
    mod = Module()
    mod.register_buffer('x', [1, 2])
    assert mod.x == [1, 2]
    mod.register_buffer('x', [3, 4])
    assert mod.x == [3, 4]
    assert mod._buffers['x'] == [3, 4]
    state = mod.state_dict()
    assert state['x'] == [3, 4]

  def test_register_buffer_replace_persistence(self) -> None:
    """Re-registering changes persistence flag."""
    mod = Module()
    mod.register_buffer('x', 1, persistent=False)
    assert 'x' not in mod.state_dict()
    mod.register_buffer('x', 2, persistent=True)
    assert 'x' in mod.state_dict()
    assert mod.state_dict()['x'] == 2

  def test_non_json_buffer_raises_typeerror(self) -> None:
    """Non-JSON-serializable buffer raises TypeError at state_dict() time."""
    mod = Module()
    mod.register_buffer('obj', object())
    with pytest.raises(TypeError, match='not JSON-serializable'):
      mod.state_dict()

  def test_register_buffer_clears_parameter(self) -> None:
    """Buffer registration clears a prior parameter entry for the same name."""
    mod = Module()
    mod.p = Parameter()
    assert 'p' in mod._parameters
    mod.register_buffer('p', 99)
    assert 'p' not in mod._parameters
    assert mod._buffers['p'] == 99

  def test_register_buffer_clears_module(self) -> None:
    """Buffer registration clears a prior child module entry for the same name."""
    mod = Module()
    mod.c = _SimpleModule()
    assert 'c' in mod._modules
    mod.register_buffer('c', 'data')
    assert 'c' not in mod._modules
    assert mod._buffers['c'] == 'data'

  def test_setattr_clears_buffer(self) -> None:
    """Direct assignment clears a prior buffer entry."""
    mod = Module()
    mod.register_buffer('x', [1, 2, 3])
    assert 'x' in mod._buffers
    mod.x = 42
    assert 'x' not in mod._buffers
    assert mod.x == 42

  def test_parameter_assignment_clears_buffer(self) -> None:
    """Assigning a Parameter to a buffer name clears the buffer."""
    mod = Module()
    mod.register_buffer('x', 'buf')
    mod.x = Parameter()
    assert 'x' not in mod._buffers
    assert 'x' in mod._parameters

  def test_module_assignment_clears_buffer(self) -> None:
    """Assigning a Module to a buffer name clears the buffer."""
    mod = Module()
    mod.register_buffer('x', 'buf')
    mod.x = _SimpleModule()
    assert 'x' not in mod._buffers
    assert 'x' in mod._modules

  def test_register_buffer_reserved_name_raises(self) -> None:
    """Buffer with reserved name raises ValueError."""
    mod = Module()
    with pytest.raises(ValueError, match='reserved name'):
      mod.register_buffer('state_dict', 42)

  def test_non_persistent_buffer_not_loaded(self) -> None:
    """Non-persistent buffer absent from state_dict is ignored on load."""
    mod = Module()
    mod.register_buffer('ephemeral', 99, persistent=False)
    mod.p = Parameter()
    state = mod.state_dict()
    assert 'ephemeral' not in state

    mod2 = Module()
    mod2.register_buffer('ephemeral', 0, persistent=False)
    mod2.p = Parameter()
    mod2.load_state_dict(state)
    assert mod2.ephemeral == 0

  def test_buffer_json_types(self) -> None:
    """All JSON-native types serialize successfully."""
    mod = Module()
    mod.register_buffer('d', {'a': 1})
    mod.register_buffer('l', [1, 2, 3])
    mod.register_buffer('s', 'hello')
    mod.register_buffer('f', 0.123)
    mod.register_buffer('i', 42)
    mod.register_buffer('b', value=True)
    mod.register_buffer('n', None)
    state = mod.state_dict()
    assert len(state) == 7

  def test_nested_non_json_buffer_raises(self) -> None:
    """Nested non-JSON-serializable value raises TypeError."""
    mod = Module()
    mod.register_buffer('nested', {'inner': object()})
    with pytest.raises(TypeError, match='not JSON-serializable'):
      mod.state_dict()


# -- requires_grad_ tests --


class TestRequiresGrad:
  def test_requires_grad_false_freezes_all(self) -> None:
    mod = Module()
    mod.p1 = Parameter(requires_grad=True)
    mod.p2 = Parameter(requires_grad=True)
    mod.requires_grad_(requires_grad=False)
    assert mod.p1.requires_grad is False
    assert mod.p2.requires_grad is False

  def test_requires_grad_recursive(self) -> None:
    """Child parameters are also frozen."""
    parent = Module()
    child = Module()
    child.w = Parameter(requires_grad=True)
    parent.child = child
    parent.requires_grad_(requires_grad=False)
    assert child.w.requires_grad is False

  def test_requires_grad_returns_self(self) -> None:
    mod = Module()
    assert mod.requires_grad_(requires_grad=False) is mod

  def test_requires_grad_true_unfreezes(self) -> None:
    mod = Module()
    mod.p = Parameter(requires_grad=False)
    mod.requires_grad_(requires_grad=True)
    assert mod.p.requires_grad is True

  def test_requires_grad_no_params_noop(self) -> None:
    """No error when module has no parameters."""
    mod = Module()
    result = mod.requires_grad_(requires_grad=False)
    assert result is mod

  def test_requires_grad_deeply_nested(self) -> None:
    """Three-level nesting freezes all parameters."""
    root = Module()
    mid = Module()
    leaf = Module()
    leaf.w = Parameter(requires_grad=True)
    mid.leaf = leaf
    root.mid = mid
    root.p = Parameter(requires_grad=True)
    root.requires_grad_(requires_grad=False)
    assert root.p.requires_grad is False
    assert leaf.w.requires_grad is False


# -- IncompatibleKeys tests --


class TestIncompatibleKeys:
  def test_incompatible_keys_dataclass(self) -> None:
    ik = IncompatibleKeys(['a'], [])
    assert ik.missing_keys == ['a']
    assert ik.unexpected_keys == []

  def test_incompatible_keys_round_trip(self) -> None:
    """Fields survive construction and equality."""
    ik = IncompatibleKeys(missing_keys=['x', 'y'], unexpected_keys=['z'])
    assert ik.missing_keys == ['x', 'y']
    assert ik.unexpected_keys == ['z']
    ik2 = IncompatibleKeys(missing_keys=['x', 'y'], unexpected_keys=['z'])
    assert ik == ik2

  def test_incompatible_keys_empty(self) -> None:
    ik = IncompatibleKeys([], [])
    assert not ik.missing_keys
    assert not ik.unexpected_keys


# -- strict load_state_dict tests --


class TestLoadStateDictStrict:
  def test_load_state_dict_strict_missing(self) -> None:
    """strict=True with missing keys raises RuntimeError."""
    mod = Module()
    mod.p = Parameter()
    with pytest.raises(RuntimeError, match='Missing') as exc_info:
      mod.load_state_dict({}, strict=True)
    assert 'p' in str(exc_info.value)
    assert 'strict=False' in str(exc_info.value)

  def test_load_state_dict_strict_unexpected(self) -> None:
    """strict=True with extra root key raises RuntimeError."""
    mod = Module()
    with pytest.raises(RuntimeError, match='Unexpected') as exc_info:
      mod.load_state_dict({'phantom': 42}, strict=True)
    assert 'phantom' in str(exc_info.value)

  def test_load_state_dict_strict_both_missing_and_unexpected(self) -> None:
    """RuntimeError includes both missing and unexpected keys."""
    mod = Module()
    mod.p = Parameter()
    with pytest.raises(RuntimeError) as exc_info:
      mod.load_state_dict({'phantom': 42}, strict=True)
    msg = str(exc_info.value)
    assert 'Missing' in msg
    assert 'Unexpected' in msg

  def test_load_state_dict_non_strict(self) -> None:
    """Partial dict loads overlapping keys; result lists sorted correctly."""
    mod = Module()
    mod.p1 = Parameter(requires_grad=False)
    mod.p2 = Parameter(requires_grad=False)

    partial_state = {'p1': Parameter(requires_grad=True).to_dict()}
    result = mod.load_state_dict(partial_state, strict=False)

    assert mod.p1.requires_grad is True
    assert mod.p2.requires_grad is False
    assert result.missing_keys == ['p2']
    assert result.unexpected_keys == []

  def test_load_state_dict_strict_success_empty_lists(self) -> None:
    """Successful strict load returns empty IncompatibleKeys."""
    mod = Module()
    mod.p = Parameter(requires_grad=True)
    state = mod.state_dict()
    result = mod.load_state_dict(state, strict=True)
    assert isinstance(result, IncompatibleKeys)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

  def test_load_state_dict_non_strict_unexpected(self) -> None:
    """Non-strict load tolerates extra keys."""
    mod = Module()
    result = mod.load_state_dict({'extra': 'data'}, strict=False)
    assert result.unexpected_keys == ['extra']
    assert result.missing_keys == []

  def test_nested_strict_load_dotted_keys(self) -> None:
    """Strict mismatch reports full dotted keys from root perspective."""
    parent = Module()
    child = Module()
    child.w = Parameter()
    parent.layer = child

    with pytest.raises(RuntimeError) as exc_info:
      parent.load_state_dict({}, strict=True)
    assert 'layer.w' in str(exc_info.value)

  def test_nested_load_round_trip(self) -> None:
    """Save and load nested module state preserves values."""
    parent = Module()
    child = Module()
    child.w = Parameter(requires_grad=True)
    parent.layer = child

    state = parent.state_dict()

    parent2 = Module()
    child2 = Module()
    child2.w = Parameter(requires_grad=False)
    parent2.layer = child2

    result = parent2.load_state_dict(state, strict=True)
    assert child2.w.requires_grad is True
    assert result.missing_keys == []
    assert result.unexpected_keys == []

  def test_load_state_dict_with_buffers(self) -> None:
    """Buffers round-trip through state_dict/load_state_dict."""
    mod = Module()
    mod.register_buffer('stats', {'count': 10, 'mean': 0.5})
    state = mod.state_dict()

    mod2 = Module()
    mod2.register_buffer('stats', {})
    result = mod2.load_state_dict(state, strict=True)
    assert mod2.stats == {'count': 10, 'mean': 0.5}
    assert result.missing_keys == []

  def test_load_state_dict_buffer_and_param_mixed(self) -> None:
    """Module with both params and buffers loads correctly."""
    mod = Module()
    mod.p = Parameter(requires_grad=True)
    mod.register_buffer('b', [1, 2])
    state = mod.state_dict()
    assert 'p' in state
    assert 'b' in state

    mod2 = Module()
    mod2.p = Parameter(requires_grad=False)
    mod2.register_buffer('b', [])
    result = mod2.load_state_dict(state, strict=True)
    assert mod2.p.requires_grad is True
    assert mod2.b == [1, 2]
    assert result.missing_keys == []
    assert result.unexpected_keys == []

  def test_strict_load_missing_buffer_key(self) -> None:
    """Strict load catches missing buffer key."""
    mod = Module()
    mod.register_buffer('x', 42)
    with pytest.raises(RuntimeError, match='Missing'):
      mod.load_state_dict({}, strict=True)

  def test_deeply_nested_strict_load(self) -> None:
    """Three-level nesting with strict load succeeds on exact match."""
    root = Module()
    mid = Module()
    leaf = Module()
    leaf.w = Parameter()
    leaf.register_buffer('b', 'data')
    mid.leaf = leaf
    root.mid = mid

    state = root.state_dict()
    assert 'mid.leaf.w' in state
    assert 'mid.leaf.b' in state

    root2 = Module()
    mid2 = Module()
    leaf2 = Module()
    leaf2.w = Parameter()
    leaf2.register_buffer('b', '')
    mid2.leaf = leaf2
    root2.mid = mid2

    result = root2.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert leaf2.b == 'data'

  def test_sorted_missing_and_unexpected(self) -> None:
    """Missing and unexpected lists are alphabetically sorted."""
    mod = Module()
    mod.z_param = Parameter()
    mod.a_param = Parameter()
    result = mod.load_state_dict({'x_extra': 1, 'b_extra': 2}, strict=False)
    assert result.missing_keys == ['a_param', 'z_param']
    assert result.unexpected_keys == ['b_extra', 'x_extra']


# -- _state_dict_keys tests --


class TestStateDictKeys:
  def test_state_dict_keys_matches_state_dict(self) -> None:
    """Keys from state_dict_keys match the actual state_dict keys."""
    parent = Module()
    child = Module()
    child.w = Parameter()
    child.register_buffer('b', 'data')
    parent.layer = child
    parent.p = Parameter()
    parent.register_buffer('stats', {'count': 0})

    assert parent.state_dict_keys() == set(parent.state_dict().keys())

  def test_state_dict_keys_excludes_non_persistent(self) -> None:
    """Non-persistent buffers are excluded from state_dict_keys."""
    mod = Module()
    mod.register_buffer('persistent', 1)
    mod.register_buffer('ephemeral', 2, persistent=False)
    mod.p = Parameter()

    keys = mod.state_dict_keys()
    assert 'persistent' in keys
    assert 'p' in keys
    assert 'ephemeral' not in keys

  def test_state_dict_keys_empty_module(self) -> None:
    """Empty module has no keys."""
    mod = Module()
    assert mod.state_dict_keys() == set()

  def test_state_dict_keys_deep_nesting(self) -> None:
    """Grandchild parameters produce dotted keys."""
    root = Module()
    mid = Module()
    leaf = Module()
    leaf.param = Parameter()
    mid.leaf = leaf
    root.mid = mid

    keys = root.state_dict_keys()
    assert 'mid.leaf.param' in keys
    assert keys == set(root.state_dict().keys())


# -- zero_grad tests --


class TestModuleZeroGrad:
  def test_module_zero_grad_clears_grad_and_accumulator(self) -> None:
    """zero_grad sets grad and grad_accumulator to None on owned parameters."""
    mod = Module()
    mod.p = Parameter()
    mod.p.grad = NumericGradient(value=1.0)
    mod.p.grad_accumulator = object()

    mod.zero_grad()
    assert mod.p.grad is None
    assert mod.p.grad_accumulator is None

  def test_module_zero_grad_recursive(self) -> None:
    """zero_grad clears nested child module parameters."""
    parent = Module()
    child = Module()
    child.w = Parameter()
    child.w.grad = NumericGradient(value=2.0)
    child.w.grad_accumulator = object()
    parent.child = child

    parent.zero_grad()
    assert child.w.grad is None
    assert child.w.grad_accumulator is None

  def test_module_zero_grad_matches_optimizer_cleanup(self) -> None:
    """Module.zero_grad produces same result as Optimizer.zero_grad for same params."""
    mod_a = Module()
    mod_a.p = Parameter()
    mod_a.p.grad = NumericGradient(value=3.0)
    mod_a.p.grad_accumulator = object()

    mod_b = Module()
    mod_b.p = Parameter()
    mod_b.p.grad = NumericGradient(value=3.0)
    mod_b.p.grad_accumulator = object()

    mod_a.zero_grad()

    opt = NoOpOptimizer([mod_b.p])
    opt.zero_grad()

    assert mod_a.p.grad is None
    assert mod_a.p.grad_accumulator is None
    assert mod_b.p.grad is None
    assert mod_b.p.grad_accumulator is None

  def test_module_zero_grad_no_params_noop(self) -> None:
    """zero_grad on module with no parameters is a no-op."""
    mod = Module()
    mod.zero_grad()

  def test_module_zero_grad_already_none(self) -> None:
    """zero_grad when grad is already None does not raise."""
    mod = Module()
    mod.p = Parameter()
    assert mod.p.grad is None
    assert mod.p.grad_accumulator is None
    mod.zero_grad()
    assert mod.p.grad is None
    assert mod.p.grad_accumulator is None
