"""Tests for ScalarParameter (Plan 19).

Covers construction, value property, snapshot/restore round-trips, JSON types,
schema_entry, to_dict/from_dict checkpoint serialization, load_from_dict,
render(), optimizer participation, FileStore integration, and error paths.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.module.module import Module
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import (
  COMMON_WRONG_KWARGS,
  Parameter,
  ScalarParameter,
)
from autopilot.core.snapshot import ParameterSchemaEntry
from pathlib import Path
from typing import Any
import json
import pytest


class TestConstruction:
  """ScalarParameter construction and basic behavior."""

  def test_default_value_is_none(self) -> None:
    sp = ScalarParameter()
    assert sp.value is None

  def test_string_value(self) -> None:
    sp = ScalarParameter(value='hello')
    assert sp.value == 'hello'

  def test_integer_value(self) -> None:
    sp = ScalarParameter(value=42)
    assert sp.value == 42

  def test_float_value(self) -> None:
    sp = ScalarParameter(value=0.75)
    assert sp.value == 0.75

  def test_boolean_value(self) -> None:
    sp = ScalarParameter(value=True)
    assert sp.value is True

  def test_dict_value(self) -> None:
    sp = ScalarParameter(value={'k': 'v', 'n': 1})
    assert sp.value == {'k': 'v', 'n': 1}

  def test_list_value(self) -> None:
    sp = ScalarParameter(value=[1, 2, 'three'])
    assert sp.value == [1, 2, 'three']

  def test_none_value_explicit(self) -> None:
    sp = ScalarParameter(value=None)
    assert sp.value is None

  def test_nested_structure(self) -> None:
    nested = {'a': [1, 2], 'b': {'c': True}}
    sp = ScalarParameter(value=nested)
    assert sp.value == nested

  def test_isinstance_parameter(self) -> None:
    sp = ScalarParameter(value='x')
    assert isinstance(sp, Parameter)

  def test_requires_grad_default_true(self) -> None:
    sp = ScalarParameter(value='x')
    assert sp.requires_grad is True

  def test_requires_grad_false(self) -> None:
    sp = ScalarParameter(value='x', requires_grad=False)
    assert sp.requires_grad is False

  def test_has_id(self) -> None:
    sp = ScalarParameter(value='x')
    assert sp.id
    assert len(sp.id) > 0


class TestValueProperty:
  """value getter/setter behavior."""

  def test_getter(self) -> None:
    sp = ScalarParameter(value=10)
    assert sp.value == 10

  def test_setter(self) -> None:
    sp = ScalarParameter(value=10)
    sp.value = 20
    assert sp.value == 20

  def test_setter_to_none(self) -> None:
    sp = ScalarParameter(value='hello')
    sp.value = None
    assert sp.value is None

  def test_setter_type_change(self) -> None:
    sp = ScalarParameter(value=42)
    sp.value = 'now a string'
    assert sp.value == 'now a string'


class TestSnapshotRestore:
  """snapshot()/restore() round-trip through JSON text."""

  def test_string_round_trip(self) -> None:
    sp = ScalarParameter(value='hello world')
    content = sp.snapshot()
    assert 'value.json' in content
    assert json.loads(content['value.json']) == 'hello world'
    sp2 = ScalarParameter()
    sp2.restore(content)
    assert sp2.value == 'hello world'

  def test_integer_round_trip(self) -> None:
    sp = ScalarParameter(value=42)
    content = sp.snapshot()
    sp2 = ScalarParameter()
    sp2.restore(content)
    assert sp2.value == 42

  def test_float_round_trip(self) -> None:
    sp = ScalarParameter(value=0.75)
    content = sp.snapshot()
    sp2 = ScalarParameter()
    sp2.restore(content)
    assert sp2.value == pytest.approx(0.75)

  def test_dict_round_trip(self) -> None:
    sp = ScalarParameter(value={'k': 'v', 'n': [1, 2]})
    content = sp.snapshot()
    sp2 = ScalarParameter()
    sp2.restore(content)
    assert sp2.value == {'k': 'v', 'n': [1, 2]}

  def test_list_round_trip(self) -> None:
    sp = ScalarParameter(value=[1, 'two', None, True])
    content = sp.snapshot()
    sp2 = ScalarParameter()
    sp2.restore(content)
    assert sp2.value == [1, 'two', None, True]

  def test_none_round_trip(self) -> None:
    sp = ScalarParameter(value=None)
    content = sp.snapshot()
    sp2 = ScalarParameter(value='overwritten')
    sp2.restore(content)
    assert sp2.value is None

  def test_boolean_round_trip(self) -> None:
    sp = ScalarParameter(value=False)
    content = sp.snapshot()
    sp2 = ScalarParameter()
    sp2.restore(content)
    assert sp2.value is False

  def test_restore_empty_dict_noop(self) -> None:
    sp = ScalarParameter(value='original')
    sp.restore({})
    assert sp.value == 'original'

  def test_restore_malformed_non_empty_raises(self) -> None:
    sp = ScalarParameter(value='original')
    with pytest.raises(StoreError, match=r'value\.json'):
      sp.restore({'other_key': 'data'})

  def test_restore_valid_content(self) -> None:
    sp = ScalarParameter(value='old')
    sp.restore({'value.json': json.dumps(42)})
    assert sp.value == 42

  def test_snapshot_returns_single_key(self) -> None:
    sp = ScalarParameter(value='test')
    content = sp.snapshot()
    assert len(content) == 1
    assert list(content.keys()) == ['value.json']


class TestSnapshotValueRestoreValue:
  """snapshot_value()/restore_value() public helpers."""

  def test_snapshot_value_returns_current(self) -> None:
    sp = ScalarParameter(value={'a': 1})
    assert sp.snapshot_value() == {'a': 1}

  def test_restore_value_sets_value(self) -> None:
    sp = ScalarParameter()
    sp.restore_value({'key': 'val'})
    assert sp.value == {'key': 'val'}

  def test_restore_value_none(self) -> None:
    sp = ScalarParameter(value='something')
    sp.restore_value(None)
    assert sp.value is None


class TestSchemaEntry:
  """schema_entry() metadata."""

  def test_type_name(self) -> None:
    sp = ScalarParameter(value='x')
    entry = sp.schema_entry()
    assert isinstance(entry, ParameterSchemaEntry)
    assert entry.type_name == 'ScalarParameter'

  def test_source_none(self) -> None:
    sp = ScalarParameter()
    entry = sp.schema_entry()
    assert entry.source is None

  def test_pattern_none(self) -> None:
    sp = ScalarParameter()
    entry = sp.schema_entry()
    assert entry.pattern is None

  def test_name_empty(self) -> None:
    sp = ScalarParameter()
    entry = sp.schema_entry()
    assert not entry.name


class TestSerialization:
  """to_dict()/from_dict() checkpoint serialization."""

  def test_to_dict_includes_value(self) -> None:
    sp = ScalarParameter(value='hello')
    d = sp.to_dict()
    assert 'value' in d
    assert d['value'] == 'hello'

  def test_to_dict_includes_requires_grad(self) -> None:
    sp = ScalarParameter(value=1, requires_grad=False)
    d = sp.to_dict()
    assert d['requires_grad'] is False

  def test_from_dict_round_trip_string(self) -> None:
    sp = ScalarParameter(value='hello')
    d = sp.to_dict()
    sp2 = ScalarParameter.from_dict(d)
    assert sp2.value == 'hello'
    assert sp2.requires_grad is True

  def test_from_dict_round_trip_dict(self) -> None:
    sp = ScalarParameter(value={'x': [1, 2]})
    d = sp.to_dict()
    sp2 = ScalarParameter.from_dict(d)
    assert sp2.value == {'x': [1, 2]}

  def test_from_dict_round_trip_none(self) -> None:
    sp = ScalarParameter(value=None)
    d = sp.to_dict()
    sp2 = ScalarParameter.from_dict(d)
    assert sp2.value is None

  def test_from_dict_preserves_id(self) -> None:
    sp = ScalarParameter(value=42)
    d = sp.to_dict()
    sp2 = ScalarParameter.from_dict(d)
    assert sp2.id == sp.id

  def test_from_dict_preserves_requires_grad(self) -> None:
    sp = ScalarParameter(value=42, requires_grad=False)
    d = sp.to_dict()
    sp2 = ScalarParameter.from_dict(d)
    assert sp2.requires_grad is False

  def test_from_dict_missing_value_defaults_none(self) -> None:
    d = {'requires_grad': True}
    sp = ScalarParameter.from_dict(d)
    assert sp.value is None


class TestLoadFromDict:
  """load_from_dict() checkpoint reload into existing instance."""

  def test_load_updates_value(self) -> None:
    sp = ScalarParameter(value='old')
    sp.load_from_dict({'value': 'new', 'requires_grad': True})
    assert sp.value == 'new'

  def test_load_updates_requires_grad(self) -> None:
    sp = ScalarParameter(value='x', requires_grad=True)
    sp.load_from_dict({'value': 'x', 'requires_grad': False})
    assert sp.requires_grad is False

  def test_load_missing_value_no_change(self) -> None:
    sp = ScalarParameter(value='original')
    sp.load_from_dict({'requires_grad': True})
    assert sp.value == 'original'

  def test_load_preserves_identity(self) -> None:
    sp = ScalarParameter(value='x')
    original_id = sp.id
    sp.load_from_dict({'value': 'y'})
    assert sp.id == original_id


class TestRender:
  """render() prompt-facing description."""

  def test_render_string(self) -> None:
    sp = ScalarParameter(value='hello')
    assert sp.render() == "'hello'"

  def test_render_int(self) -> None:
    sp = ScalarParameter(value=42)
    assert sp.render() == '42'

  def test_render_none(self) -> None:
    sp = ScalarParameter()
    assert sp.render() == 'None'

  def test_render_dict(self) -> None:
    sp = ScalarParameter(value={'a': 1})
    r = sp.render()
    assert "'a'" in r
    assert '1' in r


class TestErrorPaths:
  """Error handling: non-JSON-serializable, corrupt restore, banned kwargs."""

  def test_non_serializable_value_raises_store_error(self) -> None:
    sp = ScalarParameter(value=object())
    with pytest.raises(StoreError, match='not JSON-serializable'):
      sp.snapshot()

  def test_non_serializable_error_includes_type(self) -> None:
    sp = ScalarParameter(value=object())
    with pytest.raises(StoreError, match='object'):
      sp.snapshot()

  def test_non_serializable_set_raises(self) -> None:
    sp = ScalarParameter(value={1, 2, 3})
    with pytest.raises(StoreError, match='not JSON-serializable'):
      sp.snapshot()

  def test_corrupt_json_restore_raises_store_error(self) -> None:
    sp = ScalarParameter()
    with pytest.raises(StoreError, match='not valid JSON'):
      sp.restore({'value.json': 'not-valid-json{{'})

  def test_corrupt_restore_includes_raw_content(self) -> None:
    bad_text = '{invalid!!'
    sp = ScalarParameter()
    with pytest.raises(StoreError, match='invalid'):
      sp.restore({'value.json': bad_text})

  def test_banned_kwargs_still_rejected(self) -> None:
    with pytest.raises(TypeError, match='data'):
      ScalarParameter(data='x')

  def test_text_kwarg_rejected(self) -> None:
    with pytest.raises(TypeError, match='text'):
      ScalarParameter(text='x')

  def test_content_kwarg_rejected(self) -> None:
    with pytest.raises(TypeError, match='content'):
      ScalarParameter(content='x')

  def test_prompt_kwarg_rejected(self) -> None:
    with pytest.raises(TypeError, match='prompt'):
      ScalarParameter(prompt='x')

  def test_value_kwarg_is_allowed(self) -> None:
    sp = ScalarParameter(value='allowed')
    assert sp.value == 'allowed'


class TestOptimizerParticipation:
  """ScalarParameter works with Optimizer zero_grad and step loops."""

  def test_zero_grad_clears_grad(self) -> None:
    from autopilot.core.gradient import Gradient

    sp = ScalarParameter(value='x')
    sp.grad = Gradient()
    assert sp.grad is not None

    class TestOpt(Optimizer):
      def step(self) -> None:
        pass

    opt = TestOpt([sp])
    opt.zero_grad()
    assert sp.grad is None

  def test_in_param_groups(self) -> None:
    class TestOpt(Optimizer):
      def step(self) -> None:
        pass

    sp = ScalarParameter(value=10)
    opt = TestOpt([sp])
    assert sp in opt.parameters

  def test_module_parameters_include_scalar(self) -> None:
    sp = ScalarParameter(value='prompt text')

    class TestModule(Module):
      def __init__(self) -> None:
        super().__init__()
        self.config = sp

      def forward(self, *args: Any, **kwargs: Any) -> Any:
        return None

    mod = TestModule()
    params = list(mod.parameters())
    assert sp in params


class TestFileStoreIntegration:
  """FileStore snapshot/checkout round-trip with ScalarParameter."""

  def _make_store_with_scalar(
    self, tmp_path: Path, value: Any = 'hello'
  ) -> tuple[FileStore, ScalarParameter]:
    """Create a FileStore with a ScalarParameter registered."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path = tmp_path / '.autopilot'
    config.store_path.mkdir(parents=True, exist_ok=True)
    param = ScalarParameter(value=value)
    store = FileStore(config)
    store.register_parameters({'scalar': param})
    return store, param

  def test_snapshot_writes_blob(self, tmp_path: Path) -> None:
    store, _param = self._make_store_with_scalar(tmp_path, value='hello')
    store.snapshot('exp-1', epoch=0)
    objects_dir = store.config.store_path / 'objects'
    blobs = list(objects_dir.rglob('*'))
    blob_files = [b for b in blobs if b.is_file()]
    assert len(blob_files) >= 1

  def test_snapshot_checkout_round_trip(self, tmp_path: Path) -> None:
    store, param = self._make_store_with_scalar(tmp_path, value={'key': 42})
    store.snapshot('exp-1', epoch=0)
    param.value = 'changed'
    assert param.value == 'changed'
    store.checkout('exp-1', epoch=0)
    assert param.value == {'key': 42}

  def test_snapshot_none_value(self, tmp_path: Path) -> None:
    store, param = self._make_store_with_scalar(tmp_path, value=None)
    store.snapshot('exp-1', epoch=0)
    param.value = 'something'
    store.checkout('exp-1', epoch=0)
    assert param.value is None

  def test_snapshot_integer_value(self, tmp_path: Path) -> None:
    store, param = self._make_store_with_scalar(tmp_path, value=99)
    store.snapshot('exp-1', epoch=0)
    param.value = 0
    store.checkout('exp-1', epoch=0)
    assert param.value == 99

  def test_snapshot_complex_dict(self, tmp_path: Path) -> None:
    val = {'prompt': 'You are helpful', 'temperature': 0.7, 'tags': ['a', 'b']}
    store, param = self._make_store_with_scalar(tmp_path, value=val)
    store.snapshot('exp-1', epoch=0)
    param.value = None
    store.checkout('exp-1', epoch=0)
    assert param.value == val

  def test_multiple_epochs(self, tmp_path: Path) -> None:
    store, param = self._make_store_with_scalar(tmp_path, value='epoch-0')
    store.snapshot('exp-1', epoch=0)
    param.value = 'epoch-1'
    store.snapshot('exp-1', epoch=1)
    store.checkout('exp-1', epoch=0)
    assert param.value == 'epoch-0'
    store.checkout('exp-1', epoch=1)
    assert param.value == 'epoch-1'

  def test_mixed_with_path_parameter(self, tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'file.txt').write_text('file content')
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path = tmp_path / '.autopilot'
    config.store_path.mkdir(parents=True, exist_ok=True)
    path_param = PathParameter(source=str(src), pattern='*')
    scalar_param = ScalarParameter(value='scalar-val')
    store = FileStore(config)
    store.register_parameters({'files': path_param, 'config': scalar_param})
    store.snapshot('exp-1', epoch=0)
    scalar_param.value = 'changed'
    (src / 'file.txt').write_text('changed')
    store.checkout('exp-1', epoch=0)
    assert scalar_param.value == 'scalar-val'
    assert (src / 'file.txt').read_text() == 'file content'

  def test_schema_embedded_in_manifest(self, tmp_path: Path) -> None:
    store, _param = self._make_store_with_scalar(tmp_path, value='test')
    store.snapshot('exp-1', epoch=0)
    snapshots_dir = store.config.store_path / 'snapshots' / 'exp-1'
    manifest_file = snapshots_dir / 'epoch_0.json'
    data = json.loads(manifest_file.read_text())
    schema = data.get('schema')
    assert schema is not None
    params = schema.get('parameters', [])
    scalar_entries = [p for p in params if p['type_name'] == 'ScalarParameter']
    assert len(scalar_entries) == 1
    assert scalar_entries[0]['name'] == 'scalar'
    assert scalar_entries[0]['source'] is None
    assert scalar_entries[0]['pattern'] is None


class TestNoCouplingInFileStore:
  """FileStore must not import or isinstance-check ScalarParameter."""

  def test_store_py_does_not_mention_scalar_parameter(self) -> None:
    import autopilot.ai.store.file_store as store_mod

    source_path = Path(store_mod.__file__)
    source_text = source_path.read_text(encoding='utf-8')
    assert 'ScalarParameter' not in source_text, (
      'ai/store/file_store.py must not reference ScalarParameter -- '
      'stores observe parameters only through Parameter.snapshot()/restore()'
    )


class TestKwargGuardIntegration:
  """__init_subclass__ kwarg guard works correctly with ScalarParameter."""

  def test_value_allowed_for_scalar_parameter(self) -> None:
    sp = ScalarParameter(value='test')
    assert sp.value == 'test'

  def test_value_still_rejected_for_plain_parameter(self) -> None:
    with pytest.raises(TypeError, match='value'):
      Parameter(value='bad')  # ty: ignore[unknown-argument]

  def test_value_rejected_for_other_subclass(self) -> None:
    class CustomParam(Parameter):
      pass

    with pytest.raises(TypeError, match='value'):
      CustomParam(value='bad')  # ty: ignore[unknown-argument]

  def test_data_rejected_for_scalar_parameter(self) -> None:
    with pytest.raises(TypeError, match='data'):
      ScalarParameter(data='bad')

  def test_all_non_value_banned_kwargs_rejected(self) -> None:
    for kwarg in COMMON_WRONG_KWARGS - {'value'}:
      with pytest.raises(TypeError, match=kwarg):
        ScalarParameter(**{kwarg: 'bad'})


class TestStashIntegration:
  """ScalarParameter works with FileStore.stash()/stash_pop()."""

  def test_stash_and_pop_round_trip(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path = tmp_path / '.autopilot'
    config.store_path.mkdir(parents=True, exist_ok=True)
    param = ScalarParameter(value='before-stash')
    store = FileStore(config)
    store.register_parameters({'scalar': param})
    store.stash()
    param.value = 'after-stash'
    store.stash_pop()
    assert param.value == 'before-stash'
