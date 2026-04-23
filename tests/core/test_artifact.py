"""Tests for experiment artifact system, artifact I/O, and ArtifactOwner mixin."""

from autopilot.core.artifacts.artifact import Artifact, JSONArtifact, JSONLArtifact, TextArtifact
from autopilot.core.artifacts.experiment import (
  BaselineArtifact,
  CostArtifact,
  EventsArtifact,
  ReportArtifact,
  RunStateArtifact,
)
from autopilot.core.artifacts.owner import ArtifactOwner
from autopilot.core.callbacks.callback import Callback
from autopilot.core.models import Event
from pathlib import Path
import pytest


class TestArtifactBase:
  def test_filename_property(self) -> None:
    a = Artifact('test.json')
    assert a.filename == 'test.json'

  def test_scope_property(self) -> None:
    a = Artifact('test.json')
    assert a.scope == 'experiment'

  def test_scope_epoch(self) -> None:
    a = Artifact('test.json', scope='epoch')
    assert a.scope == 'epoch'

  def test_resolve_path_experiment_scope(self, tmp_path: Path) -> None:
    a = Artifact('test.json')
    assert a.resolve_path(tmp_path) == tmp_path / 'test.json'

  def test_resolve_path_epoch_scope(self, tmp_path: Path) -> None:
    a = Artifact('test.json', scope='epoch')
    assert a.resolve_path(tmp_path, epoch=3) == tmp_path / 'epoch_3' / 'test.json'

  def test_resolve_path_epoch_required(self, tmp_path: Path) -> None:
    a = Artifact('test.json', scope='epoch')
    with pytest.raises(ValueError, match='epoch required'):
      a.resolve_path(tmp_path)

  def test_exists_true(self, tmp_path: Path) -> None:
    (tmp_path / 'test.json').write_text('{}')
    a = Artifact('test.json')
    assert a.exists(tmp_path)

  def test_exists_false(self, tmp_path: Path) -> None:
    a = Artifact('test.json')
    assert not a.exists(tmp_path)

  def test_clear_removes_file(self, tmp_path: Path) -> None:
    (tmp_path / 'test.json').write_text('{}')
    a = Artifact('test.json')
    a.clear(tmp_path)
    assert not (tmp_path / 'test.json').exists()

  def test_clear_nonexistent_noop(self, tmp_path: Path) -> None:
    a = Artifact('test.json')
    a.clear(tmp_path)

  def test_schema_default_none(self) -> None:
    assert Artifact('x').schema() is None

  def test_validate_default_noop(self) -> None:
    Artifact('x').validate({'anything': True})

  def test_serialize_passthrough(self) -> None:
    data = {'key': 'value'}
    assert Artifact('x').serialize(data) is data

  def test_deserialize_passthrough(self) -> None:
    raw = {'key': 'value'}
    assert Artifact('x').deserialize(raw) is raw

  def test_write_raises(self, tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
      Artifact('x').write({}, tmp_path)

  def test_read_raises(self, tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
      Artifact('x').read(tmp_path)

  def test_update_raises(self, tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
      Artifact('x').update({}, tmp_path)

  def test_append_raises(self, tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
      Artifact('x').append({}, tmp_path)

  def test_repr(self) -> None:
    a = Artifact('test.json', scope='epoch')
    r = repr(a)
    assert 'test.json' in r
    assert 'epoch' in r


class TestJSONArtifact:
  def test_write_creates_json(self, tmp_path: Path) -> None:
    a = JSONArtifact('data.json')
    a.write({'key': 'value'}, tmp_path)
    assert (tmp_path / 'data.json').exists()

  def test_read_raw(self, tmp_path: Path) -> None:
    a = JSONArtifact('data.json')
    a.write({'key': 'value'}, tmp_path)
    assert a.read_raw(tmp_path) == {'key': 'value'}

  def test_read_deserializes(self, tmp_path: Path) -> None:
    class TypedArtifact(JSONArtifact):
      def deserialize(self, raw):
        return {'typed': True, **raw}

    a = TypedArtifact('data.json')
    a.write({'key': 'value'}, tmp_path)
    result = a.read(tmp_path)
    assert result['typed'] is True

  def test_read_nonexistent_returns_none(self, tmp_path: Path) -> None:
    a = JSONArtifact('nope.json')
    assert a.read(tmp_path) is None

  def test_update_shallow_merge(self, tmp_path: Path) -> None:
    a = JSONArtifact('data.json')
    a.write({'a': 1, 'b': 2}, tmp_path)
    a.update({'b': 3, 'c': 4}, tmp_path)
    result = a.read_raw(tmp_path)
    assert result == {'a': 1, 'b': 3, 'c': 4}

  def test_update_custom_merge(self, tmp_path: Path) -> None:
    class DeepMerge(JSONArtifact):
      def merge(self, existing, new):
        return {**existing, **new, 'merged': True}

    a = DeepMerge('data.json')
    a.write({'a': 1}, tmp_path)
    a.update({'b': 2}, tmp_path)
    assert a.read_raw(tmp_path)['merged'] is True

  def test_mkdir_parents(self, tmp_path: Path) -> None:
    a = JSONArtifact('data.json', scope='epoch')
    a.write({'x': 1}, tmp_path, epoch=5)
    assert (tmp_path / 'epoch_5' / 'data.json').exists()

  def test_epoch_scoped_json(self, tmp_path: Path) -> None:
    a = JSONArtifact('metrics.json', scope='epoch')
    a.write({'acc': 0.9}, tmp_path, epoch=1)
    a.write({'acc': 0.95}, tmp_path, epoch=2)
    assert a.read(tmp_path, epoch=1) == {'acc': 0.9}
    assert a.read(tmp_path, epoch=2) == {'acc': 0.95}


class TestJSONLArtifact:
  def test_append_single_record(self, tmp_path: Path) -> None:
    a = JSONLArtifact('log.jsonl')
    a.append({'event': 'start'}, tmp_path)
    records = a.read_raw(tmp_path)
    assert len(records) == 1

  def test_append_multiple_records(self, tmp_path: Path) -> None:
    a = JSONLArtifact('log.jsonl')
    a.append({'event': 'start'}, tmp_path)
    a.append({'event': 'end'}, tmp_path)
    records = a.read_raw(tmp_path)
    assert len(records) == 2

  def test_write_full_replace(self, tmp_path: Path) -> None:
    a = JSONLArtifact('log.jsonl')
    a.append({'old': True}, tmp_path)
    a.write([{'new': True}], tmp_path)
    records = a.read_raw(tmp_path)
    assert len(records) == 1
    assert records[0]['new'] is True

  def test_read_raw_returns_list(self, tmp_path: Path) -> None:
    a = JSONLArtifact('log.jsonl')
    a.append({'a': 1}, tmp_path)
    result = a.read_raw(tmp_path)
    assert isinstance(result, list)

  def test_read_deserializes_each(self, tmp_path: Path) -> None:
    class TypedJSONL(JSONLArtifact):
      def deserialize(self, raw):
        return {'typed': True, **raw}

    a = TypedJSONL('log.jsonl')
    a.append({'a': 1}, tmp_path)
    result = a.read(tmp_path)
    assert result[0]['typed'] is True

  def test_read_empty_returns_empty(self, tmp_path: Path) -> None:
    a = JSONLArtifact('log.jsonl')
    assert a.read_raw(tmp_path) == []

  def test_read_nonexistent_returns_empty(self, tmp_path: Path) -> None:
    a = JSONLArtifact('nope.jsonl')
    assert a.read_raw(tmp_path) == []


class TestTextArtifact:
  def test_write_creates_text(self, tmp_path: Path) -> None:
    a = TextArtifact('notes.txt')
    a.write('hello', tmp_path)
    assert (tmp_path / 'notes.txt').read_text() == 'hello'

  def test_append_adds_to_end(self, tmp_path: Path) -> None:
    a = TextArtifact('notes.txt')
    a.write('hello', tmp_path)
    a.append(' world', tmp_path)
    assert a.read_raw(tmp_path) == 'hello world'

  def test_read_raw_returns_string(self, tmp_path: Path) -> None:
    a = TextArtifact('notes.txt')
    a.write('content', tmp_path)
    assert isinstance(a.read_raw(tmp_path), str)

  def test_read_nonexistent_returns_none(self, tmp_path: Path) -> None:
    a = TextArtifact('nope.txt')
    assert a.read_raw(tmp_path) is None

  def test_serialize_called(self, tmp_path: Path) -> None:
    class UpperText(TextArtifact):
      def serialize(self, data):
        return data.upper()

    a = UpperText('notes.txt')
    a.write('hello', tmp_path)
    assert a.read_raw(tmp_path) == 'HELLO'


class TestEventsArtifact:
  def test_schema_returns_event_structure(self) -> None:
    a = EventsArtifact()
    s = a.schema()
    assert s['record_type'] == 'Event'

  def test_validate_event_object(self) -> None:
    a = EventsArtifact()
    a.validate(Event(timestamp='2024-01-01', event_type='test'))

  def test_validate_event_dict(self) -> None:
    a = EventsArtifact()
    a.validate({'timestamp': '2024-01-01', 'event_type': 'test'})

  def test_validate_invalid_raises(self) -> None:
    a = EventsArtifact()
    with pytest.raises(ValueError):
      a.validate({'only_timestamp': '2024-01-01'})

  def test_serialize_event_to_dict(self) -> None:
    a = EventsArtifact()
    e = Event(timestamp='2024-01-01', event_type='test', message='hi')
    d = a.serialize(e)
    assert d['event_type'] == 'test'

  def test_deserialize_dict_to_event(self) -> None:
    a = EventsArtifact()
    d = {'timestamp': '2024-01-01', 'event_type': 'test'}
    e = a.deserialize(d)
    assert isinstance(e, Event)

  def test_append_and_read_round_trip(self, tmp_path: Path) -> None:
    a = EventsArtifact()
    e = Event(timestamp='2024-01-01', event_type='created', message='new')
    a.append(e, tmp_path)
    result = a.read(tmp_path)
    assert len(result) == 1
    assert isinstance(result[0], Event)
    assert result[0].event_type == 'created'


class TestBaselineArtifact:
  def test_validate_requires_epoch_and_metrics(self) -> None:
    a = BaselineArtifact()
    with pytest.raises(ValueError):
      a.validate({'only_epoch': 1})

  def test_merge_default_newest_wins(self) -> None:
    a = BaselineArtifact()
    old = {'epoch': 1, 'metrics': {'acc': 0.8}}
    new = {'epoch': 2, 'metrics': {'acc': 0.9}}
    assert a.merge(old, new) == new

  def test_write_and_read_round_trip(self, tmp_path: Path) -> None:
    a = BaselineArtifact()
    a.write({'epoch': 1, 'metrics': {'acc': 0.8}}, tmp_path)
    result = a.read(tmp_path)
    assert result['epoch'] == 1


class TestRunStateArtifact:
  def test_validate_status_enum(self) -> None:
    a = RunStateArtifact()
    a.validate({'status': 'running'})
    with pytest.raises(ValueError):
      a.validate({'status': 'invalid'})

  def test_merge_incremental(self) -> None:
    a = RunStateArtifact()
    existing = {'epoch': 1, 'status': 'running'}
    new = {'status': 'completed', 'stop_reason': 'plateau'}
    merged = a.merge(existing, new)
    assert merged['epoch'] == 1
    assert merged['status'] == 'completed'
    assert merged['stop_reason'] == 'plateau'

  def test_update_round_trip(self, tmp_path: Path) -> None:
    a = RunStateArtifact()
    a.write({'epoch': 1, 'status': 'running'}, tmp_path)
    a.update({'status': 'completed'}, tmp_path)
    result = a.read(tmp_path)
    assert result['epoch'] == 1
    assert result['status'] == 'completed'


class TestCostArtifact:
  def test_write_and_read(self, tmp_path: Path) -> None:
    a = CostArtifact()
    a.write({'epoch': 0, 'wall_clock_s': 5.0}, tmp_path)
    result = a.read(tmp_path)
    assert result['wall_clock_s'] == 5.0


class TestReportArtifact:
  def test_serialize_string_passthrough(self) -> None:
    a = ReportArtifact()
    assert a.serialize('hello') == 'hello'

  def test_serialize_dict_to_markdown(self) -> None:
    a = ReportArtifact()
    result = a.serialize({'Summary': 'Good results'})
    assert '# Experiment Report' in result
    assert '## Summary' in result

  def test_update_appends_section(self, tmp_path: Path) -> None:
    a = ReportArtifact()
    a.write('# Start\n', tmp_path)
    a.update('## New Section\n', tmp_path)
    content = a.read_raw(tmp_path)
    assert '# Start' in content
    assert '## New Section' in content


# artifact i/o round-trip tests

_json = JSONArtifact('metrics.json', scope='epoch')
_jsonl = JSONLArtifact('data.jsonl', scope='epoch')
_exp_json = JSONArtifact('summary.json', scope='experiment')
_exp_jsonl = JSONLArtifact('log.jsonl', scope='experiment')


class TestEpochArtifacts:
  def test_write_and_read_round_trip(self, tmp_path):
    _json.write({'accuracy': 0.8}, tmp_path, epoch=1)
    result = _json.read_raw(tmp_path, epoch=1)
    assert result == {'accuracy': 0.8}

  def test_append_and_read_lines(self, tmp_path):
    _jsonl.append({'item': 'a'}, tmp_path, epoch=1)
    _jsonl.append({'item': 'b'}, tmp_path, epoch=1)
    _jsonl.append({'item': 'c'}, tmp_path, epoch=1)
    lines = _jsonl.read_raw(tmp_path, epoch=1)
    assert len(lines) == 3

  def test_epoch_dir_auto_created(self, tmp_path):
    _json.write({'ok': True}, tmp_path, epoch=5)
    assert (tmp_path / 'epoch_5' / 'metrics.json').exists()

  def test_read_nonexistent_returns_none(self, tmp_path):
    assert _json.read_raw(tmp_path, epoch=99) is None

  def test_read_lines_nonexistent_returns_empty(self, tmp_path):
    assert _jsonl.read_raw(tmp_path, epoch=99) == []


class TestExperimentLevelArtifacts:
  def test_write_and_read(self, tmp_path):
    _exp_json.write({'total': 5}, tmp_path)
    result = _exp_json.read_raw(tmp_path)
    assert result == {'total': 5}

  def test_append_and_read_lines(self, tmp_path):
    _exp_jsonl.append({'epoch': 1}, tmp_path)
    _exp_jsonl.append({'epoch': 2}, tmp_path)
    lines = _exp_jsonl.read_raw(tmp_path)
    assert len(lines) == 2

  def test_read_nonexistent(self, tmp_path):
    assert _exp_json.read_raw(tmp_path) is None

  def test_read_lines_nonexistent(self, tmp_path):
    assert _exp_jsonl.read_raw(tmp_path) == []

  def test_empty_payload(self, tmp_path):
    _exp_json.write({}, tmp_path)
    result = _exp_json.read_raw(tmp_path)
    assert result == {}


# artifactowner tests


class SimpleOwner(ArtifactOwner):
  def __init__(self):
    self.__init_artifacts__()


class TestArtifactOwnerAutoRegistration:
  def test_assigns_artifact_registers(self):
    owner = SimpleOwner()
    art = JSONArtifact('test.json')
    owner.data = art
    assert owner.artifacts['data'] is art

  def test_non_artifact_not_registered(self):
    owner = SimpleOwner()
    owner.foo = 'bar'
    assert 'foo' not in owner.artifacts

  def test_multiple_artifacts(self):
    owner = SimpleOwner()
    owner.a = JSONArtifact('a.json')
    owner.b = JSONLArtifact('b.jsonl')
    assert len(owner.artifacts) == 2
    assert 'a' in owner.artifacts
    assert 'b' in owner.artifacts

  def test_init_artifacts_idempotent(self):
    owner = SimpleOwner()
    owner.data = JSONArtifact('test.json')
    owner.__init_artifacts__()
    assert owner.artifacts == {}

  def test_artifacts_property_returns_copy(self):
    owner = SimpleOwner()
    owner.data = JSONArtifact('test.json')
    arts = owner.artifacts
    arts['extra'] = JSONArtifact('extra.json')
    assert 'extra' not in owner.artifacts

  def test_replace_artifact(self):
    owner = SimpleOwner()
    owner.data = JSONArtifact('old.json')
    owner.data = JSONArtifact('new.json')
    assert owner.artifacts['data'].filename == 'new.json'

  def test_assign_none_to_artifact_name(self):
    owner = SimpleOwner()
    owner.data = JSONArtifact('test.json')
    owner.data = None
    assert owner.artifacts['data'].filename == 'test.json'
    assert owner.data is None


class TestArtifactOwnerMRO:
  def test_callback_and_artifact_owner(self):
    class CbOwner(ArtifactOwner, Callback):
      def __init__(self):
        self.__init_artifacts__()

    obj = CbOwner()
    obj.my_artifact = JSONArtifact('test.json')
    assert 'my_artifact' in obj.artifacts
    assert obj.state_dict() == {}
    obj.on_fit_start(trainer=None)

  def test_multiple_callbacks_with_artifact_owner(self):
    class Multi(ArtifactOwner, Callback):
      def __init__(self):
        self.__init_artifacts__()
        self.art1 = JSONArtifact('a.json')
        self.art2 = JSONLArtifact('b.jsonl')

    obj = Multi()
    assert len(obj.artifacts) == 2
