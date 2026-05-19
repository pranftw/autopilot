"""Tests for DictMixin serialization round-trips."""

from autopilot.core.diagnostics import DiagnosisEntry, NodeScore
from autopilot.core.enums import Status
from autopilot.core.models import (
  CommandRecord,
  DatasetEntry,
  DatasetSnapshot,
  Event,
  Result,
)
from autopilot.core.serialization import DictMixin, serialize
from autopilot.core.snapshot import FileEntry, SnapshotManifest
from autopilot.core.store.types import (
  ConflictEntry,
  DiffEntry,
  DiffResult,
  MergeAnalysisResult,
  MergeClassification,
  MergeIndex,
  MergeStrategy,
  SnapshotEntry,
  StatusEntry,
  StatusResult,
)
from autopilot.core.types import GateResult
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TestDictMixinBasic:
  def test_simple_round_trip(self):
    @dataclass
    class Simple(DictMixin):
      name: str
      value: int = 0

    obj = Simple(name='test', value=42)
    d = obj.to_dict()
    assert d == {'name': 'test', 'value': 42}
    obj2 = Simple.from_dict(d)
    assert obj2.name == 'test'
    assert obj2.value == 42

  def test_unknown_keys_ignored(self):
    @dataclass
    class Simple(DictMixin):
      name: str

    obj = Simple.from_dict({'name': 'test', 'extra': 'ignored'})
    assert obj.name == 'test'

  def test_nested_mixin(self):
    @dataclass
    class Inner(DictMixin):
      x: int

    @dataclass
    class Outer(DictMixin):
      inner: Inner
      label: str

    outer = Outer(inner=Inner(x=5), label='wrap')
    d = outer.to_dict()
    assert d == {'inner': {'x': 5}, 'label': 'wrap'}

  def test_list_of_mixin(self):
    @dataclass
    class Item(DictMixin):
      n: int

    @dataclass
    class Container(DictMixin):
      items: list

    container = Container(items=[Item(n=1), Item(n=2)])
    d = container.to_dict()
    assert d['items'] == [{'n': 1}, {'n': 2}]

  def test_dict_of_mixin_values(self):
    @dataclass
    class Score(DictMixin):
      val: float

    @dataclass
    class ScoreMap(DictMixin):
      scores: dict

    sm = ScoreMap(scores={'a': Score(val=0.9), 'b': Score(val=0.5)})
    d = sm.to_dict()
    assert d['scores'] == {'a': {'val': 0.9}, 'b': {'val': 0.5}}

  def test_none_values_preserved(self):
    @dataclass
    class Nullable(DictMixin):
      x: int | None = None
      y: str | None = None

    obj = Nullable()
    d = obj.to_dict()
    assert d == {'x': None, 'y': None}
    obj2 = Nullable.from_dict(d)
    assert obj2.x is None
    assert obj2.y is None


class TestEventDictMixin:
  def test_round_trip(self):
    e = Event(timestamp='2024-01-01T00:00:00', event_type='created')
    d = e.to_dict()
    e2 = Event.from_dict(d)
    assert e2.timestamp == '2024-01-01T00:00:00'
    assert e2.event_type == 'created'


class TestCommandRecordDictMixin:
  def test_round_trip(self):
    cr = CommandRecord(timestamp='now', command='train', args=['--epoch', '1'])
    d = cr.to_dict()
    cr2 = CommandRecord.from_dict(d)
    assert cr2.command == 'train'
    assert cr2.args == ['--epoch', '1']


class TestResultDictMixin:
  def test_round_trip(self):
    from autopilot.core.constraint import ConstraintResult

    gate = ConstraintResult(
      name='MinGate', passed=True, metric='accuracy', value=0.9, threshold='>= 0.8'
    )
    r = Result(metrics={'accuracy': 0.9}, gates=[gate], summary='good')
    d = r.to_dict()
    r2 = Result.from_dict(d)
    assert r2.metrics == {'accuracy': 0.9}
    assert r2.passed is True
    assert r2.summary == 'good'
    assert len(r2.gates) == 1
    assert r2.gates[0].name == 'MinGate'


class TestDatasetEntryDictMixin:
  def test_round_trip(self):
    de = DatasetEntry(name='train', split='train', path='/data/train.jsonl', rows=100)
    d = de.to_dict()
    de2 = DatasetEntry.from_dict(d)
    assert de2.name == 'train'
    assert de2.rows == 100


class TestDatasetSnapshotDictMixin:
  def test_round_trip_with_entries(self):
    entry = DatasetEntry(name='test', split='test', path='/data/test.jsonl')
    snap = DatasetSnapshot(created_at='2024-01-01', entries=[entry])
    d = snap.to_dict()
    snap2 = DatasetSnapshot.from_dict(d)
    assert len(snap2.entries) == 1
    assert isinstance(snap2.entries[0], DatasetEntry)
    assert snap2.entries[0].name == 'test'


class TestEnumSerialization:
  def test_statusserializes_to_value(self):
    assert serialize(Status.pending) == 'pending'
    assert serialize(Status.running) == 'running'
    assert serialize(Status.completed) == 'completed'

  def test_gate_resultserializes_to_value(self):
    assert serialize(GateResult.PASSED) == 'pass'
    assert serialize(GateResult.FAIL) == 'fail'

  def test_enum_in_list(self):
    result = serialize([Status.pending, Status.failed])
    assert result == ['pending', 'failed']

  def test_enum_in_dict(self):
    result = serialize({'status': Status.running})
    assert result == {'status': 'running'}

  def test_enum_in_dataclass(self):
    @dataclass
    class WithStatus(DictMixin):
      name: str
      status: Status

    obj = WithStatus(name='test', status=Status.completed)
    d = obj.to_dict()
    assert d == {'name': 'test', 'status': 'completed'}


class TestFileEntryDictMixin:
  def test_round_trip(self):
    fe = FileEntry(digest='abc', size=100, mtime=1.0)
    d = fe.to_dict()
    assert d == {'digest': 'abc', 'size': 100, 'mtime': 1.0, 'original_path': None}
    fe2 = FileEntry.from_dict(d)
    assert fe2.digest == 'abc'


class TestSnapshotManifestDictMixin:
  def test_round_trip_with_entries(self):
    sm = SnapshotManifest(
      epoch=1,
      timestamp='now',
      entries={'file.py': FileEntry(digest='abc', size=100, mtime=1.0)},
    )
    d = sm.to_dict()
    sm2 = SnapshotManifest.from_dict(d)
    assert sm2.epoch == 1
    assert isinstance(sm2.entries['file.py'], FileEntry)


class TestDiffEntryDictMixin:
  def test_round_trip(self):
    de = DiffEntry(path='file.py', status='modified', old_hash='a', new_hash='b')
    d = de.to_dict()
    de2 = DiffEntry.from_dict(d)
    assert de2.path == 'file.py'
    assert de2.status == 'modified'


class TestStatusEntryDictMixin:
  def test_round_trip(self):
    se = StatusEntry(path='file.py', status='added')
    d = se.to_dict()
    se2 = StatusEntry.from_dict(d)
    assert se2.path == 'file.py'


class TestSnapshotEntryDictMixin:
  def test_round_trip(self):
    se = SnapshotEntry(epoch=1, timestamp='now', file_count=5)
    d = se.to_dict()
    se2 = SnapshotEntry.from_dict(d)
    assert se2.epoch == 1
    assert se2.file_count == 5


class TestDiagnosisEntryDictMixin:
  def test_round_trip(self):
    de = DiagnosisEntry(category='syntax', count=3, sample_errors=['e1', 'e2'])
    d = de.to_dict()
    de2 = DiagnosisEntry.from_dict(d)
    assert de2.category == 'syntax'
    assert de2.count == 3
    assert de2.sample_errors == ['e1', 'e2']


class TestNodeScoreDictMixin:
  def test_round_trip(self):
    ns = NodeScore(total=10, failed=2, error_rate=0.2)
    d = ns.to_dict()
    ns2 = NodeScore.from_dict(d)
    assert ns2.total == 10
    assert ns2.failed == 2
    assert ns2.error_rate == 0.2


class TestFromDictAsymmetry:
  """Confirm that base from_dict does NOT reverse serialize for nested types."""

  def test_nested_dictmixin_becomes_plain_dict(self):
    @dataclass
    class Inner(DictMixin):
      x: int

    @dataclass
    class Outer(DictMixin):
      inner: Any
      label: str

    outer = Outer(inner=Inner(x=5), label='wrap')
    d = outer.to_dict()
    restored = Outer.from_dict(d)
    assert isinstance(restored.inner, dict)
    assert not isinstance(restored.inner, Inner)
    assert restored.inner == {'x': 5}

  def test_enum_field_becomes_string(self):
    @dataclass
    class WithStatus(DictMixin):
      name: str
      status: Status

    obj = WithStatus(name='test', status=Status.completed)
    d = obj.to_dict()
    restored = WithStatus.from_dict(d)
    assert isinstance(restored.status, str)
    assert restored.status == 'completed'
    assert not isinstance(restored.status, Status)

  def test_override_from_dict_restores_round_trip(self):

    class Color(Enum):
      red = 'red'
      blue = 'blue'

    @dataclass
    class Inner(DictMixin):
      value: int

    @dataclass
    class RoundTrippable(DictMixin):
      inner: Inner
      color: Color
      label: str

      @classmethod
      def from_dict(cls, data: dict[str, Any]) -> 'RoundTrippable':
        return cls(
          inner=Inner.from_dict(data['inner']),
          color=Color(data['color']),
          label=data['label'],
        )

    original = RoundTrippable(inner=Inner(value=42), color=Color.red, label='test')
    d = original.to_dict()
    restored = RoundTrippable.from_dict(d)
    assert isinstance(restored.inner, Inner)
    assert restored.inner.value == 42
    assert isinstance(restored.color, Color)
    assert restored.color is Color.red
    assert restored.label == 'test'


class TestNullSafety:
  """Test that from_dict handles JSON null (None) for collection fields."""

  def test_null_dict_field_does_not_crash(self):
    @dataclass
    class WithDict(DictMixin):
      data: dict[str, Any] = field(default_factory=dict)

    result = WithDict.from_dict({'data': None})
    assert result.data is None

  def test_null_list_field_does_not_crash(self):
    @dataclass
    class WithList(DictMixin):
      items: list[str] = field(default_factory=list)

    result = WithList.from_dict({'items': None})
    assert result.items is None

  def test_missing_field_uses_default(self):
    @dataclass
    class WithDefault(DictMixin):
      name: str = 'default_name'
      count: int = 0
      tags: list[str] = field(default_factory=list)

    result = WithDefault.from_dict({})
    assert result.name == 'default_name'
    assert result.count == 0
    assert result.tags == []

  def test_extra_keys_filtered_out(self):
    @dataclass
    class Known(DictMixin):
      known: int = 0

    result = Known.from_dict({'known': 1, 'unknown': 2})
    assert result.known == 1
    assert not hasattr(result, 'unknown')

  def test_snapshot_manifest_null_entries(self):
    sm = SnapshotManifest.from_dict(
      {
        'epoch': 1,
        'timestamp': 'now',
        'entries': None,
      }
    )
    assert sm.entries == {}

  def test_diff_result_null_entries(self):
    dr = DiffResult.from_dict({'entries': None})
    assert dr.entries == []

  def test_status_result_null_entries(self):
    sr = StatusResult.from_dict({'entries': None})
    assert sr.entries == []

  def test_merge_index_null_conflicts(self):
    idx = MergeIndex.from_dict(
      {
        'conflicts': None,
        'resolved': None,
      }
    )
    assert idx.conflicts == {}
    assert idx.resolved == {}


class TestCoerceNoneToEmpty:
  """Test the _coerce_none_to_empty static helper."""

  def test_replaces_none_with_empty_dict(self):
    data = {'entries': None, 'name': 'test'}
    result = DictMixin._coerce_none_to_empty(data, {'entries': dict})
    assert result == {'entries': {}, 'name': 'test'}

  def test_replaces_none_with_empty_list(self):
    data = {'items': None}
    result = DictMixin._coerce_none_to_empty(data, {'items': list})
    assert result == {'items': []}

  def test_preserves_existing_values(self):
    data = {'entries': {'a': 1}, 'items': [1, 2]}
    result = DictMixin._coerce_none_to_empty(data, {'entries': dict, 'items': list})
    assert result == {'entries': {'a': 1}, 'items': [1, 2]}

  def test_missing_keys_not_added(self):
    data = {'name': 'test'}
    result = DictMixin._coerce_none_to_empty(data, {'entries': dict})
    assert 'entries' not in result

  def test_does_not_mutate_original(self):
    data = {'entries': None}
    DictMixin._coerce_none_to_empty(data, {'entries': dict})
    assert data['entries'] is None


class TestStoreDataclassRoundTrips:
  """Round-trip tests for all store dataclasses."""

  def test_snapshot_manifest_round_trip(self):
    sm = SnapshotManifest(
      epoch=3,
      timestamp='2024-01-01T00:00:00',
      entries={
        'param_0/config.json': FileEntry(digest='abc123', size=256, mtime=1.0),
        'param_1/prompt.txt': FileEntry(digest='def456', size=512, mtime=2.0),
      },
    )
    d = sm.to_dict()
    sm2 = SnapshotManifest.from_dict(d)
    assert sm2.epoch == 3
    assert sm2.timestamp == '2024-01-01T00:00:00'
    assert len(sm2.entries) == 2
    assert isinstance(sm2.entries['param_0/config.json'], FileEntry)
    assert sm2.entries['param_0/config.json'].digest == 'abc123'

  def test_diff_result_round_trip(self):
    dr = DiffResult(
      entries=[
        DiffEntry(path='file.py', status='added', new_hash='abc'),
        DiffEntry(path='old.py', status='deleted', old_hash='def'),
        DiffEntry(path='mod.py', status='modified', old_hash='a', new_hash='b'),
      ]
    )
    d = dr.to_dict()
    dr2 = DiffResult.from_dict(d)
    assert len(dr2.entries) == 3
    assert isinstance(dr2.entries[0], DiffEntry)
    assert dr2.added()[0].path == 'file.py'
    assert dr2.deleted()[0].path == 'old.py'
    assert dr2.modified()[0].path == 'mod.py'

  def test_merge_index_round_trip(self):
    idx = MergeIndex(
      conflicts={
        'file.txt': ConflictEntry(
          key='file.txt',
          ours=FileEntry(digest='h', size=1, mtime=0.0),
        ),
      },
      resolved={'ok.py': FileEntry(digest='g', size=2, mtime=0.0)},
      experiment_id='exp-a',
      source_experiment_id='exp-b',
      strategy=MergeStrategy.normal,
      preview_token='token123',
    )
    data = idx.to_dict()
    idx2 = MergeIndex.from_dict(data)
    assert not idx2.is_resolved()
    assert 'file.txt' in idx2.conflicts
    assert idx2.resolved['ok.py'].digest == 'g'
    assert idx2.strategy == MergeStrategy.normal
    assert idx2.preview_token == 'token123'

  def test_status_result_round_trip(self):
    sr = StatusResult(
      entries=[
        StatusEntry(path='a.py', status='modified'),
        StatusEntry(path='b.py', status='added'),
        StatusEntry(path='c.py', status='unchanged'),
      ]
    )
    d = sr.to_dict()
    sr2 = StatusResult.from_dict(d)
    assert len(sr2.entries) == 3
    assert isinstance(sr2.entries[0], StatusEntry)
    assert sr2.modified()[0].path == 'a.py'
    assert sr2.added()[0].path == 'b.py'
    assert sr2.unchanged()[0].path == 'c.py'

  def test_snapshot_manifest_empty_entries(self):
    sm = SnapshotManifest(epoch=0, timestamp='t')
    d = sm.to_dict()
    sm2 = SnapshotManifest.from_dict(d)
    assert sm2.entries == {}

  def test_diff_result_empty_entries(self):
    dr = DiffResult()
    d = dr.to_dict()
    dr2 = DiffResult.from_dict(d)
    assert dr2.entries == []

  def test_merge_analysis_result_round_trip(self):
    result = MergeAnalysisResult(
      can_fast_forward=False,
      has_conflicts=True,
      conflict_count=2,
      ancestor_epoch=1,
      classification=MergeClassification.conflict,
    )
    data = result.to_dict()
    result2 = MergeAnalysisResult.from_dict(data)
    assert result2.has_conflicts is True
    assert result2.conflict_count == 2
    assert result2.classification == 'conflict'
