"""Tests for DictMixin serialization round-trips."""

from autopilot.core.diagnostics import DiagnosisEntry, NodeScore
from autopilot.core.models import (
  CommandRecord,
  DatasetEntry,
  DatasetSnapshot,
  Event,
  HyperparamSet,
  Manifest,
  Promotion,
  Result,
)
from autopilot.core.serialization import DictMixin
from autopilot.core.store import (
  DiffEntry,
  FileEntry,
  SnapshotEntry,
  SnapshotManifest,
  StatusEntry,
)
from dataclasses import dataclass


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


class TestManifestDictMixin:
  def test_round_trip(self):
    m = Manifest(slug='test', title='Test', current_epoch=3)
    d = m.to_dict()
    m2 = Manifest.from_dict(d)
    assert m2.slug == 'test'
    assert m2.title == 'Test'
    assert m2.current_epoch == 3

  def test_is_decided_false(self):
    m = Manifest(slug='test')
    assert not m.is_decided

  def test_is_decided_true(self):
    m = Manifest(slug='test', decision='promote')
    assert m.is_decided

  def test_unknown_keys_filtered(self):
    m = Manifest.from_dict({'slug': 'test', 'unknown': 'extra'})
    assert m.slug == 'test'

  def test_to_json(self):
    m = Manifest(slug='test')
    j = m.to_json()
    assert '"slug": "test"' in j


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
    r = Result(metrics={'accuracy': 0.9}, passed=True, summary='good')
    d = r.to_dict()
    r2 = Result.from_dict(d)
    assert r2.metrics == {'accuracy': 0.9}
    assert r2.passed is True
    assert r2.summary == 'good'


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


class TestHyperparamSetDictMixin:
  def test_round_trip(self):
    h = HyperparamSet(version=2, values={'lr': 0.01}, locked=True)
    d = h.to_dict()
    h2 = HyperparamSet.from_dict(d)
    assert h2.version == 2
    assert h2.values == {'lr': 0.01}
    assert h2.locked is True


class TestPromotionDictMixin:
  def test_round_trip(self):
    p = Promotion(timestamp='now', decision='promote', reason='good')
    d = p.to_dict()
    p2 = Promotion.from_dict(d)
    assert p2.decision == 'promote'


class TestFileEntryDictMixin:
  def test_round_trip(self):
    fe = FileEntry(hash='abc', size=100, mtime=1.0)
    d = fe.to_dict()
    assert d == {'hash': 'abc', 'size': 100, 'mtime': 1.0}
    fe2 = FileEntry.from_dict(d)
    assert fe2.hash == 'abc'


class TestSnapshotManifestDictMixin:
  def test_round_trip_with_entries(self):
    sm = SnapshotManifest(
      epoch=1,
      timestamp='now',
      entries={'file.py': FileEntry(hash='abc', size=100, mtime=1.0)},
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
