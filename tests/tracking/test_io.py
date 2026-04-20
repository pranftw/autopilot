"""Tests for tracking/io.py shared I/O primitives."""

from autopilot.core.errors import TrackingError
from autopilot.tracking.io import append_jsonl, atomic_write_json, read_json, read_jsonl
import pytest


class TestAtomicWriteJson:
  def test_round_trip(self, tmp_path):
    path = tmp_path / 'test.json'
    payload = {'key': 'value', 'number': 42}
    atomic_write_json(path, payload)
    result = read_json(path)
    assert result == payload

  def test_creates_parents(self, tmp_path):
    path = tmp_path / 'nested' / 'deep' / 'test.json'
    atomic_write_json(path, {'ok': True})
    assert path.exists()
    assert read_json(path) == {'ok': True}

  def test_no_partial_on_serialization_failure(self, tmp_path):
    path = tmp_path / 'test.json'
    with pytest.raises(TrackingError):
      atomic_write_json(path, {'bad': object()})
    assert not path.exists()


class TestAppendJsonl:
  def test_single_record(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    append_jsonl(path, {'a': 1})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1

  def test_multiple_records(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    for i in range(5):
      append_jsonl(path, {'i': i})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 5


class TestReadJsonl:
  def test_missing_file(self, tmp_path):
    path = tmp_path / 'missing.jsonl'
    assert read_jsonl(path) == []

  def test_empty_file(self, tmp_path):
    path = tmp_path / 'empty.jsonl'
    path.write_text('')
    assert read_jsonl(path) == []

  def test_unicode(self, tmp_path):
    path = tmp_path / 'unicode.jsonl'
    append_jsonl(path, {'text': 'hello'})
    records = read_jsonl(path)
    assert records[0]['text'] == 'hello'

  def test_corrupt_strict_raises(self, tmp_path):
    path = tmp_path / 'bad.jsonl'
    path.write_text('{"ok": 1}\nnot json\n{"ok": 2}\n')
    with pytest.raises(TrackingError):
      read_jsonl(path, strict=True)

  def test_corrupt_tolerant_skips(self, tmp_path):
    path = tmp_path / 'bad.jsonl'
    path.write_text('{"ok": 1}\nnot json\n{"ok": 2}\n')
    records = read_jsonl(path, strict=False)
    assert len(records) == 2
    assert records[0] == {'ok': 1}
    assert records[1] == {'ok': 2}

  def test_partial_write_no_trailing_newline(self, tmp_path):
    path = tmp_path / 'partial.jsonl'
    path.write_text('{"a": 1}\n{"b": 2}')
    records = read_jsonl(path)
    assert len(records) == 2


class TestReadJson:
  def test_missing_returns_none(self, tmp_path):
    path = tmp_path / 'missing.json'
    assert read_json(path) is None

  def test_invalid_json_raises(self, tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('not valid json {{{')
    with pytest.raises(TrackingError):
      read_json(path)

  def test_valid_json_reads(self, tmp_path):
    path = tmp_path / 'good.json'
    path.write_text('{"hello": "world"}')
    assert read_json(path) == {'hello': 'world'}


class TestReadJsonlStrictNonDict:
  def test_strict_non_dict_line_raises(self, tmp_path):
    path = tmp_path / 'array.jsonl'
    path.write_text('{"ok": 1}\n[1, 2, 3]\n')
    with pytest.raises(TrackingError):
      read_jsonl(path, strict=True)

  def test_tolerant_non_dict_skipped(self, tmp_path):
    path = tmp_path / 'array.jsonl'
    path.write_text('{"ok": 1}\n[1, 2, 3]\n{"ok": 2}\n')
    records = read_jsonl(path, strict=False)
    assert len(records) == 2
