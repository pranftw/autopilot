"""Tests for tracking/io.py shared I/O primitives."""

from autopilot.core.errors import TrackingError
from autopilot.tracking.io import (
  append_jsonl,
  atomic_write_json,
  exclusive_create,
  iter_jsonl_lines,
  read_json,
  read_json_dict,
  read_jsonl,
  utc_now_iso,
)
from unittest.mock import patch
import concurrent.futures
import json
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

  def test_list_payload(self, tmp_path):
    path = tmp_path / 'list.json'
    payload = [{'name': 'a'}, {'name': 'b'}]
    atomic_write_json(path, payload)
    result = read_json(path)
    assert result == payload

  def test_overwrites_existing(self, tmp_path):
    path = tmp_path / 'test.json'
    atomic_write_json(path, {'first': True})
    atomic_write_json(path, {'second': True})
    result = read_json(path)
    assert result == {'second': True}


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


class TestReadJsonBinaryFile:
  def test_read_json_binary_file_raises_tracking_error(self, tmp_path):
    path = tmp_path / 'binary.json'
    path.write_bytes(b'\x80\x81\x82\xff\xfe')
    with pytest.raises(TrackingError, match='failed to read JSON'):
      read_json(path)

  def test_read_json_dict_binary_file_raises_tracking_error(self, tmp_path):
    path = tmp_path / 'binary.json'
    path.write_bytes(b'\x80\x81\x82\xff\xfe')
    with pytest.raises(TrackingError, match='failed to read JSON'):
      read_json_dict(path, 'test')


class TestReadJsonlBinaryFile:
  def test_read_jsonl_binary_file_raises_tracking_error(self, tmp_path):
    path = tmp_path / 'binary.jsonl'
    path.write_bytes(b'\x80\x81\x82\xff\xfe')
    with pytest.raises(TrackingError, match='failed to read JSONL'):
      read_jsonl(path)


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


class TestUtcNowIso:
  def test_returns_string(self):
    result = utc_now_iso()
    assert isinstance(result, str)
    assert len(result) > 10

  def test_contains_utc_offset(self):
    result = utc_now_iso()
    assert '+00:00' in result


class TestReadJsonDict:
  def test_round_trip(self, tmp_path):
    path = tmp_path / 'test.json'
    atomic_write_json(path, {'k': 'v'})
    assert read_json_dict(path, 'test') == {'k': 'v'}

  def test_root_array_raises(self, tmp_path):
    path = tmp_path / 'arr.json'
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(TrackingError) as exc_info:
      read_json_dict(path, 'test')
    assert 'list' in str(exc_info.value)

  def test_empty_file_raises(self, tmp_path):
    path = tmp_path / 'empty.json'
    path.write_text('')
    with pytest.raises(TrackingError) as exc_info:
      read_json_dict(path, 'test')
    assert 'NoneType' in str(exc_info.value)

  def test_missing_file_raises(self, tmp_path):
    path = tmp_path / 'missing.json'
    with pytest.raises(TrackingError) as exc_info:
      read_json_dict(path, 'test')
    assert 'NoneType' in str(exc_info.value)


class TestIterJsonlLines:
  def test_skips_blank_and_strips(self, tmp_path):
    path = tmp_path / 'lines.txt'
    path.write_text('  a\n\n b \n', encoding='utf-8')
    assert list(iter_jsonl_lines(path)) == ['a', 'b']

  def test_missing_file_is_empty(self, tmp_path):
    path = tmp_path / 'nonexistent.txt'
    assert list(iter_jsonl_lines(path)) == []

  def test_empty_file(self, tmp_path):
    path = tmp_path / 'empty.txt'
    path.write_text('', encoding='utf-8')
    assert list(iter_jsonl_lines(path)) == []

  def test_utf8_round_trip(self, tmp_path):
    path = tmp_path / 'unicode.txt'
    path.write_text('\u00e9l\u00e8ve\n', encoding='utf-8')
    assert list(iter_jsonl_lines(path)) == ['\u00e9l\u00e8ve']

  def test_invalid_utf8_raises_tracking_error(self, tmp_path):
    path = tmp_path / 'binary.txt'
    path.write_bytes(b'\x80\x81')
    with pytest.raises(TrackingError, match='failed to read lines'):
      list(iter_jsonl_lines(path))

  def test_oserror_during_read(self, tmp_path):
    path = tmp_path / 'oserror.txt'
    path.write_text('line1\n', encoding='utf-8')

    def _exploding_open(*args, **kwargs):
      msg = 'disk error'
      raise OSError(msg)

    with (
      patch.object(type(path), 'open', _exploding_open),
      pytest.raises(TrackingError, match='failed to read lines') as exc_info,
    ):
      list(iter_jsonl_lines(path))

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, OSError)


class TestExclusiveCreate:
  def test_writes_empty_file(self, tmp_path):
    path = tmp_path / 'new.lock'
    exclusive_create(path)
    assert path.is_file()
    assert path.stat().st_size == 0

  def test_existing_raises(self, tmp_path):
    path = tmp_path / 'existing.lock'
    path.write_text('', encoding='utf-8')
    with pytest.raises(FileExistsError):
      exclusive_create(path)

  def test_creates_parent_dirs(self, tmp_path):
    path = tmp_path / 'a' / 'b' / 'c' / 'file.lock'
    exclusive_create(path)
    assert path.is_file()

  def test_concurrent(self, tmp_path):
    path = tmp_path / 'race.lock'
    results = []

    def attempt():
      try:
        exclusive_create(path)
      except FileExistsError:
        return 'exists'
      else:
        return 'ok'

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
      futures = [pool.submit(attempt) for _ in range(2)]
      results = [f.result() for f in futures]

    assert results.count('ok') == 1
    assert results.count('exists') == 1

  def test_oserror_not_exists(self, tmp_path):
    path = tmp_path / 'perm.lock'

    def _exploding_open(*args, **kwargs):
      msg = 'denied'
      raise PermissionError(msg)

    with (
      patch.object(type(path), 'open', _exploding_open),
      pytest.raises(TrackingError, match='failed to create'),
    ):
      exclusive_create(path)
