"""Concurrency and edge-case tests for tracking/io.py.

Tests:
  1. 10 threads x 50 writes to same JSON file -- all succeed, final file valid
  2. 10 threads x 50 appends to same JSONL -- 500 lines, all valid
  3. Unique temp files don't leak on success
  4. Temp files cleaned up on failure
  5. atomic_write_json with read-only parent dir -> TrackingError
  6. atomic_write_json with unicode path
  7. read_json on empty file -> returns None
  8. read_json with BOM -> strip BOM before parsing
"""

from autopilot.core.errors import TrackingError
from autopilot.tracking.io import append_jsonl, atomic_write_json, read_json, read_jsonl
from pathlib import Path
import pytest
import sys
import threading


class TestConcurrentAtomicWriteJson:
  """10 threads, each writing 50 times to same JSON file."""

  def test_concurrent_writes_all_succeed(self, tmp_path):
    path = tmp_path / 'shared.json'
    errors: list[Exception] = []
    num_threads = 10
    writes_per_thread = 50

    def writer(thread_id):
      for i in range(writes_per_thread):
        try:
          atomic_write_json(path, {'thread': thread_id, 'write': i})
        except TrackingError as exc:
          errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(num_threads)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    assert not errors, f'concurrent writes failed: {errors}'
    result = read_json(path)
    assert isinstance(result, dict)
    assert 'thread' in result
    assert 'write' in result


class TestConcurrentAppendJsonl:
  """10 threads, each appending 50 records to same JSONL."""

  def test_concurrent_appends_produce_500_lines(self, tmp_path):
    path = tmp_path / 'shared.jsonl'
    errors: list[Exception] = []
    num_threads = 10
    appends_per_thread = 50

    def appender(thread_id):
      for i in range(appends_per_thread):
        try:
          append_jsonl(path, {'thread': thread_id, 'record': i})
        except TrackingError as exc:
          errors.append(exc)

    threads = [threading.Thread(target=appender, args=(t,)) for t in range(num_threads)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    assert not errors, f'concurrent appends failed: {errors}'
    records = read_jsonl(path)
    assert len(records) == num_threads * appends_per_thread


class TestTempFileCleanup:
  """Verify temp files don't leak on success and are cleaned on failure."""

  def test_no_temp_file_leak_on_success(self, tmp_path):
    path = tmp_path / 'clean.json'
    atomic_write_json(path, {'ok': True})
    remaining = list(tmp_path.glob('*.tmp'))
    assert remaining == [], f'leaked temp files: {remaining}'

  def test_temp_file_cleaned_on_serialization_failure(self, tmp_path):
    path = tmp_path / 'fail.json'
    with pytest.raises(TrackingError):
      atomic_write_json(path, {'bad': object()})
    remaining = list(tmp_path.glob('*.tmp'))
    assert remaining == [], f'leaked temp files after failure: {remaining}'

  def test_temp_file_cleaned_on_os_error(self, tmp_path, monkeypatch):
    path = tmp_path / 'fail.json'
    original_write_text = Path.write_text

    def failing_write_text(self, *args, **kwargs):
      original_write_text(self, *args, **kwargs)
      msg = 'simulated disk failure'
      raise OSError(msg)

    monkeypatch.setattr(Path, 'write_text', failing_write_text)
    with pytest.raises(TrackingError, match='failed to write JSON'):
      atomic_write_json(path, {'data': 1})
    monkeypatch.undo()
    remaining = list(tmp_path.glob('*.tmp'))
    assert remaining == [], f'leaked temp files after OS error: {remaining}'


class TestAtomicWriteJsonEdgeCases:
  """Edge cases: read-only parent dir, unicode path."""

  def test_readonly_parent_dir_raises_tracking_error(self, tmp_path):
    readonly_dir = tmp_path / 'readonly'
    readonly_dir.mkdir()
    Path(readonly_dir).chmod(0o444)
    path = readonly_dir / 'subdir' / 'file.json'
    try:
      with pytest.raises(TrackingError):
        atomic_write_json(path, {'data': 1})
    finally:
      Path(readonly_dir).chmod(0o755)

  @pytest.mark.skipif(
    sys.platform == 'win32',
    reason='Windows does not support unicode filenames as broadly',
  )
  def test_unicode_path(self, tmp_path):
    path = tmp_path / 'datos_\u00e9special.json'
    payload = {'clave': 'valor', 'n\u00famero': 42}
    atomic_write_json(path, payload)
    result = read_json(path)
    assert result == payload


class TestReadJsonEdgeCases:
  """Empty file and BOM handling for read_json."""

  def test_empty_file_returns_none(self, tmp_path):
    path = tmp_path / 'empty.json'
    path.write_text('')
    assert read_json(path) is None

  def test_whitespace_only_file_returns_none(self, tmp_path):
    path = tmp_path / 'whitespace.json'
    path.write_text('   \n  \n  ')
    assert read_json(path) is None

  def test_bom_stripped_before_parsing(self, tmp_path):
    path = tmp_path / 'bom.json'
    content = '\ufeff{"key": "value"}'
    path.write_text(content, encoding='utf-8')
    result = read_json(path)
    assert result == {'key': 'value'}

  def test_bom_with_array(self, tmp_path):
    path = tmp_path / 'bom_array.json'
    content = '\ufeff[1, 2, 3]'
    path.write_text(content, encoding='utf-8')
    result = read_json(path)
    assert result == [1, 2, 3]

  def test_bom_only_file_returns_none(self, tmp_path):
    path = tmp_path / 'bom_only.json'
    path.write_text('\ufeff', encoding='utf-8')
    assert read_json(path) is None

  def test_bom_with_whitespace_returns_none(self, tmp_path):
    path = tmp_path / 'bom_ws.json'
    path.write_text('\ufeff   \n  ', encoding='utf-8')
    assert read_json(path) is None
