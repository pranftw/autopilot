"""Tests for dataset fingerprinting utilities."""

from autopilot.ai.fingerprint import (
  DatasetFingerprint,
  compute_fingerprint,
  detect_drift,
  fingerprint_directory,
  fingerprint_file,
  fingerprint_jsonl,
)
from pathlib import Path
import json
import pytest


class TestFingerprintFile:
  """Tests for fingerprint_file."""

  def test_identical_contents_identical_hash(self, tmp_path: Path) -> None:
    """Two files with same content produce the same hash."""
    (tmp_path / 'a.txt').write_text('hello world', encoding='utf-8')
    (tmp_path / 'b.txt').write_text('hello world', encoding='utf-8')
    assert fingerprint_file(tmp_path / 'a.txt') == fingerprint_file(tmp_path / 'b.txt')

  def test_different_contents_different_hash(self, tmp_path: Path) -> None:
    """Files with different content produce different hashes."""
    (tmp_path / 'a.txt').write_text('hello world', encoding='utf-8')
    (tmp_path / 'b.txt').write_text('goodbye world', encoding='utf-8')
    assert fingerprint_file(tmp_path / 'a.txt') != fingerprint_file(tmp_path / 'b.txt')

  def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
    """Non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match='file not found'):
      fingerprint_file(tmp_path / 'nonexistent.txt')


class TestFingerprintDirectory:
  """Tests for fingerprint_directory."""

  def test_empty_directory(self, tmp_path: Path) -> None:
    """Empty directory produces a deterministic hash (SHA-256 of empty bytes)."""
    d = tmp_path / 'empty'
    d.mkdir()
    h = fingerprint_directory(d)
    assert isinstance(h, str)
    assert len(h) == 64

  def test_order_independent(self, tmp_path: Path) -> None:
    """Two files -- discovery order does not affect hash (sorted paths)."""
    d = tmp_path / 'data'
    d.mkdir()
    (d / 'z_file.txt').write_text('second', encoding='utf-8')
    (d / 'a_file.txt').write_text('first', encoding='utf-8')
    h1 = fingerprint_directory(d)
    d2 = tmp_path / 'data2'
    d2.mkdir()
    (d2 / 'a_file.txt').write_text('first', encoding='utf-8')
    (d2 / 'z_file.txt').write_text('second', encoding='utf-8')
    assert h1 == fingerprint_directory(d2)

  def test_pattern_excludes_files(self, tmp_path: Path) -> None:
    """Pattern filters to only matching files."""
    d = tmp_path / 'mixed'
    d.mkdir()
    (d / 'keep.txt').write_text('yes', encoding='utf-8')
    (d / 'skip.log').write_text('no', encoding='utf-8')
    h_all = fingerprint_directory(d, '**/*')
    h_txt = fingerprint_directory(d, '*.txt')
    assert h_all != h_txt

  def test_nested_relative_paths_sorted(self, tmp_path: Path) -> None:
    """Nested dirs produce stable hashes with lexicographic relative paths."""
    d = tmp_path / 'nested'
    d.mkdir()
    (d / 'sub').mkdir()
    (d / 'sub' / 'inner.txt').write_text('deep', encoding='utf-8')
    (d / 'top.txt').write_text('shallow', encoding='utf-8')
    h = fingerprint_directory(d)
    assert isinstance(h, str)
    assert len(h) == 64


class TestFingerprintJsonl:
  """Tests for fingerprint_jsonl."""

  def test_same_lines_same_hash(self, tmp_path: Path) -> None:
    """Identical JSONL files produce the same hash."""
    lines = [json.dumps({'a': 1}), json.dumps({'b': 2})]
    content = '\n'.join(lines) + '\n'
    (tmp_path / 'a.jsonl').write_text(content, encoding='utf-8')
    (tmp_path / 'b.jsonl').write_text(content, encoding='utf-8')
    assert fingerprint_jsonl(tmp_path / 'a.jsonl') == fingerprint_jsonl(tmp_path / 'b.jsonl')

  def test_line_order_change_different_hash(self, tmp_path: Path) -> None:
    """Swapping line order changes the hash (order-sensitive)."""
    (tmp_path / 'orig.jsonl').write_text('{"a":1}\n{"b":2}\n', encoding='utf-8')
    (tmp_path / 'swap.jsonl').write_text('{"b":2}\n{"a":1}\n', encoding='utf-8')
    assert fingerprint_jsonl(tmp_path / 'orig.jsonl') != fingerprint_jsonl(tmp_path / 'swap.jsonl')

  def test_duplicate_lines_preserved(self, tmp_path: Path) -> None:
    """Hash reflects duplicate lines (not deduplicated)."""
    (tmp_path / 'one.jsonl').write_text('{"x":1}\n', encoding='utf-8')
    (tmp_path / 'two.jsonl').write_text('{"x":1}\n{"x":1}\n', encoding='utf-8')
    assert fingerprint_jsonl(tmp_path / 'one.jsonl') != fingerprint_jsonl(tmp_path / 'two.jsonl')


class TestDatasetFingerprintRoundTrip:
  """Tests for DatasetFingerprint serialization."""

  def test_to_dict_from_dict_equality(self) -> None:
    """Round-trip via DictMixin preserves all fields."""
    fp = DatasetFingerprint(
      paths=['/data/train.jsonl', '/data/val.jsonl'],
      hashes=['abc123', 'def456'],
      row_count=1000,
      bundle_hash='bundle789',
      timestamp='2026-01-01T00:00:00+00:00',
    )
    restored = DatasetFingerprint.from_dict(fp.to_dict())
    assert restored.paths == fp.paths
    assert restored.hashes == fp.hashes
    assert restored.row_count == fp.row_count
    assert restored.bundle_hash == fp.bundle_hash
    assert restored.timestamp == fp.timestamp

  def test_optional_row_count_serialization(self) -> None:
    """row_count=None serializes and restores correctly."""
    fp = DatasetFingerprint(paths=['p'], hashes=['h'])
    d = fp.to_dict()
    assert d['row_count'] is None
    restored = DatasetFingerprint.from_dict(d)
    assert restored.row_count is None

  def test_optional_row_count_when_set(self) -> None:
    """row_count serializes when explicitly set."""
    fp = DatasetFingerprint(paths=['p'], hashes=['h'], row_count=42)
    d = fp.to_dict()
    assert d['row_count'] == 42
    restored = DatasetFingerprint.from_dict(d)
    assert restored.row_count == 42


class TestComputeFingerprint:
  """Tests for compute_fingerprint."""

  def test_multiple_roots_non_empty_bundle_hash(self, tmp_path: Path) -> None:
    """Multiple dataset roots produce a non-empty bundle_hash."""
    d1 = tmp_path / 'ds1'
    d1.mkdir()
    (d1 / 'data.txt').write_text('dataset one', encoding='utf-8')
    d2 = tmp_path / 'ds2'
    d2.mkdir()
    (d2 / 'data.txt').write_text('dataset two', encoding='utf-8')
    fp = compute_fingerprint([d1, d2])
    assert fp.bundle_hash is not None
    assert len(fp.bundle_hash) == 64

  def test_paths_hashes_aligned_lengths(self, tmp_path: Path) -> None:
    """paths and hashes lists have the same length."""
    f1 = tmp_path / 'a.txt'
    f1.write_text('one', encoding='utf-8')
    f2 = tmp_path / 'b.txt'
    f2.write_text('two', encoding='utf-8')
    fp = compute_fingerprint([f1, f2])
    assert len(fp.paths) == 2
    assert len(fp.hashes) == 2

  def test_empty_paths_returns_empty_fingerprint(self) -> None:
    """Empty paths input yields empty fingerprint with an empty-input bundle hash."""
    fp = compute_fingerprint([])
    assert fp.paths == []
    assert fp.hashes == []
    assert fp.bundle_hash is not None
    assert fp.timestamp is not None

  def test_timestamp_populated(self, tmp_path: Path) -> None:
    """compute_fingerprint sets timestamp via utc_now_iso."""
    f = tmp_path / 'data.txt'
    f.write_text('content', encoding='utf-8')
    fp = compute_fingerprint([f])
    assert fp.timestamp is not None
    assert len(fp.timestamp) > 0


class TestDetectDrift:
  """Tests for detect_drift."""

  def test_identical_fingerprints_no_drift(self) -> None:
    """Identical fingerprints return False."""
    fp = DatasetFingerprint(
      paths=['/a'],
      hashes=['abc'],
      bundle_hash='xyz',
    )
    assert detect_drift(fp, fp) is False

  def test_changed_single_hash_drift(self) -> None:
    """Changed single file hash returns True."""
    before = DatasetFingerprint(paths=['/a'], hashes=['abc'], bundle_hash='x')
    after = DatasetFingerprint(paths=['/a'], hashes=['def'], bundle_hash='y')
    assert detect_drift(before, after) is True

  def test_mismatched_path_lengths_drift(self) -> None:
    """Different path list lengths return True (structural mismatch)."""
    before = DatasetFingerprint(paths=['/a'], hashes=['abc'], bundle_hash='x')
    after = DatasetFingerprint(paths=['/a', '/b'], hashes=['abc', 'def'], bundle_hash='y')
    assert detect_drift(before, after) is True

  def test_same_hashes_changed_timestamp_no_drift(self) -> None:
    """Same hashes and paths with different timestamps return False."""
    before = DatasetFingerprint(
      paths=['/a'],
      hashes=['abc'],
      bundle_hash='x',
      timestamp='2026-01-01T00:00:00',
    )
    after = DatasetFingerprint(
      paths=['/a'],
      hashes=['abc'],
      bundle_hash='x',
      timestamp='2026-06-15T12:00:00',
    )
    assert detect_drift(before, after) is False


class TestLargeFileStreaming:
  """Verify streaming reads don't exhaust memory for larger files."""

  def test_large_file_hashing(self, tmp_path: Path) -> None:
    """A 100 KiB file hashes successfully via streaming path."""
    f = tmp_path / 'large.bin'
    data = b'x' * (100 * 1024)
    f.write_bytes(data)
    h = fingerprint_file(f)
    assert isinstance(h, str)
    assert len(h) == 64
