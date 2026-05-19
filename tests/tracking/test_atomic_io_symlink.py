"""Tests for atomic I/O symlink safety (FRICTION-012).

Verifies that atomic_write_json follows symlinks so that the symlink inode
is preserved and the resolved target file is atomically replaced. Also
covers append_jsonl small-record regression.
"""

from autopilot.tracking.io import append_jsonl, atomic_write_json, read_json
import json


class TestAtomicWriteThroughSymlink:
  """atomic_write_json must follow symlinks to the real file."""

  def test_atomic_write_through_symlink_updates_content(self, tmp_path):
    target = tmp_path / 'real.json'
    target.write_text(json.dumps({'old': True}, indent=2), encoding='utf-8')
    link = tmp_path / 'link.json'
    link.symlink_to(target)

    new_payload = {'new': True, 'count': 42}
    atomic_write_json(link, new_payload)

    result = json.loads(link.read_text(encoding='utf-8'))
    assert result == new_payload

  def test_atomic_write_preserves_symlink(self, tmp_path):
    target = tmp_path / 'real.json'
    target.write_text('{}', encoding='utf-8')
    link = tmp_path / 'link.json'
    link.symlink_to(target)

    atomic_write_json(link, {'updated': True})

    assert link.is_symlink()

  def test_atomic_write_updates_resolved_target(self, tmp_path):
    target = tmp_path / 'real.json'
    target.write_text(json.dumps({'before': 1}), encoding='utf-8')
    link = tmp_path / 'link.json'
    link.symlink_to(target)

    payload = {'after': 2}
    atomic_write_json(link, payload)

    resolved_content = json.loads(target.read_text(encoding='utf-8'))
    assert resolved_content == payload

  def test_atomic_write_json_regular_file(self, tmp_path):
    path = tmp_path / 'plain.json'
    payload = {'key': 'value', 'nested': [1, 2, 3]}
    atomic_write_json(path, payload)
    result = read_json(path)
    assert result == payload
    assert not path.is_symlink()

  def test_atomic_write_json_nonexistent_parent(self, tmp_path):
    path = tmp_path / 'a' / 'b' / 'c' / 'deep.json'
    payload = {'created': True}
    atomic_write_json(path, payload)
    assert path.exists()
    assert read_json(path) == payload


class TestAtomicWriteSymlinkChain:
  """Symlink chains (link -> link -> file) must also resolve correctly."""

  def test_double_symlink_chain(self, tmp_path):
    target = tmp_path / 'real.json'
    target.write_text('{}', encoding='utf-8')
    mid = tmp_path / 'mid.json'
    mid.symlink_to(target)
    outer = tmp_path / 'outer.json'
    outer.symlink_to(mid)

    payload = {'chain': 'resolved'}
    atomic_write_json(outer, payload)

    assert outer.is_symlink()
    assert mid.is_symlink()
    assert json.loads(target.read_text(encoding='utf-8')) == payload
    assert json.loads(outer.read_text(encoding='utf-8')) == payload


class TestAppendJsonlSmallRecord:
  """Small JSONL records append as single lines."""

  def test_append_jsonl_small_record(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    record = {'event': 'test', 'value': 1}
    append_jsonl(path, record)

    text = path.read_text(encoding='utf-8')
    assert text.endswith('\n')
    lines = text.strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed == record
