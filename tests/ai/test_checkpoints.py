"""Tests for autopilot.ai.checkpoints."""

from autopilot.ai.checkpoints import CheckpointIO, CheckpointManager
from autopilot.core.errors import AIError
from pathlib import Path
from pydantic import BaseModel
import json
import pytest


class _Mini(BaseModel):
  type: str = 'result'
  item_id: str = 'id'
  timestamp: str = 't'
  payload: dict


def _minimal_event(extra: dict | None = None) -> _Mini:
  payload = dict(extra) if extra is not None else {}
  return _Mini(payload=payload)


class InMemoryCheckpointIO(CheckpointIO):
  """Test double: stores JSONL events in memory keyed by resolved path string."""

  def __init__(self) -> None:
    self._store: dict[str, list[dict]] = {}

  def save_event(self, path: Path, event: BaseModel) -> None:
    key = str(path.resolve())
    if key not in self._store:
      self._store[key] = []
    self._store[key].append(json.loads(event.model_dump_json()))

  def load(self, path: Path) -> list[dict]:
    key = str(path.resolve())
    return list(self._store.get(key, []))

  def remove(self, path: Path) -> None:
    self._store.pop(str(path.resolve()), None)


class TestCheckpointIO:
  def test_save_event_creates_file(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    io = CheckpointIO()
    io.save_event(path, _minimal_event())
    assert path.is_file()

  def test_save_event_appends(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    io = CheckpointIO()
    io.save_event(path, _minimal_event())
    io.save_event(path, _minimal_event())
    lines = path.read_text(encoding='utf-8').strip().split('\n')
    assert len(lines) == 2

  def test_load_empty_file(self, tmp_path: Path) -> None:
    path = tmp_path / 'empty.jsonl'
    path.write_text('', encoding='utf-8')
    assert CheckpointIO().load(path) == []

  def test_load_nonexistent_file(self, tmp_path: Path) -> None:
    path = tmp_path / 'missing.jsonl'
    assert CheckpointIO().load(path) == []

  def test_load_round_trip(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    io = CheckpointIO()
    for i in range(3):
      io.save_event(path, _minimal_event(extra={'i': i}))
    loaded = io.load(path)
    assert len(loaded) == 3
    assert [d['payload']['i'] for d in loaded] == [0, 1, 2]

  def test_creates_parent_dirs(self, tmp_path: Path) -> None:
    path = tmp_path / 'nested' / 'deep' / 'c.jsonl'
    CheckpointIO().save_event(path, _minimal_event())
    assert path.is_file()

  def test_invalid_json_raises(self, tmp_path: Path) -> None:
    path = tmp_path / 'bad.jsonl'
    path.write_text('not json\n', encoding='utf-8')
    with pytest.raises(AIError, match='invalid JSON'):
      CheckpointIO().load(path)

  def test_remove_deletes_file(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    path.write_text('x\n', encoding='utf-8')
    CheckpointIO().remove(path)
    assert not path.exists()

  def test_remove_nonexistent_no_error(self, tmp_path: Path) -> None:
    path = tmp_path / 'nope.jsonl'
    CheckpointIO().remove(path)


class TestCheckpointManager:
  def test_save_header(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_header('h1', 'sub')
    assert m.header is not None
    assert m.header['type'] == 'header'
    assert m.header['subsystem'] == 'sub'
    assert m.header['config_hash'] == 'h1'

  def test_save_and_load_events(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_event('ping', 'x', {'k': 1})
    loaded = CheckpointIO().load(path)
    assert len(loaded) == 1
    assert loaded[0]['type'] == 'ping'
    assert loaded[0]['item_id'] == 'x'
    assert loaded[0]['payload']['k'] == 1

  def test_is_completed_result_type(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_event('result', 'item1', {})
    assert m.is_completed('item1')

  def test_is_completed_error_not_counted(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_event('error', 'e1', {})
    assert not m.is_completed('e1')

  def test_is_completed_skip_not_counted(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_event('skip', 's1', {})
    assert not m.is_completed('s1')

  def test_completed_ids(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_event('result', 'a', {})
    m.save_event('result', 'b', {})
    m.save_event('error', 'c', {})
    assert m.completed_ids() == {'a', 'b'}

  def test_resume_sees_prior_events(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m1 = CheckpointManager(path)
    m1.save_event('result', 'r1', {})
    m2 = CheckpointManager(path)
    assert m2.completed_ids() == {'r1'}

  def test_resume_skips_completed(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m1 = CheckpointManager(path)
    m1.save_event('result', 'done', {})
    m2 = CheckpointManager(path)
    assert m2.is_completed('done')

  def test_save_state_load_state(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_state('k1', {'a': 1})
    assert m.load_state('k1') == {'a': 1}

  def test_save_state_multiple_keys(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_state('x', {'v': 1})
    m.save_state('y', {'v': 2})
    assert m.load_state('x') == {'v': 1}
    assert m.load_state('y') == {'v': 2}

  def test_save_state_overwrite(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_state('same', {'v': 1})
    m.save_state('same', {'v': 2})
    assert m.load_state('same') == {'v': 2}

  def test_summary_counts(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    for _ in range(3):
      m.save_event('result', 'r', {})
    m.save_event('error', 'e', {})
    for _ in range(2):
      m.save_event('skip', 's', {})
    assert m.summary() == {'result': 3, 'error': 1, 'skip': 2}

  def test_header_config_hash(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_header('abc123', 'gen')
    assert m.header is not None
    assert m.header['config_hash'] == 'abc123'

  def test_header_args_stored(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_header('h', 'sub', {'x': 1})
    assert m.args == {'x': 1}

  def test_update_args(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_header('h', 'sub', {'a': 1})
    m.update_args({'b': 2})
    assert m.args == {'a': 1, 'b': 2}

  def test_update_args_overrides(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    m.save_header('h', 'sub', {'a': 1, 'b': 1})
    m.update_args({'a': 2, 'c': 3})
    assert m.args == {'a': 2, 'b': 1, 'c': 3}

  def test_resume_with_updated_args(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m1 = CheckpointManager(path)
    m1.save_header('h', 'sub', {'x': 1})
    m1.update_args({'y': 2})
    m2 = CheckpointManager(path)
    assert m2.args == {'x': 1, 'y': 2}

  def test_args_empty_by_default(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m = CheckpointManager(path)
    assert m.args == {}

  def test_incremental_fault_tolerance(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    m1 = CheckpointManager(path)
    m1.save_event('result', 'a', {})
    with path.open('ab') as fh:
      fh.write(b'\nGARBAGE{{{')
    try:
      m2 = CheckpointManager(path)
      assert m2.is_completed('a')
    except AIError:
      pass


class TestCustomCheckpointIO:
  def test_in_memory_backend(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    io = InMemoryCheckpointIO()
    io.save_event(path, _minimal_event({'n': 1}))
    io.save_event(path, _minimal_event({'n': 2}))
    assert io.load(path)[0]['payload']['n'] == 1
    assert io.load(path)[1]['payload']['n'] == 2

  def test_custom_io_used_by_manager(self, tmp_path: Path) -> None:
    path = tmp_path / 'c.jsonl'
    io = InMemoryCheckpointIO()
    m = CheckpointManager(path, io=io)
    m.save_event('result', 'z', {})
    key = str(path.resolve())
    assert key in io._store
    assert len(io._store[key]) == 1
    assert io._store[key][0]['item_id'] == 'z'
