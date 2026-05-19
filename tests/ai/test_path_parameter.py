"""Tests for PathParameter working_root, bind/unbind, BUG-006 dir filter, and schema."""

from autopilot.ai.parameter import PathParameter
from autopilot.core.snapshot import ParameterSchemaEntry
from pathlib import Path
import pytest


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
  """Create a source directory with test files and a subdirectory."""
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'main.py').write_text('print("hello")', encoding='utf-8')
  (src / 'util.py').write_text('# util', encoding='utf-8')
  (src / 'subdir').mkdir()
  (src / 'subdir' / 'nested.py').write_text('# nested', encoding='utf-8')
  return src


class TestWorkingRootDefault:
  """working_root defaults to source when unbound."""

  def test_working_root_default_is_source(self, source_dir: Path) -> None:
    param = PathParameter(source=str(source_dir))
    assert param.working_root == str(source_dir)
    assert param._working_root is None


class TestBindUnbind:
  """bind sets working_root, unbind resets to source."""

  def test_bind_sets_working_root(self, source_dir: Path) -> None:
    param = PathParameter(source=str(source_dir))
    param.bind('/tmp/worktree/src')
    assert param.working_root == '/tmp/worktree/src'
    assert param._working_root == '/tmp/worktree/src'

  def test_unbind_resets_to_source(self, source_dir: Path) -> None:
    param = PathParameter(source=str(source_dir))
    param.bind('/tmp/wt')
    param.unbind()
    assert param.working_root == str(source_dir)
    assert param._working_root is None

  def test_independent_binds(self, source_dir: Path) -> None:
    p1 = PathParameter(source=str(source_dir))
    p2 = PathParameter(source=str(source_dir))
    p1.bind('/tmp/wt1')
    p2.bind('/tmp/wt2')
    assert p1.working_root == '/tmp/wt1'
    assert p2.working_root == '/tmp/wt2'
    p1.unbind()
    assert p1.working_root == str(source_dir)
    assert p2.working_root == '/tmp/wt2'


class TestMatchedFilesBound:
  """matched_files uses working_root when bound."""

  def test_matched_files_bound(self, source_dir: Path, tmp_path: Path) -> None:
    wt = tmp_path / 'worktree'
    wt.mkdir()
    (wt / 'new_file.py').write_text('# new', encoding='utf-8')

    param = PathParameter(source=str(source_dir))
    param.bind(str(wt))
    files = param.matched_files()
    assert len(files) == 1
    assert files[0].name == 'new_file.py'

  def test_matched_files_skips_dirs(self, source_dir: Path) -> None:
    param = PathParameter(source=str(source_dir), pattern='**/*')
    files = param.matched_files()
    for f in files:
      assert f.is_file(), f'{f} is not a file (BUG-006)'
    names = [f.name for f in files]
    assert 'subdir' not in names


class TestRenderShowsWorkingRoot:
  """render() displays working_root paths."""

  def test_render_shows_working_root(self, source_dir: Path) -> None:
    param = PathParameter(source=str(source_dir))
    render_before = param.render()
    assert str(source_dir) in render_before

    wt = source_dir.parent / 'worktree'
    wt.mkdir()
    (wt / 'file.txt').write_text('hi', encoding='utf-8')
    param.bind(str(wt))
    render_after = param.render()
    assert str(wt) in render_after
    assert str(source_dir) not in render_after


class TestSnapshotRestore:
  """snapshot/restore route through working_root."""

  def test_snapshot_captures_worktree(self, source_dir: Path, tmp_path: Path) -> None:
    wt = tmp_path / 'wt'
    wt.mkdir()
    (wt / 'a.txt').write_text('worktree content', encoding='utf-8')

    param = PathParameter(source=str(source_dir))
    param.bind(str(wt))
    snap = param.snapshot()
    assert 'a.txt' in snap
    assert snap['a.txt'] == 'worktree content'

  def test_restore_writes_worktree(self, source_dir: Path, tmp_path: Path) -> None:
    wt = tmp_path / 'wt'
    wt.mkdir()

    param = PathParameter(source=str(source_dir))
    param.bind(str(wt))
    param.restore({'restored.txt': 'hello'})
    assert (wt / 'restored.txt').read_text(encoding='utf-8') == 'hello'
    assert not (source_dir / 'restored.txt').exists()


class TestToDictOmitsWorkingRoot:
  """to_dict serializes source, not working_root."""

  def test_to_dict_omits_working_root(self, source_dir: Path) -> None:
    param = PathParameter(source=str(source_dir), pattern='*.py')
    param.bind('/tmp/wt')
    data = param.to_dict()
    assert data['source'] == str(source_dir)
    assert data['pattern'] == '*.py'
    assert '_working_root' not in data
    assert 'working_root' not in data


class TestSchemaEntryPathParam:
  """PathParameter.schema_entry() includes source and pattern."""

  def test_schema_entry_path_param(self, source_dir: Path) -> None:
    param = PathParameter(source=str(source_dir), pattern='**/*.md')
    entry = param.schema_entry()
    assert isinstance(entry, ParameterSchemaEntry)
    assert entry.type_name == 'PathParameter'
    assert entry.source == str(source_dir)
    assert entry.pattern == '**/*.md'
    assert not entry.name


class TestFromDictRoundTrip:
  """to_dict / from_dict round-trip preserves source, pattern, id."""

  def test_from_dict_round_trip(self, source_dir: Path) -> None:
    param = PathParameter(source=str(source_dir), pattern='*.py')
    data = param.to_dict()
    restored = PathParameter.from_dict(data)
    assert restored.source == param.source
    assert restored.pattern == param.pattern
    assert restored._working_root is None
