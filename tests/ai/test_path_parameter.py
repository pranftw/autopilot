"""Tests for PathParameter."""

from autopilot.ai.parameter import PathParameter
from autopilot.core.module import Module
from autopilot.core.parameter import Parameter
from pathlib import Path


class TestPathParameterBase:
  def test_path_parameter_source_and_pattern(self) -> None:
    p = PathParameter(source='/tmp/prompts', pattern='**/*.md')
    assert p.source == '/tmp/prompts'
    assert p.pattern == '**/*.md'

  def test_path_parameter_default_pattern_star_star(self) -> None:
    p = PathParameter(source='/tmp')
    assert p.pattern == '**/*'

  def test_path_parameter_is_parameter_subclass(self) -> None:
    p = PathParameter(source='/tmp')
    assert isinstance(p, Parameter)

  def test_path_parameter_to_dict(self) -> None:
    p = PathParameter(source='/tmp', pattern='*.tf')
    d = p.to_dict()
    assert d['source'] == '/tmp'
    assert d['pattern'] == '*.tf'
    assert d['requires_grad'] is True

  def test_path_parameter_from_dict_round_trip(self) -> None:
    p = PathParameter(source='/tmp/code', pattern='**/*.py', metrics={'n': 1.0})
    d = p.to_dict()
    p2 = PathParameter.from_dict(d)
    assert p2.source == '/tmp/code'
    assert p2.pattern == '**/*.py'
    assert p2.metrics == {'n': 1.0}


class TestPathParameterModuleIntegration:
  def test_path_parameter_registered_in_module(self) -> None:
    mod = Module()
    p = PathParameter(source='/tmp')
    mod.prompts = p
    params = list(mod.parameters())
    assert p in params

  def test_path_parameter_files_on_disk(self, tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.py').write_text('pass')
    (src / 'b.py').write_text('pass')
    (src / 'readme.md').write_text('# hi')

    p = PathParameter(source=str(src), pattern='*.py')
    files = p.matched_files()
    assert len(files) == 2
    names = {f.name for f in files}
    assert names == {'a.py', 'b.py'}

  def test_path_parameter_missing_source(self) -> None:
    p = PathParameter(source='/nonexistent/path')
    assert p.matched_files() == []


class TestPathParameterRender:
  def test_path_parameter_render(self, tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.py').write_text('pass')
    (src / 'b.py').write_text('pass')
    p = PathParameter(source=str(src), pattern='*.py')
    output = p.render()
    assert f'Editable files ({src})' in output
    assert 'a.py' in output
    assert 'b.py' in output

  def test_path_parameter_render_empty(self) -> None:
    p = PathParameter(source='/nonexistent/path')
    assert p.render() == ''


class TestPathParameterSnapshot:
  def test_path_parameter_snapshot(self, tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.txt').write_text('hello')
    (src / 'b.txt').write_text('world')
    p = PathParameter(source=str(src), pattern='*.txt')
    snap = p.snapshot()
    assert snap == {'a.txt': 'hello', 'b.txt': 'world'}

  def test_path_parameter_snapshot_empty(self) -> None:
    p = PathParameter(source='/nonexistent/path')
    assert p.snapshot() == {}

  def test_path_parameter_restore(self, tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.txt').write_text('original')
    p = PathParameter(source=str(src), pattern='*.txt')
    snap = p.snapshot()
    (src / 'a.txt').write_text('modified')
    p.restore(snap)
    assert (src / 'a.txt').read_text() == 'original'

  def test_path_parameter_restore_creates_dirs(self, tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    p = PathParameter(source=str(src))
    p.restore({'nested/deep/file.txt': 'content'})
    assert (src / 'nested' / 'deep' / 'file.txt').read_text() == 'content'
