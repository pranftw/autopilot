"""Tests for project management command."""

from autopilot.cli.commands.project import ProjectCommand, ProjectInit
from autopilot.cli.context import CLIContext
from autopilot.cli.main import build_parser
from autopilot.cli.output import Output
from pathlib import Path
import argparse
import autopilot.core.paths as paths
import pytest


def _ctx(tmp_path: Path) -> CLIContext:
  return CLIContext(workspace=tmp_path, output=Output(use_json=True))


def _args(**kwargs) -> argparse.Namespace:
  kwargs.setdefault('bare', False)
  return argparse.Namespace(**kwargs)


class TestProjectInitParser:
  def test_requires_name(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['project', 'init'])

  def test_parses_name(self) -> None:
    parser = build_parser()
    args = parser.parse_args(['project', 'init', 'my-project'])
    assert args.name == 'my-project'


class TestProjectListParser:
  def test_parses_list(self) -> None:
    parser = build_parser()
    args = parser.parse_args(['project', 'list'])
    assert args.project_action == 'list'


class TestProjectDoctorParser:
  def test_requires_name(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['project', 'doctor'])


class TestProjectInitHandler:
  def test_creates_directory_structure(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ProjectInit()(ctx, _args(name='p1'))
    proj = paths.root(tmp_path, 'p1')
    assert proj.is_dir()
    assert (proj / 'ai').is_dir()
    assert paths.experiments(tmp_path, 'p1').is_dir()
    assert paths.datasets(tmp_path, 'p1').is_dir()
    assert paths.records(tmp_path, 'p1').is_dir()
    assert (paths.records(tmp_path, 'p1') / 'promotions').is_dir()
    assert (paths.records(tmp_path, 'p1') / 'notes').is_dir()

  def test_idempotent(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ProjectInit()(ctx, _args(name='p1'))
    ProjectInit()(ctx, _args(name='p1'))
    assert paths.root(tmp_path, 'p1').is_dir()

  def test_creates_skeleton_files(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ProjectInit()(ctx, _args(name='p1', bare=False))
    proj = paths.root(tmp_path, 'p1')
    assert paths.project_cli(tmp_path, 'p1').is_file()
    assert (proj / 'module.py').is_file()
    assert (proj / 'data.py').is_file()
    cli_content = paths.project_cli(tmp_path, 'p1').read_text()
    assert "project='p1'" in cli_content

  def test_bare_skips_skeleton(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ProjectInit()(ctx, _args(name='p1', bare=True))
    assert not paths.project_cli(tmp_path, 'p1').exists()
    proj = paths.root(tmp_path, 'p1')
    assert not (proj / 'module.py').exists()

  def test_doctor_passes_after_init(self, tmp_path: Path, capsys) -> None:
    ctx = _ctx(tmp_path)
    ProjectInit()(ctx, _args(name='p1', bare=False))
    capsys.readouterr()
    ProjectCommand().doctor(ctx, _args(name='p1'))
    import json

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['healthy'] is True


class TestProjectListHandler:
  def test_lists_discovered_projects(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    for name in ('alpha', 'beta'):
      ProjectInit()(ctx, _args(name=name))
    ProjectCommand().list(ctx, _args())

  def test_empty_when_no_projects(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ProjectCommand().list(ctx, _args())


class TestProjectDoctorHandler:
  def test_healthy_project(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ProjectInit()(ctx, _args(name='p1', bare=False))
    ProjectCommand().doctor(ctx, _args(name='p1'))

  def test_missing_dirs_reported(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ProjectCommand().doctor(ctx, _args(name='nonexistent'))

  def test_missing_cli_py_reported(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ProjectInit()(ctx, _args(name='p1', bare=True))
    ProjectCommand().doctor(ctx, _args(name='p1'))
