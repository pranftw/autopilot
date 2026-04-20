"""Tests for project resolution and CLI parser construction."""

from autopilot.cli.context import _resolve_project
from autopilot.cli.main import build_parser
from pathlib import Path
import autopilot.core.paths as paths
import pytest


def _setup_project_dir(tmp_path: Path, name: str) -> None:
  pdir = paths.projects_dir(tmp_path)
  (pdir / name).mkdir(parents=True, exist_ok=True)


class TestResolveProject:
  def test_explicit_flag_wins(self, tmp_path: Path) -> None:
    result = _resolve_project(tmp_path, 'explicit')
    assert result == 'explicit'

  def test_returns_none_when_no_project(self, tmp_path: Path) -> None:
    result = _resolve_project(tmp_path, '')
    assert result is None

  def test_explicit_over_cwd_detection(self, tmp_path: Path) -> None:
    _setup_project_dir(tmp_path, 'default-proj')
    result = _resolve_project(tmp_path, 'explicit')
    assert result == 'explicit'

  def test_detects_project_from_cwd_under_projects(
    self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    _setup_project_dir(tmp_path, 'myproj')
    proj_home = paths.projects_dir(tmp_path) / 'myproj'
    (proj_home / 'ai').mkdir(parents=True)
    monkeypatch.chdir(proj_home)
    assert _resolve_project(tmp_path, '') == 'myproj'


class TestBuildParser:
  def test_has_project_flag(self) -> None:
    parser = build_parser()
    args = parser.parse_args(['workspace', 'doctor', '-p', 'my-proj', '--workspace', '.'])
    assert args.project == 'my-proj'

  def test_project_long_flag(self) -> None:
    parser = build_parser()
    args = parser.parse_args(['workspace', 'doctor', '--project', 'my-proj', '--workspace', '.'])
    assert args.project == 'my-proj'

  def test_project_default_empty(self) -> None:
    parser = build_parser()
    args = parser.parse_args(['workspace', 'doctor', '--workspace', '.'])
    assert args.project == ''

  def test_project_command_registered(self) -> None:
    parser = build_parser()
    args = parser.parse_args(['project', 'list', '--workspace', '.'])
    assert args.command == 'project'

