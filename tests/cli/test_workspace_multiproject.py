"""Tests for workspace command multi-project support."""

from autopilot.cli.commands.workspace import WorkspaceCommand
from autopilot.cli.context import CLIContext
from autopilot.cli.output import Output
from pathlib import Path
import argparse
import autopilot.core.paths as paths


def _ctx(tmp_path: Path) -> CLIContext:
  return CLIContext(workspace=tmp_path, output=Output(use_json=True))


def _args() -> argparse.Namespace:
  return argparse.Namespace()


class TestWorkspaceInit:
  def test_creates_projects_dir(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    WorkspaceCommand().init(ctx, _args())
    assert paths.projects_dir(tmp_path).is_dir()

  def test_does_not_create_workspace_toml(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    WorkspaceCommand().init(ctx, _args())
    assert not (tmp_path / 'autopilot' / 'workspace.toml').exists()

  def test_idempotent(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    WorkspaceCommand().init(ctx, _args())
    WorkspaceCommand().init(ctx, _args())
    assert paths.projects_dir(tmp_path).is_dir()


class TestWorkspaceDoctor:
  def test_runs_after_init(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    WorkspaceCommand().init(ctx, _args())
    WorkspaceCommand().doctor(ctx, _args())

  def test_detects_missing_layout(self, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    WorkspaceCommand().doctor(ctx, _args())
