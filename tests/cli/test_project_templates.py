"""Tests for project template resolution from package.

Covers:
  - Workspace template override: workspace file wins when present
  - Package fallback: workspace missing -> bundle still produces output files
"""

from autopilot.cli.commands.project import _read_template
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
from unittest.mock import patch
import pytest


class TestProjectTemplateWorkspaceOverride:
  def test_workspace_file_wins(self, tmp_path: Path) -> None:
    """When workspace template exists, it is used over package bundle."""
    config = AutoPilotConfig(workspace=tmp_path)
    tpl_dir = config.templates_path / 'project'
    tpl_dir.mkdir(parents=True)
    (tpl_dir / 'cli.py').write_text('workspace-cli-{name}')

    result = _read_template(config, 'cli.py', name='myproj')
    assert result == 'workspace-cli-myproj'


class TestProjectTemplatePackageFallback:
  def test_package_fallback(self, tmp_path: Path) -> None:
    """When workspace template is missing, package bundle is used."""
    config = AutoPilotConfig(workspace=tmp_path)

    fake_content = 'bundled-template-{name}'

    class FakePath:
      def read_text(self, encoding='utf-8'):
        return fake_content

    with patch('autopilot.cli.commands.project.importlib.resources.files') as mock_files:
      chain = mock_files.return_value.joinpath.return_value
      chain.joinpath.return_value.joinpath.return_value = FakePath()
      result = _read_template(config, 'cli.py', name='testproj')

    assert result == 'bundled-template-testproj'

  def test_both_missing_raises(self, tmp_path: Path) -> None:
    """When both workspace and package templates are missing, FileNotFoundError is raised."""
    config = AutoPilotConfig(workspace=tmp_path)

    with patch('autopilot.cli.commands.project.importlib.resources.files') as mock_files:
      chain = mock_files.return_value.joinpath.return_value
      leaf = chain.joinpath.return_value.joinpath.return_value
      leaf.read_text.side_effect = FileNotFoundError
      with pytest.raises(FileNotFoundError, match='not found'):
        _read_template(config, 'nonexistent.py')
