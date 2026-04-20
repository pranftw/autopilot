"""Canonical config helpers for JSON overrides and path resolution."""

from autopilot.core.errors import ConfigError
from pathlib import Path
from typing import Any
import autopilot.core.paths as paths
import json


def load_json(path: Path) -> dict[str, Any]:
  """Load a JSON file and return its contents as a dict."""
  if not path.exists():
    raise ConfigError(f'config file not found: {path}')
  try:
    with open(path) as f:
      return json.load(f)
  except json.JSONDecodeError as e:
    raise ConfigError(f'invalid JSON in {path}: {e}') from e


def resolve_experiment_dir(
  workspace: Path,
  slug: str,
  project: str | None = None,
) -> Path:
  """Return the experiment directory path for a given slug."""
  return paths.experiment(workspace, slug, project)


def resolve_records_dir(workspace: Path, project: str | None = None) -> Path:
  """Return the records directory path."""
  return paths.records(workspace, project)


def list_projects(workspace: Path) -> list[str]:
  """Discover project names: subdirectories of the projects directory."""
  pdir = paths.projects_dir(workspace)
  if not pdir.exists():
    return []
  names = []
  for child in pdir.iterdir():
    if child.is_dir():
      names.append(child.name)
  return sorted(names)


def merge_overrides(
  base: dict[str, Any],
  overrides: dict[str, Any],
) -> dict[str, Any]:
  """Shallow merge overrides into base config."""
  merged = dict(base)
  merged.update(overrides)
  return merged
