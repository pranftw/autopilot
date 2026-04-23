"""Canonical config helpers for JSON overrides and path resolution."""

from autopilot.core.errors import ConfigError, TrackingError
from autopilot.tracking.io import read_json
from pathlib import Path
from typing import Any
import autopilot.core.paths as paths


def load_json(path: Path) -> dict[str, Any]:
  """Load a JSON file and return its contents as a dict."""
  try:
    data = read_json(path)
  except TrackingError as exc:
    raise ConfigError(str(exc)) from exc
  if data is None:
    raise ConfigError(f'config file not found: {path}')
  return data


def resolve_experiment_dir(
  workspace: Path,
  slug: str,
  project: str | None = None,
) -> Path:
  """Return the experiment directory path for a given slug."""
  return paths.experiment(workspace, slug, project)


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
