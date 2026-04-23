"""Shared test fixtures for the autopilot test suite."""

from autopilot.core.models import Manifest, Result
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
import pytest


@pytest.fixture
def sample_manifest() -> Manifest:
  return Manifest(
    slug='test-exp',
    title='Test Experiment',
    current_epoch=1,
    idea='test idea',
    hypothesis='test hypothesis',
  )


@pytest.fixture
def sample_result() -> Result:
  return Result(
    metrics={'accuracy': 0.85, 'f1': 0.80},
    passed=True,
    summary='test run complete',
  )


@pytest.fixture
def tmp_experiment_dir(tmp_path: Path, sample_manifest: Manifest) -> Path:
  exp_dir = tmp_path / sample_manifest.slug
  exp_dir.mkdir()
  atomic_write_json(exp_dir / 'manifest.json', sample_manifest.to_dict())
  return exp_dir


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
  ws = tmp_path / 'workspace'
  ws.mkdir()
  autopilot = ws / 'autopilot'
  autopilot.mkdir()
  projects = autopilot / 'projects'
  projects.mkdir()
  experiments = autopilot / 'experiments'
  experiments.mkdir()
  return ws
