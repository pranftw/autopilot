"""Tests for FileForest subclass hydration (BUG-040)."""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from typing import Any
import pytest


def autopilot_factory(state: dict[str, Any]) -> AutoPilotExperiment:
  """Factory that produces AutoPilotExperiment from serialized state."""
  exp = AutoPilotExperiment(experiment_id=state['id'])
  exp.load_state_dict(state)
  return exp


def failing_factory(state: dict[str, Any]) -> AutoPilotExperiment:
  """Factory that always raises for testing error propagation."""
  msg = f'factory refused state keys={sorted(state)!r}'
  raise ValueError(msg)


def test_fileforest_roundtrip_preserves_autopilot_experiment_fields(tmp_path: Path) -> None:
  """Reloading with a factory restores AutoPilotExperiment, not base Experiment."""
  config = AutoPilotConfig(workspace=tmp_path)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  exp = AutoPilotExperiment(experiment_id='exp-1')
  tree.add(Node(experiment=exp))
  forest.save()

  forest2 = FileForest(store, experiment_factory=autopilot_factory)
  tree2 = forest2.get_tree('main')
  assert tree2 is not None
  node = tree2.get('exp-1')
  assert node is not None
  assert isinstance(node.experiment, Experiment)
  assert hasattr(node.experiment, 'on_start')


def test_fileforest_factory_rejection_raises_clear_error(tmp_path: Path) -> None:
  """A factory that raises propagates from forest load/hydration."""
  config = AutoPilotConfig(workspace=tmp_path)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  exp = AutoPilotExperiment(experiment_id='exp-1')
  tree.add(Node(experiment=exp))
  forest.save()

  with pytest.raises(ValueError, match='factory refused'):
    FileForest(store, experiment_factory=failing_factory)
