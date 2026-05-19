"""Shared test fixtures for the autopilot test suite."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import TrackingError
from autopilot.core.experiment import Experiment
from autopilot.core.models import Result
from autopilot.core.node import Node
from pathlib import Path
from typing import Any
import pytest
import threading


@pytest.fixture
def sample_experiment() -> Experiment:
  return Experiment(experiment_id='test-exp', hypothesis='test hypothesis')


@pytest.fixture
def sample_result() -> Result:
  return Result(
    metrics={'accuracy': 0.85, 'f1': 0.80},
    summary='test run complete',
  )


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


@pytest.fixture
def workspace_with_store_and_forest(tmp_path: Path) -> dict[str, Any]:
  """Workspace with a FileStore and FileForest ready for CLI and integration tests.

  Creates the on-disk layout, a store, a forest with one active tree
  containing a completed experiment, and exposes the config.

  Returns:
    Dict with keys ``workspace``, ``config``, ``store``, ``forest``.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  exp = Experiment(experiment_id='seed-exp', hypothesis='seed')
  exp.start()
  exp.complete(metrics={'score': 0.75})
  tree.add(Node(experiment=exp))
  forest.switch('main')
  forest.save()
  return {
    'workspace': ws,
    'config': config,
    'store': store,
    'forest': forest,
  }


@pytest.fixture
def concurrent_forest_writer(tmp_path: Path):
  """Helper fixture that spawns concurrent forest writers under the forest lock.

  Returns a callable ``run(n_writers, n_writes_each)`` that launches
  ``n_writers`` threads, each performing ``n_writes_each`` tree-create +
  save operations.  After joining, returns ``(forest, errors)`` where
  ``errors`` is a list of exceptions raised by any thread.
  """

  def _run(
    n_writers: int = 3,
    n_writes_each: int = 5,
  ) -> tuple[FileForest, list[Exception]]:
    ws = tmp_path / 'concurrent_ws'
    ws.mkdir(exist_ok=True)
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    errors: list[Exception] = []
    lock = threading.Lock()

    def writer(writer_id: int) -> None:
      for i in range(n_writes_each):
        try:
          tree_name = f'writer-{writer_id}-tree-{i}'
          forest.create_tree(tree_name)
        except (ValueError, OSError, TrackingError) as exc:
          with lock:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    fresh = FileForest(store)
    return fresh, errors

  return _run
