"""Tests for FileForest concurrent save with file locking."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import TrackingError
from autopilot.core.node import Node
from autopilot.tracking.file_lock import AutopilotFileLock, ConcurrentMutationError
from tests.core.conftest import completed_exp
import pytest
import threading
import time


@pytest.fixture
def config(tmp_path):
  return AutoPilotConfig(workspace=tmp_path)


@pytest.fixture
def store(config):
  return FileStore(config)


MAX_RETRIES = 10


def test_concurrent_forest_saves_preserve_all_trees(store):
  """Two threads saving different trees -- both present after reload.

  Each thread retries on lock contention (fail-fast default), reloading
  state before each attempt to avoid lost updates.
  """
  barrier = threading.Barrier(2, timeout=5)
  errors: list[Exception] = []

  def save_tree(name: str, exp_id: str):
    barrier.wait()
    for _attempt in range(MAX_RETRIES):
      try:
        ff = FileForest(store)
        tree = ff.create_tree(name)
        exp = completed_exp(exp_id, {'accuracy': 0.5})
        tree.add(Node(experiment=exp))
      except (TrackingError, ValueError):
        time.sleep(0.01)
      else:
        return
    errors.append(RuntimeError(f'failed to save tree {name!r} after {MAX_RETRIES} retries'))

  t1 = threading.Thread(target=save_tree, args=('tree-a', 'exp-a'))
  t2 = threading.Thread(target=save_tree, args=('tree-b', 'exp-b'))
  t1.start()
  t2.start()
  t1.join(timeout=10)
  t2.join(timeout=10)

  assert not errors, f'thread errors: {errors}'

  ff_final = FileForest(store)
  tree_names = {t.name for t in ff_final.list_trees()}
  assert 'tree-a' in tree_names
  assert 'tree-b' in tree_names


def test_forest_save_raises_on_lock_contention(store):
  """Lock held externally -- forest save raises ConcurrentMutationError."""
  lock_path = store.config.store_path / 'forest.lock'
  lock_path.parent.mkdir(parents=True, exist_ok=True)

  ff = FileForest(store)

  external_lock = AutopilotFileLock(lock_path)
  external_lock.acquire()
  try:
    with pytest.raises(ConcurrentMutationError, match='concurrent mutation'):
      ff.save()
  finally:
    external_lock.release()


def test_forest_trees_persist_after_reload(store):
  """Create trees, save, reload in new FileForest -- trees visible."""
  ff1 = FileForest(store)
  ff1.create_tree('alpha', description='first')
  ff1.create_tree('beta', description='second')

  ff2 = FileForest(store)
  assert len(ff2.list_trees()) == 2
  alpha = ff2.get_tree('alpha')
  assert alpha is not None
  assert alpha.description == 'first'
  beta = ff2.get_tree('beta')
  assert beta is not None
  assert beta.description == 'second'


def test_forest_lock_path(store):
  """Lock file is at store_path / 'forest.lock'."""
  expected = store.config.store_path / 'forest.lock'

  lock = AutopilotFileLock(expected)
  lock.acquire()
  try:
    ff = FileForest(store)
    with pytest.raises(ConcurrentMutationError, match='concurrent mutation'):
      ff.save()
  finally:
    lock.release()
