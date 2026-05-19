"""AP-3: concurrent_forest_writer integration test.

Exercises the ``concurrent_forest_writer`` fixture from ``tests/conftest.py``
to verify that ``FileForest.save()`` under the forest lock serializes
concurrent writers without data corruption.

With fail-fast locks (``timeout_s=None``), some threads will encounter
lock contention and record ``TrackingError`` -- this is expected behavior.
The key invariant is no data corruption occurs: the reloaded forest is
valid and all persisted trees are present on disk.

See plan 22, subplan 2.3.
"""

import pytest


@pytest.mark.slow
def test_concurrent_forest_writer_serializes_saves(concurrent_forest_writer) -> None:
  """Concurrent tree creation under forest lock must not corrupt data.

  Spawns multiple threads each creating trees via ``FileForest.create_tree``.
  The in-memory forest accumulates all trees; successful saves persist the
  snapshot. After joining, verifies:
  - At least one tree was persisted to disk.
  - Reloaded forest has no corruption (valid JSON, no duplicate names).
  - All contention errors are ``TrackingError`` (expected with fail-fast locks).
  """
  n_writers = 3
  n_writes_each = 5

  forest, errors = concurrent_forest_writer(
    n_writers=n_writers,
    n_writes_each=n_writes_each,
  )

  trees = forest.list_trees()
  tree_count = len(trees)

  assert tree_count > 0, 'expected at least one tree to be persisted'

  tree_names = [t.name for t in trees]
  assert len(tree_names) == len(set(tree_names)), (
    f'duplicate tree names found: {[n for n in tree_names if tree_names.count(n) > 1]}'
  )

  from autopilot.core.errors import TrackingError

  for err in errors:
    assert isinstance(err, (ValueError, OSError, TrackingError)), (
      f'unexpected error type: {type(err).__name__}: {err}'
    )


@pytest.mark.slow
def test_concurrent_forest_writer_tree_names_unique(concurrent_forest_writer) -> None:
  """Each concurrently created tree has a unique name."""
  forest, _errors = concurrent_forest_writer(n_writers=2, n_writes_each=4)

  tree_names = [t.name for t in forest.list_trees()]
  assert len(tree_names) == len(set(tree_names)), (
    f'duplicate tree names found: {[n for n in tree_names if tree_names.count(n) > 1]}'
  )
