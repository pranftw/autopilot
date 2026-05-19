"""Concurrent store operations regression tests.

Verifies that under fail-fast lock semantics, exactly one thread succeeds
for mutually exclusive store operations (stash/stash-pop), and the other
raises ConcurrentMutationError.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.tracking.file_lock import ConcurrentMutationError
from pathlib import Path
import threading


def _setup_store_with_snapshot(
  tmp_path: Path,
) -> tuple[FileStore, Path]:
  """Create workspace, store, forest, and initial snapshot for stash tests."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  src = ws / 'src'
  src.mkdir()
  (src / 'main.py').write_text('print("hello")\n', encoding='utf-8')

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  param = PathParameter(source=str(src), pattern='**/*.py')
  store = FileStore(config)
  store.register_parameters({'source': param})

  store.snapshot('root', 0)
  store.branch('stash-exp')
  store.snapshot('stash-exp', 1, context='initial')

  return store, ws


def test_concurrent_stash_operations(tmp_path: Path) -> None:
  """Two threads: one stash, other stash-pop; exactly one succeeds under fail-fast lock."""
  store, _ws = _setup_store_with_snapshot(tmp_path)

  store.stash(context='pre-test stash')

  results: dict[str, str] = {}
  errors: dict[str, BaseException] = {}
  barrier = threading.Barrier(2, timeout=5)

  def thread_stash() -> None:
    barrier.wait()
    try:
      store.stash(context='thread stash')
      results['stash'] = 'success'
    except ConcurrentMutationError as exc:
      errors['stash'] = exc
      results['stash'] = 'contention'
    except (StoreError, OSError) as exc:
      errors['stash'] = exc
      results['stash'] = 'error'

  def thread_pop() -> None:
    barrier.wait()
    try:
      store.stash_pop(context='thread pop')
      results['pop'] = 'success'
    except ConcurrentMutationError as exc:
      errors['pop'] = exc
      results['pop'] = 'contention'
    except (StoreError, OSError) as exc:
      errors['pop'] = exc
      results['pop'] = 'error'

  t1 = threading.Thread(target=thread_stash)
  t2 = threading.Thread(target=thread_pop)
  t1.start()
  t2.start()
  t1.join(timeout=10)
  t2.join(timeout=10)

  assert 'stash' in results, 'stash thread did not complete'
  assert 'pop' in results, 'pop thread did not complete'

  outcomes = [results['stash'], results['pop']]
  success_count = outcomes.count('success')
  contention_count = outcomes.count('contention')

  assert success_count + contention_count == 2, (
    f'unexpected error in concurrent stash: stash={results.get("stash")}, '
    f'pop={results.get("pop")}, errors={errors}'
  )
  assert success_count == 1, f'exactly one thread must succeed under fail-fast lock; got {outcomes}'
  assert contention_count == 1, (
    f'exactly one thread must see ConcurrentMutationError; got {outcomes}'
  )

  for key, exc in errors.items():
    assert isinstance(exc, ConcurrentMutationError), (
      f'thread {key} raised unexpected {type(exc).__name__}: {exc}'
    )

  stash_list = store.stash_list()
  assert isinstance(stash_list, list)
  refs = store.load_refs()
  assert 'stash-exp' in refs.get('branches', {})
