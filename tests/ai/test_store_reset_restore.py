"""Tests for Store.reset_and_restore() and CLI store branch --reset --restore.

Covers plan 09 of dogfood-v8: atomic branch tip reset plus working-tree
file restore/clear.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.tracking.io import read_jsonl
from pathlib import Path
import pytest


def _make_store(
  tmp_path: Path,
  files: dict[str, str] | None = None,
) -> tuple[FileStore, Path, PathParameter]:
  """Create a FileStore with a single PathParameter for testing."""
  if files is None:
    files = {'main.py': 'print("hello")\n'}
  src = tmp_path / 'src'
  src.mkdir(parents=True, exist_ok=True)
  for name, content in files.items():
    (src / name).write_text(content)
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path = tmp_path / '.autopilot'
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  return store, src, param


def _read_reflog(store: FileStore) -> list[dict]:
  """Read all reflog entries from the store."""
  return read_jsonl(store.config.store_path / 'reflog.jsonl', strict=False)


# -- 4.1 Store API -----------------------------------------------------------


class TestResetAndRestoreToEpoch:
  """reset_and_restore with a target epoch restores files and sets tip."""

  def test_reset_and_restore_to_epoch_1(self, tmp_path: Path) -> None:
    """3 epochs snapshotted; reset to epoch 1 restores that content."""
    store, src, _param = _make_store(tmp_path)

    (src / 'main.py').write_text('v0\n')
    store.snapshot('exp', 0)

    (src / 'main.py').write_text('v1\n')
    store.snapshot('exp', 1)

    (src / 'main.py').write_text('v2\n')
    store.snapshot('exp', 2)

    store.reset_and_restore('exp', 1)

    refs = store.load_refs()
    assert refs['branches']['exp']['latest_epoch'] == 1
    assert (src / 'main.py').read_text() == 'v1\n'
    assert refs['HEAD'] == 'exp'


class TestResetAndRestoreToNone:
  """reset_and_restore with epoch=None clears files and sets tip to -1."""

  def test_reset_and_restore_to_none_clears_files(self, tmp_path: Path) -> None:
    """Snapshot then reset_and_restore(None) clears tracked files."""
    store, src, _param = _make_store(tmp_path)

    (src / 'main.py').write_text('content\n')
    store.snapshot('exp', 0)

    assert (src / 'main.py').exists()
    store.reset_and_restore('exp', None)

    refs = store.load_refs()
    assert refs['branches']['exp']['latest_epoch'] == -1
    assert not (src / 'main.py').exists()


class TestResetAndRestoreErrors:
  """Error paths for reset_and_restore."""

  def test_reset_and_restore_nonexistent_branch_raises(self, tmp_path: Path) -> None:
    """StoreError mentions branch name when branch is missing."""
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp', 0)

    with pytest.raises(StoreError, match='missing-exp'):
      store.reset_and_restore('missing-exp', 0)

  def test_reset_and_restore_nonexistent_epoch_raises(self, tmp_path: Path) -> None:
    """StoreError mentions epoch number when epoch does not exist."""
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp', 0)

    with pytest.raises(StoreError, match='99'):
      store.reset_and_restore('exp', 99)

  def test_reset_and_restore_epoch_beyond_tip_raises(self, tmp_path: Path) -> None:
    """StoreError when epoch exceeds current latest_epoch."""
    store, src, _param = _make_store(tmp_path)
    (src / 'main.py').write_text('v0\n')
    store.snapshot('exp', 0)
    (src / 'main.py').write_text('v1\n')
    store.snapshot('exp', 1)
    (src / 'main.py').write_text('v2\n')
    store.snapshot('exp', 2)

    with pytest.raises(StoreError, match='5'):
      store.reset_and_restore('exp', 5)


class TestResetAndRestoreReflog:
  """Reflog entries produced by reset_and_restore."""

  def test_reset_and_restore_reflog_entries(self, tmp_path: Path) -> None:
    """Last two reflog entries are reset_branch then checkout."""
    store, src, _param = _make_store(tmp_path)
    (src / 'main.py').write_text('v0\n')
    store.snapshot('exp', 0)
    (src / 'main.py').write_text('v1\n')
    store.snapshot('exp', 1)

    store.reset_and_restore('exp', 0)

    entries = _read_reflog(store)
    assert len(entries) >= 2
    last_two = entries[-2:]
    assert last_two[0]['operation'] == 'reset_branch'
    assert last_two[0]['experiment_id'] == 'exp'
    assert last_two[0]['new_epoch'] == 0
    assert last_two[1]['operation'] == 'checkout'
    assert last_two[1]['experiment_id'] == 'exp'
    assert last_two[1]['new_epoch'] == 0

  def test_reset_and_restore_context_threaded(self, tmp_path: Path) -> None:
    """Both reflog entries carry the provided context string."""
    store, src, _param = _make_store(tmp_path)
    (src / 'main.py').write_text('v0\n')
    store.snapshot('exp', 0)

    store.reset_and_restore('exp', 0, context='rewind for re-run')

    entries = _read_reflog(store)
    last_two = entries[-2:]
    assert last_two[0]['context'] == 'rewind for re-run'
    assert last_two[1]['context'] == 'rewind for re-run'


# -- 4.2 CLI -----------------------------------------------------------------


class TestCLIStoreBranchResetRestore:
  """CLI integration tests for store branch --reset --restore."""

  def test_cli_store_branch_reset_restore(self, tmp_path: Path) -> None:
    """--reset --restore --epoch 1 restores epoch 1 files and sets tip."""
    from autopilot.ai.forest import FileForest
    from autopilot.core.experiment import Experiment
    from autopilot.core.node import Node
    from tests.cli.conftest import run_cli

    ws = tmp_path / 'ws'
    ws.mkdir()
    src = ws / 'src'
    src.mkdir()
    (src / 'main.py').write_text('v0\n')

    config = AutoPilotConfig(workspace=ws)
    store = FileStore(config)
    param = PathParameter(source=str(src), pattern='*')
    store.register_parameters({'source': param})

    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')

    node = Node(experiment=Experiment(experiment_id='exp-1'))
    tree.add(node)
    forest.save()

    (src / 'main.py').write_text('v0\n')
    store.snapshot('exp-1', 0)
    (src / 'main.py').write_text('v1\n')
    store.snapshot('exp-1', 1)
    (src / 'main.py').write_text('v2\n')
    store.snapshot('exp-1', 2)

    result = run_cli(
      ws,
      [
        '--experiment',
        'exp-1',
        '--epoch',
        '1',
        'store',
        'branch',
        '--reset',
        '--restore',
        '--source',
        str(src),
      ],
    )

    assert result.get('ok') is True
    payload = result['result']
    assert payload['experiment_id'] == 'exp-1'
    assert payload['reset'] is True
    assert payload['restore'] is True
    assert payload['epoch'] == 1

    refs = store.load_refs()
    assert refs['branches']['exp-1']['latest_epoch'] == 1
    assert (src / 'main.py').read_text() == 'v1\n'

  def test_cli_store_branch_reset_restore_no_epoch(self, tmp_path: Path) -> None:
    """--reset --restore without --epoch clears files and sets tip to -1."""
    from autopilot.ai.forest import FileForest
    from autopilot.core.experiment import Experiment
    from autopilot.core.node import Node
    from tests.cli.conftest import run_cli

    ws = tmp_path / 'ws'
    ws.mkdir()
    src = ws / 'src'
    src.mkdir()
    (src / 'main.py').write_text('content\n')

    config = AutoPilotConfig(workspace=ws)
    store = FileStore(config)
    param = PathParameter(source=str(src), pattern='*')
    store.register_parameters({'source': param})

    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')

    node = Node(experiment=Experiment(experiment_id='exp-1'))
    tree.add(node)
    forest.save()

    store.snapshot('exp-1', 0)

    result = run_cli(
      ws,
      [
        '--experiment',
        'exp-1',
        'store',
        'branch',
        '--reset',
        '--restore',
        '--source',
        str(src),
      ],
    )

    assert result.get('ok') is True
    payload = result['result']
    assert payload['epoch'] is None

    refs = store.load_refs()
    assert refs['branches']['exp-1']['latest_epoch'] == -1
    assert not (src / 'main.py').exists()

  def test_cli_restore_requires_reset(self, tmp_path: Path) -> None:
    """--restore without --reset exits non-zero with actionable message."""
    from autopilot.ai.forest import FileForest
    from autopilot.core.experiment import Experiment
    from autopilot.core.node import Node
    from tests.cli.conftest import run_cli

    ws = tmp_path / 'ws'
    ws.mkdir()
    src = ws / 'src'
    src.mkdir()
    (src / 'main.py').write_text('content\n')

    config = AutoPilotConfig(workspace=ws)
    store = FileStore(config)
    param = PathParameter(source=str(src), pattern='*')
    store.register_parameters({'source': param})

    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')

    node = Node(experiment=Experiment(experiment_id='exp-1'))
    tree.add(node)
    forest.save()

    store.snapshot('exp-1', 0)

    with pytest.raises(SystemExit):
      run_cli(
        ws,
        [
          '--experiment',
          'exp-1',
          'store',
          'branch',
          '--restore',
          '--source',
          str(src),
        ],
      )
