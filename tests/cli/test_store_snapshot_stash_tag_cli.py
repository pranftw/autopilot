"""Regression tests for store CLI integration fixes (Dogfood V3, plan 05).

Tests cover:
  - P0#4: forest-backed stash parameter registration from latest manifest
  - P0#6: snapshot context forwarding to manifests/reflog
  - P3#32: idempotent snapshots with --force escape hatch
  - P3#33: tag name validation UX (slash rejection, valid chars accepted)
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.store.helpers import register_parameters_from_latest_manifest
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.tracking.io import read_jsonl
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
from typing import Any
import pytest


def _make_workspace_with_snapshot(
  tmp_path: Path,
  experiment_id: str = 'exp-test',
) -> dict[str, Any]:
  """Create a workspace with a forest, store, and one snapshot.

  Returns dict with keys: workspace, config, store, forest, source_dir, param.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  source_dir = ws / 'src'
  source_dir.mkdir()
  (source_dir / 'main.py').write_text('print("hello")', encoding='utf-8')

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)

  param = PathParameter(source=str(source_dir), pattern='*.py')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot(experiment_id, 0, context='initial')

  forest = FileForest(store)
  tree = forest.create_tree('main')
  exp = Experiment(experiment_id=experiment_id, hypothesis='test')
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
    'source_dir': source_dir,
    'param': param,
  }


def _read_reflog(store: FileStore) -> list[dict[str, Any]]:
  """Read all reflog entries from the store."""
  path = store.config.store_path / 'reflog.jsonl'
  return read_jsonl(path, strict=False)


class TestStashFromForestStore:
  """P0#4: store stash/stash-pop via forest-backed store with parameter registration."""

  def test_stash_from_forest_store_with_prior_snapshot(self, tmp_path: Path) -> None:
    """With a prior snapshot (schema present), store stash succeeds."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    ws = ctx['workspace']

    result = run_cli(
      ws,
      ['--experiment', 'exp-test', 'store', 'stash'],
    )
    assert result['ok'] is True
    assert 'result' in result
    assert result['result']['entry_count'] > 0

  def test_stash_from_forest_store_no_schema_fails(self, tmp_path: Path) -> None:
    """Forest store + experiment with no snapshots: stash fails with clear message."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='no-snap-exp', hypothesis='test')
    exp.start()
    exp.complete(metrics={'score': 0.5})
    tree.add(Node(experiment=exp))
    forest.switch('main')
    forest.save()

    with pytest.raises((StoreError, SystemExit)):
      run_cli(ws, ['--experiment', 'no-snap-exp', 'store', 'stash'])

  def test_stash_pop_from_forest_store(self, tmp_path: Path) -> None:
    """After a successful stash, stash-pop restores state and empties the stack."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    ws = ctx['workspace']

    run_cli(ws, ['--experiment', 'exp-test', 'store', 'stash'])

    stash_list = run_cli_no_context(ws, ['store', 'stash-list'])
    assert stash_list['result']['count'] == 1

    pop_result = run_cli(ws, ['--experiment', 'exp-test', 'store', 'stash-pop'])
    assert pop_result['ok'] is True
    assert pop_result['result']['entry_count'] > 0

    stash_list_after = run_cli_no_context(ws, ['store', 'stash-list'])
    assert stash_list_after['result']['count'] == 0


class TestRegisterParametersFromManifest:
  """Unit tests for register_parameters_from_latest_manifest helper."""

  def test_registers_path_parameter(self, tmp_path: Path) -> None:
    """Parameters are registered from the manifest schema."""
    ctx = _make_workspace_with_snapshot(tmp_path)

    fresh_store = FileStore(ctx['config'])
    assert len(fresh_store._param_names) == 0

    register_parameters_from_latest_manifest(fresh_store, 'exp-test')
    assert 'source' in fresh_store._param_names
    assert isinstance(fresh_store._param_names['source'], PathParameter)

  def test_no_snapshots_raises(self, tmp_path: Path) -> None:
    """StoreError raised when no snapshots exist for the experiment."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    store._acquire_lock()
    try:
      refs = store.load_refs()
      refs.setdefault('branches', {})['empty-exp'] = {
        'latest_epoch': -1,
        'parent_id': None,
        'parent_epoch': None,
      }
      store.save_refs(refs)
    finally:
      store._release_lock()

    with pytest.raises(StoreError, match='no snapshots exist'):
      register_parameters_from_latest_manifest(store, 'empty-exp')

  def test_no_schema_in_manifest_raises(self, tmp_path: Path) -> None:
    """StoreError raised when the tip manifest has no parameter schema."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    source_dir = ws / 'src'
    source_dir.mkdir()
    (source_dir / 'f.txt').write_text('hello')

    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    param = PathParameter(source=str(source_dir), pattern='*')
    store.register_parameters({'source': param})
    store.snapshot('schema-exp', 0)

    manifest_path = config.snapshots_path / 'schema-exp' / 'epoch_0.json'
    import json

    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    del data['schema']
    manifest_path.write_text(json.dumps(data), encoding='utf-8')

    fresh_store = FileStore(config)
    with pytest.raises(StoreError, match='no parameter schema'):
      register_parameters_from_latest_manifest(fresh_store, 'schema-exp')

  def test_unsupported_type_name_raises(self, tmp_path: Path) -> None:
    """StoreError raised when schema contains an unsupported parameter type."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    source_dir = ws / 'src'
    source_dir.mkdir()
    (source_dir / 'f.txt').write_text('hello')

    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    param = PathParameter(source=str(source_dir), pattern='*')
    store.register_parameters({'source': param})
    store.snapshot('type-exp', 0)

    manifest_path = config.snapshots_path / 'type-exp' / 'epoch_0.json'
    import json

    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    data['schema']['parameters'][0]['type_name'] = 'ScalarParameter'
    manifest_path.write_text(json.dumps(data), encoding='utf-8')

    fresh_store = FileStore(config)
    with pytest.raises(StoreError, match='unsupported type'):
      register_parameters_from_latest_manifest(fresh_store, 'type-exp')


class TestSnapshotForwardsContext:
  """P0#6: store snapshot forwards --context to manifests and reflog."""

  def test_snapshot_forwards_context_to_reflog(self, tmp_path: Path) -> None:
    """Snapshot reflog entry contains the CLI --context value."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    ws = ctx['workspace']
    store = ctx['store']

    reflog_before = _read_reflog(store)
    snapshot_entries_before = [e for e in reflog_before if e['operation'] == 'snapshot']

    (ctx['source_dir'] / 'main.py').write_text('print("v2")', encoding='utf-8')
    run_cli(
      ws,
      [
        '--experiment',
        'exp-test',
        'store',
        'snapshot',
        '--source',
        str(ctx['source_dir']),
        '--pattern',
        '*.py',
      ],
    )

    reflog_after = _read_reflog(store)
    snapshot_entries_after = [e for e in reflog_after if e['operation'] == 'snapshot']
    new_entries = snapshot_entries_after[len(snapshot_entries_before) :]
    assert len(new_entries) >= 1
    assert new_entries[-1]['context'] == 'test'


class TestSnapshotIdempotent:
  """P3#32: idempotent snapshots skip when unchanged, --force overrides."""

  def test_snapshot_idempotent_no_changes(self, tmp_path: Path) -> None:
    """Second snapshot with no file changes is skipped (epoch unchanged)."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    ws = ctx['workspace']

    result = run_cli(
      ws,
      [
        '--experiment',
        'exp-test',
        'store',
        'snapshot',
        '--source',
        str(ctx['source_dir']),
        '--pattern',
        '*.py',
      ],
    )
    assert result['ok'] is True
    assert result['result']['skipped'] is True
    assert result['result']['epoch'] == 0

  def test_snapshot_force_bypasses_idempotent(self, tmp_path: Path) -> None:
    """With --force, a new epoch is created even when files are unchanged."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    ws = ctx['workspace']

    result = run_cli(
      ws,
      [
        '--experiment',
        'exp-test',
        'store',
        'snapshot',
        '--source',
        str(ctx['source_dir']),
        '--pattern',
        '*.py',
        '--force',
      ],
    )
    assert result['ok'] is True
    assert result['result']['skipped'] is False
    assert result['result']['epoch'] == 1


class TestTagValidation:
  """P3#33: tag name validation error message and valid chars."""

  def test_tag_slash_rejected_with_message(self, tmp_path: Path) -> None:
    """Tag name with '/' is rejected with a message listing allowed characters."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    ws = ctx['workspace']

    with pytest.raises(SystemExit):
      run_cli(
        ws,
        [
          '--experiment',
          'exp-test',
          'store',
          'tag',
          'create',
          'release/v1',
        ],
      )

  def test_tag_slash_error_mentions_allowed_chars(self, tmp_path: Path) -> None:
    """Error text for slash tag explicitly mentions allowed character classes."""
    from autopilot.core.store.types import validate_tag_name

    with pytest.raises(StoreError, match='Slashes') as exc_info:
      validate_tag_name('release/v1')
    assert 'ASCII letters' in str(exc_info.value)
    assert 'digits' in str(exc_info.value)

  def test_tag_valid_chars_accepted(self, tmp_path: Path) -> None:
    """Tag names using '.', '_', and '-' succeed when the target epoch exists."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    ws = ctx['workspace']

    for tag_name in ['v1.0', 'release_candidate', 'my-tag-2']:
      result = run_cli(
        ws,
        [
          '--experiment',
          'exp-test',
          'store',
          'tag',
          'create',
          tag_name,
        ],
      )
      assert result['ok'] is True
      assert result['result']['tag'] == tag_name
      assert result['result']['epoch'] == 0


class TestIdempotentSnapshotAPI:
  """API-level tests for FileStore.snapshot idempotent behavior."""

  def test_identical_content_returns_prior_manifest(self, tmp_path: Path) -> None:
    """When content is identical, snapshot returns the prior manifest."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    store = ctx['store']

    result = store.snapshot('exp-test', 1)
    assert result.epoch == 0

    refs = store.load_refs()
    assert refs['branches']['exp-test']['latest_epoch'] == 0

  def test_force_creates_new_epoch(self, tmp_path: Path) -> None:
    """With force=True, a new epoch is always created."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    store = ctx['store']

    result = store.snapshot('exp-test', 1, force=True)
    assert result.epoch == 1

    refs = store.load_refs()
    assert refs['branches']['exp-test']['latest_epoch'] == 1

  def test_changed_content_creates_new_epoch(self, tmp_path: Path) -> None:
    """When content changes, a new epoch is created normally."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    store = ctx['store']

    (ctx['source_dir'] / 'main.py').write_text('print("changed")', encoding='utf-8')
    result = store.snapshot('exp-test', 1)
    assert result.epoch == 1

    refs = store.load_refs()
    assert refs['branches']['exp-test']['latest_epoch'] == 1

  def test_context_only_change_requires_force(self, tmp_path: Path) -> None:
    """Different context but same files is skipped without --force."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    store = ctx['store']

    result = store.snapshot('exp-test', 1, context='different context')
    assert result.epoch == 0

    result_forced = store.snapshot('exp-test', 1, context='forced context', force=True)
    assert result_forced.epoch == 1
