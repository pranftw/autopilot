"""Tests for FileStore.doctor() and CLI store doctor command.

Covers:
  - Healthy store reports healthy (2.1)
  - Shard directory not counted as blob (2.1)
  - Missing blob detected (4.1)
  - Orphan blob detected (4.1)
  - Corrupt manifest surfaces as unhealthy (2.2)
  - CLI store doctor no context required (2.2)
  - CLI store doctor JSON envelope (4.2)
  - refs.json branch consistency checks
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
import json


def _make_store_with_snapshot(tmp_path: Path) -> tuple[FileStore, AutoPilotConfig]:
  """Create a FileStore with a single snapshot for testing.

  Returns:
    Tuple of (store, config) after one snapshot at epoch 0.
  """
  prompts_dir = tmp_path / 'prompts'
  prompts_dir.mkdir()
  (prompts_dir / 'main.txt').write_text('hello world')

  config = AutoPilotConfig(workspace=tmp_path)
  param = PathParameter(source=str(prompts_dir), pattern='*.txt')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('exp-001', 0)
  FileForest(store).save()
  return store, config


# 2.1 -- FileStore.doctor() library tests


class TestStoreDoctorHealthyStore:
  """test_store_doctor_healthy_store_reports_healthy -- empty or single-snapshot fixture."""

  def test_healthy_store_with_snapshot(self, tmp_path: Path) -> None:
    """Single-snapshot store reports healthy with zero orphans."""
    store, _ = _make_store_with_snapshot(tmp_path)
    report = store.doctor_report()
    assert report['healthy'] is True
    assert report['orphan_count'] == 0
    assert report['manifest_errors'] == []
    assert report['missing_blobs'] == []
    assert report['orphan_blobs'] == []
    assert report['refs_issues'] == []

  def test_healthy_empty_store(self, tmp_path: Path) -> None:
    """Empty store with no snapshots is healthy."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    FileForest(store).save()
    report = store.doctor_report()
    assert report['healthy'] is True
    assert report['orphan_count'] == 0

  def test_healthy_store_multiple_epochs(self, tmp_path: Path) -> None:
    """Store with multiple sequential snapshots is healthy."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'main.txt').write_text('v1')

    config = AutoPilotConfig(workspace=tmp_path)
    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})

    store.snapshot('exp-001', 0)
    (prompts_dir / 'main.txt').write_text('v2')
    store.snapshot('exp-001', 1)

    FileForest(store).save()
    report = store.doctor_report()
    assert report['healthy'] is True
    assert report['orphan_count'] == 0


class TestStoreDoctorShardDirectory:
  """test_store_doctor_shard_directory_not_counted_as_blob."""

  def test_shard_dir_not_counted_as_blob(self, tmp_path: Path) -> None:
    """Empty shard directory does not produce false orphan entries."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    store.register_parameters({})
    FileForest(store).save()

    shard = config.objects_path / 'ab'
    shard.mkdir(parents=True, exist_ok=True)

    report = store.doctor_report()
    assert report['healthy'] is True
    assert report['orphan_blobs'] == []
    assert report['orphan_count'] == 0

  def test_subdir_inside_shard_not_counted(self, tmp_path: Path) -> None:
    """Subdirectory inside a shard is not counted as a blob."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    store.register_parameters({})
    FileForest(store).save()

    shard = config.objects_path / 'ab'
    shard.mkdir(parents=True, exist_ok=True)
    nested = shard / 'nested_dir'
    nested.mkdir()

    report = store.doctor_report()
    assert report['orphan_blobs'] == []
    assert report['orphan_count'] == 0

  def test_non_hex_shard_ignored(self, tmp_path: Path) -> None:
    """Directory under objects/ with non-hex name is ignored."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    store.register_parameters({})
    FileForest(store).save()

    bogus = config.objects_path / 'zz'
    bogus.mkdir(parents=True, exist_ok=True)
    (bogus / 'somefile').write_text('data')

    report = store.doctor_report()
    assert report['orphan_blobs'] == []
    assert report['orphan_count'] == 0


# 4.1 -- missing and orphan blob detection


class TestStoreDoctorMissingBlob:
  """test_store_doctor_missing_blob_detected."""

  def test_missing_blob_detected(self, tmp_path: Path) -> None:
    """Manifest references digest not on disk; digest in missing_blobs."""
    store, config = _make_store_with_snapshot(tmp_path)

    manifest = store.load_snapshot('exp-001', 0)
    referenced_digest = next(iter(manifest.entries.values())).digest

    blob_path = config.objects_path / referenced_digest[:2] / referenced_digest[2:]
    blob_path.unlink()

    report = store.doctor_report()
    assert report['healthy'] is False
    assert referenced_digest in report['missing_blobs']

  def test_multiple_missing_blobs(self, tmp_path: Path) -> None:
    """Multiple missing blobs all appear in the report."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'a.txt').write_text('file a')
    (prompts_dir / 'b.txt').write_text('file b')

    config = AutoPilotConfig(workspace=tmp_path)
    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})
    manifest = store.snapshot('exp-001', 0)
    FileForest(store).save()

    for entry in manifest.entries.values():
      blob_path = config.objects_path / entry.digest[:2] / entry.digest[2:]
      blob_path.unlink()

    report = store.doctor_report()
    assert report['healthy'] is False
    assert len(report['missing_blobs']) == len(manifest.entries)


class TestStoreDoctorOrphanBlob:
  """test_store_doctor_orphan_blob_detected."""

  def test_orphan_blob_detected(self, tmp_path: Path) -> None:
    """Extra file under objects/ab/ not referenced; listed and orphan_count >= 1."""
    store, config = _make_store_with_snapshot(tmp_path)

    orphan_digest = 'ff' + 'a1' * 31
    orphan_dir = config.objects_path / orphan_digest[:2]
    orphan_dir.mkdir(parents=True, exist_ok=True)
    (orphan_dir / orphan_digest[2:]).write_text('orphan content')

    report = store.doctor_report()
    assert report['healthy'] is True
    assert orphan_digest in report['orphan_blobs']
    assert report['orphan_count'] >= 1

  def test_multiple_orphan_blobs(self, tmp_path: Path) -> None:
    """Multiple orphan blobs all appear in the report."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    store.register_parameters({})
    FileForest(store).save()

    orphans = []
    for i in range(3):
      digest = f'{i:02x}' + 'bb' * 31
      shard = config.objects_path / digest[:2]
      shard.mkdir(parents=True, exist_ok=True)
      (shard / digest[2:]).write_text(f'orphan {i}')
      orphans.append(digest)

    report = store.doctor_report()
    assert report['orphan_count'] == 3
    for digest in orphans:
      assert digest in report['orphan_blobs']


# 2.2 -- corrupt manifest detection


class TestStoreDoctorCorruptManifest:
  """test_store_doctor_reports_corrupt_manifest."""

  def test_corrupt_manifest_reports_unhealthy(self, tmp_path: Path) -> None:
    """Malformed JSON in a snapshot surfaces as healthy=False with manifest_errors."""
    store, config = _make_store_with_snapshot(tmp_path)

    snap_dir = config.store_path / 'snapshots' / 'exp-001'
    snap_files = list(snap_dir.glob('*.json'))
    assert len(snap_files) == 1
    snap_files[0].write_text('not valid json')

    report = store.doctor_report()
    assert report['healthy'] is False
    assert len(report['manifest_errors']) >= 1

  def test_empty_json_object_manifest(self, tmp_path: Path) -> None:
    """Empty JSON object {} in snapshot is detected as corrupt."""
    store, config = _make_store_with_snapshot(tmp_path)

    snap_dir = config.store_path / 'snapshots' / 'exp-001'
    snap_files = list(snap_dir.glob('*.json'))
    snap_files[0].write_text(json.dumps({}))

    report = store.doctor_report()
    assert report['healthy'] is False
    assert len(report['manifest_errors']) >= 1


# refs consistency checks


class TestStoreDoctorRefsConsistency:
  """Branch consistency checks in refs.json."""

  def test_missing_tip_snapshot_reported(self, tmp_path: Path) -> None:
    """Branch tip pointing to non-existent snapshot surfaces as refs_issues."""
    store, config = _make_store_with_snapshot(tmp_path)

    snap_path = config.store_path / 'snapshots' / 'exp-001' / 'epoch_0.json'
    snap_path.unlink()

    report = store.doctor_report()
    assert report['healthy'] is False
    assert any('tip epoch' in issue for issue in report['refs_issues'])

  def test_corrupt_refs_json_reported(self, tmp_path: Path) -> None:
    """Corrupt refs.json is reported as refs_issues."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    store.register_parameters({})
    FileForest(store).save()

    config.refs_file.write_text('broken json')
    report = store.doctor_report()
    assert report['healthy'] is False
    assert any('refs.json' in issue for issue in report['refs_issues'])


# 4.2 -- CLI tests


class TestStoreDoctorCLI:
  """CLI store doctor subcommand tests."""

  def test_store_doctor_cli_no_context_required(self, tmp_path: Path) -> None:
    """store doctor runs via run_cli_no_context with exit 0 (no --context needed)."""
    from autopilot.ai.forest import FileForest

    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.save()

    envelope = run_cli_no_context(ws, ['store', 'doctor'])
    assert envelope.get('ok') is True

  def test_store_doctor_json_envelope_ok(self, tmp_path: Path) -> None:
    """--json prints envelope with ok: True for a healthy store."""
    from autopilot.ai.forest import FileForest

    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.save()

    envelope = run_cli_no_context(ws, ['store', 'doctor'])
    assert envelope.get('ok') is True
    assert envelope['result']['healthy'] is True

  def test_store_doctor_reports_corrupt_via_cli(self, tmp_path: Path) -> None:
    """CLI reports healthy=False when a manifest is corrupt."""
    from autopilot.ai.forest import FileForest

    ws = tmp_path / 'ws'
    ws.mkdir()

    prompts_dir = ws / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'main.txt').write_text('data')

    config = AutoPilotConfig(workspace=ws)
    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('exp-001', 0)

    forest = FileForest(store)
    forest.save()

    snap_dir = config.store_path / 'snapshots' / 'exp-001'
    snap_files = list(snap_dir.glob('*.json'))
    snap_files[0].write_text('bad json')

    envelope = run_cli_no_context(ws, ['store', 'doctor'])
    assert envelope.get('ok') is True
    assert envelope['result']['healthy'] is False
    assert len(envelope['result'].get('manifest_errors', [])) >= 1
