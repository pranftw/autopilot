"""Tests for manifest attestation (digest verification) on tags.

Covers TagEntry.manifest_digest field, digest computation at tag time,
verify_tag match/mismatch/pre-attestation paths, canonical JSON determinism,
and TagEntry round-trip serialization with the new field.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.ai.store.peripherals import compute_manifest_digest
from autopilot.ai.store_lock import hash_content
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.store.types import TagEntry
from pathlib import Path
from typing import Any
import json
import pytest


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
  """Workspace root with a parameter directory."""
  ws = tmp_path / 'project'
  ws.mkdir()
  prompts = ws / 'prompts'
  prompts.mkdir()
  (prompts / 'system.txt').write_text('you are helpful', encoding='utf-8')
  return ws


@pytest.fixture
def config(workspace: Path) -> AutoPilotConfig:
  """AutoPilotConfig rooted in the workspace."""
  return AutoPilotConfig(workspace=workspace)


@pytest.fixture
def store_with_snapshot(
  config: AutoPilotConfig,
  workspace: Path,
) -> tuple[FileStore, str]:
  """FileStore with one experiment branch and one snapshot at epoch 0.

  Returns:
    Tuple of (store, experiment_id).
  """
  store = FileStore(config)
  param = PathParameter(source=str(workspace / 'prompts'), pattern='*')
  store.register_parameters({'prompts': param})
  exp_id = 'exp-001'
  store.snapshot(exp_id, 0, context='initial snapshot')
  return store, exp_id


class TestTagStoresDigest:
  """test_tag_stores_digest: after tag(), persisted tag includes non-None manifest_digest."""

  def test_tag_stores_digest(
    self,
    store_with_snapshot: tuple[FileStore, str],
  ) -> None:
    store, exp_id = store_with_snapshot
    store.tag('v1.0', exp_id, 0, context='release')

    tag_entry = store.get_tag('v1.0')
    assert tag_entry is not None
    assert tag_entry.manifest_digest is not None
    assert len(tag_entry.manifest_digest) == 64

    manifest = store.load_snapshot(exp_id, 0)
    expected_digest = compute_manifest_digest(manifest)
    assert tag_entry.manifest_digest == expected_digest


class TestTagVerifyMatch:
  """test_tag_verify_match: unmodified manifest -> verified: True."""

  def test_tag_verify_match(
    self,
    store_with_snapshot: tuple[FileStore, str],
  ) -> None:
    store, exp_id = store_with_snapshot
    store.tag('v1.0', exp_id, 0, context='release')

    result = store.verify_tag('v1.0')
    assert result == {'verified': True}


class TestTagVerifyMismatch:
  """test_tag_verify_mismatch: mutated manifest -> verified: False with expected/actual."""

  def test_tag_verify_mismatch(
    self,
    store_with_snapshot: tuple[FileStore, str],
    config: AutoPilotConfig,
  ) -> None:
    store, exp_id = store_with_snapshot
    store.tag('v1.0', exp_id, 0, context='release')

    snap_path = config.store_path / 'snapshots' / exp_id / 'epoch_0.json'
    original_data = json.loads(snap_path.read_text(encoding='utf-8'))
    original_data['context'] = 'tampered context'
    snap_path.write_text(
      json.dumps(original_data, indent=2),
      encoding='utf-8',
    )

    result = store.verify_tag('v1.0')
    assert result['verified'] is False
    assert result['reason'] == 'digest mismatch'
    assert 'expected' in result
    assert 'actual' in result
    assert result['expected'] != result['actual']
    assert len(result['expected']) == 64
    assert len(result['actual']) == 64


class TestTagVerifyMissingTagRaises:
  """test_tag_verify_missing_tag_raises: unknown name -> StoreError."""

  def test_tag_verify_missing_tag_raises(
    self,
    store_with_snapshot: tuple[FileStore, str],
  ) -> None:
    store, _ = store_with_snapshot
    with pytest.raises(StoreError, match='not found'):
      store.verify_tag('nonexistent')


class TestTagPreAttestationNoDigest:
  """test_tag_pre_attestation_no_digest: tag without digest -> 'no digest available'."""

  def test_tag_pre_attestation_no_digest(
    self,
    store_with_snapshot: tuple[FileStore, str],
    config: AutoPilotConfig,
  ) -> None:
    store, exp_id = store_with_snapshot

    store.tag('v1.0', exp_id, 0, context='release')

    refs_path = config.store_path / 'refs.json'
    refs_data = json.loads(refs_path.read_text(encoding='utf-8'))
    del refs_data['tags']['v1.0']['manifest_digest']
    refs_path.write_text(json.dumps(refs_data, indent=2), encoding='utf-8')

    result = store.verify_tag('v1.0')
    assert result == {'verified': False, 'reason': 'no digest available'}


class TestTagEntryRoundtrip:
  """test_tag_entry_roundtrip: to_dict/from_dict preserves manifest_digest."""

  def test_tag_entry_roundtrip_with_digest(self) -> None:
    entry = TagEntry(
      name='v1.0',
      experiment_id='exp-001',
      epoch=5,
      context='release',
      timestamp='2026-01-01T00:00:00+00:00',
      manifest_digest='abcd1234' * 8,
    )
    data = entry.to_dict()
    restored = TagEntry.from_dict(data)
    assert restored == entry
    assert restored.manifest_digest == entry.manifest_digest

  def test_tag_entry_roundtrip_without_digest(self) -> None:
    entry = TagEntry(
      name='v1.0',
      experiment_id='exp-001',
      epoch=5,
      context='release',
      timestamp='2026-01-01T00:00:00+00:00',
    )
    data = entry.to_dict()
    restored = TagEntry.from_dict(data)
    assert restored == entry
    assert restored.manifest_digest is None


class TestTagEntryFromDictMissingDigest:
  """test_tag_entry_from_dict_missing_digest: missing key -> None field."""

  def test_tag_entry_from_dict_missing_digest(self) -> None:
    data: dict[str, Any] = {
      'name': 'v1.0',
      'experiment_id': 'exp-001',
      'epoch': 5,
      'context': 'release',
      'timestamp': '2026-01-01T00:00:00+00:00',
    }
    entry = TagEntry.from_dict(data)
    assert entry.manifest_digest is None


class TestTagDigestCanonicalJson:
  """test_tag_digest_canonical_json: key order/whitespace do not change digest."""

  def test_canonical_json_key_order_independent(self) -> None:
    """Changing key order in input dict does not change digest."""
    dict_a = {'epoch': 0, 'timestamp': 'ts', 'entries': {}, 'context': None}
    dict_b = {'context': None, 'entries': {}, 'timestamp': 'ts', 'epoch': 0}

    canonical_a = json.dumps(dict_a, sort_keys=True, separators=(',', ':'))
    canonical_b = json.dumps(dict_b, sort_keys=True, separators=(',', ':'))
    assert canonical_a == canonical_b

    digest_a = hash_content(canonical_a)
    digest_b = hash_content(canonical_b)
    assert digest_a == digest_b

  def test_canonical_json_contract(
    self,
    store_with_snapshot: tuple[FileStore, str],
  ) -> None:
    """Verify exact canonical JSON contract used by compute_manifest_digest."""
    store, exp_id = store_with_snapshot
    manifest = store.load_snapshot(exp_id, 0)
    manifest_dict = manifest.to_dict()

    canonical = json.dumps(manifest_dict, sort_keys=True, separators=(',', ':'))
    expected_digest = hash_content(canonical)

    actual_digest = compute_manifest_digest(manifest)
    assert actual_digest == expected_digest

  def test_pretty_printed_manifest_still_verifies(
    self,
    store_with_snapshot: tuple[FileStore, str],
    config: AutoPilotConfig,
  ) -> None:
    """Verify that pretty-printed manifest on disk still yields same digest.

    The digest is computed from the manifest dict, not raw file bytes.
    """
    store, exp_id = store_with_snapshot
    store.tag('v1.0', exp_id, 0, context='release')

    snap_path = config.store_path / 'snapshots' / exp_id / 'epoch_0.json'
    data = json.loads(snap_path.read_text(encoding='utf-8'))
    snap_path.write_text(json.dumps(data, indent=4), encoding='utf-8')

    result = store.verify_tag('v1.0')
    assert result == {'verified': True}
