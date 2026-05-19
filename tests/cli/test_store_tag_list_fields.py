"""Tests for ``store tag list`` JSON manifest_digest field (Sub-plan 03, section 2.3).

Verifies that each tag row in the JSON output includes ``manifest_digest``:
hex string when present, ``null`` for pre-attestation tags.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
from typing import Any


def _make_workspace_with_tag(tmp_path: Path) -> dict[str, Any]:
  """Create a workspace with a store, snapshot, and one tag."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  source_dir = ws / 'src'
  source_dir.mkdir()
  (source_dir / 'main.py').write_text('print("hello")', encoding='utf-8')

  config = AutoPilotConfig(workspace=ws)
  config.init_workspace()
  config.store_path.mkdir(parents=True, exist_ok=True)

  param = PathParameter(source=str(source_dir), pattern='*.py')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('exp-tag', 0, context='initial')

  forest = FileForest(store)
  tree = forest.create_tree('main')
  exp = Experiment(experiment_id='exp-tag', hypothesis='tag test')
  exp.start()
  exp.complete(metrics={'score': 0.9})
  tree.add(Node(experiment=exp))
  forest.switch('main')
  forest.save()

  store.tag('v1.0', 'exp-tag', 0, context='release')

  return {'workspace': ws, 'config': config, 'store': store}


class TestStoreTagListIncludesManifestDigest:
  """store tag list JSON rows include manifest_digest field."""

  def test_store_tag_list_includes_manifest_digest(self, tmp_path: Path) -> None:
    """Tag created with digest support includes a 64-char hex manifest_digest."""
    ctx = _make_workspace_with_tag(tmp_path)
    ws = ctx['workspace']

    envelope = run_cli_no_context(
      ws,
      ['--experiment', 'exp-tag', 'store', 'tag', 'list'],
    )
    result = envelope.get('result', envelope)
    tags = result['tags']
    assert len(tags) == 1
    tag = tags[0]
    assert 'manifest_digest' in tag
    digest = tag['manifest_digest']
    assert isinstance(digest, str)
    assert len(digest) == 64


class TestStoreTagListManifestDigestNullForOld:
  """store tag list returns null manifest_digest for pre-attestation tags."""

  def test_store_tag_list_manifest_digest_null_for_old(self, tmp_path: Path) -> None:
    """Fixture without digest yields null in the manifest_digest field."""
    ctx = _make_workspace_with_tag(tmp_path)
    ws = ctx['workspace']
    store = ctx['store']

    refs = store.load_refs()
    refs['tags']['v1.0'].pop('manifest_digest', None)
    store.save_refs(refs)

    envelope = run_cli_no_context(
      ws,
      ['--experiment', 'exp-tag', 'store', 'tag', 'list'],
    )
    result = envelope.get('result', envelope)
    tags = result['tags']
    assert len(tags) == 1
    tag = tags[0]
    assert 'manifest_digest' in tag
    assert tag['manifest_digest'] is None
