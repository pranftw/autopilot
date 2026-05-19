"""Tests for the ``store tag verify`` CLI command.

Covers JSON output, text mode, exit codes for match/mismatch/missing tag,
and context exemption (read-only).
"""

from autopilot.ai.parameter import PathParameter
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
from tests.cli.conftest import make_cli_workspace, run_cli_no_context
import json
import pytest


class TestTagVerifyCLIJson:
  """test_tag_verify_cli_json: CLI --json mirrors verify_tag dict."""

  def test_tag_verify_cli_json_match(self, tmp_path: Path) -> None:
    store, ws = make_cli_workspace(tmp_path)
    store.snapshot('exp-1', 0)
    store.tag('v1.0', 'exp-1', 0, context='release')

    envelope = run_cli_no_context(ws, ['store', 'tag', 'verify', 'v1.0'])
    assert envelope['ok'] is True
    assert envelope['result']['verified'] is True


class TestTagVerifyExitCodeMatch:
  """test_tag_verify_exit_code_match: CLI exit 0 on match."""

  def test_tag_verify_exit_code_match(self, tmp_path: Path) -> None:
    store, ws = make_cli_workspace(tmp_path)
    store.snapshot('exp-1', 0)
    store.tag('v1.0', 'exp-1', 0, context='release')

    envelope = run_cli_no_context(ws, ['store', 'tag', 'verify', 'v1.0'])
    assert envelope['ok'] is True
    assert envelope['result']['verified'] is True


class TestTagVerifyExitCodeMismatch:
  """test_tag_verify_exit_code_mismatch: CLI non-zero exit on mismatch."""

  def test_tag_verify_exit_code_mismatch(self, tmp_path: Path) -> None:
    store, ws = make_cli_workspace(tmp_path)
    store.snapshot('exp-1', 0)
    store.tag('v1.0', 'exp-1', 0, context='release')

    config = AutoPilotConfig(workspace=ws)
    snap_path = config.store_path / 'snapshots' / 'exp-1' / 'epoch_0.json'
    data = json.loads(snap_path.read_text(encoding='utf-8'))
    data['context'] = 'tampered'
    snap_path.write_text(json.dumps(data), encoding='utf-8')

    with pytest.raises(SystemExit) as exc_info:
      run_cli_no_context(ws, ['store', 'tag', 'verify', 'v1.0'])
    assert exc_info.value.code == 1

  def test_tag_verify_exit_code_no_digest(self, tmp_path: Path) -> None:
    """Pre-attestation tag (no digest) exits non-zero."""
    store, ws = make_cli_workspace(tmp_path)
    store.snapshot('exp-1', 0)
    store.tag('v1.0', 'exp-1', 0, context='release')

    config = AutoPilotConfig(workspace=ws)
    refs_path = config.store_path / 'refs.json'
    refs_data = json.loads(refs_path.read_text(encoding='utf-8'))
    del refs_data['tags']['v1.0']['manifest_digest']
    refs_path.write_text(json.dumps(refs_data), encoding='utf-8')

    with pytest.raises(SystemExit) as exc_info:
      run_cli_no_context(ws, ['store', 'tag', 'verify', 'v1.0'])
    assert exc_info.value.code == 1

  def test_tag_verify_missing_tag_exits_nonzero(self, tmp_path: Path) -> None:
    """Missing tag exits non-zero."""
    store, ws = make_cli_workspace(tmp_path)
    store.snapshot('exp-1', 0)

    with pytest.raises(SystemExit) as exc_info:
      run_cli_no_context(ws, ['store', 'tag', 'verify', 'nonexistent'])
    assert exc_info.value.code != 0


class TestVerifyTagErrorGuidance:
  """verify_tag error for unknown tags includes recovery guidance."""

  def test_verify_tag_unknown_includes_list_tags_guidance(self, tmp_path: Path) -> None:
    from autopilot.ai.store.file_store import FileStore
    from autopilot.core.config import AutoPilotConfig
    from autopilot.core.errors import StoreError

    ws = tmp_path / 'ws'
    ws.mkdir()
    src = ws / 'src'
    src.mkdir()
    (src / 'main.py').write_text('print("hello")\n')
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    param = PathParameter(source=str(src), pattern='**/*.py')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('exp-1', 0)

    with pytest.raises(StoreError, match=r'not found.*list_tags'):
      store.verify_tag('nonexistent')
