"""Tests for CLI store merge commands: analysis, preview, apply, resolve."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.core.config import AutoPilotConfig
from autopilot.core.snapshot import FileEntry
from autopilot.core.store.types import MergeIndex, MergeStrategy
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from tests.cli.conftest import run_cli
import contextlib
import io
import json
import pytest


def _setup_workspace(tmp_path: Path) -> Path:
  """Create a workspace directory."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


def _setup_diverged_store(
  ws: Path,
  base_files: dict[str, str],
  ours_edits: dict[str, str],
  theirs_edits: dict[str, str],
) -> tuple[FileStore, Path]:
  """Set up a workspace with root -> exp-a (ours) and exp-b (theirs) diverged."""
  src = ws / 'src'
  src.mkdir(parents=True, exist_ok=True)
  for name, content in base_files.items():
    (src / name).write_text(content)

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})

  store.snapshot('root', 0)
  store.branch('exp-a')
  store.branch('exp-b')

  store.checkout('exp-a', 0)
  for name, content in ours_edits.items():
    (src / name).write_text(content)
  store.snapshot('exp-a', 1)

  store.checkout('exp-b', 0)
  for name, content in theirs_edits.items():
    (src / name).write_text(content)
  store.snapshot('exp-b', 1)

  return store, src


def _run_merge_cli(ws: Path, argv: list[str]) -> dict:
  """Run a merge CLI command and capture JSON output."""
  return run_cli(ws, argv)


class TestCliMergeAnalysis:
  def test_json_fields(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    result = _run_merge_cli(ws, ['store', 'merge-analysis', 'exp-a', 'exp-b'])
    r = result['result']
    assert 'classification' in r
    assert 'can_fast_forward' in r
    assert 'has_conflicts' in r
    assert 'conflict_count' in r
    assert 'ancestor_epoch' in r
    assert r['has_conflicts'] is True
    assert r['conflict_count'] >= 1
    assert r['classification'] == 'conflict'

  def test_human_table_no_crash(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    parser = build_parser()
    argv = [
      'store',
      'merge-analysis',
      'exp-a',
      'exp-b',
      '--workspace',
      str(ws),
    ]
    args = parser.parse_args(argv)
    ctx = build_context(args)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      args.handler(ctx, args)
    output = buf.getvalue()
    assert 'conflict' in output.lower() or 'OK' in output

  def test_unknown_experiment_errors(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    FileStore(config)

    with pytest.raises(SystemExit):
      _run_merge_cli(ws, ['store', 'merge-analysis', 'nonexistent', 'also-missing'])

  def test_fast_forward_classification(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    src = ws / 'src'
    src.mkdir(parents=True, exist_ok=True)
    (src / 'f.txt').write_text('base')
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    param = PathParameter(source=str(src), pattern='*')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('root', 0)
    store.branch('feature')
    store.checkout('feature', 0)
    (src / 'f.txt').write_text('advanced')
    store.snapshot('feature', 1)

    result = _run_merge_cli(ws, ['store', 'merge-analysis', 'root', 'feature'])
    assert result['result']['classification'] == 'fast_forward'
    assert result['result']['can_fast_forward'] is True


class TestCliMergePreview:
  def test_writes_cache_file(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    result = _run_merge_cli(ws, ['store', 'merge-preview', 'exp-a', 'exp-b'])
    token = result['result']['preview_token']
    assert token is not None

    cache_path = ws / '.autopilot' / 'merge_preview' / f'{token}.json'
    assert cache_path.is_file()

    data = json.loads(cache_path.read_text())
    roundtripped = MergeIndex.from_dict(data)
    assert roundtripped.preview_token == token

  def test_json_conflict_details(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    result = _run_merge_cli(ws, ['store', 'merge-preview', 'exp-a', 'exp-b'])
    r = result['result']
    assert 'conflicts' in r
    assert 'resolved' in r
    assert 'preview_token' in r
    assert 'strategy' in r
    assert r['strategy'] == 'normal'
    assert r['experiment_id'] == 'exp-a'
    assert r['source_experiment_id'] == 'exp-b'

    for sides in r['conflicts'].values():
      for side_name in ['ancestor', 'ours', 'theirs']:
        side = sides[side_name]
        if side is not None:
          assert 'digest' in side
          assert 'size' in side

  def test_strategy_ours_autoresolves(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    result = _run_merge_cli(
      ws,
      ['store', 'merge-preview', 'exp-a', 'exp-b', '--strategy', 'ours'],
    )
    r = result['result']
    assert len(r['conflicts']) == 0
    assert r['strategy'] == 'ours'

  def test_unknown_experiment_errors(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    FileStore(config)

    with pytest.raises(SystemExit):
      _run_merge_cli(ws, ['store', 'merge-preview', 'nonexistent', 'missing'])


class TestCliMergeApply:
  def test_applies_from_cache(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    preview = _run_merge_cli(
      ws,
      ['store', 'merge-preview', 'exp-a', 'exp-b', '--strategy', 'ours'],
    )
    token = preview['result']['preview_token']

    result = _run_merge_cli(ws, ['store', 'merge-apply', '--token', token])
    r = result['result']
    assert 'epoch' in r
    assert r['epoch'] == 2
    assert 'file_count' in r
    assert r['experiment_id'] == 'exp-a'

  def test_invalid_token_errors(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    FileStore(config)

    with pytest.raises(SystemExit):
      _run_merge_cli(ws, ['store', 'merge-apply', '--token', 'bogus-token'])

  def test_unresolved_conflicts_errors(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    preview = _run_merge_cli(ws, ['store', 'merge-preview', 'exp-a', 'exp-b'])
    token = preview['result']['preview_token']
    assert len(preview['result']['conflicts']) > 0

    with pytest.raises(SystemExit):
      _run_merge_cli(ws, ['store', 'merge-apply', '--token', token])

  def test_cache_deleted_after_apply(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    preview = _run_merge_cli(
      ws,
      ['store', 'merge-preview', 'exp-a', 'exp-b', '--strategy', 'theirs'],
    )
    token = preview['result']['preview_token']
    cache_path = ws / '.autopilot' / 'merge_preview' / f'{token}.json'
    assert cache_path.is_file()

    _run_merge_cli(ws, ['store', 'merge-apply', '--token', token])
    assert not cache_path.is_file()


class TestCliMergeResolve:
  def test_resolve_ours(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    preview = _run_merge_cli(ws, ['store', 'merge-preview', 'exp-a', 'exp-b'])
    token = preview['result']['preview_token']
    conflict_key = next(iter(preview['result']['conflicts']))

    result = _run_merge_cli(
      ws,
      ['store', 'merge-resolve', '--token', token, conflict_key, '--ours'],
    )
    assert result['result']['key'] == conflict_key
    assert result['result']['resolution'] == 'ours'

  def test_resolve_theirs(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    preview = _run_merge_cli(ws, ['store', 'merge-preview', 'exp-a', 'exp-b'])
    token = preview['result']['preview_token']
    conflict_key = next(iter(preview['result']['conflicts']))

    result = _run_merge_cli(
      ws,
      ['store', 'merge-resolve', '--token', token, conflict_key, '--theirs'],
    )
    assert result['result']['key'] == conflict_key
    assert result['result']['resolution'] == 'theirs'

  def test_resolve_content_reads_file(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    preview = _run_merge_cli(ws, ['store', 'merge-preview', 'exp-a', 'exp-b'])
    token = preview['result']['preview_token']
    conflict_key = next(iter(preview['result']['conflicts']))

    content_file = tmp_path / 'resolved.txt'
    content_file.write_text('manually resolved content\n')

    result = _run_merge_cli(
      ws,
      ['store', 'merge-resolve', '--token', token, conflict_key, '--content', str(content_file)],
    )
    assert result['result']['key'] == conflict_key
    assert result['result']['resolution'] == 'content'
    assert 'path' in result['result']

  def test_content_non_utf8_raises(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    preview = _run_merge_cli(ws, ['store', 'merge-preview', 'exp-a', 'exp-b'])
    token = preview['result']['preview_token']
    conflict_key = next(iter(preview['result']['conflicts']))

    binary_file = tmp_path / 'binary.bin'
    binary_file.write_bytes(b'\x80\x81\x82\xff')

    with pytest.raises(SystemExit):
      _run_merge_cli(
        ws,
        ['store', 'merge-resolve', '--token', token, conflict_key, '--content', str(binary_file)],
      )

  def test_unknown_key_errors(self, tmp_path: Path) -> None:
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    preview = _run_merge_cli(ws, ['store', 'merge-preview', 'exp-a', 'exp-b'])
    token = preview['result']['preview_token']

    with pytest.raises(SystemExit):
      _run_merge_cli(
        ws,
        ['store', 'merge-resolve', '--token', token, 'nonexistent/key', '--ours'],
      )


class TestIntegrationPreviewResolveApply:
  def test_full_flow(self, tmp_path: Path) -> None:
    """preview -> resolve each conflict -> apply: epoch advances."""
    ws = _setup_workspace(tmp_path)
    _setup_diverged_store(
      ws,
      base_files={'f.txt': 'base\n'},
      ours_edits={'f.txt': 'ours\n'},
      theirs_edits={'f.txt': 'theirs\n'},
    )
    preview = _run_merge_cli(ws, ['store', 'merge-preview', 'exp-a', 'exp-b'])
    token = preview['result']['preview_token']
    conflict_keys = list(preview['result']['conflicts'])

    for key in conflict_keys:
      _run_merge_cli(
        ws,
        ['store', 'merge-resolve', '--token', token, key, '--ours'],
      )

    apply_result = _run_merge_cli(ws, ['store', 'merge-apply', '--token', token])
    assert apply_result['result']['epoch'] == 2
    assert apply_result['result']['experiment_id'] == 'exp-a'


class TestMergeIndexCacheRoundtrip:
  def test_roundtrip_via_cli_paths(self, tmp_path: Path) -> None:
    """Write MergeIndex dict manually then load to ensure CLI persistence compat."""
    cache_dir = tmp_path / '.autopilot' / 'merge_preview'
    cache_dir.mkdir(parents=True, exist_ok=True)

    from autopilot.core.store.types import ConflictEntry

    index = MergeIndex(
      conflicts={
        'a/b.txt': ConflictEntry(
          key='a/b.txt',
          ancestor=FileEntry(digest='abc123', size=10, mtime=0.0),
          ours=FileEntry(digest='def456', size=12, mtime=0.0),
          theirs=None,
        ),
      },
      resolved={'c/d.txt': FileEntry(digest='ghi789', size=5, mtime=0.0)},
      experiment_id='exp-1',
      source_experiment_id='exp-2',
      strategy=MergeStrategy.normal,
      preview_token='tok-123',
    )
    raw = index.to_dict()
    path = cache_dir / 'tok-123.json'
    atomic_write_json(path, raw)

    loaded = json.loads(path.read_text())
    roundtripped = MergeIndex.from_dict(loaded)
    assert roundtripped.preview_token == 'tok-123'
    assert roundtripped.experiment_id == 'exp-1'
    assert roundtripped.strategy == MergeStrategy.normal


class TestConflictEntryJsonProjection:
  def test_required_keys_survive_decode(self, tmp_path: Path) -> None:
    """CLI projects conflict entries to {digest, size}; verify round-trip."""
    entry = FileEntry(digest='sha256:abc', size=42, mtime=1.0)
    projected = {'digest': entry.digest, 'size': entry.size}
    assert projected['digest'] == 'sha256:abc'
    assert projected['size'] == 42

    full = FileEntry.from_dict({**projected, 'mtime': 0.0})
    assert full.digest == 'sha256:abc'
    assert full.size == 42
