"""AP-4: CLI-level store checkout --dry-run tests.

Verifies that ``store checkout --dry-run`` performs read-only validation
(experiment existence, epoch validity, schema matching) without mutating
any files on disk.  The ``--dry-run`` flag is a global CLI flag handled
by ``ctx.dry_run`` -- no production code changes needed.

See plan 22, subplan 2.4.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
from tests.cli.conftest import run_cli
import hashlib


def _file_digest(path: Path) -> str:
  """SHA-256 hex digest of a file's contents."""
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup_workspace_with_snapshot(tmp_path: Path) -> tuple[Path, Path, Path]:
  """Create a workspace with a store, register a param, and snapshot epoch 0.

  Returns:
    Tuple of (workspace_path, param_file_path, source_dir).
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  source_dir = ws / 'params'
  source_dir.mkdir()
  param_file = source_dir / 'prompt.txt'
  param_file.write_text('original snapshot content', encoding='utf-8')

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  param = PathParameter(source=str(source_dir), pattern='**/*.txt')
  store.register_parameters({'prompts': param})
  store.snapshot('exp-dry', 0)

  param_file.write_text('MUTATED after snapshot', encoding='utf-8')

  return ws, param_file, source_dir


def test_checkout_dry_run_leaves_worktree_unchanged(tmp_path: Path) -> None:
  """Dry-run checkout must not mutate any files on disk.

  AP-4: snapshot epoch 0, mutate file, run checkout --dry-run, verify
  file digest is unchanged (still the mutated content, not restored).
  """
  ws, param_file, source_dir = _setup_workspace_with_snapshot(tmp_path)

  digest_before = _file_digest(param_file)
  content_before = param_file.read_text(encoding='utf-8')

  result = run_cli(
    ws,
    [
      '--experiment',
      'exp-dry',
      '--epoch',
      '0',
      '--dry-run',
      'store',
      'checkout',
      '--source',
      str(source_dir),
    ],
  )

  assert result['ok'] is True
  assert result['result']['dry_run'] is True
  assert result['result']['command'] == 'checkout'
  assert result['result']['experiment'] == 'exp-dry'
  assert result['result']['epoch'] == 0

  digest_after = _file_digest(param_file)
  content_after = param_file.read_text(encoding='utf-8')

  assert digest_before == digest_after, (
    f'dry-run checkout mutated file: before={content_before!r}, after={content_after!r}'
  )
  assert content_after == 'MUTATED after snapshot', (
    'dry-run checkout should not restore snapshot content'
  )


def test_checkout_dry_run_reports_validation_info(tmp_path: Path) -> None:
  """Dry-run checkout JSON includes files_to_restore and schema info."""
  ws, _param_file, source_dir = _setup_workspace_with_snapshot(tmp_path)

  result = run_cli(
    ws,
    [
      '--experiment',
      'exp-dry',
      '--epoch',
      '0',
      '--dry-run',
      'store',
      'checkout',
      '--source',
      str(source_dir),
    ],
  )

  assert 'files_to_restore' in result['result']
  assert isinstance(result['result']['files_to_restore'], int)
  assert result['result']['files_to_restore'] > 0
  assert 'schema_match' in result['result']
  assert 'schema_mismatch' in result['result']
