"""Tests for store CLI registration, parsing, FileStore integration, and worktrees."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.cli.primitives import ArgparseCLIError
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
from tests.cli.conftest import run_cli
import contextlib
import io
import json
import pytest


def _src(tmp_path: Path) -> Path:
  src = tmp_path / 'src'
  src.mkdir(parents=True, exist_ok=True)
  (src / 'main.py').write_text('v0')
  (src / 'util.py').write_text('u0')
  return src


def _parse(argv: list[str]):
  parser = build_parser()
  return parser.parse_args(argv)


class TestStoreParser:
  def test_store_command_parses(self) -> None:
    args = _parse(
      [
        'store',
        'status',
        '--workspace',
        '/tmp',
        '--experiment',
        'e',
        '--source',
        '/tmp/s',
      ]
    )
    assert args.command == 'store'
    assert args.store_action == 'status'

  def test_create_parses(self) -> None:
    args = _parse(
      [
        'store',
        'create',
        '--workspace',
        '/w',
        '--experiment',
        'slug',
        '--source',
        '/w/code',
      ]
    )
    assert args.store_action == 'create'

  def test_snapshot_parses(self) -> None:
    args = _parse(
      [
        'store',
        'snapshot',
        '--workspace',
        '/w',
        '--experiment',
        'slug',
        '--source',
        '/w/code',
      ]
    )
    assert args.store_action == 'snapshot'

  def test_checkout_parses(self) -> None:
    args = _parse(
      [
        'store',
        'checkout',
        '--workspace',
        '/w',
        '--experiment',
        'slug',
        '--source',
        '/w/code',
        '--epoch',
        '0',
      ]
    )
    assert args.store_action == 'checkout'
    assert args.epoch == 0

  def test_diff_parses(self) -> None:
    args = _parse(
      [
        'store',
        'diff',
        '--workspace',
        '/w',
        '--experiment',
        'a',
        '--source',
        '/w/code',
        '--epoch-a',
        '0',
        '--epoch-b',
        '1',
      ]
    )
    assert args.store_action == 'diff'
    assert args.epoch_a == 0
    assert args.epoch_b == 1

  def test_branch_parses(self) -> None:
    args = _parse(
      [
        'store',
        'branch',
        '--workspace',
        '/w',
        '--experiment',
        'feature',
        '--source',
        '/w/code',
      ]
    )
    assert args.store_action == 'branch'

  def test_merge_parses(self) -> None:
    args = _parse(
      [
        'store',
        'merge',
        '--workspace',
        '/w',
        '--experiment',
        'a',
        '--source',
        '/w/code',
        '--from-experiment',
        'feature',
        '--merge-epoch',
        '1',
      ]
    )
    assert args.store_action == 'merge'
    assert args.from_experiment == 'feature'
    assert args.merge_epoch == 1

  def test_log_parses(self) -> None:
    args = _parse(
      [
        'store',
        'log',
        '--workspace',
        '/w',
        '--experiment',
        'slug',
        '--source',
        '/w/code',
      ]
    )
    assert args.store_action == 'log'

  def test_status_parses(self) -> None:
    args = _parse(
      [
        'store',
        'status',
        '--workspace',
        '/w',
        '--experiment',
        'slug',
        '--source',
        '/w/code',
      ]
    )
    assert args.store_action == 'status'

  def test_promote_parses(self) -> None:
    args = _parse(
      [
        'store',
        'promote',
        '--workspace',
        '/w',
        '--experiment',
        'slug',
        '--source',
        '/w/code',
        '--epoch',
        '1',
      ]
    )
    assert args.store_action == 'promote'

  def test_create_requires_source(self) -> None:
    parser = build_parser()
    with pytest.raises(ArgparseCLIError):
      parser.parse_args(
        [
          'store',
          'create',
          '--workspace',
          '/w',
          '--experiment',
          'slug',
        ]
      )

  def test_merge_requires_from_experiment(self) -> None:
    parser = build_parser()
    with pytest.raises(ArgparseCLIError):
      parser.parse_args(
        [
          'store',
          'merge',
          '--workspace',
          '/w',
          '--experiment',
          'a',
          '--source',
          '/w/code',
        ]
      )


class TestStoreCliFileStore:
  def test_create_and_snapshot(self, tmp_path: Path) -> None:
    src = _src(tmp_path)
    parser = build_parser()
    argv_create = [
      'store',
      'create',
      '--workspace',
      str(tmp_path),
      '--experiment',
      'exp-cli',
      '--source',
      str(src),
      '--store',
      str(tmp_path / 'dotstore'),
    ]
    args = parser.parse_args(argv_create)
    ctx = build_context(args)
    args.handler(ctx, args)

    (src / 'main.py').write_text('v1')
    argv_snap = [
      'store',
      'snapshot',
      '--workspace',
      str(tmp_path),
      '--experiment',
      'exp-cli',
      '--source',
      str(src),
      '--store',
      str(tmp_path / 'dotstore'),
    ]
    args2 = parser.parse_args(argv_snap)
    ctx2 = build_context(args2)
    args2.handler(ctx2, args2)

    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path = tmp_path / 'dotstore'
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})
    entries = store.log('exp-cli')
    assert entries[-1].epoch == 1

  def test_status_json_envelope(self, tmp_path: Path) -> None:
    src = _src(tmp_path)
    parser = build_parser()
    argv = [
      'store',
      'create',
      '--json',
      '--workspace',
      str(tmp_path),
      '--experiment',
      'exp-json',
      '--source',
      str(src),
      '--store',
      str(tmp_path / 's'),
    ]
    args = parser.parse_args(argv)
    ctx = build_context(args)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      args.handler(ctx, args)
    first = json.loads(buf.getvalue())
    assert first['ok'] is True
    assert 'result' in first
    assert 'messages' in first

    buf2 = io.StringIO()
    argv_status = [
      'store',
      'status',
      '--json',
      '--workspace',
      str(tmp_path),
      '--experiment',
      'exp-json',
      '--source',
      str(src),
      '--store',
      str(tmp_path / 's'),
    ]
    args2 = parser.parse_args(argv_status)
    ctx2 = build_context(args2)
    with contextlib.redirect_stdout(buf2):
      args2.handler(ctx2, args2)
    second = json.loads(buf2.getvalue())
    assert second['ok'] is True
    assert 'entries' in second['result']


class TestStoreWorktreeList:
  def test_empty_worktrees(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    result = run_cli(ws, ['store', 'worktree', 'list'])
    assert result['result']['worktrees'] == []

  def test_shows_active_worktrees(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    src = ws / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})
    store.snapshot('exp-wt', 0)
    store.create_worktree('exp-wt')

    result = run_cli(ws, ['store', 'worktree', 'list'])
    assert 'exp-wt' in result['result']['worktrees']

  def test_json_valid(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    result = run_cli(ws, ['store', 'worktree', 'list'])
    assert 'ok' in result
    assert 'worktrees' in result['result']


class TestStoreWorktreeCreate:
  def test_creates_worktree(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    src = ws / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})
    store.snapshot('exp-wt2', 0)

    result = run_cli(ws, ['store', 'worktree', 'create', 'exp-wt2'])
    assert result['result']['ok'] is True
    assert result['result']['experiment_id'] == 'exp-wt2'
    assert 'path' in result['result']

    list_result = run_cli(ws, ['store', 'worktree', 'list'])
    assert 'exp-wt2' in list_result['result']['worktrees']

  def test_json_output(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    src = ws / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})
    store.snapshot('exp-json-wt', 0)

    result = run_cli(ws, ['store', 'worktree', 'create', 'exp-json-wt'])
    assert 'ok' in result
    assert result['result']['ok'] is True
