"""Tests for experiment notes write --body and --file flags (FR#24).

Validates:
  - Inline --body writes notes correctly
  - --file reads UTF-8 content from disk
  - Mutual exclusivity of --body and --file
  - Missing file path error
  - Binary file rejection
  - Neither flag provided error
  - Invalid UTF-8 file rejection
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
from tests.cli.conftest import run_cli, seed_tree_with_experiments
import contextlib
import io
import json
import pytest


def _run_cli_expect_fail(workspace: Path, argv: list[str]) -> dict[str, str]:
  """Run a CLI command expecting failure; return the JSON error envelope.

  Args:
    workspace: Workspace root directory.
    argv: CLI argument tokens.

  Returns:
    Parsed JSON envelope from captured stdout (contains 'error' key).

  Raises:
    AssertionError: If the command does not raise SystemExit.
  """
  parser = build_parser()
  full_argv = [*list(argv), '--workspace', str(workspace), '--json', '--context', 'test']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)

  buf = io.StringIO()
  with pytest.raises(SystemExit) as exc_info, contextlib.redirect_stdout(buf):
    parsed.handler(ctx, parsed)

  assert exc_info.value.code != 0
  output = buf.getvalue().strip()
  if output:
    return json.loads(output)
  return {}


@pytest.fixture
def notes_workspace(tmp_path: Path) -> Path:
  """Workspace with a running experiment for notes tests."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [{'id': 'exp-notes', 'hypothesis': 'notes test', 'status': 'running'}],
  )
  return ws


class TestNotesWriteFromFile:
  """--file flag reads UTF-8 content from a file path."""

  def test_notes_write_from_file(self, notes_workspace: Path, tmp_path: Path) -> None:
    """Create temp file with known UTF-8 content; assert notes match file content."""
    notes_file = tmp_path / 'my_notes.txt'
    notes_file.write_text('These are my experiment notes.\nLine two.', encoding='utf-8')

    content = 'These are my experiment notes.\nLine two.'
    result = run_cli(
      notes_workspace,
      ['experiment', 'notes', 'write', 'exp-notes', '--file', str(notes_file)],
    )
    assert result['ok'] is True
    assert result['result']['bytes_written'] == len(content.encode('utf-8'))

    from tests.cli.conftest import run_cli_no_context

    show_result = run_cli_no_context(notes_workspace, ['experiment', 'notes', 'show', 'exp-notes'])
    assert show_result['ok'] is True
    assert show_result['result']['notes'] == 'These are my experiment notes.\nLine two.'


class TestNotesWriteBodyInline:
  """--body flag writes inline text."""

  def test_notes_write_body_inline(self, notes_workspace: Path) -> None:
    """--body 'hello' persists notes and JSON result matches."""
    result = run_cli(
      notes_workspace,
      ['experiment', 'notes', 'write', 'exp-notes', '--body', 'hello world'],
    )
    assert result['ok'] is True
    assert result['result']['bytes_written'] == len(b'hello world')

    from tests.cli.conftest import run_cli_no_context

    show_result = run_cli_no_context(notes_workspace, ['experiment', 'notes', 'show', 'exp-notes'])
    assert show_result['ok'] is True
    assert show_result['result']['notes'] == 'hello world'


class TestNotesWriteValidationErrors:
  """Validation and error paths for notes write."""

  def test_notes_write_file_and_body_mutually_exclusive(
    self, notes_workspace: Path, tmp_path: Path
  ) -> None:
    """Both flags set; expect non-zero exit with mutual exclusivity message."""
    notes_file = tmp_path / 'dummy.txt'
    notes_file.write_text('content', encoding='utf-8')

    envelope = _run_cli_expect_fail(
      notes_workspace,
      [
        'experiment',
        'notes',
        'write',
        'exp-notes',
        '--body',
        'inline',
        '--file',
        str(notes_file),
      ],
    )
    assert 'mutually exclusive' in envelope['error']

  def test_notes_write_file_missing(self, notes_workspace: Path, tmp_path: Path) -> None:
    """--file points to nonexistent path; assert 'File not found' in error."""
    missing = tmp_path / 'does_not_exist.txt'

    envelope = _run_cli_expect_fail(
      notes_workspace,
      ['experiment', 'notes', 'write', 'exp-notes', '--file', str(missing)],
    )
    assert 'File not found' in envelope['error']
    assert str(missing) in envelope['error']

  def test_notes_write_file_binary_rejected(self, notes_workspace: Path, tmp_path: Path) -> None:
    """File with NUL byte in first 8192 bytes is rejected."""
    binary_file = tmp_path / 'binary.dat'
    binary_file.write_bytes(b'some text\x00more binary data')

    envelope = _run_cli_expect_fail(
      notes_workspace,
      ['experiment', 'notes', 'write', 'exp-notes', '--file', str(binary_file)],
    )
    assert 'Binary files not supported' in envelope['error']

  def test_notes_write_neither_body_nor_file(self, notes_workspace: Path) -> None:
    """Omit both flags; assert failure telling user to provide one."""
    envelope = _run_cli_expect_fail(
      notes_workspace,
      ['experiment', 'notes', 'write', 'exp-notes'],
    )
    assert '--body' in envelope['error'] or '--file' in envelope['error']

  def test_notes_write_file_invalid_utf8(self, notes_workspace: Path, tmp_path: Path) -> None:
    """File with invalid UTF-8 bytes is rejected with path in error."""
    bad_file = tmp_path / 'bad_encoding.txt'
    bad_file.write_bytes(b'\x80\x81\x82\x83')

    envelope = _run_cli_expect_fail(
      notes_workspace,
      ['experiment', 'notes', 'write', 'exp-notes', '--file', str(bad_file)],
    )
    assert 'not valid UTF-8' in envelope['error']
    assert str(bad_file) in envelope['error']
