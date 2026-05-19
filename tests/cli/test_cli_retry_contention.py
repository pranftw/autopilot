"""Tests for --retry global flag: exponential backoff on ConcurrentMutationError."""

from autopilot.cli.command import CLI
from autopilot.cli.context import CLIContext, build_context
from autopilot.cli.main import build_parser
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.tracking.file_lock import LOCK_RETRY_AFTER_MS, ConcurrentMutationError
from pathlib import Path
from unittest.mock import patch
import argparse
import contextlib
import io
import json
import pytest

CONTENTION_OP = 'test_op'


def _build_ctx(
  workspace: Path,
  *,
  use_json: bool = False,
  retry_max: int = 0,
  wait_timeout_ms: int | None = None,
  dry_run: bool = False,
) -> CLIContext:
  """Build a CLIContext for retry tests."""
  config = AutoPilotConfig(workspace=workspace)
  return CLIContext(
    workspace=workspace,
    config=config,
    output=Output(use_json=use_json),
    context='test',
    retry_max=retry_max,
    wait_timeout_ms=wait_timeout_ms,
    dry_run=dry_run,
  )


class TestRetryZeroIsFailFast:
  """Default retry_max=0: first ConcurrentMutationError bubbles immediately."""

  def test_retry_zero_is_fail_fast(self, tmp_path):
    ctx = _build_ctx(tmp_path, retry_max=0)
    call_count = 0

    def handler(c, a):
      nonlocal call_count
      call_count += 1
      raise ConcurrentMutationError(CONTENTION_OP)

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()

    with pytest.raises(SystemExit):
      cli.dispatch(ctx, args)

    assert call_count == 1


class TestRetrySucceedsOnSecondAttempt:
  """First dispatch raises ConcurrentMutationError, second succeeds."""

  def test_retry_succeeds_on_second_attempt(self, tmp_path):
    ctx = _build_ctx(tmp_path, use_json=True, retry_max=3)
    call_count = 0

    def handler(c, a):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise ConcurrentMutationError(CONTENTION_OP)
      c.output.result({'status': 'done'})

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), patch('autopilot.cli.command.time.sleep') as mock_sleep:
      cli.dispatch(ctx, args)

    assert call_count == 2
    mock_sleep.assert_called_once()

    output = buf.getvalue().strip()
    envelope = json.loads(output)
    assert envelope['ok'] is True
    assert envelope['retry_attempts'] == 1


class TestRetryExhaustedRaises:
  """Always contention: after 1+N attempts, final ConcurrentMutationError propagates."""

  def test_retry_exhausted_raises(self, tmp_path):
    ctx = _build_ctx(tmp_path, retry_max=2)
    call_count = 0

    def handler(c, a):
      nonlocal call_count
      call_count += 1
      raise ConcurrentMutationError(CONTENTION_OP)

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()

    with pytest.raises(SystemExit) as exc_info, patch('autopilot.cli.command.time.sleep'):
      cli.dispatch(ctx, args)

    assert exc_info.value.code == 1
    assert call_count == 3  # 1 initial + 2 retries


class TestRetryBackoffTiming:
  """Verify exponential backoff: 100ms, 200ms, 400ms from LOCK_RETRY_AFTER_MS."""

  def test_retry_backoff_timing(self, tmp_path):
    ctx = _build_ctx(tmp_path, retry_max=3)
    call_count = 0

    def handler(c, a):
      nonlocal call_count
      call_count += 1
      if call_count <= 3:
        raise ConcurrentMutationError(CONTENTION_OP)

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()

    with patch('autopilot.cli.command.time.sleep') as mock_sleep:
      cli.dispatch(ctx, args)

    assert call_count == 4  # 3 failures + 1 success
    sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
    base_s = LOCK_RETRY_AFTER_MS / 1000.0
    assert sleep_calls == [base_s, base_s * 2, base_s * 4]


class TestRetryAndWaitMutuallyExclusive:
  """--retry and --wait cannot be combined."""

  def test_retry_and_wait_mutually_exclusive(self, tmp_path):
    ctx = _build_ctx(tmp_path, retry_max=3, wait_timeout_ms=5000)

    def handler(c, a):
      pass

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()

    with pytest.raises(SystemExit) as exc_info:
      cli.dispatch(ctx, args)
    assert exc_info.value.code == 1

  def test_mutual_exclusion_message_references_both_flags(self, tmp_path):
    ctx = _build_ctx(tmp_path, use_json=True, retry_max=1, wait_timeout_ms=0)

    def handler(c, a):
      pass

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
      cli.dispatch(ctx, args)

    output = buf.getvalue()
    assert '--retry' in output
    assert '--wait' in output

  def test_retry_zero_with_wait_allowed(self, tmp_path):
    """retry_max=0 (default) with --wait is fine -- no mutual exclusion."""
    ctx = _build_ctx(tmp_path, retry_max=0, wait_timeout_ms=5000)

    def handler(c, a):
      pass

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()
    cli.dispatch(ctx, args)


class TestRetryJsonEnvelopeIncludesAttempts:
  """Success JSON includes retry_attempts key when retries consumed."""

  def test_retry_json_envelope_includes_attempts(self, tmp_path):
    ctx = _build_ctx(tmp_path, use_json=True, retry_max=2)
    call_count = 0

    def handler(c, a):
      nonlocal call_count
      call_count += 1
      if call_count <= 2:
        raise ConcurrentMutationError(CONTENTION_OP)
      c.output.result({'status': 'done'})

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), patch('autopilot.cli.command.time.sleep'):
      cli.dispatch(ctx, args)

    envelope = json.loads(buf.getvalue().strip())
    assert envelope['ok'] is True
    assert envelope['retry_attempts'] == 2
    assert isinstance(envelope['retry_attempts'], int)

  def test_no_retry_attempts_key_on_first_success(self, tmp_path):
    """No retry_attempts in envelope when handler succeeds on first try."""
    ctx = _build_ctx(tmp_path, use_json=True, retry_max=3)

    def handler(c, a):
      c.output.result({'status': 'done'})

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      cli.dispatch(ctx, args)

    envelope = json.loads(buf.getvalue().strip())
    assert envelope['ok'] is True
    assert 'retry_attempts' not in envelope


class TestRetryTextModeNoJson:
  """Success in text mode: no machine-readable retry_attempts field."""

  def test_retry_text_mode_no_json(self, tmp_path):
    ctx = _build_ctx(tmp_path, use_json=False, retry_max=2)
    call_count = 0

    def handler(c, a):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise ConcurrentMutationError(CONTENTION_OP)
      print('success')

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with (
      contextlib.redirect_stdout(out_buf),
      contextlib.redirect_stderr(err_buf),
      patch('autopilot.cli.command.time.sleep'),
    ):
      cli.dispatch(ctx, args)

    assert 'retry_attempts' not in out_buf.getvalue()
    assert 'retry_attempts' not in err_buf.getvalue()
    assert 'success' in out_buf.getvalue()


class TestRetryExhaustedExitCode:
  """Exhausted retries produce non-zero exit code."""

  def test_retry_exhausted_exit_code(self, tmp_path):
    ctx = _build_ctx(tmp_path, retry_max=1)

    def handler(c, a):
      raise ConcurrentMutationError(CONTENTION_OP)

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()

    with pytest.raises(SystemExit) as exc_info, patch('autopilot.cli.command.time.sleep'):
      cli.dispatch(ctx, args)

    assert exc_info.value.code != 0

  def test_retry_exhausted_json_envelope(self, tmp_path):
    """Exhausted retries preserve concurrent_mutation JSON envelope."""
    ctx = _build_ctx(tmp_path, use_json=True, retry_max=1)

    def handler(c, a):
      raise ConcurrentMutationError(CONTENTION_OP)

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()

    buf = io.StringIO()
    with (
      contextlib.redirect_stdout(buf),
      pytest.raises(SystemExit),
      patch('autopilot.cli.command.time.sleep'),
    ):
      cli.dispatch(ctx, args)

    envelope = json.loads(buf.getvalue().strip())
    assert envelope['ok'] is False
    assert envelope['error_code'] == 'concurrent_mutation'
    assert envelope['retry_after_ms'] == LOCK_RETRY_AFTER_MS


class TestRetryDryRunNoRetry:
  """--dry-run with --retry N > 0: retry loop does not trigger."""

  def test_retry_dry_run_no_retry(self, tmp_path):
    ctx = _build_ctx(tmp_path, retry_max=3, dry_run=True)
    call_count = 0

    def handler(c, a):
      nonlocal call_count
      call_count += 1
      raise ConcurrentMutationError(CONTENTION_OP)

    args = argparse.Namespace(handler=handler, command='tree create')
    cli = CLI()

    with pytest.raises(SystemExit), patch('autopilot.cli.command.time.sleep') as mock_sleep:
      cli.dispatch(ctx, args)

    assert call_count == 1
    mock_sleep.assert_not_called()


class TestRetryFlagParsing:
  """Test --retry flag argparse wiring."""

  def test_retry_flag_parsed(self, tmp_path):
    parser = build_parser()
    args = parser.parse_args(['--retry', '3', '--workspace', str(tmp_path), 'status'])
    assert args.retry == 3

  def test_retry_flag_default_zero(self, tmp_path):
    parser = build_parser()
    args = parser.parse_args(['--workspace', str(tmp_path), 'status'])
    assert args.retry == 0

  def test_retry_wired_to_context(self, tmp_path):
    parser = build_parser()
    args = parser.parse_args(['--retry', '5', '--workspace', str(tmp_path), 'status'])
    ctx = build_context(args)
    assert ctx.retry_max == 5

  def test_retry_default_context_zero(self, tmp_path):
    parser = build_parser()
    args = parser.parse_args(['--workspace', str(tmp_path), 'status'])
    ctx = build_context(args)
    assert ctx.retry_max == 0
