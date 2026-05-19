"""Tests for CLI JSON error envelope, quick-start help, optional hypothesis, and slug resolution.

Covers sub-plan 01 of dogfood-v2:
  - Argparse errors emit JSON envelope when --json is present
  - Argparse errors emit stderr usage text when --json is absent
  - Exit code 2 for usage errors
  - --help exits 0 with usage text
  - Quick-start epilog in root help
  - experiment add works without --hypothesis
  - Active-tree experiment id prefix resolution

Covers sub-plan 01 of dogfood-v2-fixes (BUG-001, BUG-002):
  - flush_error always includes error_code (default 'handler_error')
  - ctx.fail threads error_code to flush_error
  - Handler exceptions produce error_code in JSON envelope
  - No-subcommand path with --json emits JSON envelope with 'cli_usage' code
"""

from autopilot.cli.command import CLI, Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import (
  EXPERIMENT_SLUG_PREFIX_MIN_LEN,
  CLIError,
  resolve_active_tree_experiment_node,
)
from autopilot.cli.main import AutoPilotCLI
from autopilot.cli.output import Output
from autopilot.cli.primitives import ArgparseCLIError, AutopilotArgumentParser
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.core.tree import Tree
from pathlib import Path
from tests.cli.conftest import run_cli
from typing import Any
from unittest.mock import MagicMock
import json
import pytest

# -- 2.1: argparse error routing --


class TestArgparseCLIError:
  """Tests for ArgparseCLIError exception class."""

  def test_attributes(self) -> None:
    exc = ArgparseCLIError('bad flag', exit_code=2)
    assert exc.message == 'bad flag'
    assert exc.exit_code == 2
    assert str(exc) == 'bad flag'

  def test_default_exit_code(self) -> None:
    exc = ArgparseCLIError('oops')
    assert exc.exit_code == 2


class TestAutopilotArgumentParser:
  """Tests for AutopilotArgumentParser overrides."""

  def test_error_raises(self) -> None:
    parser = AutopilotArgumentParser()
    with pytest.raises(ArgparseCLIError, match='test error') as exc_info:
      parser.error('test error')
    assert exc_info.value.exit_code == 2

  def test_exit_raises_with_status(self) -> None:
    with pytest.raises(ArgparseCLIError) as exc_info:
      AutopilotArgumentParser().exit(status=0, message='help')
    assert exc_info.value.exit_code == 0

  def test_exit_default_message(self) -> None:
    with pytest.raises(ArgparseCLIError) as exc_info:
      AutopilotArgumentParser().exit(status=1)
    assert not exc_info.value.message
    assert exc_info.value.exit_code == 1


class TestArgparseErrorJsonEnvelope:
  """Tests for JSON envelope on argparse failures via run_direct."""

  def test_argparse_error_json_envelope(self, capsys: pytest.CaptureFixture) -> None:
    cli = AutoPilotCLI()
    with pytest.raises(SystemExit) as exc_info:
      cli.run_direct(argv=['--json', 'nosuchcmd'])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload['ok'] is False
    assert 'error' in payload
    assert payload['error_code'] == 'cli_usage'

  def test_argparse_error_no_json(self, capsys: pytest.CaptureFixture) -> None:
    cli = AutoPilotCLI()
    with pytest.raises(SystemExit) as exc_info:
      cli.run_direct(argv=['nosuchcmd'])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert 'error:' in captured.err

  def test_subparser_error_json(self, capsys: pytest.CaptureFixture) -> None:
    cli = AutoPilotCLI()
    with pytest.raises(SystemExit) as exc_info:
      cli.run_direct(argv=['--json', 'experiment'])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload['ok'] is False
    assert payload['error_code'] == 'cli_usage'

  def test_missing_required_arg_json(self, capsys: pytest.CaptureFixture) -> None:
    cli = AutoPilotCLI()
    with pytest.raises(SystemExit) as exc_info:
      cli.run_direct(argv=['--json', 'experiment', 'complete'])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload['ok'] is False
    assert payload['error_code'] == 'cli_usage'

  def test_exit_code_is_2(self) -> None:
    cli = AutoPilotCLI()
    with pytest.raises(SystemExit) as exc_info:
      cli.run_direct(argv=['nosuchcmd'])
    assert exc_info.value.code == 2


class TestHelpBehavior:
  """Tests for --help routing through ArgparseCLIError."""

  def test_help_exits_zero(self, capsys: pytest.CaptureFixture) -> None:
    cli = AutoPilotCLI()
    with pytest.raises(SystemExit) as exc_info:
      cli.run_direct(argv=['--help'])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert 'usage:' in captured.out.lower() or 'Usage:' in captured.out

  def test_help_with_json_first_argv_order(self, capsys: pytest.CaptureFixture) -> None:
    cli = AutoPilotCLI()
    with pytest.raises(SystemExit) as exc_info:
      cli.run_direct(argv=['--json', '--help'])
    assert exc_info.value.code == 0


# -- 2.2: quick-start epilog --


class TestQuickStartEpilog:
  """Tests for the quick-start epilog in root help."""

  def test_help_shows_quickstart(self) -> None:
    help_text = CLI().build_parser().format_help()
    assert 'Quick start' in help_text

  def test_quickstart_contains_workspace_init(self) -> None:
    help_text = AutoPilotCLI().build_parser().format_help()
    assert 'workspace init' in help_text
    assert 'tree create' in help_text
    assert 'experiment add' in help_text
    assert 'query --json' in help_text


# -- 2.3: optional --hypothesis --


class TestExperimentAddWithoutHypothesis:
  """Tests for experiment add with optional --hypothesis."""

  def test_experiment_add_without_hypothesis(
    self,
    cli_workspace: Path,
    cli_forest: Any,
  ) -> None:
    cli_forest.create_tree('main')
    cli_forest.switch('main')
    cli_forest.save()

    envelope = run_cli(
      cli_workspace,
      ['experiment', 'add', '--id', 'test-exp-01'],
    )
    result = envelope['result']
    assert result['ok'] is True
    assert result['experiment_id'] == 'test-exp-01'
    assert result['hypothesis'] is None

  def test_experiment_add_with_hypothesis(
    self,
    cli_workspace: Path,
    cli_forest: Any,
  ) -> None:
    cli_forest.create_tree('main')
    cli_forest.switch('main')
    cli_forest.save()

    envelope = run_cli(
      cli_workspace,
      ['experiment', 'add', '--id', 'test-exp-02', '--hypothesis', 'test hyp'],
    )
    result = envelope['result']
    assert result['ok'] is True
    assert result['hypothesis'] == 'test hyp'


# -- 2.4: slug / prefix resolution --


def _make_tree_with_experiments(experiment_ids: list[str]) -> Tree:
  """Create a tree with experiments for slug resolution testing."""
  mock_store = MagicMock()
  mock_store.branch_exists.return_value = False
  tree = Tree(name='test', store=mock_store)
  for eid in experiment_ids:
    exp = Experiment(experiment_id=eid, hypothesis='h')
    tree.add(Node(experiment=exp))
  return tree


class TestSlugResolution:
  """Tests for resolve_active_tree_experiment_node."""

  def test_exact_match(self) -> None:
    tree = _make_tree_with_experiments(['abcd11112222', 'abcd22223333'])
    node = resolve_active_tree_experiment_node(tree, 'abcd11112222')
    assert node is not None
    assert node.experiment.id == 'abcd11112222'

  def test_unique_prefix_resolves(self) -> None:
    tree = _make_tree_with_experiments(['abcd11112222', 'abcd22223333'])
    node = resolve_active_tree_experiment_node(tree, 'abcd1111')
    assert node is not None
    assert node.experiment.id == 'abcd11112222'

  def test_ambiguous_prefix_raises(self) -> None:
    tree = _make_tree_with_experiments(['abcd11112222', 'abcd11113333'])
    with pytest.raises(CLIError, match='ambiguous prefix') as exc_info:
      resolve_active_tree_experiment_node(tree, 'abcd1111')
    error_msg = str(exc_info.value)
    assert 'abcd11112222' in error_msg
    assert 'abcd11113333' in error_msg
    ids_in_msg = [s.strip() for s in error_msg.split('experiments:')[1].split(';')[0].split(',')]
    assert ids_in_msg == sorted(ids_in_msg)

  def test_short_token_not_resolved(self) -> None:
    tree = _make_tree_with_experiments(['abcd11112222'])
    node = resolve_active_tree_experiment_node(tree, 'abcd')
    assert node is None

  def test_prefix_min_length(self) -> None:
    assert EXPERIMENT_SLUG_PREFIX_MIN_LEN == 8

  def test_no_match_returns_none(self) -> None:
    tree = _make_tree_with_experiments(['abcd11112222'])
    node = resolve_active_tree_experiment_node(tree, 'zzzzzzzz')
    assert node is None

  def test_exact_match_shorter_than_min_prefix(self) -> None:
    tree = _make_tree_with_experiments(['short'])
    node = resolve_active_tree_experiment_node(tree, 'short')
    assert node is not None
    assert node.experiment.id == 'short'

  def test_empty_tree_returns_none(self) -> None:
    tree = _make_tree_with_experiments([])
    node = resolve_active_tree_experiment_node(tree, 'anything1')
    assert node is None


# -- 3.1: flush_error envelope completeness (BUG-002) --


class TestFlushErrorEnvelopeCompleteness:
  """Tests for error_code in Output.flush_error."""

  def test_flush_error_includes_default_error_code(self, capsys: pytest.CaptureFixture) -> None:
    output = Output(use_json=True)
    output.flush_error('something went wrong')
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is False
    assert envelope['error'] == 'something went wrong'
    assert envelope['error_code'] == 'handler_error'

  def test_flush_error_includes_explicit_error_code(self, capsys: pytest.CaptureFixture) -> None:
    output = Output(use_json=True)
    output.flush_error('not found', error_code='custom')
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['error_code'] == 'custom'

  def test_flush_error_preserves_buffered_messages(self, capsys: pytest.CaptureFixture) -> None:
    output = Output(use_json=True)
    output.info('step one')
    output.info('step two')
    output.flush_error('failed after steps')
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['error_code'] == 'handler_error'
    messages = envelope['messages']
    assert len(messages) == 2
    assert messages[0]['message'] == 'step one'
    assert messages[1]['message'] == 'step two'
    assert output._json_buffer == []

  def test_flush_error_noop_when_not_json(self, capsys: pytest.CaptureFixture) -> None:
    output = Output(use_json=False)
    output.flush_error('should not appear')
    captured = capsys.readouterr()
    assert not captured.out


# -- 3.2: ctx.fail envelope (BUG-002) --


class TestCtxFailErrorCode:
  """Tests for error_code threading through CLIContext.fail."""

  def test_ctx_fail_json_includes_error_code(self, capsys: pytest.CaptureFixture) -> None:
    ctx = CLIContext(output=Output(use_json=True))
    with pytest.raises(SystemExit) as exc_info:
      ctx.fail('bad request')
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is False
    assert envelope['error'] == 'bad request'
    assert envelope['error_code'] == 'handler_error'

  def test_ctx_fail_custom_error_code(self, capsys: pytest.CaptureFixture) -> None:
    ctx = CLIContext(output=Output(use_json=True))
    with pytest.raises(SystemExit):
      ctx.fail('missing', error_code='not_found')
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['error_code'] == 'not_found'


# -- 3.3: handler exception path --


class TestHandlerExceptionErrorCode:
  """Tests for error_code in dispatch exception handler."""

  def test_handler_exception_json_has_error_code(self, capsys: pytest.CaptureFixture) -> None:
    class _FailingCommand(Command):
      name = 'boom'

      def forward(self, ctx: Any, args: Any) -> None:
        msg = 'handler kaboom'
        raise RuntimeError(msg)

    cli = CLI()
    cli.boom = _FailingCommand()

    with pytest.raises(SystemExit) as exc_info:
      cli.run_direct(argv=['--json', '--context', 'test', 'boom'])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split('\n') if line.strip()]
    envelope = json.loads(lines[-1])
    assert envelope['ok'] is False
    assert envelope['error_code'] == 'handler_error'
    assert 'handler kaboom' in envelope['error']


# -- 3.4: no-subcommand path (BUG-001) --


class TestNoSubcommandJsonEnvelope:
  """Tests for --json with no subcommand emitting JSON envelope."""

  def test_no_subcommand_json_envelope(self, capsys: pytest.CaptureFixture) -> None:
    cli = AutoPilotCLI()
    with pytest.raises(SystemExit) as exc_info:
      cli.run_direct(argv=['--json'])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is False
    assert envelope['error_code'] == 'cli_usage'
    assert envelope['error'] == 'no command specified'

  def test_no_subcommand_plaintext(self, capsys: pytest.CaptureFixture) -> None:
    cli = AutoPilotCLI()
    with pytest.raises(SystemExit) as exc_info:
      cli.run_direct(argv=[])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert 'usage:' in captured.out.lower()
