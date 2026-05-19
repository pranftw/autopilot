"""Tests for CLI UX fixes (plan 10).

Covers BUG-011 (JSON-pure execute), BUG-015 (--proposal-id required),
BUG-016 (--epoch defaults to None), BUG-020 (--store in merge commands),
BUG-021 (identical-digest excluded from conflicts), FRICTION-007
(execution record captures experiment id).
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.execute import ExecuteCommand
from autopilot.cli.commands.propose import ProposeCommand
from autopilot.cli.commands.store.helpers import open_forest_store
from autopilot.cli.context import CLIContext
from autopilot.cli.main import build_parser
from autopilot.cli.output import Output
from autopilot.cli.primitives import ArgparseCLIError
from autopilot.core.config import AutoPilotConfig
from autopilot.core.store.types import MergeClassification, MergeStrategy
from autopilot.tracking.executions import create_execution_record
from pathlib import Path
import argparse
import contextlib
import io
import json
import pytest

# -- BUG-011: execute JSON output purity ------------------------------------


def test_execute_json_output_is_pure_json(tmp_path: Path) -> None:
  """In JSON mode, subprocess stdout must NOT appear on sys.stdout directly.

  Only the JSON envelope should be written; subprocess output is inside
  the result fields.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  ctx = CLIContext(
    workspace=ws,
    config=config,
    output=Output(use_json=True),
  )
  cmd = ExecuteCommand()
  args = argparse.Namespace(
    code="print('hello_from_subprocess')",
    module=None,
    extra_args=[],
    handler=cmd,
  )

  stdout_buf = io.StringIO()
  with contextlib.redirect_stdout(stdout_buf), contextlib.suppress(SystemExit):
    cmd.forward(ctx, args)

  raw_output = stdout_buf.getvalue()
  assert 'hello_from_subprocess' not in raw_output.split('\n')[0] or raw_output.startswith('{')
  envelope = json.loads(raw_output)
  assert envelope['result']['stdout'].strip() == 'hello_from_subprocess'


def test_execute_text_mode_shows_subprocess_output(tmp_path: Path) -> None:
  """In text mode (no --json), subprocess stdout appears on sys.stdout."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  ctx = CLIContext(
    workspace=ws,
    config=config,
    output=Output(use_json=False),
  )
  cmd = ExecuteCommand()
  args = argparse.Namespace(
    code="print('visible_output')",
    module=None,
    extra_args=[],
    handler=cmd,
  )

  stdout_buf = io.StringIO()
  with contextlib.redirect_stdout(stdout_buf), contextlib.suppress(SystemExit):
    cmd.forward(ctx, args)

  raw_output = stdout_buf.getvalue()
  assert 'visible_output' in raw_output


# -- BUG-015: --proposal-id required ----------------------------------------


def test_propose_verify_requires_proposal_id() -> None:
  """Omitting --proposal-id on propose verify exits with argparse error."""
  parser = build_parser()
  with pytest.raises(ArgparseCLIError) as exc_info:
    parser.parse_args(
      ['propose', 'verify', '--experiment', 'exp1', '--epoch', '0', '--context', 'test']
    )
  assert exc_info.value.exit_code != 0


# -- BUG-016: --epoch default -----------------------------------------------


def test_epoch_flag_defaults_to_none() -> None:
  """When --epoch is omitted, parsed args have epoch=None."""
  parser = build_parser()
  parsed = parser.parse_args(['status', '--workspace', '/tmp', '--context', 'test'])
  assert parsed.epoch is None


def test_epoch_flag_accepts_explicit_zero() -> None:
  """--epoch 0 is explicitly parsed as integer 0, not None."""
  parser = build_parser()
  parsed = parser.parse_args(['status', '--workspace', '/tmp', '--epoch', '0', '--context', 'test'])
  assert parsed.epoch == 0


# -- BUG-020: merge commands respect --store flag ----------------------------


def test_merge_commands_respect_store_flag(tmp_path: Path) -> None:
  """When --store is passed to merge commands, that path overrides the default."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  custom_store = tmp_path / 'custom_store'
  custom_store.mkdir(parents=True)

  config = AutoPilotConfig(workspace=ws)
  ctx = CLIContext(
    workspace=ws,
    config=config,
    output=Output(use_json=True),
  )

  args = argparse.Namespace(store=str(custom_store))
  open_forest_store(ctx, args)

  assert ctx.config.store_path == custom_store


# -- BUG-021: identical digest excluded from conflict count ------------------


def test_identical_digest_excluded_from_conflicts(tmp_path: Path) -> None:
  """Keys changed on both sides but with same digest are not conflicts."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)

  source_dir = tmp_path / 'src'
  source_dir.mkdir()
  (source_dir / 'file.txt').write_text('content_v2')

  param = PathParameter(source=str(source_dir), pattern='**/*')
  store = FileStore(config)
  store.register_parameters({'source': param})

  store.snapshot('exp_a', 0)
  store.branch('exp_b')
  (source_dir / 'file.txt').write_text('same_new_content')
  store.snapshot('exp_a', 1)
  (source_dir / 'file.txt').write_text('same_new_content')
  store.snapshot('exp_b', 1)

  result = store.merge_analysis('exp_a', 'exp_b')
  assert result.conflict_count == 0
  assert not result.has_conflicts


def test_divergent_digest_counted_as_conflict(tmp_path: Path) -> None:
  """Keys changed on both sides with different digests are counted as conflicts."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)

  source_dir = tmp_path / 'src'
  source_dir.mkdir()
  (source_dir / 'file.txt').write_text('original')

  param = PathParameter(source=str(source_dir), pattern='**/*')
  store = FileStore(config)
  store.register_parameters({'source': param})

  store.snapshot('exp_a', 0)
  store.branch('exp_b')
  (source_dir / 'file.txt').write_text('ours_change')
  store.snapshot('exp_a', 1)
  (source_dir / 'file.txt').write_text('theirs_change')
  store.snapshot('exp_b', 1)

  result = store.merge_analysis('exp_a', 'exp_b')
  assert result.conflict_count == 1
  assert result.has_conflicts
  assert result.classification == MergeClassification.conflict


# -- FRICTION-007: execution record captures experiment ----------------------


def test_execution_record_captures_experiment_id() -> None:
  """ExecutionRecord includes the experiment field from CLI context."""
  record = create_execution_record(
    command='optimize train',
    args=['--experiment', 'my-exp'],
    duration_ms=42.0,
    exit_code=0,
    experiment='my-exp',
    project='proj',
    context='test run',
  )
  assert record.experiment == 'my-exp'
  assert record.project == 'proj'
  assert record.context == 'test run'

  serialized = record.to_dict()
  assert serialized['experiment'] == 'my-exp'


# -- E2E: propose create then verify ----------------------------------------


def test_propose_create_then_verify_e2e(tmp_path: Path) -> None:
  """End-to-end: create a proposal then verify it.

  Verifies the verify command reads the proposal and produces a verdict.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  exp_dir = config.experiment_path(slug='test-exp')
  exp_dir.mkdir(parents=True)

  ctx = CLIContext(
    workspace=ws,
    config=config,
    experiment='test-exp',
    epoch=0,
    output=Output(use_json=True),
    context='test',
  )

  propose_cmd = ProposeCommand()

  create_args = argparse.Namespace(
    target=None,
    hypothesis='test hypothesis',
    category='test',
    handler=propose_cmd,
    propose_action='create',
  )

  create_buf = io.StringIO()
  with contextlib.redirect_stdout(create_buf):
    propose_cmd.create(ctx, create_args)

  create_output = json.loads(create_buf.getvalue())
  proposal_id = create_output['result']['proposal_id']

  verify_args = argparse.Namespace(
    proposal_id=proposal_id,
    handler=propose_cmd,
    propose_action='verify',
    higher_is_better=None,
    lower_is_better=None,
  )

  ctx_verify = CLIContext(
    workspace=ws,
    config=config,
    experiment='test-exp',
    epoch=0,
    output=Output(use_json=True),
    context='verify test',
  )

  verify_buf = io.StringIO()
  with contextlib.redirect_stdout(verify_buf):
    propose_cmd.verify(ctx_verify, verify_args)

  verify_output = json.loads(verify_buf.getvalue())
  assert verify_output['result']['proposal_id'] == proposal_id
  assert verify_output['result']['verdict'] in {'improved', 'regressed', 'inconclusive'}


# -- E2E: merge full workflow ------------------------------------------------


def test_merge_full_workflow_e2e(tmp_path: Path) -> None:
  """End-to-end merge workflow: analysis -> preview -> resolve -> apply.

  Sets up two divergent experiments, performs the full merge cycle with
  token coupling, and verifies a new epoch is persisted.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)

  source_dir = tmp_path / 'src'
  source_dir.mkdir()
  (source_dir / 'shared.txt').write_text('base')
  (source_dir / 'diverge.txt').write_text('base')

  param = PathParameter(source=str(source_dir), pattern='**/*')
  store = FileStore(config)
  store.register_parameters({'source': param})

  store.snapshot('main', 0)
  store.branch('feature')

  (source_dir / 'diverge.txt').write_text('main_change')
  store.snapshot('main', 1)

  (source_dir / 'diverge.txt').write_text('feature_change')
  store.snapshot('feature', 1)

  analysis = store.merge_analysis('main', 'feature')
  assert analysis.has_conflicts

  merge_index = store.merge_preview('main', 'feature', strategy=MergeStrategy.normal)
  assert not merge_index.is_resolved()
  token = merge_index.preview_token
  assert token is not None

  for key in list(merge_index.conflicts):
    merge_index.resolve_ours(key)

  assert merge_index.is_resolved()

  manifest = store.merge_apply(merge_index)
  assert manifest.epoch == 2

  log_entries = store.log('main')
  assert log_entries[-1].epoch == 2
