"""Tests for Plan 19: query & stabilize UX enhancements.

Covers:
  2.1 -- Query JSON includes created_at and started_at timestamps.
  2.2 -- --metric-gt / --metric-lt reject '=' separator with guidance.
  2.3 -- --compact omits context_log from JSON rows.
  2.4 -- Stabilize prefix no-match produces explicit message.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.stabilize import StabilizeCommand
from autopilot.cli.context import CLIContext
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.core.context import ContextEntry
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
from unittest.mock import MagicMock
import contextlib
import io
import json
import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seed_query_workspace(ws: Path, *, add_context: bool = False) -> FileStore:
  """Create a workspace with a tree containing two experiments."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree = forest.create_tree('main')
  forest.switch('main')

  exp_a = Experiment(experiment_id='exp-a', hypothesis='first')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9, 'loss': 0.1})
  if add_context:
    entry = ContextEntry.create(reason='initial run', source='user')
    exp_a.context_log.record(entry)
  tree.add(Node(experiment=exp_a))

  exp_b = Experiment(experiment_id='exp-b', hypothesis='second')
  tree.add(Node(experiment=exp_b))

  forest.save()
  return store


def _make_stabilize_ctx(tmp_path: Path, *, use_json: bool = True) -> MagicMock:
  """Build a mock CLIContext for stabilize tests."""
  config = AutoPilotConfig(workspace=tmp_path)
  ctx = MagicMock()
  ctx.config = config
  ctx.output = Output(use_json=use_json)
  ctx.wait_timeout_ms = None
  ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
  return ctx


def _stabilize_args(
  experiment_id: str = 'exp-001',
  parameter_prefix: str | None = None,
) -> MagicMock:
  args = MagicMock()
  args.experiment_id = experiment_id
  args.parameter_prefix = parameter_prefix
  return args


# ---------------------------------------------------------------------------
# 2.1 -- query JSON includes created_at / started_at
# ---------------------------------------------------------------------------


class TestQueryTimestamps:
  """created_at and started_at appear in query JSON output."""

  def test_query_json_includes_created_at_and_started_at(self, tmp_path: Path) -> None:
    """Every experiment dict has created_at and started_at keys."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_query_workspace(ws)

    result = run_cli_no_context(ws, ['query'])
    experiments = result['result']['experiments']

    assert len(experiments) == 2
    for exp_row in experiments:
      assert 'created_at' in exp_row
      assert 'started_at' in exp_row

    completed = next(e for e in experiments if e['id'] == 'exp-a')
    assert completed['created_at'] is not None
    assert completed['started_at'] is not None

    pending = next(e for e in experiments if e['id'] == 'exp-b')
    assert pending['created_at'] is not None
    assert pending['started_at'] is None

  def test_query_best_json_includes_timestamps(self, tmp_path: Path) -> None:
    """--best envelope includes created_at and started_at."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_query_workspace(ws)

    result = run_cli_no_context(ws, ['query', '--best', 'accuracy'])
    best = result['result']['best']
    assert 'created_at' in best
    assert 'started_at' in best
    assert best['started_at'] is not None


# ---------------------------------------------------------------------------
# 2.2 -- metric-gt / metric-lt reject '=' separator
# ---------------------------------------------------------------------------


class TestMetricSeparatorValidation:
  """--metric-gt and --metric-lt reject '=' misuse with guidance."""

  def test_query_metric_gt_rejects_equals_separator(self, tmp_path: Path) -> None:
    """--metric-gt with '=' separator triggers ctx.fail with name:value guidance."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_query_workspace(ws)

    from autopilot.cli.context import build_context
    from autopilot.cli.main import build_parser

    parser = build_parser()
    argv = [
      'query',
      '--metric-gt',
      'acc=0.5',
      '--workspace',
      str(ws),
      '--json',
    ]
    parsed = parser.parse_args(argv)
    ctx = build_context(parsed)

    buf = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stdout(buf):
      parsed.handler(ctx, parsed)

    output = buf.getvalue().strip()
    payload = json.loads(output)
    assert payload['ok'] is False
    assert 'name:value' in payload['error']

  def test_query_metric_lt_rejects_equals_separator(self, tmp_path: Path) -> None:
    """--metric-lt with '=' separator triggers ctx.fail with name:value guidance."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_query_workspace(ws)

    from autopilot.cli.context import build_context
    from autopilot.cli.main import build_parser

    parser = build_parser()
    argv = [
      'query',
      '--metric-lt',
      'loss=0.5',
      '--workspace',
      str(ws),
      '--json',
    ]
    parsed = parser.parse_args(argv)
    ctx = build_context(parsed)

    buf = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stdout(buf):
      parsed.handler(ctx, parsed)

    output = buf.getvalue().strip()
    payload = json.loads(output)
    assert payload['ok'] is False
    assert 'name:value' in payload['error']

  def test_query_metric_gt_accepts_colon_separator(self, tmp_path: Path) -> None:
    """--metric-gt with ':' separator works correctly."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_query_workspace(ws)

    result = run_cli_no_context(ws, ['query', '--metric-gt', 'accuracy:0.5'])
    experiments = result['result']['experiments']
    assert len(experiments) == 1
    assert experiments[0]['id'] == 'exp-a'

  def test_query_metric_lt_accepts_colon_separator(self, tmp_path: Path) -> None:
    """--metric-lt with ':' separator works correctly."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_query_workspace(ws)

    result = run_cli_no_context(ws, ['query', '--metric-lt', 'loss:0.5'])
    experiments = result['result']['experiments']
    assert len(experiments) == 1
    assert experiments[0]['id'] == 'exp-a'


# ---------------------------------------------------------------------------
# 2.3 -- --compact omits context_log
# ---------------------------------------------------------------------------


class TestQueryCompact:
  """--compact flag omits context_log from JSON output."""

  def test_query_compact_omits_context_log(self, tmp_path: Path) -> None:
    """With --compact, rows lack context_log; without it, rows include full list."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_query_workspace(ws, add_context=True)

    normal = run_cli_no_context(ws, ['query'])
    normal_rows = normal['result']['experiments']
    assert all('context_log' in row for row in normal_rows)
    completed_row = next(r for r in normal_rows if r['id'] == 'exp-a')
    assert len(completed_row['context_log']) >= 1

    compact = run_cli_no_context(ws, ['query', '--compact'])
    compact_rows = compact['result']['experiments']
    assert all('context_log' not in row for row in compact_rows)

  def test_query_best_compact_omits_context_log(self, tmp_path: Path) -> None:
    """--best with --compact omits context_log from best envelope."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_query_workspace(ws, add_context=True)

    normal = run_cli_no_context(ws, ['query', '--best', 'accuracy'])
    assert 'context_log' in normal['result']['best']

    compact = run_cli_no_context(ws, ['query', '--best', 'accuracy', '--compact'])
    assert 'context_log' not in compact['result']['best']


# ---------------------------------------------------------------------------
# 2.4 -- stabilize prefix no-match message
# ---------------------------------------------------------------------------


class TestStabilizePrefixMessage:
  """Stabilize messaging distinguishes prefix no-match from general empty."""

  def test_stabilize_prefix_no_match_message(self, tmp_path: Path, capsys) -> None:
    """--parameter-prefix with no matching keys produces explicit no-match message."""
    ctx = _make_stabilize_ctx(tmp_path, use_json=False)
    config = ctx.config

    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'main.txt').write_text('content')

    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'prompts': param})

    forest = FileForest(store)
    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='exp-001', hypothesis='test')
    exp.start()
    exp.complete(metrics={})
    tree.add(Node(experiment=exp))
    forest.switch('main')
    forest.save()

    store.snapshot('exp-001', 0)

    cmd = StabilizeCommand()
    cmd.forward(ctx, _stabilize_args('exp-001', parameter_prefix='nomatch'))

    captured = capsys.readouterr()
    assert 'no parameters matched' in captured.out
    assert 'nomatch' in captured.out

  def test_stabilize_no_prefix_empty_message_unchanged(self, tmp_path: Path, capsys) -> None:
    """Without --parameter-prefix, empty copy uses standard 'No files' message."""
    ctx = _make_stabilize_ctx(tmp_path, use_json=False)
    config = ctx.config

    snapshots_dir = config.snapshots_path / 'exp-001'
    snapshots_dir.mkdir(parents=True)
    atomic_write_json(
      snapshots_dir / 'epoch_0.json',
      {'epoch': 0, 'timestamp': '2024-01-01T00:00:00Z', 'entries': {}},
    )

    cmd = StabilizeCommand()
    cmd.forward(ctx, _stabilize_args('exp-001'))

    captured = capsys.readouterr()
    assert 'No files to stabilize' in captured.out

  def test_stabilize_prefix_match_copies_normally(self, tmp_path: Path, capsys) -> None:
    """When prefix matches, files are copied normally."""
    ctx = _make_stabilize_ctx(tmp_path, use_json=True)
    config = ctx.config

    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'main.txt').write_text('content')

    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'prompts': param})

    forest = FileForest(store)
    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='exp-001', hypothesis='test')
    exp.start()
    exp.complete(metrics={})
    tree.add(Node(experiment=exp))
    forest.switch('main')
    forest.save()

    store.snapshot('exp-001', 0)

    cmd = StabilizeCommand()
    cmd.forward(ctx, _stabilize_args('exp-001', parameter_prefix='prompts'))

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert len(envelope['result']['copied']) == 1
