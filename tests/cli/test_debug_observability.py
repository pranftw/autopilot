"""Tests for Plan 22: Observability & Context CLI Enhancements.

Covers:
  - debug executions list default limit raised to 50
  - debug executions list --summary mode
  - debug module-gradients rename (old name absent)
  - debug gradients from context log
  - debug params at epoch
  - debug optimizer-decisions lists files
  - debug cost --detail nested breakdown
  - Trainer fit emit traceback context on failure
  - ConfigSnapshotCallback emits state at fit start
  - Context exemptions for new debug commands
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.command import _BASE_CONTEXT_EXEMPT, CLI
from autopilot.cli.commands.debug import (
  EXEC_LIST_DEFAULT_LIMIT,
  DebugCommand,
  ExecutionsCommand,
)
from autopilot.cli.context import CLIContext
from autopilot.cli.primitives import collect_subcommands
from autopilot.core.callbacks.callback import Callback
from autopilot.core.callbacks.config_snapshot import ConfigSnapshotCallback
from autopilot.core.config import AutoPilotConfig
from autopilot.core.context import ContextEntry
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.tracking.executions import ExecutionRecord, log_execution
from autopilot.tracking.io import atomic_write_json, utc_now_iso
from pathlib import Path
from tests.cli.conftest import make_mock_cli_context, run_cli_no_context
from tests.doubles import NoopEvalModule
from typing import Any
from unittest.mock import MagicMock
import json
import pytest

# -- shared helpers --


def _make_ctx(
  tmp_path: Path,
  *,
  use_json: bool = True,
  experiment: str | None = 'exp-001',
) -> MagicMock:
  """Build a mock CLIContext with config-backed experiment_path."""
  ctx = make_mock_cli_context(
    tmp_path,
    use_json=use_json,
    experiment=experiment,
    module=None,
    wait_timeout_ms=None,
  )
  ctx.experiment_path = CLIContext.experiment_path.__get__(ctx, type(ctx))
  return ctx


def _args(**overrides: Any) -> MagicMock:
  """Build a mock argparse Namespace with given attribute overrides."""
  a = MagicMock()
  for k, v in overrides.items():
    setattr(a, k, v)
  return a


def _seed_executions(ws: Path, count: int) -> None:
  """Seed a workspace with N execution records."""
  config = AutoPilotConfig(workspace=ws)
  config.autopilot_path.mkdir(parents=True, exist_ok=True)
  for i in range(count):
    record = ExecutionRecord(
      timestamp=utc_now_iso(),
      command=f'cmd-{i}',
      args=[],
      duration_ms=float(i * 10),
      exit_code=0,
      context=f'context for record {i}',
      experiment=f'exp-{i}',
    )
    log_execution(config.executions_path, record)


# -- 2.1 tests --


class TestExecListDefaultLimit:
  """EXEC_LIST_DEFAULT_LIMIT raised to 50."""

  def test_default_limit_is_50(self) -> None:
    """Constant reflects plan requirement."""
    assert EXEC_LIST_DEFAULT_LIMIT == 50

  def test_list_shows_50_by_default(self, tmp_path: Path, capsys) -> None:
    """With > 50 records, default list shows last 50."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_executions(ws, 60)
    result = run_cli_no_context(ws, ['debug', 'executions', 'list'])
    rows = result['result']['executions']
    assert len(rows) == 50


class TestExecListSummary:
  """--summary flag produces compact output in text mode."""

  def test_summary_text_mode(self, tmp_path: Path, capsys) -> None:
    """--summary in text mode emits compact columns without args."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_executions(ws, 5)
    ctx = _make_ctx(ws, use_json=False, experiment=None)
    ctx.config = AutoPilotConfig(workspace=ws)
    a = _args(
      limit=EXEC_LIST_DEFAULT_LIMIT,
      filter_command=None,
      failures=False,
      summary=True,
      context_contains=None,
    )
    cmd = ExecutionsCommand()
    cmd.list_executions(ctx, a)
    captured = capsys.readouterr()
    assert 'args' not in captured.out.split('\n')[0] if captured.out else True

  def test_summary_json_unchanged(self, tmp_path: Path, capsys) -> None:
    """--summary flag does not affect JSON output shape."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_executions(ws, 3)
    result = run_cli_no_context(ws, ['debug', 'executions', 'list', '--summary'])
    rows = result['result']['executions']
    assert len(rows) == 3
    assert 'args' in rows[0]


# -- 2.3 tests: gradients rename --


class TestDebugModuleGradientsRename:
  """Old 'gradients' command renamed to 'module-gradients'."""

  def test_module_gradients_subcommand_exists(self) -> None:
    """DebugCommand has a 'module-gradients' subcommand."""
    cmd = DebugCommand()
    sub_names = [meta.name for meta, _ in collect_subcommands(cmd)]
    assert 'module-gradients' in sub_names

  def test_old_gradients_name_replaced(self) -> None:
    """'gradients' subcommand now refers to context-log version, not module."""
    cmd = DebugCommand()
    sub_map = {meta.name: fn for meta, fn in collect_subcommands(cmd)}
    assert 'gradients' in sub_map
    assert sub_map['gradients'].__name__ == 'gradients'
    assert sub_map['module-gradients'].__name__ == 'module_gradients'


# -- 2.3 tests: debug gradients from context log --


class TestDebugGradientsFromContextLog:
  """debug gradients extracts gradient summaries from experiment context log."""

  def test_filters_gradient_entries(self, tmp_path: Path, capsys) -> None:
    """Synthetic context entries with gradient_summaries are returned."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='exp-grad', hypothesis='test')
    exp.start()
    exp.add_context(
      'gradient journal',
      source='trainer',
      metadata={'gradient_summaries': [{'prompt': 'simplify wording'}]},
    )
    exp.add_context('some other entry', source='user')
    exp.complete(metrics={'accuracy': 0.9})
    tree.add(Node(experiment=exp))
    forest.switch('main')
    forest.save()

    result = run_cli_no_context(ws, ['--experiment', 'exp-grad', 'debug', 'gradients'])
    entries = result['result']['entries']
    assert len(entries) == 1
    assert entries[0]['source'] == 'trainer'
    assert 'gradient_summaries' in entries[0]['metadata']

  def test_empty_context_log(self, tmp_path: Path, capsys) -> None:
    """Experiment with no gradient entries yields empty list."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='exp-empty', hypothesis='test')
    exp.start()
    exp.complete(metrics={'accuracy': 0.5})
    tree.add(Node(experiment=exp))
    forest.switch('main')
    forest.save()

    result = run_cli_no_context(ws, ['--experiment', 'exp-empty', 'debug', 'gradients'])
    assert result['result']['count'] == 0


# -- 2.2 tests: debug params --


class TestDebugParams:
  """debug params reads parameter state at a store epoch."""

  def test_reads_snapshot_content(self, tmp_path: Path, capsys) -> None:
    """Known fixture snapshot epoch prints expected file key and content."""
    from autopilot.ai.parameter import PathParameter

    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    config.objects_path.mkdir(parents=True, exist_ok=True)

    source_dir = ws / 'params'
    source_dir.mkdir()
    (source_dir / 'prompt.txt').write_text('hello world')

    store = FileStore(config)
    param = PathParameter(source=str(source_dir), pattern='*.txt')
    store.register_parameters({'prompt': param})
    store.snapshot('exp-params', 0, context='initial')

    forest = FileForest(store)
    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='exp-params', hypothesis='test')
    exp.start()
    exp.complete(metrics={})
    tree.add(Node(experiment=exp))
    forest.switch('main')
    forest.save()

    result = run_cli_no_context(ws, ['--experiment', 'exp-params', 'debug', 'params', '0'])
    files = result['result']['files']
    assert len(files) > 0
    key = next(iter(files))
    assert 'prompt' in key
    assert files[key]['content'] == 'hello world'

  def test_missing_experiment_fails(self, tmp_path: Path) -> None:
    """Missing experiment slug produces failure."""
    ctx = _make_ctx(tmp_path, experiment=None)
    ctx.output = MagicMock()
    cmd = DebugCommand()
    with pytest.raises(SystemExit):
      cmd.params(ctx, _args(epoch='0'))


# -- 2.4 tests: optimizer-decisions --


class TestDebugOptimizerDecisions:
  """debug optimizer-decisions surfaces .optimization/ artifacts."""

  def test_lists_files(self, tmp_path: Path, capsys) -> None:
    """Temp .optimization/ file appears in output."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    exp_dir = config.experiment_path(slug='exp-001')
    opt_dir = exp_dir / '.optimization'
    opt_dir.mkdir(parents=True)
    (opt_dir / 'epoch_0.md').write_text('# Epoch 0 feedback\nSimplify prompts.')

    ctx = _make_ctx(ws)
    cmd = DebugCommand()
    cmd.optimizer_decisions(ctx, _args())

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    files = envelope['result']['files']
    assert len(files) == 1
    assert files[0]['name'] == 'epoch_0.md'
    assert files[0]['size'] > 0
    assert 'Simplify' in (files[0].get('preview') or '')

  def test_missing_dir_empty_result(self, tmp_path: Path, capsys) -> None:
    """Missing .optimization/ yields empty result, not error."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.experiment_path(slug='exp-001').mkdir(parents=True)

    ctx = _make_ctx(ws)
    cmd = DebugCommand()
    cmd.optimizer_decisions(ctx, _args())

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['count'] == 0


# -- 2.5 tests: cost --detail --


class TestDebugCostDetail:
  """debug cost --detail renders nested breakdown."""

  def test_nested_breakdown_renders(self, tmp_path: Path, capsys) -> None:
    """Nested dict in cost summary expands under --detail."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    exp_dir = config.experiment_path(slug='exp-001')
    exp_dir.mkdir(parents=True)
    cost_data = {
      'total_usd': 5.0,
      'breakdown': {
        'generation': 3.0,
        'judging': 2.0,
      },
    }
    atomic_write_json(exp_dir / 'cost_summary.json', cost_data)

    ctx = _make_ctx(ws, use_json=False)
    cmd = DebugCommand()
    cmd.cost(ctx, _args(detail=True))
    captured = capsys.readouterr()
    assert 'breakdown.generation' in captured.out
    assert 'breakdown.judging' in captured.out

  def test_detail_json_includes_full_data(self, tmp_path: Path, capsys) -> None:
    """JSON mode includes nested data regardless of --detail."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    exp_dir = config.experiment_path(slug='exp-001')
    exp_dir.mkdir(parents=True)
    cost_data = {
      'total_usd': 5.0,
      'breakdown': {'generation': 3.0, 'judging': 2.0},
    }
    atomic_write_json(exp_dir / 'cost_summary.json', cost_data)

    ctx = _make_ctx(ws)
    cmd = DebugCommand()
    cmd.cost(ctx, _args(detail=False))

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['breakdown']['generation'] == 3.0


# -- 2.6 tests: fit traceback context --


class ContextCapture(Callback):
  """Callback that captures context entries emitted during training."""

  def __init__(self) -> None:
    super().__init__()
    self.entries: list[ContextEntry] = []

  def on_context_emit(self, trainer, module, entry) -> None:
    """Record context entry."""
    self.entries.append(entry)


class FailingModule(NoopEvalModule):
  """Module that raises during training_step."""

  def training_step(self, batch, batch_idx):
    msg = 'intentional training failure'
    raise RuntimeError(msg)


class TestFitEmitTracebackContext:
  """Trainer emits traceback in context metadata on failure path."""

  def test_traceback_in_failure_context(self) -> None:
    """Exception during fit -> context metadata has 'traceback' and 'exception_type'."""
    exp = Experiment(experiment_id='exp-fail', hypothesis='test')
    capture = ContextCapture()
    trainer = Trainer(experiment=exp, callbacks=[capture])

    module = FailingModule()
    with pytest.raises(RuntimeError, match='intentional training failure'):
      trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

    failure_entries = [
      e for e in capture.entries if e.source == 'trainer' and 'traceback' in e.metadata
    ]
    assert len(failure_entries) >= 1
    entry = failure_entries[0]
    assert 'RuntimeError' in entry.metadata['traceback']
    assert entry.metadata['exception_type'] == 'RuntimeError'
    assert 'intentional training failure' in entry.reason


# -- 2.7 tests: ConfigSnapshotCallback --


class TestConfigSnapshotCallback:
  """ConfigSnapshotCallback emits module state at fit start."""

  def test_emits_state_dict(self) -> None:
    """Callback emits context with module_state_dict in metadata."""
    exp = Experiment(experiment_id='exp-snap', hypothesis='test')
    capture = ContextCapture()
    snapshot_cb = ConfigSnapshotCallback()
    trainer = Trainer(experiment=exp, callbacks=[capture, snapshot_cb])

    module = NoopEvalModule()
    module.test_param = Parameter()

    trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

    snapshot_entries = [e for e in capture.entries if e.reason == 'config snapshot at fit start']
    assert len(snapshot_entries) == 1
    metadata = snapshot_entries[0].metadata
    assert 'module_state_dict' in metadata


# -- context exemption tests --


class TestDebugSubcommandsContextExemptions:
  """New read-only debug commands are context-exempt."""

  def test_module_gradients_exempt(self) -> None:
    """'debug module-gradients' is exempt from --context."""
    assert 'debug module-gradients' in _BASE_CONTEXT_EXEMPT

  def test_gradients_exempt(self) -> None:
    """'debug gradients' is exempt from --context."""
    assert 'debug gradients' in _BASE_CONTEXT_EXEMPT

  def test_params_exempt(self) -> None:
    """'debug params' is exempt from --context."""
    assert 'debug params' in _BASE_CONTEXT_EXEMPT

  def test_optimizer_decisions_exempt(self) -> None:
    """'debug optimizer-decisions' is exempt from --context."""
    assert 'debug optimizer-decisions' in _BASE_CONTEXT_EXEMPT

  def test_all_new_commands_via_requires_context(self) -> None:
    """CLI.requires_context returns False for all new read-only debug commands."""
    cli = CLI()
    new_commands = [
      'debug module-gradients',
      'debug gradients',
      'debug params',
      'debug optimizer-decisions',
    ]
    for cmd in new_commands:
      assert not cli.requires_context(cmd), f'{cmd!r} should be exempt'
