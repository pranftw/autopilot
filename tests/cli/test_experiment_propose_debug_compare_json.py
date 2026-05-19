"""Tests for CLI fixes sub-plan 03 (BUG-010 through BUG-020).

Covers:
  - BUG-010: optimize loop JSON final_metrics from epochs[-1]
  - BUG-011: optimize train/validate JSON includes metrics, error_message, feedback
  - BUG-012: optimize set-hparams persists in experiment notes
  - BUG-013: experiment remove wraps Tree.remove
  - BUG-014: experiment notes show/write
  - BUG-015: experiment compare always emits result
  - BUG-016: execute JSON includes stdout/stderr
  - BUG-017: debug executions data in result not messages
  - BUG-018: report compare uses MetricsComparator with significance
  - BUG-019: query --all-trees cross-tree mode
  - BUG-020: report compare multi-way (N experiments)
"""

from autopilot.cli.commands.debug import ExecutionsCommand
from autopilot.cli.commands.execute import ExecuteCommand
from autopilot.cli.commands.experiment.compare import ExperimentCompare
from autopilot.cli.commands.experiment.lifecycle import ExperimentRemove
from autopilot.cli.commands.experiment.metadata import ExperimentNotes
from autopilot.cli.commands.optimize import OptimizeCommand
from autopilot.cli.commands.query import QueryCommand
from autopilot.cli.commands.report.command import ReportCommand
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.tracking.executions import ExecutionRecord, log_execution
from autopilot.tracking.io import utc_now_iso
from pathlib import Path
from tests.cli.conftest import seed_tree_with_experiments
from unittest.mock import MagicMock, patch
import json
import pytest


def _ws_ctx(tmp_path: Path, use_json: bool = True) -> CLIContext:
  config = AutoPilotConfig(workspace=tmp_path)
  return CLIContext(workspace=tmp_path, config=config, output=Output(use_json=use_json))


def _mock_observation(
  success: bool = True,
  metrics: dict | None = None,
  error_message: str | None = None,
  feedback: str | None = None,
):
  obs = MagicMock()
  obs.success = success
  obs.metrics = metrics or {}
  obs.error_message = error_message
  obs.feedback = feedback
  obs.metadata = {}
  return obs


# Section 4.1: Optimize (BUG-010, BUG-011, BUG-012)


class TestOptimizeLoopJson:
  """BUG-010: optimize loop JSON final_metrics from last epoch."""

  def test_final_metrics_non_empty(self, tmp_path: Path, capsys) -> None:
    """Last epoch has metrics -> final_metrics is non-empty dict."""
    ctx = _ws_ctx(tmp_path)
    ctx.experiment = 'test-exp'
    ctx.module = MagicMock()
    ctx.datamodule = None

    mock_trainer = MagicMock()
    mock_trainer.callbacks = []
    mock_trainer.dry_run = False
    mock_trainer.logger = None
    mock_trainer.policy = None
    mock_trainer.experiment = None
    mock_trainer.config = ctx.config
    mock_trainer.accumulate_grad_batches = 1
    mock_trainer.fit.return_value = {
      'epochs': [
        {'metrics': {'accuracy': 0.7}},
        {'metrics': {'accuracy': 0.9}},
      ],
      'total_epochs': 2,
      'stop_reason': None,
      'last_good_epoch': 1,
    }
    ctx.trainer = mock_trainer

    exp_dir = ctx.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = OptimizeCommand()
    args = MagicMock(max_epochs=2, strategy='conservative')
    with patch('autopilot.cli.commands.optimize._build_loop_trainer', return_value=mock_trainer):
      cmd.loop(ctx, args)

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['final_metrics'] == {'accuracy': 0.9}

  def test_final_metrics_matches_last_epoch(self, tmp_path: Path, capsys) -> None:
    """final_metrics equals epochs[-1]['metrics']."""
    ctx = _ws_ctx(tmp_path)
    ctx.experiment = 'test-exp'
    ctx.module = MagicMock()
    ctx.datamodule = None

    epoch_metrics = {'loss': 0.1, 'f1': 0.95}
    mock_trainer = MagicMock()
    mock_trainer.callbacks = []
    mock_trainer.dry_run = False
    mock_trainer.logger = None
    mock_trainer.policy = None
    mock_trainer.experiment = None
    mock_trainer.config = ctx.config
    mock_trainer.accumulate_grad_batches = 1
    mock_trainer.fit.return_value = {
      'epochs': [{'metrics': epoch_metrics}],
      'total_epochs': 1,
      'stop_reason': 'plateau',
      'last_good_epoch': 0,
    }
    ctx.trainer = mock_trainer

    exp_dir = ctx.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = OptimizeCommand()
    args = MagicMock(max_epochs=5, strategy='conservative')
    with patch('autopilot.cli.commands.optimize._build_loop_trainer', return_value=mock_trainer):
      cmd.loop(ctx, args)

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['final_metrics'] == epoch_metrics
    assert envelope['result']['stop_reason'] == 'plateau'

  def test_includes_stop_reason(self, tmp_path: Path, capsys) -> None:
    """Result includes stop_reason field."""
    ctx = _ws_ctx(tmp_path)
    ctx.experiment = 'test-exp'
    ctx.module = MagicMock()
    ctx.datamodule = None

    mock_trainer = MagicMock()
    mock_trainer.callbacks = []
    mock_trainer.dry_run = False
    mock_trainer.logger = None
    mock_trainer.policy = None
    mock_trainer.experiment = None
    mock_trainer.config = ctx.config
    mock_trainer.accumulate_grad_batches = 1
    mock_trainer.fit.return_value = {
      'epochs': [{'metrics': {}}],
      'total_epochs': 1,
      'stop_reason': 'callback_stop',
      'last_good_epoch': 0,
    }
    ctx.trainer = mock_trainer

    exp_dir = ctx.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = OptimizeCommand()
    args = MagicMock(max_epochs=10, strategy='conservative')
    with patch('autopilot.cli.commands.optimize._build_loop_trainer', return_value=mock_trainer):
      cmd.loop(ctx, args)

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert 'stop_reason' in envelope['result']

  def test_empty_epochs_guard(self, tmp_path: Path, capsys) -> None:
    """No epochs -> final_metrics is {} without raising."""
    ctx = _ws_ctx(tmp_path)
    ctx.experiment = 'test-exp'
    ctx.module = MagicMock()
    ctx.datamodule = None

    mock_trainer = MagicMock()
    mock_trainer.callbacks = []
    mock_trainer.dry_run = False
    mock_trainer.logger = None
    mock_trainer.policy = None
    mock_trainer.experiment = None
    mock_trainer.config = ctx.config
    mock_trainer.accumulate_grad_batches = 1
    mock_trainer.fit.return_value = {
      'epochs': [],
      'total_epochs': 0,
      'stop_reason': 'callback_stop',
      'last_good_epoch': -1,
    }
    ctx.trainer = mock_trainer

    exp_dir = ctx.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = OptimizeCommand()
    args = MagicMock(max_epochs=10, strategy='conservative')
    with patch('autopilot.cli.commands.optimize._build_loop_trainer', return_value=mock_trainer):
      cmd.loop(ctx, args)

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['final_metrics'] == {}


class TestOptimizeTrainJson:
  """BUG-011: optimize train JSON includes metrics, error_message."""

  def test_has_metrics_key(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    ctx.experiment = 'test-exp'
    ctx.split = None
    ctx.epoch = 0
    ctx.dry_run = False

    obs = _mock_observation(success=True, metrics={'accuracy': 0.8})
    ctx.module = MagicMock(return_value=obs)
    ctx.trainer = MagicMock()

    exp_dir = ctx.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = OptimizeCommand()
    args = MagicMock(limit=0)
    cmd.train.forward(ctx, args)

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert 'metrics' in envelope['result']
    assert envelope['result']['metrics'] == {'accuracy': 0.8}

  def test_has_error_when_failed(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    ctx.experiment = 'test-exp'
    ctx.split = None
    ctx.epoch = 0
    ctx.dry_run = False

    obs = _mock_observation(success=False, error_message='timeout exceeded')
    ctx.module = MagicMock(return_value=obs)
    ctx.trainer = MagicMock()

    exp_dir = ctx.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = OptimizeCommand()
    args = MagicMock(limit=0)
    cmd.train.forward(ctx, args)

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['error_message'] == 'timeout exceeded'


class TestOptimizeValidateJson:
  """BUG-011: optimize validate JSON includes metrics, feedback."""

  def test_has_metrics(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    ctx.experiment = 'test-exp'
    ctx.epoch = 0
    ctx.dry_run = False

    obs = _mock_observation(success=True, metrics={'f1': 0.85}, feedback='good improvement')
    ctx.module = MagicMock(return_value=obs)
    ctx.trainer = MagicMock()

    exp_dir = ctx.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = OptimizeCommand()
    args = MagicMock()
    cmd.validate.forward(ctx, args)

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert 'metrics' in envelope['result']

  def test_has_feedback(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    ctx.experiment = 'test-exp'
    ctx.epoch = 0
    ctx.dry_run = False

    obs = _mock_observation(success=True, feedback='needs more diversity')
    ctx.module = MagicMock(return_value=obs)
    ctx.trainer = MagicMock()

    exp_dir = ctx.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = OptimizeCommand()
    args = MagicMock()
    cmd.validate.forward(ctx, args)

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert 'feedback' in envelope['result']
    assert envelope['result']['feedback'] == 'needs more diversity'


class TestOptimizeSetHparams:
  """BUG-012: set-hparams persists in experiment notes."""

  def _setup_forest(self, tmp_path: Path) -> tuple[CLIContext, str]:
    ctx = _ws_ctx(tmp_path)
    ctx.config.store_path.mkdir(parents=True, exist_ok=True)
    from autopilot.ai.forest import FileForest
    from autopilot.ai.store.file_store import FileStore

    store = FileStore(ctx.config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [{'id': 'exp-1', 'hypothesis': 'test', 'status': 'running'}],
    )
    ctx.experiment = 'exp-1'
    ctx.hyperparams_file = None
    return ctx, 'exp-1'

  def test_persists_in_notes(self, tmp_path: Path, capsys) -> None:
    ctx, eid = self._setup_forest(tmp_path)
    cmd = OptimizeCommand()
    args = MagicMock(values='{"lr": 0.001, "batch_size": 32}')
    cmd.set_hparams(ctx, args)

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['hparams'] == {'lr': 0.001, 'batch_size': 32}
    assert envelope['result']['experiment_id'] == eid

    forest = load_forest(ctx)
    tree = forest.active
    assert tree is not None
    node = tree.get(eid)
    assert node is not None
    assert node.experiment.notes is not None
    notes = json.loads(node.experiment.notes)
    assert notes['hparams'] == {'lr': 0.001, 'batch_size': 32}

  def test_idempotent_merge(self, tmp_path: Path, capsys) -> None:
    ctx, _eid = self._setup_forest(tmp_path)
    cmd = OptimizeCommand()

    args = MagicMock(values='{"lr": 0.01}')
    cmd.set_hparams(ctx, args)
    capsys.readouterr()

    args2 = MagicMock(values='{"batch_size": 64}')
    cmd.set_hparams(ctx, args2)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['hparams'] == {'lr': 0.01, 'batch_size': 64}

  def test_invalid_json_fails(self, tmp_path: Path) -> None:
    ctx, _ = self._setup_forest(tmp_path)
    cmd = OptimizeCommand()
    args = MagicMock(values='{invalid json')
    with pytest.raises(SystemExit) as exc_info:
      cmd.set_hparams(ctx, args)
    assert exc_info.value.code == 1


# Section 4.2: Experiment (BUG-013, BUG-014, BUG-015)


class TestExperimentRemove:
  """BUG-013: experiment remove wraps Tree.remove."""

  def _setup(self, tmp_path: Path) -> CLIContext:
    ctx = _ws_ctx(tmp_path)
    ctx.config.store_path.mkdir(parents=True, exist_ok=True)
    from autopilot.ai.forest import FileForest
    from autopilot.ai.store.file_store import FileStore

    store = FileStore(ctx.config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [
        {'id': 'root', 'hypothesis': 'root exp', 'status': 'completed', 'metrics': {}},
        {'id': 'child1', 'hypothesis': 'child', 'status': 'running', 'parent': 'root'},
        {'id': 'child2', 'hypothesis': 'child2', 'status': 'pending', 'parent': 'root'},
      ],
    )
    return ctx

  def test_remove_deletes_node(self, tmp_path: Path, capsys) -> None:
    ctx = self._setup(tmp_path)
    cmd = ExperimentRemove()
    args = MagicMock(id='child1', cascade=False)
    cmd.forward(ctx, args)

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['removed'] == 'child1'

    forest = load_forest(ctx)
    tree = forest.active
    assert tree is not None
    assert tree.get('child1') is None

  def test_remove_cascade_removes_descendants(self, tmp_path: Path, capsys) -> None:
    ctx = self._setup(tmp_path)
    cmd = ExperimentRemove()
    args = MagicMock(id='root', cascade=True)
    cmd.forward(ctx, args)

    forest = load_forest(ctx)
    tree = forest.active
    assert tree is not None
    assert tree.get('root') is None
    assert tree.get('child1') is None
    assert tree.get('child2') is None

  def test_remove_unknown_id_fails(self, tmp_path: Path) -> None:
    ctx = self._setup(tmp_path)
    cmd = ExperimentRemove()
    args = MagicMock(id='nonexistent', cascade=False)
    with pytest.raises(SystemExit) as exc_info:
      cmd.forward(ctx, args)
    assert exc_info.value.code == 1


class TestExperimentNotes:
  """BUG-014: experiment notes show/write."""

  def _setup(self, tmp_path: Path) -> CLIContext:
    ctx = _ws_ctx(tmp_path)
    ctx.config.store_path.mkdir(parents=True, exist_ok=True)
    from autopilot.ai.forest import FileForest
    from autopilot.ai.store.file_store import FileStore

    store = FileStore(ctx.config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [{'id': 'exp-n', 'hypothesis': 'notes test', 'status': 'running'}],
    )
    return ctx

  def test_write_then_show(self, tmp_path: Path, capsys) -> None:
    ctx = self._setup(tmp_path)
    notes_cmd = ExperimentNotes()

    args_write = MagicMock(id='exp-n', body='hello notes', file=None)
    notes_cmd.write.forward(ctx, args_write)
    capsys.readouterr()

    args_show = MagicMock(id='exp-n')
    notes_cmd.show.forward(ctx, args_show)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['notes'] == 'hello notes'

  def test_unknown_id_fails(self, tmp_path: Path) -> None:
    ctx = self._setup(tmp_path)
    notes_cmd = ExperimentNotes()
    args = MagicMock(id='ghost')
    with pytest.raises(SystemExit) as exc_info:
      notes_cmd.show.forward(ctx, args)
    assert exc_info.value.code == 1


class TestExperimentCompareResult:
  """BUG-015: experiment compare always emits result."""

  def _setup(self, tmp_path: Path) -> CLIContext:
    ctx = _ws_ctx(tmp_path, use_json=False)
    ctx.config.store_path.mkdir(parents=True, exist_ok=True)
    from autopilot.ai.forest import FileForest
    from autopilot.ai.store.file_store import FileStore

    store = FileStore(ctx.config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [
        {'id': 'a', 'hypothesis': 'A', 'status': 'completed', 'metrics': {'acc': 0.8}},
        {'id': 'b', 'hypothesis': 'B', 'status': 'completed', 'metrics': {'acc': 0.9}},
      ],
    )
    return ctx

  def test_json_has_result_wrapper(self, tmp_path: Path, capsys) -> None:
    ctx = self._setup(tmp_path)
    ctx.output = Output(use_json=True)
    cmd = ExperimentCompare()
    args = MagicMock(a='a', b='b', weights=None)
    cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert 'result' in envelope
    assert 'deltas' in envelope['result']

  def test_table_mode_still_populates_result(self, tmp_path: Path, capsys) -> None:
    """Non-JSON table mode still calls ctx.output.result() with structured data."""
    ctx = self._setup(tmp_path)
    mock_output = MagicMock(spec=Output)
    mock_output.use_json = False
    ctx.output = mock_output
    cmd = ExperimentCompare()
    args = MagicMock(a='a', b='b', weights=None)
    cmd.forward(ctx, args)
    mock_output.result.assert_called_once()
    payload = mock_output.result.call_args[0][0]
    assert payload['a'] == 'a'
    assert payload['b'] == 'b'
    assert 'deltas' in payload

  def test_deltas_structure(self, tmp_path: Path, capsys) -> None:
    ctx = self._setup(tmp_path)
    ctx.output = Output(use_json=True)
    cmd = ExperimentCompare()
    args = MagicMock(a='a', b='b', weights=None)
    cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    deltas = envelope['result']['deltas']
    deltas_by_metric = {d['metric']: d for d in deltas}
    assert 'acc' in deltas_by_metric
    assert deltas_by_metric['acc']['delta'] == pytest.approx(0.1)


# Section 4.3: Execute (BUG-016)


def _extract_json_envelope(text: str) -> dict:
  """Extract JSON envelope from output that may have subprocess text prepended."""
  start = text.find('{')
  if start == -1:
    return {}
  return json.loads(text[start:])


class TestExecuteJson:
  """BUG-016: execute JSON includes stdout/stderr."""

  def test_includes_stdout(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    cmd = ExecuteCommand()
    args = MagicMock(code='print("hello")', module=None, extra_args=[])

    with patch('subprocess.run') as mock_run:
      mock_run.return_value = MagicMock(stdout='hello\n', stderr='', returncode=0)
      cmd.forward(ctx, args)

    captured = capsys.readouterr()
    envelope = _extract_json_envelope(captured.out)
    assert 'stdout' in envelope['result']
    assert 'hello' in envelope['result']['stdout']

  def test_includes_stderr(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    cmd = ExecuteCommand()
    args = MagicMock(code='import sys; sys.stderr.write("err")', module=None, extra_args=[])

    with patch('subprocess.run') as mock_run:
      mock_run.return_value = MagicMock(stdout='', stderr='err', returncode=0)
      cmd.forward(ctx, args)

    captured = capsys.readouterr()
    envelope = _extract_json_envelope(captured.out)
    assert 'stderr' in envelope['result']
    assert envelope['result']['stderr'] == 'err'

  def test_streams_on_nonzero_exit(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    cmd = ExecuteCommand()
    args = MagicMock(code='exit(1)', module=None, extra_args=[])

    with patch('subprocess.run') as mock_run:
      mock_run.return_value = MagicMock(
        stdout='partial output', stderr='error detail', returncode=1
      )
      with pytest.raises(SystemExit):
        cmd.forward(ctx, args)

    captured = capsys.readouterr()
    envelope = _extract_json_envelope(captured.out)
    assert envelope['result']['stdout'] == 'partial output'
    assert envelope['result']['stderr'] == 'error detail'


# Section 4.4: Debug (BUG-017)


class TestDebugExecutions:
  """BUG-017: debug executions data in result not messages."""

  def _seed_executions(self, ctx: CLIContext) -> None:
    ctx.config.executions_path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(3):
      rec = ExecutionRecord(
        timestamp=utc_now_iso(),
        command=f'cmd-{i}',
        args=['autopilot', f'cmd-{i}'],
        duration_ms=100.0 + i,
        exit_code=0,
        stdout=f'output-{i}',
        stderr=None,
        experiment=None,
        project=None,
        extra={},
      )
      log_execution(ctx.config.executions_path, rec)

  def test_list_payload_in_result(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    self._seed_executions(ctx)
    cmd = ExecutionsCommand()
    args = MagicMock(limit=20, filter_command=None, failures=False, context_contains=None)
    cmd.list_executions(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert 'executions' in envelope['result']
    assert envelope['result']['count'] == 3

  def test_show_payload_in_result(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    self._seed_executions(ctx)
    cmd = ExecutionsCommand()
    args = MagicMock(index=0)
    cmd.show_execution(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert 'execution' in envelope['result']
    assert 'command' in envelope['result']['execution']

  def test_tail_payload_in_result(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    self._seed_executions(ctx)
    cmd = ExecutionsCommand()
    args = MagicMock(limit=10)
    cmd.tail_executions(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert 'tail' in envelope['result']
    assert envelope['result']['count'] == 3
    assert envelope['result']['total'] == 3

  def test_messages_not_primary_carrier(self, tmp_path: Path, capsys) -> None:
    """Messages should not contain bulk execution data as primary payload."""
    ctx = _ws_ctx(tmp_path)
    self._seed_executions(ctx)
    cmd = ExecutionsCommand()
    args = MagicMock(limit=20, filter_command=None, failures=False, context_contains=None)
    cmd.list_executions(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    for msg in envelope.get('messages', []):
      if msg.get('type') == 'table':
        continue
      assert 'executions' not in msg.get('payload', {})


# Section 4.5: Report & MetricsComparator (BUG-018, BUG-020)


class TestReportCompare:
  """BUG-018 + BUG-020: MetricsComparator with significance, multi-way."""

  def _setup(self, tmp_path: Path) -> CLIContext:
    ctx = _ws_ctx(tmp_path)
    ctx.config.store_path.mkdir(parents=True, exist_ok=True)
    from autopilot.ai.forest import FileForest
    from autopilot.ai.store.file_store import FileStore

    store = FileStore(ctx.config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [
        {
          'id': 'baseline',
          'hypothesis': 'base',
          'status': 'completed',
          'metrics': {'accuracy': 0.7, 'loss': 0.5},
        },
        {
          'id': 'cand1',
          'hypothesis': 'candidate 1',
          'status': 'completed',
          'metrics': {'accuracy': 0.85, 'loss': 0.3},
        },
        {
          'id': 'cand2',
          'hypothesis': 'candidate 2',
          'status': 'completed',
          'metrics': {'accuracy': 0.9, 'loss': 0.2},
        },
      ],
    )
    return ctx

  def test_uses_metrics_comparator_significance(self, tmp_path: Path, capsys) -> None:
    ctx = self._setup(tmp_path)
    cmd = ReportCommand()
    args = MagicMock(
      slugs=['baseline', 'cand1'],
      lower_metric=None,
      union_metrics=False,
      all_trees=False,
    )
    cmd.compare.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    comparisons = envelope['result']['metric_comparisons']
    assert len(comparisons) == 1
    for delta in comparisons[0]:
      assert 'significant' in delta

  def test_multi_way_three_summaries(self, tmp_path: Path, capsys) -> None:
    ctx = self._setup(tmp_path)
    cmd = ReportCommand()
    args = MagicMock(
      slugs=['baseline', 'cand1', 'cand2'],
      lower_metric=None,
      union_metrics=False,
      all_trees=False,
    )
    cmd.compare.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert len(envelope['result']['summaries']) == 3

  def test_multi_way_two_delta_blocks(self, tmp_path: Path, capsys) -> None:
    ctx = self._setup(tmp_path)
    cmd = ReportCommand()
    args = MagicMock(
      slugs=['baseline', 'cand1', 'cand2'],
      lower_metric=None,
      union_metrics=False,
      all_trees=False,
    )
    cmd.compare.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert len(envelope['result']['metric_comparisons']) == 2

  def test_single_slug_fails(self, tmp_path: Path) -> None:
    ctx = self._setup(tmp_path)
    cmd = ReportCommand()
    args = MagicMock(slugs=['baseline'], lower_metric=None, union_metrics=False, all_trees=False)
    with pytest.raises(SystemExit) as exc_info:
      cmd.compare.forward(ctx, args)
    assert exc_info.value.code == 1

  def test_missing_baseline_slug_fails(self, tmp_path: Path) -> None:
    ctx = self._setup(tmp_path)
    cmd = ReportCommand()
    args = MagicMock(
      slugs=['nonexistent', 'cand1'],
      lower_metric=None,
      union_metrics=False,
      all_trees=False,
    )
    forest = load_forest(ctx)
    exp_dir = forest.store.config.experiment_path(slug='nonexistent')
    exp_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit) as exc_info:
      cmd.compare.forward(ctx, args)
    assert exc_info.value.code == 1


# Section 4.6: Query cross-tree (BUG-019)


class TestQueryCrossTree:
  """BUG-019: query --all-trees cross-tree mode."""

  def _setup(self, tmp_path: Path) -> CLIContext:
    ctx = _ws_ctx(tmp_path)
    ctx.config.store_path.mkdir(parents=True, exist_ok=True)
    from autopilot.ai.forest import FileForest
    from autopilot.ai.store.file_store import FileStore

    store = FileStore(ctx.config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [{'id': 'main-exp', 'hypothesis': 'main tree exp', 'status': 'completed', 'metrics': {}}],
    )
    seed_tree_with_experiments(
      forest,
      'secondary',
      [
        {
          'id': 'sec-exp',
          'hypothesis': 'secondary tree exp',
          'status': 'completed',
          'metrics': {},
        }
      ],
    )
    forest.switch('main')
    forest.save()
    return ctx

  def test_all_trees_includes_secondary_tree(self, tmp_path: Path, capsys) -> None:
    ctx = self._setup(tmp_path)
    cmd = QueryCommand()
    args = MagicMock(
      completed=True,
      failed=False,
      running=False,
      pending=False,
      terminal=False,
      cancelled=False,
      filter=None,
      metric_gt=None,
      metric_lt=None,
      best=None,
      lower=False,
      higher=False,
      all_trees=True,
      context_contains=None,
      context_source=None,
      context_after=None,
      created_after=None,
      created_before=None,
      case_sensitive=False,
      include_invalidated=False,
      deployed=False,
      sort=None,
      compact=False,
      spec_version=None,
    )
    cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    ids = [e['id'] for e in envelope['result']['experiments']]
    assert 'sec-exp' in ids

  def test_active_only_excludes_secondary_tree(self, tmp_path: Path, capsys) -> None:
    ctx = self._setup(tmp_path)
    cmd = QueryCommand()
    args = MagicMock(
      completed=True,
      failed=False,
      running=False,
      pending=False,
      terminal=False,
      cancelled=False,
      filter=None,
      metric_gt=None,
      metric_lt=None,
      best=None,
      lower=False,
      higher=False,
      all_trees=False,
      context_contains=None,
      context_source=None,
      context_after=None,
      created_after=None,
      created_before=None,
      case_sensitive=False,
      include_invalidated=False,
      deployed=False,
      sort=None,
      compact=False,
      spec_version=None,
    )
    cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    ids = [e['id'] for e in envelope['result']['experiments']]
    assert 'sec-exp' not in ids
    assert 'main-exp' in ids

  def test_all_trees_json_shape(self, tmp_path: Path, capsys) -> None:
    ctx = self._setup(tmp_path)
    cmd = QueryCommand()
    args = MagicMock(
      completed=False,
      failed=False,
      running=False,
      pending=False,
      terminal=False,
      cancelled=False,
      filter=None,
      metric_gt=None,
      metric_lt=None,
      best=None,
      lower=False,
      higher=False,
      all_trees=True,
      context_contains=None,
      context_source=None,
      context_after=None,
      created_after=None,
      created_before=None,
      case_sensitive=False,
      include_invalidated=False,
      deployed=False,
      sort=None,
      compact=False,
      spec_version=None,
    )
    cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert 'experiments' in envelope['result']
    assert isinstance(envelope['result']['experiments'], list)


# Section 4.7: Additional error / edge cases


class TestExperimentCompareEdgeCases:
  """Edge cases for experiment compare."""

  def test_missing_operand_fails(self, tmp_path: Path) -> None:
    ctx = _ws_ctx(tmp_path)
    ctx.config.store_path.mkdir(parents=True, exist_ok=True)
    from autopilot.ai.forest import FileForest
    from autopilot.ai.store.file_store import FileStore

    store = FileStore(ctx.config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [{'id': 'a', 'hypothesis': 'A', 'status': 'completed', 'metrics': {}}],
    )

    cmd = ExperimentCompare()
    args = MagicMock(a='a', b='nonexistent')
    with pytest.raises(SystemExit) as exc_info:
      cmd.forward(ctx, args)
    assert exc_info.value.code == 1
