"""Tests for CLI command fixes (sub-plan 12).

Covers:
  - Bug 26: workspace init creates all dirs doctor expects
  - Bug 27: project init errors on missing templates dir
  - Bug 28: epoch 0 treated as valid epoch
  - Bug 29: trace inspect --depth removed
  - Bug 30: workspace tree correct connectors
  - Bug 30a: optimize set-hparams bad JSON -> error
  - Bug 30b: ai judge summarize missing summary -> error
  - Bug 46b: str='' defaults -> str | None = None
"""

from autopilot.cli.commands.ai import JudgeCommand, JudgeSummarize
from autopilot.cli.commands.diagnose import DiagnoseCommand
from autopilot.cli.commands.optimize import OptimizeCommand
from autopilot.cli.commands.project import ProjectInit
from autopilot.cli.commands.trace import TraceCommand
from autopilot.cli.commands.workspace import WorkspaceCommand, tree_lines
from autopilot.cli.context import CLIContext
from autopilot.cli.main import build_parser
from autopilot.cli.output import Output
from autopilot.core.artifacts.epoch import DataArtifact
from autopilot.core.config import AutoPilotConfig
from autopilot.core.module.module import Module
from autopilot.core.parameter import Parameter
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from unittest.mock import MagicMock
import argparse
import json
import pytest


def _ws_ctx(tmp_path: Path, use_json: bool = True) -> CLIContext:
  config = AutoPilotConfig(workspace=tmp_path)
  return CLIContext(workspace=tmp_path, config=config, output=Output(use_json=use_json))


def _args(**kwargs) -> MagicMock:
  return MagicMock(**kwargs)


class TestWorkspaceInitDoctorHealthy:
  """Bug 26: workspace init + doctor -> healthy."""

  def test_init_then_doctor_healthy(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    WorkspaceCommand().init(ctx, _args())
    capsys.readouterr()

    WorkspaceCommand().doctor(ctx, _args(repair=False))
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['healthy'] is True
    assert all(envelope['result']['checks'].values())

  def test_init_creates_experiments_dir(self, tmp_path: Path) -> None:
    ctx = _ws_ctx(tmp_path)
    WorkspaceCommand().init(ctx, _args())
    assert ctx.config.experiments_path.is_dir()

  def test_init_creates_records_dir(self, tmp_path: Path) -> None:
    ctx = _ws_ctx(tmp_path)
    WorkspaceCommand().init(ctx, _args())
    assert ctx.config.records_path.is_dir()

  def test_init_creates_datasets_dir(self, tmp_path: Path) -> None:
    ctx = _ws_ctx(tmp_path)
    WorkspaceCommand().init(ctx, _args())
    assert ctx.config.datasets_path.is_dir()

  def test_init_creates_projects_dir(self, tmp_path: Path) -> None:
    ctx = _ws_ctx(tmp_path)
    WorkspaceCommand().init(ctx, _args())
    assert ctx.config.projects_path.is_dir()

  def test_init_creates_autopilot_dir(self, tmp_path: Path) -> None:
    ctx = _ws_ctx(tmp_path)
    WorkspaceCommand().init(ctx, _args())
    assert ctx.config.autopilot_path.is_dir()

  def test_init_idempotent(self, tmp_path: Path) -> None:
    ctx = _ws_ctx(tmp_path)
    WorkspaceCommand().init(ctx, _args())
    WorkspaceCommand().init(ctx, _args())
    assert ctx.config.experiments_path.is_dir()
    assert ctx.config.records_path.is_dir()
    assert ctx.config.datasets_path.is_dir()

  def test_doctor_unhealthy_before_init(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    WorkspaceCommand().doctor(ctx, _args(repair=False))
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['healthy'] is False
    assert len(envelope['result']['issues']) > 0


class TestProjectInitMissingTemplates:
  """Bug 27 + BUG-073: project init gracefully handles missing templates.

  With BUG-073 package fallback, ProjectInit no longer exits with code 1
  when workspace templates are absent -- it falls back to the package bundle
  or silently skips missing templates. These tests verify the new behavior.
  """

  def test_no_workspace_templates_succeeds(self, tmp_path: Path, capsys) -> None:
    """ProjectInit succeeds even without workspace templates (fallback or skip)."""
    ctx = _ws_ctx(tmp_path)
    ctx.config.init_workspace()
    templates_dir = ctx.config.templates_path / 'project'
    assert not templates_dir.exists()

    ProjectInit()(ctx, argparse.Namespace(name='test-proj', bare=False))
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['status'] == 'initialized'

  def test_bare_skips_template_check(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path)
    ctx.config.init_workspace()
    ProjectInit()(ctx, argparse.Namespace(name='test-proj', bare=True))
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['status'] == 'initialized'


class TestEpochZeroValid:
  """Bug 28: epoch=0 treated as valid, not as missing."""

  def test_trace_collect_epoch_zero(self, tmp_path: Path, capsys) -> None:
    ctx = MagicMock()
    ctx.epoch = 0
    ctx.output = Output(use_json=True)
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir(parents=True)
    ctx.experiment_path.return_value = exp_dir
    DataArtifact().append({'id': 'item1'}, exp_dir, epoch=0)

    cmd = TraceCommand()
    args = MagicMock(epoch=0, limit=0)
    cmd.collect(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['epoch'] == 0

  def test_trace_collect_epoch_none_errors(self, tmp_path: Path) -> None:
    ctx = MagicMock()
    ctx.epoch = None
    ctx.output = Output(use_json=False)
    ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
    cmd = TraceCommand()
    args = MagicMock(epoch=None, limit=0)
    with pytest.raises(SystemExit) as exc_info:
      cmd.collect(ctx, args)
    assert exc_info.value.code == 1

  def test_trace_inspect_epoch_zero(self, tmp_path: Path, capsys) -> None:
    ctx = MagicMock()
    ctx.epoch = 0
    ctx.output = Output(use_json=True)
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir(parents=True)
    ctx.experiment_path.return_value = exp_dir
    DataArtifact().append({'id': 'node_x', 'success': True}, exp_dir, epoch=0)

    cmd = TraceCommand()
    args = MagicMock(node='node_x', epoch=0)
    cmd.inspect_trace(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['count'] == 1

  def test_diagnose_run_epoch_zero(self, tmp_path: Path, capsys) -> None:
    ctx = MagicMock()
    ctx.epoch = 0
    ctx.output = Output(use_json=True)
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir(parents=True)
    ctx.experiment_path.return_value = exp_dir

    cmd = DiagnoseCommand()
    args = MagicMock(epoch=0, category=None, node=None)
    cmd.run_diagnose(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['epoch'] == 0

  def test_ai_distribution_epoch_zero(self, tmp_path: Path, capsys) -> None:
    ctx = MagicMock()
    ctx.epoch = 0
    ctx.output = Output(use_json=True)
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir(parents=True)
    ctx.experiment_path.return_value = exp_dir
    DataArtifact().append({'success': True, 'metadata': {}}, exp_dir, epoch=0)

    cmd = JudgeCommand()
    args = MagicMock(epoch=0)
    cmd.distribution(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['epoch'] == 0


class TestTraceDepthRemoved:
  """Bug 29: --depth flag removed from trace inspect."""

  def test_no_depth_argument(self) -> None:
    parser = build_parser()
    args = parser.parse_args(['trace', 'inspect', '--node', 'x', '--epoch', '1'])
    assert not hasattr(args, 'depth')


class TestWorkspaceTreeConnectors:
  """Bug 30: tree connectors differ for last vs non-last entries."""

  def test_last_entry_uses_backslash_connector(self, tmp_path: Path) -> None:
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'only_child').mkdir()
    lines = tree_lines(root)
    assert len(lines) == 1
    assert lines[0].startswith('\\-- ')

  def test_non_last_uses_plus_connector(self, tmp_path: Path) -> None:
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'aaa').mkdir()
    (root / 'bbb').mkdir()
    lines = tree_lines(root)
    assert len(lines) == 2
    assert lines[0].startswith('+-- ')
    assert lines[1].startswith('\\-- ')

  def test_nested_tree_connectors(self, tmp_path: Path) -> None:
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'aaa').mkdir()
    (root / 'bbb').mkdir()
    (root / 'aaa' / 'child').mkdir()
    lines = tree_lines(root)
    assert '+-- aaa' in lines[0]
    assert '|   \\-- child' in lines[1]
    assert '\\-- bbb' in lines[2]

  def test_tree_display_via_command(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path, use_json=False)
    ctx.config.init_workspace()
    WorkspaceCommand().tree(ctx, _args())
    captured = capsys.readouterr()
    text = captured.out
    assert '\\-- ' in text or '+-- ' in text


class TestOptimizeSetHparamsBadJson:
  """Bug 30a: bad JSON in set-hparams -> exit 1."""

  def test_bad_json_exits(self, tmp_path: Path) -> None:
    ctx = _ws_ctx(tmp_path)
    cmd = OptimizeCommand()
    args = MagicMock(values='{invalid', hyperparams_file=None)
    ctx.hyperparams_file = None
    with pytest.raises(SystemExit) as exc_info:
      cmd.set_hparams(ctx, args)
    assert exc_info.value.code == 1

  def test_bad_json_error_message(self, tmp_path: Path, capsys) -> None:
    ctx = _ws_ctx(tmp_path, use_json=False)
    cmd = OptimizeCommand()
    args = MagicMock(values='{invalid', hyperparams_file=None)
    ctx.hyperparams_file = None
    with pytest.raises(SystemExit):
      cmd.set_hparams(ctx, args)
    captured = capsys.readouterr()
    assert 'invalid JSON for --values' in captured.err

  def test_valid_json_succeeds(self, tmp_path: Path, capsys) -> None:
    from autopilot.ai.forest import FileForest
    from autopilot.ai.store.file_store import FileStore
    from autopilot.core.experiment import Experiment
    from autopilot.core.node import Node

    ctx = _ws_ctx(tmp_path)
    ctx.hyperparams_file = None
    ctx.experiment = 'test-exp'
    ctx.config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(ctx.config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')
    exp = Experiment(experiment_id='test-exp', hypothesis='testing')
    exp.start()
    node = Node(experiment=exp)
    tree.add(node)
    forest.save()

    cmd = OptimizeCommand()
    args = MagicMock(values='{"lr": 0.01}')
    cmd.set_hparams(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['hparams'] == {'lr': 0.01}
    assert envelope['result']['experiment_id'] == 'test-exp'


class TestJudgeSummarizeMissingSummary:
  """Bug 30b: judge summarize with missing summary -> exit 1."""

  def test_missing_summary_key(self, tmp_path: Path) -> None:
    judge_file = tmp_path / 'judge_output.json'
    atomic_write_json(judge_file, {'results': [1, 2, 3]})

    ctx = MagicMock()
    ctx.judge = MagicMock()
    ctx.output = Output(use_json=True)
    ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
    cmd = JudgeSummarize()
    args = MagicMock(judge_input=str(judge_file))
    with pytest.raises(SystemExit) as exc_info:
      cmd.forward(ctx, args)
    assert exc_info.value.code == 1

  def test_none_file(self, tmp_path: Path) -> None:
    judge_file = tmp_path / 'nonexistent.json'

    ctx = MagicMock()
    ctx.judge = MagicMock()
    ctx.output = Output(use_json=True)
    ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
    cmd = JudgeSummarize()
    args = MagicMock(judge_input=str(judge_file))
    with pytest.raises(SystemExit) as exc_info:
      cmd.forward(ctx, args)
    assert exc_info.value.code == 1

  def test_missing_summary_error_message(self, tmp_path: Path, capsys) -> None:
    judge_file = tmp_path / 'judge_output.json'
    atomic_write_json(judge_file, {'results': []})

    ctx = MagicMock()
    ctx.judge = MagicMock()
    ctx.output = Output(use_json=False)
    ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
    cmd = JudgeSummarize()
    args = MagicMock(judge_input=str(judge_file))
    with pytest.raises(SystemExit):
      cmd.forward(ctx, args)
    captured = capsys.readouterr()
    assert 'invalid judge input: missing summary' in captured.err

  def test_valid_summary(self, tmp_path: Path, capsys) -> None:
    judge_file = tmp_path / 'judge_output.json'
    atomic_write_json(judge_file, {'summary': {'total': 10, 'passed': 8}})

    ctx = MagicMock()
    ctx.judge = MagicMock()
    ctx.output = Output(use_json=True)
    cmd = JudgeSummarize()
    args = MagicMock(judge_input=str(judge_file))
    cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['total'] == 10


class TestStrDefaultsFix:
  """Bug 46b: str='' defaults changed to str | None = None."""

  def testtree_lines_none_prefix(self, tmp_path: Path) -> None:
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'child').mkdir()
    lines = tree_lines(root, prefix=None)
    assert len(lines) == 1
    assert lines[0] == '\\-- child'

  def testtree_lines_explicit_prefix(self, tmp_path: Path) -> None:
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'child').mkdir()
    lines = tree_lines(root, prefix='>>>')
    assert lines[0] == '>>>\\-- child'

  def test_named_modules_none_prefix(self) -> None:
    m = Module()
    child = Module()
    m.child = child
    names = [name for name, _ in m.named_modules()]
    assert names == ['', 'child']

  def test_named_modules_explicit_prefix(self) -> None:
    m = Module()
    child = Module()
    m.child = child
    names = [name for name, _ in m.named_modules(prefix='root')]
    assert names == ['root', 'root.child']

  def test_named_parameters_none_prefix(self) -> None:
    m = Module()
    m.weight = Parameter()
    names = [name for name, _ in m.named_parameters()]
    assert names == ['weight']

  def test_named_parameters_explicit_prefix(self) -> None:
    m = Module()
    m.weight = Parameter()
    names = [name for name, _ in m.named_parameters(prefix='layer')]
    assert names == ['layer.weight']


class TestInitWorkspaceDocstring:
  """Verify the docstring matches the spec."""

  def test_docstring_content(self) -> None:
    doc = AutoPilotConfig.init_workspace.__doc__
    assert doc is not None
    assert '.autopilot/' in doc
    assert '.autopilot/projects/' in doc
    assert '.autopilot/experiments/' in doc
    assert '.autopilot/records/' in doc
    assert '.autopilot/datasets/' in doc
