"""Tests for CLI type annotation fixes (plan 04).

Covers: command.py _commands typing + register() name assert, ai.py RunConfig +
narrowing, report.py manifest None guard, stabilize.py dict narrowing,
execute.py build_cmd assertions, optimize.py module/args guards, propose.py
experiment guard, query.py Node typing, expose.py command assert, policy.py
result_data narrowing.
"""

from autopilot.ai.evaluation.schemas import RetryConfig, RunConfig
from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.command import CLI, Command
from autopilot.cli.commands.ai import build_judge_config
from autopilot.cli.commands.execute import ExecuteCommand
from autopilot.cli.commands.report.compare import gather_summary
from autopilot.cli.commands.stabilize import StabilizeCommand
from autopilot.cli.context import CLIContext
from autopilot.cli.expose import ExposeCollector, expose_command
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.core.enums import Status
from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from unittest.mock import MagicMock
import argparse
import json
import pytest

# -- 2.1 command.py: _commands typed attribute + register() name assertion --


class TestCommandTypedCommands:
  """Verify _commands is a typed dict[str, Command] attribute."""

  def test_commands_attr_exists_on_instance(self) -> None:
    cmd = Command()
    assert isinstance(cmd._commands, dict)
    assert len(cmd._commands) == 0

  def test_commands_attr_on_cli_instance(self) -> None:
    cli = CLI()
    assert isinstance(cli._commands, dict)

  def test_setattr_populates_commands(self) -> None:
    class Parent(Command):
      name = 'parent'

    class Child(Command):
      name = 'child'

    p = Parent()
    p.child = Child()
    assert 'child' in p._commands
    assert isinstance(p._commands['child'], Command)

  def test_cli_setattr_populates_commands(self) -> None:
    class Leaf(Command):
      name = 'leaf'

    cli = CLI()
    cli.leaf = Leaf()
    assert 'leaf' in cli._commands


class TestRegisterNameAssertion:
  """register() must assert self.name is not None before calling add_parser."""

  def test_register_with_name_succeeds(self) -> None:
    class Named(Command):
      name = 'named'
      help = 'test'

      def forward(self, ctx, args):
        pass

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    Named().register(sub)
    args = parser.parse_args(['named', '--workspace', '.'])
    assert args.cmd == 'named'

  def test_register_without_name_raises(self) -> None:

    class NoName(Command):
      pass

    cmd = NoName()
    cmd.name = None  # type: ignore[assignment]
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    with pytest.raises(AssertionError, match='must have a name'):
      cmd.register(sub)

  def test_register_group_with_children(self) -> None:

    class Child(Command):
      name = 'child'
      help = 'a child'

      def forward(self, ctx, args):
        pass

    class Group(Command):
      name = 'group'
      help = 'a group'

      def __init__(self):
        super().__init__()
        self.child = Child()

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    Group().register(sub)
    args = parser.parse_args(['group', 'child', '--workspace', '.'])
    assert args.cmd == 'group'


# -- 2.2 ai.py: RunConfig, judge config, narrowing --


class TestBuildJudgeConfig:
  """build_judge_config uses RunConfig, not dict literal."""

  def test_minimal_defaults_returns_valid_config(self) -> None:
    args = argparse.Namespace(
      judge_input='/nonexistent/items.jsonl',
      num_parallel=0,
      max_rpm=0,
    )
    config = build_judge_config(args)
    assert isinstance(config.run, RunConfig)
    assert isinstance(config.run.retry, RetryConfig)
    assert config.run.model == 'openai:gpt-4o'
    assert config.run.num_parallel == 5
    assert config.run.retry.max_retries == 3

  def test_overrides_apply(self) -> None:
    args = argparse.Namespace(
      judge_input='/nonexistent/items.jsonl',
      num_parallel=10,
      max_rpm=200,
    )
    config = build_judge_config(args)
    assert config.run.num_parallel == 10
    assert config.run.max_rpm == 200

  def test_loads_from_file(self, tmp_path: Path) -> None:
    config_data = {
      'run': {
        'model': 'openai:gpt-4o-mini',
        'num_parallel': 3,
        'max_rpm': 50,
        'rpm_safety_margin': 0.8,
        'retry': {
          'max_retries': 2,
          'min_timeout_ms': 500,
          'max_timeout_ms': 10000,
          'backoff_factor': 1,
        },
        'max_tool_steps': 3,
        'max_output_tokens': 2048,
      },
    }
    items_dir = tmp_path / 'judge_input'
    items_dir.mkdir()
    config_path = items_dir / 'judge_config.json'
    config_path.write_text(json.dumps(config_data), encoding='utf-8')

    args = argparse.Namespace(
      judge_input=str(items_dir / 'items.jsonl'),
      num_parallel=0,
      max_rpm=0,
    )
    config = build_judge_config(args)
    assert config.run.model == 'openai:gpt-4o-mini'


class TestAiSummarizeNarrowing:
  """JudgeSummarize.forward narrows raw with isinstance(raw, dict)."""

  def test_summarize_with_dict_summary(self, tmp_path: Path, capsys) -> None:
    from autopilot.cli.commands.ai import JudgeSummarize

    judge_file = tmp_path / 'output.json'
    judge_file.write_text(json.dumps({'summary': {'total': 10}}), encoding='utf-8')

    ctx = MagicMock()
    ctx.judge = MagicMock()
    ctx.output = Output(use_json=True)
    args = argparse.Namespace(judge_input=str(judge_file))
    JudgeSummarize().forward(ctx, args)

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result'] == {'total': 10}

  def test_summarize_with_list_fails(self, tmp_path: Path) -> None:
    from autopilot.cli.commands.ai import JudgeSummarize

    judge_file = tmp_path / 'output.json'
    judge_file.write_text(json.dumps([1, 2, 3]), encoding='utf-8')

    ctx = MagicMock()
    ctx.judge = MagicMock()
    ctx.output = Output(use_json=True)
    ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
    args = argparse.Namespace(judge_input=str(judge_file))
    with pytest.raises(SystemExit):
      JudgeSummarize().forward(ctx, args)

  def test_summarize_with_none_fails(self, tmp_path: Path) -> None:
    from autopilot.cli.commands.ai import JudgeSummarize

    ctx = MagicMock()
    ctx.judge = MagicMock()
    ctx.output = Output(use_json=True)
    ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
    args = argparse.Namespace(judge_input=str(tmp_path / 'nonexistent.json'))
    with pytest.raises(SystemExit):
      JudgeSummarize().forward(ctx, args)


# -- 2.3 report.py: manifest None guard + stabilize.py dict narrowing --


class TestGatherSummaryForest:
  """gather_summary works with Forest-backed experiment resolution."""

  def test_with_node_includes_experiment_keys(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')
    exp = Experiment(experiment_id='test-slug', hypothesis='test idea')
    exp.epoch = 3
    tree.add(Node(experiment=exp))
    forest.save()
    exp_dir = config.experiment_path(slug='test-slug')
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary = gather_summary(forest, 'test-slug')
    assert summary['id'] == 'test-slug'
    assert summary['epoch'] == 3
    assert summary['hypothesis'] == 'test idea'
    assert 'event_count' in summary

  def test_without_node_raises_value_error(self, tmp_path: Path) -> None:
    """gather_summary raises ValueError for nonexistent experiment."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    exp_dir = config.experiment_path(slug='missing')
    exp_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match='not found in forest'):
      gather_summary(forest, 'missing')


class TestStabilizeForestNarrowing:
  """stabilize.py loads forest via FileStore.load_state_dict; non-dict raises StoreError."""

  def test_forest_state_as_list_raises_store_error(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config.forest_file, [1, 2, 3])

    ctx = MagicMock()
    ctx.config = config
    ctx.output = MagicMock()
    ctx.wait_timeout_ms = None
    ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))

    cmd = StabilizeCommand()
    with pytest.raises(StoreError):
      cmd.forward(ctx, MagicMock(experiment_id='exp-001'))

  def test_forest_state_as_dict_processes_trees(self, tmp_path: Path, capsys) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    snapshots_dir = config.snapshots_path / 'exp-001'
    snapshots_dir.mkdir(parents=True)
    atomic_write_json(
      snapshots_dir / 'epoch_0.json',
      {'epoch': 0, 'timestamp': '2024-01-01T00:00:00Z', 'entries': {}},
    )

    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='exp-001', hypothesis='test')
    exp.start()
    exp.complete(metrics={})
    tree.add(Node(experiment=exp))
    forest.switch('main')
    forest.save()

    ctx = MagicMock()
    ctx.config = config
    ctx.output = Output(use_json=True)
    ctx.wait_timeout_ms = None
    cmd = StabilizeCommand()
    cmd.forward(ctx, MagicMock(experiment_id='exp-001'))

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True


# -- 2.4 execute.py: build_cmd source assertions --


class TestBuildCmdAssertions:
  """build_cmd asserts source is not None for inline/module/file modes."""

  def test_inline_with_source(self) -> None:
    cmd = ExecuteCommand()
    result = cmd.build_cmd('inline', 'print(1)', [])
    assert result == ['uv', 'run', 'python', '-c', 'print(1)']

  def test_inline_none_raises(self) -> None:
    cmd = ExecuteCommand()
    with pytest.raises(AssertionError, match='inline mode requires'):
      cmd.build_cmd('inline', None, [])

  def test_module_with_source(self) -> None:
    cmd = ExecuteCommand()
    result = cmd.build_cmd('module', 'pytest', ['--verbose'])
    assert result == ['uv', 'run', 'python', '-m', 'pytest', '--verbose']

  def test_module_none_raises(self) -> None:
    cmd = ExecuteCommand()
    with pytest.raises(AssertionError, match='module mode requires'):
      cmd.build_cmd('module', None, [])

  def test_file_with_source(self) -> None:
    cmd = ExecuteCommand()
    result = cmd.build_cmd('file', 'script.py', ['--arg'])
    assert result == ['uv', 'run', 'python', 'script.py', '--arg']

  def test_file_none_raises(self) -> None:
    cmd = ExecuteCommand()
    with pytest.raises(AssertionError, match='file mode requires'):
      cmd.build_cmd('file', None, [])

  def test_stdin_returns_base(self) -> None:
    cmd = ExecuteCommand()
    result = cmd.build_cmd('stdin', None, [])
    assert result == ['uv', 'run', 'python']

  def test_all_elements_are_str(self) -> None:
    cmd = ExecuteCommand()
    for mode, source in [('inline', 'x'), ('module', 'y'), ('file', 'z')]:
      result = cmd.build_cmd(mode, source, ['--flag'])
      assert all(isinstance(elem, str) for elem in result)


# -- 2.5 propose.py: experiment guard, query.py Node typing,
#    expose.py command assert, policy.py result_data narrowing --


class TestProposeExperimentGuard:
  """propose revert guards ctx.experiment before checkout."""

  def test_revert_without_experiment_fails(self, tmp_path: Path) -> None:
    from autopilot.ai.proposal import ChangeProposal, record_proposal
    from autopilot.cli.commands.propose import ProposeCommand

    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir()
    record_proposal(
      exp_dir,
      ChangeProposal(
        proposal_id='abc',
        hypothesis='h',
        target_node='t',
        change_type='g',
        epoch=2,
        status='proposed',
      ),
    )

    ctx = MagicMock()
    ctx.experiment = None
    ctx.epoch = 1
    ctx.workspace = tmp_path
    ctx.project = None
    ctx.output = Output(use_json=True)
    ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
    ctx.experiment_path.return_value = exp_dir

    source_dir = tmp_path / 'src'
    source_dir.mkdir()

    args = MagicMock(
      proposal_id='abc',
      source=str(source_dir),
      store=None,
      pattern='**/*',
    )

    cmd = ProposeCommand()
    with pytest.raises(SystemExit) as exc_info:
      cmd.revert(ctx, args)
    assert exc_info.value.code == 1


class TestQueryNodeTyping:
  """query.py uses Node instead of object for type safety."""

  def test_render_best_with_node(self) -> None:
    from autopilot.cli.commands.query import QueryCommand

    exp = Experiment(experiment_id='exp-1', hypothesis='test')
    exp.status = Status.completed
    exp.metrics = {'acc': 0.9}
    node = Node(experiment=exp)

    ctx = MagicMock()
    ctx.output = MagicMock()
    ctx.output.use_json = True

    cmd = QueryCommand()
    cmd._render_best(ctx, node)
    ctx.output.result.assert_called_once()
    payload = ctx.output.result.call_args[0][0]
    assert payload['best']['id'] == 'exp-1'

  def test_render_all_with_nodes(self) -> None:
    from autopilot.cli.commands.query import QueryCommand

    nodes = []
    for i in range(3):
      exp = Experiment(experiment_id=f'exp-{i}', hypothesis=f'h{i}')
      exp.status = Status.completed
      exp.metrics = {'acc': 0.5 + i * 0.1}
      nodes.append(Node(experiment=exp))

    ctx = MagicMock()
    ctx.output = MagicMock()
    ctx.output.use_json = True

    cmd = QueryCommand()
    cmd._render_all(ctx, nodes)
    ctx.output.result.assert_called_once()
    payload = ctx.output.result.call_args[0][0]
    assert payload['count'] == 3
    assert len(payload['experiments']) == 3

  def test_render_all_empty(self) -> None:
    from autopilot.cli.commands.query import QueryCommand

    ctx = MagicMock()
    ctx.output = MagicMock()
    ctx.output.use_json = False

    cmd = QueryCommand()
    cmd._render_all(ctx, [])
    ctx.output.info.assert_called_once()


class TestExposeCommandAssert:
  """expose_command asserts command is not None."""

  def test_with_command_succeeds(self) -> None:
    collector = ExposeCollector()
    with expose_command(collector, 'test op', 'echo hello'):
      pass
    records = collector.to_list()
    assert len(records) == 1
    assert records[0]['command'] == 'echo hello'

  def test_without_command_raises(self) -> None:
    collector = ExposeCollector()
    with (
      pytest.raises(AssertionError, match='non-None command'),
      expose_command(collector, 'test op', None),
    ):
      pass


class TestPolicyResultNarrowing:
  """policy check uses metrics + gate precedence (plan 07)."""

  def test_policy_check_no_metrics_source(self, tmp_path: Path) -> None:
    from autopilot.cli.commands.policy import PolicyCommand

    ctx = MagicMock()
    ctx.experiment = None
    ctx.module = None
    ctx.output = MagicMock()
    ctx.fail = MagicMock(side_effect=SystemExit(1))

    args = argparse.Namespace(
      metrics_json=None,
      min_thresholds=['accuracy:0.8'],
      max_thresholds=None,
    )

    cmd = PolicyCommand()
    with pytest.raises(SystemExit):
      cmd.check(ctx, args)

    ctx.fail.assert_called_once()

  def test_policy_check_with_valid_metrics(self, tmp_path: Path) -> None:
    from autopilot.cli.commands.policy import PolicyCommand
    from autopilot.core.types import GateResult

    ctx = MagicMock()
    ctx.experiment = None
    ctx.module = MagicMock()
    ctx.module.policy = MagicMock()
    ctx.module.policy.name.return_value = 'quality_first'
    ctx.module.policy.forward.return_value = GateResult.PASSED
    ctx.output = MagicMock()

    args = argparse.Namespace(
      metrics_json='{"accuracy": 0.9}',
      min_thresholds=None,
      max_thresholds=None,
    )

    cmd = PolicyCommand()
    cmd.check(ctx, args)

    ctx.output.result.assert_called_once()
    payload = ctx.output.result.call_args[0][0]
    assert payload['gate_result'] == 'pass'
    assert payload['policy'] == 'quality_first'


# -- optimize.py: module guard and args guard --


class TestOptimizeModuleGuard:
  """optimize loop guards ctx.module before calling fit."""

  def testlog_optimize_command_without_experiment(self, tmp_path: Path) -> None:
    from autopilot.cli.commands.optimize import log_optimize_command

    ctx = MagicMock()
    ctx.experiment = None

    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir()

    log_optimize_command(exp_dir, 'train', ctx)

  def testlog_optimize_command_with_experiment(self, tmp_path: Path) -> None:
    from autopilot.cli.commands.optimize import log_optimize_command

    ctx = MagicMock()
    ctx.experiment = 'my-exp'

    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir()

    log_optimize_command(exp_dir, 'validate', ctx)

  def test_loop_without_module_fails(self, tmp_path: Path) -> None:
    from autopilot.cli.commands.optimize import OptimizeCommand

    ctx = MagicMock()
    ctx.experiment = 'test-exp'
    ctx.module = None
    ctx.output = Output(use_json=True)
    ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
    ctx.split = None
    ctx.dry_run = False
    ctx.epoch = 0
    ctx.workspace = tmp_path
    ctx.trainer = MagicMock()
    ctx.trainer.callbacks = []
    ctx.trainer.dry_run = False
    ctx.trainer.logger = None
    ctx.trainer.policy = None
    ctx.trainer.experiment = None
    ctx.trainer.config = None
    ctx.trainer.accumulate_grad_batches = 1
    ctx.datamodule = None
    ctx.experiment_path.return_value = tmp_path / 'test-exp'
    (tmp_path / 'test-exp').mkdir()

    cmd = OptimizeCommand()
    args = MagicMock(max_epochs=1, strategy='conservative')
    with pytest.raises(SystemExit) as exc_info:
      cmd.loop(ctx, args)
    assert exc_info.value.code == 1


class TestStoreCreatePathField:
  """Store create output includes path from AutoPilotConfig."""

  def test_create_result_has_path(self, tmp_path: Path) -> None:
    from autopilot.cli.commands.store.command import StoreCommand

    src_dir = tmp_path / 'src'
    src_dir.mkdir()
    (src_dir / 'file.txt').write_text('content')

    ctx = MagicMock()
    ctx.experiment = 'exp-1'
    ctx.workspace = tmp_path
    ctx.project = None
    ctx.context = 'test'
    ctx.output = Output(use_json=True)

    args = MagicMock()
    args.source = str(src_dir)
    args.store = str(tmp_path / '.store')
    args.pattern = '**/*'

    cmd = StoreCommand()
    ctx.output = MagicMock()
    cmd.create(ctx, args)

    ctx.output.result.assert_called_once()
    payload = ctx.output.result.call_args[0][0]
    assert 'path' in payload
    assert 'slug' in payload
    assert payload['slug'] == 'exp-1'
    assert payload['path'] == str(Path(args.store).resolve())
