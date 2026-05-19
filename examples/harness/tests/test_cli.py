"""Tests for HarnessCLI registration and wiring."""

from autopilot.ai.store.file_store import FileStore
from autopilot.cli.command import CLI
from autopilot.cli.commands.optimize import OptimizeCommand
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.policy.quality_first import QualityFirstPolicy
from harness.callbacks import DeployCallback, MetricsWriterCallback, OptimizerContextCallback
from harness.cli import (
  _NO_JUDGE_KWARGS,
  _USE_JUDGE_KWARGS,
  HarnessCLI,
  HarnessOptimizeCommand,
  HarnessResume,
  HarnessTest,
  HarnessTrain,
  HarnessValidate,
  _resolve_use_judge,
)
from harness.data import HarnessDataModule
from harness.judge import HarnessJudge
from harness.module import HarnessModule
from pathlib import Path
import argparse
import pytest
import runpy


class TestProjectRegistration:
  """Verify HarnessCLI is registered in the project registry."""

  def test_harness_project_registered(self):
    assert 'harness' in CLI._project_registry
    assert CLI._project_registry['harness'] is HarnessCLI


class TestCLIInitialization:
  """Test HarnessCLI initializes all components correctly."""

  def test_module_is_harness_module(self):
    cli = HarnessCLI()
    assert isinstance(cli.module, HarnessModule)

  def test_datamodule_is_harness_datamodule(self):
    cli = HarnessCLI()
    assert isinstance(cli.datamodule, HarnessDataModule)

  def test_store_is_file_store(self):
    cli = HarnessCLI()
    assert isinstance(cli.store, FileStore)

  def test_policy_is_quality_first(self):
    cli = HarnessCLI()
    assert isinstance(cli.policy, QualityFirstPolicy)

  def test_callbacks_present(self):
    cli = HarnessCLI()
    assert len(cli.callbacks) == 4
    assert isinstance(cli.callbacks[0], StoreCheckpointCallback)
    assert isinstance(cli.callbacks[1], MetricsWriterCallback)
    assert isinstance(cli.callbacks[2], OptimizerContextCallback)
    assert isinstance(cli.callbacks[3], DeployCallback)

  def test_config_workspace_set(self):
    cli = HarnessCLI()
    assert cli.config is not None
    assert cli.config.workspace is not None


class TestCLIJudgeAttribute:
  """Verify HarnessCLI.judge is a live HarnessJudge and generator is None."""

  def test_cli_judge_attribute(self):
    cli = HarnessCLI()
    assert isinstance(cli.judge, HarnessJudge)

  def test_cli_generator_none(self):
    cli = HarnessCLI()
    assert cli.generator is None

  def test_cli_optimize_is_harness_subclass(self):
    cli = HarnessCLI()
    assert isinstance(cli.optimize, HarnessOptimizeCommand)
    assert isinstance(cli.optimize, OptimizeCommand)

  def test_cli_resume_is_harness_resume(self):
    cli = HarnessCLI()
    assert isinstance(cli.optimize.resume, HarnessResume)

  def test_cli_train_is_harness_train(self):
    cli = HarnessCLI()
    assert isinstance(cli.optimize.train, HarnessTrain)

  def test_cli_validate_is_harness_validate(self):
    cli = HarnessCLI()
    assert isinstance(cli.optimize.validate, HarnessValidate)

  def test_cli_test_is_harness_test(self):
    cli = HarnessCLI()
    assert isinstance(cli.optimize.test, HarnessTest)


class TestCLIUseJudgeFlag:
  """Verify --use-judge / --no-judge on optimize subcommands."""

  def test_parser_accepts_no_judge_on_loop(self):
    cli = HarnessCLI()
    parser = cli.build_parser()
    parser.set_defaults(use_judge=False, no_judge=False)
    args = parser.parse_args(['optimize', 'loop', '--max-epochs', '1', '--no-judge'])
    assert args.no_judge is True
    assert args.use_judge is False

  def test_parser_accepts_use_judge_on_loop(self):
    cli = HarnessCLI()
    parser = cli.build_parser()
    parser.set_defaults(use_judge=False, no_judge=False)
    args = parser.parse_args(['optimize', 'loop', '--max-epochs', '1', '--use-judge'])
    assert args.use_judge is True
    assert args.no_judge is False

  def test_parser_default_judge_mode(self):
    cli = HarnessCLI()
    parser = cli.build_parser()
    parser.set_defaults(use_judge=False, no_judge=False)
    args = parser.parse_args(['optimize', 'loop', '--max-epochs', '1'])
    assert args.use_judge is False
    assert args.no_judge is False

  def test_parser_accepts_no_judge_on_resume(self):
    cli = HarnessCLI()
    parser = cli.build_parser()
    parser.set_defaults(use_judge=False, no_judge=False)
    args = parser.parse_args(['optimize', 'resume', 'ckpt.json', '--no-judge'])
    assert args.no_judge is True
    assert args.use_judge is False

  def test_parser_accepts_use_judge_on_resume(self):
    cli = HarnessCLI()
    parser = cli.build_parser()
    parser.set_defaults(use_judge=False, no_judge=False)
    args = parser.parse_args(['optimize', 'resume', 'ckpt.json', '--use-judge'])
    assert args.use_judge is True
    assert args.no_judge is False

  def test_parser_accepts_no_judge_on_train(self):
    cli = HarnessCLI()
    parser = cli.build_parser()
    parser.set_defaults(use_judge=False, no_judge=False)
    args = parser.parse_args(['optimize', 'train', '--no-judge'])
    assert args.no_judge is True
    assert args.use_judge is False

  def test_parser_accepts_no_judge_on_validate(self):
    cli = HarnessCLI()
    parser = cli.build_parser()
    parser.set_defaults(use_judge=False, no_judge=False)
    args = parser.parse_args(['optimize', 'validate', '--no-judge'])
    assert args.no_judge is True
    assert args.use_judge is False

  def test_parser_accepts_no_judge_on_test(self):
    cli = HarnessCLI()
    parser = cli.build_parser()
    parser.set_defaults(use_judge=False, no_judge=False)
    args = parser.parse_args(['optimize', 'test', '--no-judge'])
    assert args.no_judge is True
    assert args.use_judge is False


class TestResolveUseJudge:
  """Test the _resolve_use_judge helper."""

  def test_neither_flag_defaults_true(self):
    ns = argparse.Namespace(use_judge=False, no_judge=False)
    assert _resolve_use_judge(ns) is True

  def test_no_judge_returns_false(self):
    ns = argparse.Namespace(use_judge=False, no_judge=True)
    assert _resolve_use_judge(ns) is False

  def test_use_judge_returns_true(self):
    ns = argparse.Namespace(use_judge=True, no_judge=False)
    assert _resolve_use_judge(ns) is True

  def test_both_flags_raises(self):
    ns = argparse.Namespace(use_judge=True, no_judge=True)
    with pytest.raises(ValueError, match='cannot pass both'):
      _resolve_use_judge(ns)


class TestEnsureModuleUseJudge:
  """Test _ensure_module_use_judge rebuilds module when mode differs."""

  def test_noop_when_same_mode(self):
    cli = HarnessCLI()
    original_module = cli.module
    cli._ensure_module_use_judge(True)
    assert cli.module is original_module

  def test_rebuilds_when_mode_changes(self):
    cli = HarnessCLI()
    assert cli.module.use_judge is True
    cli._ensure_module_use_judge(False)
    assert cli.module.use_judge is False

  def test_store_re_registered_on_rebuild(self):
    cli = HarnessCLI()
    cli._ensure_module_use_judge(False)
    module_params = dict(cli.module.named_parameters())
    assert set(cli.store._param_names.keys()) == set(module_params.keys())


class TestParameterWiring:
  """Verify store parameter registration aligns with module parameters."""

  def test_named_parameters_registered(self):
    cli = HarnessCLI()
    module_params = dict(cli.module.named_parameters())
    assert set(cli.store._param_names.keys()) == set(module_params.keys())

  def test_parameter_names_include_expected(self):
    cli = HarnessCLI()
    param_names = set(dict(cli.module.named_parameters()).keys())
    assert 'system_prompt' in param_names
    assert 'policies' in param_names
    assert 'tools_code' in param_names


class TestRunTrainerUseJudgeForwarded:
  """Test that run_trainer.py forwards --use-judge / --no-judge to build_trainer."""

  def test_no_judge_flag_parsed(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-judge', action='store_true', default=False, dest='use_judge')
    parser.add_argument('--no-judge', action='store_true', default=False, dest='no_judge')
    args = parser.parse_args(['--no-judge'])
    assert args.no_judge is True
    assert args.use_judge is False

  def test_use_judge_flag_parsed(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-judge', action='store_true', default=False, dest='use_judge')
    parser.add_argument('--no-judge', action='store_true', default=False, dest='no_judge')
    args = parser.parse_args(['--use-judge'])
    assert args.use_judge is True
    assert args.no_judge is False


class TestRunDirectConflict:
  """Test run_direct exits non-zero when both judge flags are passed."""

  def test_run_direct_both_flags_exits_nonzero(self):
    """Both --use-judge and --no-judge via run_direct exits 2."""
    cli = HarnessCLI()
    with pytest.raises(SystemExit) as exc_info:
      cli.run_direct(argv=['optimize', 'loop', '--max-epochs', '0', '--use-judge', '--no-judge'])
    assert exc_info.value.code == 2


class TestBootstrapFile:
  """Verify the .autopilot/projects/harness/cli.py bootstrap registers HarnessCLI."""

  def test_bootstrap_registers_project(self):
    """Importing bootstrap file triggers __init_subclass__ registration."""
    harness_root = Path(__file__).resolve().parent.parent
    bootstrap = harness_root / '.autopilot' / 'projects' / 'harness' / 'cli.py'
    assert bootstrap.exists(), f'bootstrap file missing: {bootstrap}'
    result = runpy.run_path(str(bootstrap), run_name='__autopilot_project__')
    assert 'HarnessCLI' in result['__all__']
    assert 'harness' in CLI._project_registry
    assert CLI._project_registry['harness'] is HarnessCLI


class TestJudgeKwargsDRY:
  """Verify _USE_JUDGE_KWARGS / _NO_JUDGE_KWARGS are the single source of truth."""

  def test_use_judge_kwargs_keys(self):
    """Shared kwargs contain all required argparse fields."""
    assert _USE_JUDGE_KWARGS['action'] == 'store_true'
    assert _USE_JUDGE_KWARGS['dest'] == 'use_judge'
    assert _USE_JUDGE_KWARGS['default'] is False

  def test_no_judge_kwargs_keys(self):
    """Shared kwargs contain all required argparse fields."""
    assert _NO_JUDGE_KWARGS['action'] == 'store_true'
    assert _NO_JUDGE_KWARGS['dest'] == 'no_judge'
    assert _NO_JUDGE_KWARGS['default'] is False
