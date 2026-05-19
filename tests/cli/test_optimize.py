"""Tests for optimize CLI command handlers.

Covers BUG-008 (strategy removal), BUG-010 (corrupt notes warning),
BUG-011 (store/tree/forest forwarding), BUG-013 (direct feedback access).
"""

from autopilot.cli.commands.optimize import (
  OptimizeCommand,
  Train,
  Validate,
  _build_loop_trainer,
)
from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum
from pathlib import Path
from tests.cli.conftest import make_mock_cli_context
from unittest.mock import MagicMock, patch
import argparse
import json
import logging
import pytest


def _make_ctx(tmp_path: Path, use_json: bool = True) -> MagicMock:
  """Build a mock CLIContext with optimize-specific defaults."""
  trainer = MagicMock(spec=Trainer)
  trainer.callbacks = []
  trainer.dry_run = False
  trainer.logger = None
  trainer.policy = None
  trainer.experiment = None
  trainer.config = None
  trainer.accumulate_grad_batches = 1
  trainer.store = None
  trainer.tree = None
  trainer.forest = None
  return make_mock_cli_context(
    tmp_path,
    use_json=use_json,
    experiment='test-exp',
    split=None,
    epoch=0,
    module=MagicMock(),
    datamodule=None,
    trainer=trainer,
    hyperparams_file=None,
  )


class TestStrategyRemoval:
  """BUG-008: OrchestratorConfig.strategy is dead code."""

  def test_orchestrator_config_no_strategy_field(self) -> None:
    """OrchestratorConfig has no strategy attribute."""
    config = OrchestratorConfig(plateau_window=0)
    assert not hasattr(config, 'strategy')

  def test_orchestrator_config_rejects_strategy_kwarg(self) -> None:
    """Passing strategy= to OrchestratorConfig raises TypeError."""
    with pytest.raises(TypeError):
      OrchestratorConfig(strategy='conservative')  # type: ignore[ty:unknown-argument]

  def test_build_loop_trainer_no_strategy_param(self, tmp_path: Path) -> None:
    """_build_loop_trainer works without strategy parameter."""
    trainer = Trainer()
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir(parents=True)
    result = _build_loop_trainer(trainer, exp_dir)
    assert isinstance(result, Trainer)

  def test_loop_subcommand_no_strategy_flag(self) -> None:
    """The loop subcommand does not register --strategy."""
    from autopilot.cli.primitives import Argument, collect_arguments

    cmd = OptimizeCommand()
    all_args = collect_arguments(type(cmd))
    all_flags: list[str] = []
    for arg in all_args:
      if isinstance(arg, Argument):
        all_flags.extend(arg.flags)
    assert '--strategy' not in all_flags

  def test_dry_run_plan_no_strategy(self) -> None:
    """Dry run plan dict does not include strategy key."""
    orch = EpochOrchestrator()
    trainer = Trainer(loop=orch, dry_run=True)

    from autopilot.core.module.autopilot_module import AutoPilotModule
    from autopilot.core.types import EvalDatum

    class Stub(AutoPilotModule):
      def forward(self, batch):
        return EvalDatum(success=True)

      def training_step(self, batch, batch_idx):
        return EvalDatum(success=True)

      def configure_optimizers(self):
        return None

    result = trainer.fit(Stub(), train_dataloaders=[1], max_epochs=3)
    assert 'strategy' not in result.get('orchestrator_config', {})


class TestCorruptNotesWarning:
  """BUG-010: Silent experiment.notes parse failure."""

  def test_set_hparams_logs_warning_on_corrupt_notes(
    self, tmp_path: Path, caplog: pytest.LogCaptureFixture
  ) -> None:
    """When notes are not valid JSON, a warning is logged and hparams still written."""
    ctx = _make_ctx(tmp_path)
    ctx.output = MagicMock()
    ctx.context = 'test'

    mock_exp = MagicMock()
    mock_exp.id = 'exp-123'
    mock_exp.notes = 'not valid json {'

    mock_node = MagicMock()
    mock_node.experiment = mock_exp

    mock_tree = MagicMock()
    mock_forest = MagicMock()
    mock_forest.active = mock_tree

    with (
      patch('autopilot.cli.commands.optimize.load_forest', return_value=mock_forest),
      patch('autopilot.cli.commands.optimize.require_active_tree', return_value=mock_tree),
      patch('autopilot.cli.commands.optimize.require_experiment_node', return_value=mock_node),
      caplog.at_level(logging.WARNING, logger='autopilot.cli.commands.optimize'),
    ):
      cmd = OptimizeCommand()
      args = argparse.Namespace(
        command='optimize',
        optimize_action='set-hparams',
        values='{"lr": 0.01}',
      )
      cmd.set_hparams(ctx, args)

    assert 'not valid JSON' in caplog.text
    assert 'exp-123' in caplog.text
    written_notes = json.loads(mock_exp.notes)
    assert written_notes['hparams'] == {'lr': 0.01}
    mock_forest.save.assert_called_once()

  def test_set_hparams_valid_notes_no_warning(
    self, tmp_path: Path, caplog: pytest.LogCaptureFixture
  ) -> None:
    """Valid JSON notes produce no warning."""
    ctx = _make_ctx(tmp_path)
    ctx.output = MagicMock()
    ctx.context = 'test'

    mock_exp = MagicMock()
    mock_exp.id = 'exp-456'
    mock_exp.notes = '{"existing": "data"}'

    mock_node = MagicMock()
    mock_node.experiment = mock_exp

    mock_tree = MagicMock()
    mock_forest = MagicMock()
    mock_forest.active = mock_tree

    with (
      patch('autopilot.cli.commands.optimize.load_forest', return_value=mock_forest),
      patch('autopilot.cli.commands.optimize.require_active_tree', return_value=mock_tree),
      patch('autopilot.cli.commands.optimize.require_experiment_node', return_value=mock_node),
      caplog.at_level(logging.WARNING, logger='autopilot.cli.commands.optimize'),
    ):
      cmd = OptimizeCommand()
      args = argparse.Namespace(
        command='optimize',
        optimize_action='set-hparams',
        values='{"lr": 0.01}',
      )
      cmd.set_hparams(ctx, args)

    assert 'not valid JSON' not in caplog.text
    written_notes = json.loads(mock_exp.notes)
    assert written_notes['hparams'] == {'lr': 0.01}
    assert written_notes['existing'] == 'data'


class TestBuildLoopTrainerForwarding:
  """BUG-011: _build_loop_trainer drops store, tree, forest."""

  def test_forwards_store(self, tmp_path: Path) -> None:
    """Loop trainer preserves the original trainer's store."""
    mock_store = MagicMock()
    trainer = Trainer(store=mock_store)
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir(parents=True)

    result = _build_loop_trainer(trainer, exp_dir)
    assert result.store is mock_store

  def test_forwards_tree(self, tmp_path: Path) -> None:
    """Loop trainer preserves the original trainer's tree."""
    mock_tree = MagicMock()
    trainer = Trainer(tree=mock_tree)
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir(parents=True)

    result = _build_loop_trainer(trainer, exp_dir)
    assert result.tree is mock_tree

  def test_forwards_forest(self, tmp_path: Path) -> None:
    """Loop trainer preserves the original trainer's forest."""
    mock_forest = MagicMock()
    trainer = Trainer(forest=mock_forest)
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir(parents=True)

    result = _build_loop_trainer(trainer, exp_dir)
    assert result.forest is mock_forest

  def test_forwards_all_three(self, tmp_path: Path) -> None:
    """Loop trainer preserves store, tree, and forest together."""
    mock_store = MagicMock()
    mock_tree = MagicMock()
    mock_forest = MagicMock()
    trainer = Trainer(store=mock_store, tree=mock_tree, forest=mock_forest)
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir(parents=True)

    result = _build_loop_trainer(trainer, exp_dir)
    assert result.store is mock_store
    assert result.tree is mock_tree
    assert result.forest is mock_forest

  def test_none_values_preserved(self, tmp_path: Path) -> None:
    """When original trainer has None store/tree/forest, loop trainer does too."""
    trainer = Trainer()
    exp_dir = tmp_path / 'exp'
    exp_dir.mkdir(parents=True)

    result = _build_loop_trainer(trainer, exp_dir)
    assert result.store is None
    assert result.tree is None
    assert result.forest is None


class TestDirectFeedbackAccess:
  """BUG-013: getattr(observation, 'feedback', None) replaced with direct access."""

  def test_train_result_includes_feedback_value(self, tmp_path: Path, capsys) -> None:
    """EvalDatum with feedback='good' appears in train result."""
    ctx = _make_ctx(tmp_path)
    ctx.module.return_value = EvalDatum(success=True, feedback='good', metrics={'acc': 0.9})

    cmd = Train()
    args = MagicMock()
    args.limit = 0

    cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['feedback'] == 'good'

  def test_train_result_feedback_none(self, tmp_path: Path, capsys) -> None:
    """EvalDatum without feedback has feedback=None in train result."""
    ctx = _make_ctx(tmp_path)
    ctx.module.return_value = EvalDatum(success=True, metrics={'acc': 0.9})

    cmd = Train()
    args = MagicMock()
    args.limit = 0

    cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['feedback'] is None

  def test_validate_result_includes_feedback_value(self, tmp_path: Path, capsys) -> None:
    """EvalDatum with feedback='poor' appears in validate result."""
    ctx = _make_ctx(tmp_path)
    ctx.module.return_value = EvalDatum(success=True, feedback='poor', metrics={'f1': 0.7})

    cmd = Validate()
    args = MagicMock()

    cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['feedback'] == 'poor'

  def test_validate_result_feedback_none(self, tmp_path: Path, capsys) -> None:
    """EvalDatum without feedback has feedback=None in validate result."""
    ctx = _make_ctx(tmp_path)
    ctx.module.return_value = EvalDatum(success=True, metrics={'f1': 0.7})

    cmd = Validate()
    args = MagicMock()

    cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['feedback'] is None
