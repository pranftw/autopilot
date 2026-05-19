"""Tests for debug CLI: cost, parameters, gradients, and agent removal.

Covers:
  - debug cost reads cost_summary.json
  - debug cost missing file fails with guidance
  - debug parameters lists names
  - debug gradients null when cleared
  - debug gradients numeric gradient
  - debug gradients text gradient
  - agent command removed (no NotImplementedError)
  - agent not advertised in help
  - debug store prune-orphans JSON envelope
"""

from autopilot.ai.gradient import TextGradient
from autopilot.cli.commands.debug import DebugCommand, DebugStoreCommand
from autopilot.cli.context import CLIContext
from autopilot.core.gradient import NumericGradient
from autopilot.core.module.module import Module
from autopilot.core.parameter import Parameter
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from tests.cli.conftest import make_mock_cli_context
from unittest.mock import MagicMock
import json
import pytest


def _make_ctx(tmp_path: Path, use_json: bool = True) -> MagicMock:
  return make_mock_cli_context(tmp_path, use_json=use_json, experiment='exp-001', module=None)


def _args() -> MagicMock:
  return MagicMock()


class TestDebugCost:
  def test_reads_summary(self, tmp_path: Path, capsys) -> None:
    """When cost_summary.json exists, CLI returns parsed totals."""
    ctx = _make_ctx(tmp_path)
    exp_dir = ctx.config.experiment_path(slug='exp-001')
    exp_dir.mkdir(parents=True, exist_ok=True)
    cost_data = {'wall_clock_s': 42.5, 'api_calls': 10, 'tokens_used': 5000, 'epoch': 0}
    atomic_write_json(exp_dir / 'cost_summary.json', cost_data)
    ctx.experiment_path = CLIContext.experiment_path.__get__(ctx, type(ctx))

    cmd = DebugCommand()
    cmd.cost(ctx, _args())

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['wall_clock_s'] == 42.5
    assert envelope['result']['api_calls'] == 10

  def test_missing_file_fails(self, tmp_path: Path) -> None:
    """Missing experiment / missing file fails with actionable message."""
    ctx = _make_ctx(tmp_path)
    exp_dir = ctx.config.experiment_path(slug='exp-001')
    exp_dir.mkdir(parents=True, exist_ok=True)
    ctx.experiment_path = CLIContext.experiment_path.__get__(ctx, type(ctx))
    ctx.output = MagicMock()

    cmd = DebugCommand()
    with pytest.raises(SystemExit) as exc_info:
      cmd.cost(ctx, _args())
    assert exc_info.value.code == 1
    ctx.output.error.assert_called_once()
    assert 'CostTrackerCallback' in ctx.output.error.call_args[0][0]

  def test_missing_experiment_slug_fails(self, tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    ctx.experiment = None
    ctx.output = MagicMock()
    cmd = DebugCommand()
    with pytest.raises(SystemExit) as exc_info:
      cmd.cost(ctx, _args())
    assert exc_info.value.code == 1


class _SimpleModule(Module):
  """Test module with two parameters."""

  def __init__(self) -> None:
    super().__init__()
    self.prompt = Parameter()
    self.config_param = Parameter()

  def forward(self, ctx, params):
    pass


class TestDebugParameters:
  def test_lists_names(self, tmp_path: Path, capsys) -> None:
    """Synthetic module exposes parameter keys in JSON."""
    ctx = _make_ctx(tmp_path)
    ctx.module = _SimpleModule()

    cmd = DebugCommand()
    cmd.parameters(ctx, _args())

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    params = envelope['result']['parameters']
    assert 'prompt' in params
    assert 'config_param' in params
    assert params['prompt']['type'] == 'Parameter'

  def test_no_module_fails(self, tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    ctx.module = None
    ctx.output = MagicMock()

    cmd = DebugCommand()
    with pytest.raises(SystemExit) as exc_info:
      cmd.parameters(ctx, _args())
    assert exc_info.value.code == 1


class TestDebugModuleGradients:
  def test_null_when_cleared(self, tmp_path: Path, capsys) -> None:
    """After zero_grad, JSON marks absent grads distinctly."""
    ctx = _make_ctx(tmp_path)
    module = _SimpleModule()
    ctx.module = module

    cmd = DebugCommand()
    cmd.module_gradients(ctx, _args())

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    grads = envelope['result']['gradients']
    assert grads['prompt'] is None
    assert grads['config_param'] is None

  def test_numeric_gradient(self, tmp_path: Path, capsys) -> None:
    """NumericGradient shows value and type."""
    ctx = _make_ctx(tmp_path)
    module = _SimpleModule()
    module.prompt.grad = NumericGradient(value=42.5)
    ctx.module = module

    cmd = DebugCommand()
    cmd.module_gradients(ctx, _args())

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    grads = envelope['result']['gradients']
    assert grads['prompt']['type'] == 'NumericGradient'
    assert grads['prompt']['value'] == 42.5
    assert 'gradient:' in grads['prompt']['preview']

  def test_text_gradient(self, tmp_path: Path, capsys) -> None:
    """TextGradient shows type and rendered preview."""
    ctx = _make_ctx(tmp_path)
    module = _SimpleModule()
    module.prompt.grad = TextGradient(
      text='simplify the wording',
      attribution='prompt is too verbose',
      severity=0.7,
    )
    ctx.module = module

    cmd = DebugCommand()
    cmd.module_gradients(ctx, _args())

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    grads = envelope['result']['gradients']
    assert grads['prompt']['type'] == 'TextGradient'
    assert len(grads['prompt']['preview']) > 0
    assert grads['config_param'] is None


class TestAgentCommandRemoval:
  def test_no_not_implemented_error(self) -> None:
    """No code path raises bare NotImplementedError for agent surface."""
    from autopilot.cli.main import AutoPilotCLI

    cli = AutoPilotCLI()
    assert 'agent' not in cli.commands

  def test_agent_not_in_help(self) -> None:
    """Agent not advertised in help."""
    from autopilot.cli.main import AutoPilotCLI

    cli = AutoPilotCLI()
    command_names = list(cli.commands.keys())
    assert 'agent' not in command_names


class TestDebugStorePruneOrphans:
  def test_json_envelope(self, tmp_path: Path, capsys) -> None:
    """Prune-orphans returns structured JSON with removed list."""
    ctx = _make_ctx(tmp_path)

    config = ctx.config
    config.store_path.mkdir(parents=True, exist_ok=True)
    config.objects_path.mkdir(parents=True, exist_ok=True)

    cmd = DebugStoreCommand()
    cmd.prune_orphans(ctx, _args())

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['count'] == 0
    assert envelope['result']['removed'] == []
