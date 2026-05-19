"""Tests for autopilot.cli.commands.policy.PolicyCommand."""

from autopilot.cli.commands.policy import PolicyCommand
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.core.types import GateResult
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from unittest.mock import MagicMock
import argparse
import pytest


def _policy_ctx(tmp_path: Path, slug: str | None = 'exp-a') -> MagicMock:
  """Build a mock CLIContext for policy tests."""
  cfg = AutoPilotConfig(workspace=tmp_path)
  cfg.init_workspace()
  ctx = MagicMock()
  ctx.experiment = slug
  ctx.config = cfg
  ctx.module = None
  ctx.output = MagicMock(spec=Output)
  ctx.experiment_path = lambda s=None: cfg.experiment_path(slug=s or slug or 'default')
  ctx.fail = MagicMock(side_effect=SystemExit(1))
  return ctx


def _seed_result(ctx: MagicMock, slug: str, result_dict: dict | None = None) -> None:
  """Write a result.json into the experiment directory."""
  exp_dir = ctx.experiment_path(slug)
  exp_dir.mkdir(parents=True, exist_ok=True)
  payload = result_dict or {'metrics': {'score': 0.9}, 'gates': [], 'passed': True}
  atomic_write_json(exp_dir / 'result.json', payload)


def _check_args(**kwargs) -> argparse.Namespace:
  """Build argparse.Namespace for policy check with defaults."""
  defaults = {
    'metrics_json': None,
    'min_thresholds': None,
    'max_thresholds': None,
  }
  defaults.update(kwargs)
  return argparse.Namespace(**defaults)


class TestPolicyCheck:
  def test_no_metrics_no_experiment_fails(self, tmp_path: Path) -> None:
    """check without --metrics and without experiment fails."""
    ctx = _policy_ctx(tmp_path, slug=None)
    args = _check_args(min_thresholds=['accuracy:0.8'])
    with pytest.raises(SystemExit):
      PolicyCommand().check(ctx, args)
    ctx.fail.assert_called_once()
    assert '--metrics' in ctx.fail.call_args[0][0]

  def test_no_policy_source_fails(self, tmp_path: Path) -> None:
    """check with metrics but no --min/--max and no module policy fails."""
    ctx = _policy_ctx(tmp_path)
    ctx.module = None
    args = _check_args(metrics_json='{"accuracy": 0.9}')
    with pytest.raises(SystemExit):
      PolicyCommand().check(ctx, args)
    ctx.fail.assert_called_once()
    assert '--min' in ctx.fail.call_args[0][0]

  def test_module_policy_used_without_flags(self, tmp_path: Path) -> None:
    """check falls back to ctx.module.policy when no --min/--max."""
    ctx = _policy_ctx(tmp_path)
    mock_policy = MagicMock()
    mock_policy.name.return_value = 'quality_first'
    mock_policy.forward.return_value = GateResult.PASSED
    ctx.module = MagicMock()
    ctx.module.policy = mock_policy

    args = _check_args(metrics_json='{"accuracy": 0.9}')
    PolicyCommand().check(ctx, args)
    body = ctx.output.result.call_args[0][0]
    assert body['policy'] == 'quality_first'
    assert body['gate_result'] == 'pass'
    assert ctx.output.result.call_args[1]['ok'] is True

  def test_min_gate_overrides_module(self, tmp_path: Path) -> None:
    """--min flags produce a ThresholdPolicy, ignoring module policy."""
    ctx = _policy_ctx(tmp_path)
    ctx.module = MagicMock()
    ctx.module.policy = MagicMock()

    args = _check_args(
      metrics_json='{"accuracy": 0.9}',
      min_thresholds=['accuracy:0.8'],
    )
    PolicyCommand().check(ctx, args)
    body = ctx.output.result.call_args[0][0]
    assert body['policy'] == 'threshold'
    assert body['gate_result'] == 'pass'

  def test_malformed_json_fails(self, tmp_path: Path) -> None:
    """check with non-JSON --metrics calls ctx.fail."""
    ctx = _policy_ctx(tmp_path)
    args = _check_args(
      metrics_json='not json',
      min_thresholds=['accuracy:0.8'],
    )
    with pytest.raises(SystemExit):
      PolicyCommand().check(ctx, args)
    ctx.fail.assert_called_once()
    assert 'malformed' in ctx.fail.call_args[0][0]


class TestPolicyExplain:
  def test_no_slug_calls_fail(self, tmp_path: Path) -> None:
    """explain without experiment slug calls ctx.fail."""
    ctx = _policy_ctx(tmp_path, slug=None)
    with pytest.raises(SystemExit):
      PolicyCommand().explain(ctx, argparse.Namespace())
    ctx.fail.assert_called_once()
    msg = ctx.fail.call_args[0][0]
    assert 'experiment slug' in msg
    assert '--experiment' in msg

  def test_no_module_exits_nonzero(self, tmp_path: Path) -> None:
    """explain with slug but ctx.module=None exits 1 via ctx.fail."""
    ctx = _policy_ctx(tmp_path)
    ctx.module = None
    with pytest.raises(SystemExit):
      PolicyCommand().explain(ctx, argparse.Namespace())
    ctx.fail.assert_called_once()
    msg = ctx.fail.call_args[0][0]
    assert 'no module' in msg

  def test_no_policy_exits_nonzero(self, tmp_path: Path) -> None:
    """explain with module but module.policy=None exits 1 via ctx.fail."""
    ctx = _policy_ctx(tmp_path)
    ctx.module = MagicMock()
    ctx.module.policy = None
    with pytest.raises(SystemExit):
      PolicyCommand().explain(ctx, argparse.Namespace())
    ctx.fail.assert_called_once()
    msg = ctx.fail.call_args[0][0]
    assert 'no policy' in msg

  def test_no_result_exits_nonzero(self, tmp_path: Path) -> None:
    """explain with policy but no result artifact exits 1 via ctx.fail."""
    ctx = _policy_ctx(tmp_path)
    ctx.module = MagicMock()
    ctx.module.policy = MagicMock()
    exp_dir = ctx.experiment_path('exp-a')
    exp_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit):
      PolicyCommand().explain(ctx, argparse.Namespace())
    ctx.fail.assert_called_once()
    msg = ctx.fail.call_args[0][0]
    assert 'no result' in msg

  def test_happy_path_returns_explanation(self, tmp_path: Path) -> None:
    """explain happy path returns explanation string from policy."""
    ctx = _policy_ctx(tmp_path)
    _seed_result(ctx, 'exp-a')

    mock_policy = MagicMock()
    mock_policy.name.return_value = 'quality_first'
    mock_policy.forward.return_value = GateResult.PASSED
    mock_policy.explain.return_value = 'score exceeded threshold'

    ctx.module = MagicMock()
    ctx.module.policy = mock_policy

    PolicyCommand().explain(ctx, argparse.Namespace())
    body = ctx.output.result.call_args[0][0]
    assert body['slug'] == 'exp-a'
    assert body['policy'] == 'quality_first'
    assert body['explanation'] == 'score exceeded threshold'
    assert body['gate_result'] == 'pass'
    assert 'result' in body
    assert ctx.output.result.call_args[1]['ok'] is True
