"""Tests for policy check/explain exit code contract (BUG-POLICY-EXITCODE, FR-007).

Verifies that:
- GateResult.FAIL yields ok=False and SystemExit(1).
- GateResult.WARN (optional gate failure) yields ok=True and exit 0.
- GateResult.PASSED yields ok=True and exit 0.
- GateResult.SKIP yields ok=True and exit 0.
- Both check and explain follow the same contract.
"""

from autopilot.cli.commands.policy import PolicyCommand
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.core.types import GateResult
from autopilot.policy.gates import MinGate
from autopilot.policy.quality_first import QualityFirstPolicy
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from unittest.mock import MagicMock
import argparse
import pytest


def _policy_ctx(tmp_path: Path, slug: str = 'exp-a') -> MagicMock:
  """Build a mock CLIContext for policy exit code tests."""
  cfg = AutoPilotConfig(workspace=tmp_path)
  cfg.init_workspace()
  ctx = MagicMock()
  ctx.experiment = slug
  ctx.config = cfg
  ctx.module = None
  ctx.output = MagicMock(spec=Output)
  ctx.experiment_path = lambda s=None: cfg.experiment_path(slug=s or slug)
  ctx.fail = MagicMock(side_effect=SystemExit(1))
  return ctx


def _seed_result(ctx: MagicMock, slug: str, metrics: dict[str, float]) -> None:
  """Write a result.json with given metrics into the experiment directory."""
  exp_dir = ctx.experiment_path(slug)
  exp_dir.mkdir(parents=True, exist_ok=True)
  payload = {'metrics': metrics, 'gates': [], 'passed': True}
  atomic_write_json(exp_dir / 'result.json', payload)


def _check_args(**kwargs) -> argparse.Namespace:
  """Build args namespace with new check-required attributes."""
  defaults = {
    'metrics_json': None,
    'min_thresholds': None,
    'max_thresholds': None,
  }
  defaults.update(kwargs)
  return argparse.Namespace(**defaults)


class TestPolicyCheckExitCode:
  """policy check exit code contract tests."""

  def test_policy_check_fail_exits_nonzero_json(self, tmp_path: Path) -> None:
    """BUG-POLICY-EXITCODE: check FAIL yields ok=False and SystemExit(1)."""
    ctx = _policy_ctx(tmp_path)

    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.8)])
    mock_module = MagicMock()
    mock_module.policy = policy
    ctx.module = mock_module

    args = _check_args(metrics_json='{"accuracy": 0.3}')
    with pytest.raises(SystemExit) as exc_info:
      PolicyCommand().check(ctx, args)

    assert exc_info.value.code == 1
    call_kwargs = ctx.output.result.call_args[1]
    assert call_kwargs['ok'] is False
    body = ctx.output.result.call_args[0][0]
    assert body['gate_result'] == 'fail'

  def test_policy_check_pass_exits_zero(self, tmp_path: Path) -> None:
    """check with passing gates yields ok=True and no SystemExit."""
    ctx = _policy_ctx(tmp_path)

    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.8)])
    mock_module = MagicMock()
    mock_module.policy = policy
    ctx.module = mock_module

    args = _check_args(metrics_json='{"accuracy": 0.95}')
    PolicyCommand().check(ctx, args)

    call_kwargs = ctx.output.result.call_args[1]
    assert call_kwargs['ok'] is True
    body = ctx.output.result.call_args[0][0]
    assert body['gate_result'] == 'pass'

  def test_policy_check_warn_exits_zero(self, tmp_path: Path) -> None:
    """FR-007: optional gate failure (WARN) yields ok=True and exit 0."""
    ctx = _policy_ctx(tmp_path)

    required_gate = MinGate('accuracy', 0.8, required=True)
    optional_gate = MinGate('style_score', 0.5, required=False)
    policy = QualityFirstPolicy(gates=[required_gate, optional_gate])
    mock_module = MagicMock()
    mock_module.policy = policy
    ctx.module = mock_module

    args = _check_args(metrics_json='{"accuracy": 0.95, "style_score": 0.1}')
    PolicyCommand().check(ctx, args)

    call_kwargs = ctx.output.result.call_args[1]
    assert call_kwargs['ok'] is True
    body = ctx.output.result.call_args[0][0]
    assert body['gate_result'] == 'warn'

  def test_policy_check_skip_exits_zero(self, tmp_path: Path) -> None:
    """SKIP gate result yields ok=True (gate abstained)."""
    ctx = _policy_ctx(tmp_path)

    mock_policy = MagicMock()
    mock_policy.name.return_value = 'custom'
    mock_policy.forward.return_value = GateResult.SKIP
    mock_module = MagicMock()
    mock_module.policy = mock_policy
    ctx.module = mock_module

    args = _check_args(metrics_json='{"accuracy": 0.9}')
    PolicyCommand().check(ctx, args)

    call_kwargs = ctx.output.result.call_args[1]
    assert call_kwargs['ok'] is True
    body = ctx.output.result.call_args[0][0]
    assert body['gate_result'] == 'skip'

  def test_policy_check_json_ok_matches_gate_result(self, tmp_path: Path) -> None:
    """Envelope ok field matches: ok == (gate_result != 'fail')."""
    ctx = _policy_ctx(tmp_path)

    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.8)])
    mock_module = MagicMock()
    mock_module.policy = policy
    ctx.module = mock_module

    args = _check_args(metrics_json='{"accuracy": 0.3}')
    with pytest.raises(SystemExit):
      PolicyCommand().check(ctx, args)

    body = ctx.output.result.call_args[0][0]
    call_kwargs = ctx.output.result.call_args[1]
    expected_ok = body['gate_result'] != 'fail'
    assert call_kwargs['ok'] == expected_ok


class TestPolicyExplainExitCode:
  """policy explain exit code contract tests."""

  def test_policy_explain_fail_matches_check_ok_false(self, tmp_path: Path) -> None:
    """explain with FAIL gate yields ok=False and SystemExit(1), matching check."""
    ctx = _policy_ctx(tmp_path)
    _seed_result(ctx, 'exp-a', {'accuracy': 0.3})

    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.8)])
    mock_module = MagicMock()
    mock_module.policy = policy
    ctx.module = mock_module

    with pytest.raises(SystemExit) as exc_info:
      PolicyCommand().explain(ctx, argparse.Namespace())

    assert exc_info.value.code == 1
    call_kwargs = ctx.output.result.call_args[1]
    assert call_kwargs['ok'] is False
    body = ctx.output.result.call_args[0][0]
    assert body['gate_result'] == 'fail'
    assert 'explanation' in body

  def test_policy_explain_json_includes_ok_field(self, tmp_path: Path) -> None:
    """explain JSON envelope includes ok field on success path."""
    ctx = _policy_ctx(tmp_path)
    _seed_result(ctx, 'exp-a', {'accuracy': 0.95})

    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.8)])
    mock_module = MagicMock()
    mock_module.policy = policy
    ctx.module = mock_module

    PolicyCommand().explain(ctx, argparse.Namespace())

    call_kwargs = ctx.output.result.call_args[1]
    assert 'ok' in call_kwargs
    assert call_kwargs['ok'] is True
    body = ctx.output.result.call_args[0][0]
    assert body['gate_result'] == 'pass'
    assert 'explanation' in body

  def test_policy_explain_warn_exits_zero(self, tmp_path: Path) -> None:
    """explain with WARN (optional gate failure) yields ok=True."""
    ctx = _policy_ctx(tmp_path)
    _seed_result(ctx, 'exp-a', {'accuracy': 0.95, 'style_score': 0.1})

    required_gate = MinGate('accuracy', 0.8, required=True)
    optional_gate = MinGate('style_score', 0.5, required=False)
    policy = QualityFirstPolicy(gates=[required_gate, optional_gate])
    mock_module = MagicMock()
    mock_module.policy = policy
    ctx.module = mock_module

    PolicyCommand().explain(ctx, argparse.Namespace())

    call_kwargs = ctx.output.result.call_args[1]
    assert call_kwargs['ok'] is True
    body = ctx.output.result.call_args[0][0]
    assert body['gate_result'] == 'warn'
    assert 'explanation' in body

  def test_policy_explain_calls_forward_before_explain(self, tmp_path: Path) -> None:
    """explain calls policy.forward() explicitly before policy.explain()."""
    ctx = _policy_ctx(tmp_path)
    _seed_result(ctx, 'exp-a', {'accuracy': 0.95})

    call_order: list[str] = []

    mock_policy = MagicMock()
    mock_policy.name.return_value = 'quality_first'
    mock_policy.forward.side_effect = lambda r: call_order.append('forward') or GateResult.PASSED
    mock_policy.explain.side_effect = lambda r: call_order.append('explain') or 'all gates passed'

    mock_module = MagicMock()
    mock_module.policy = mock_policy
    ctx.module = mock_module

    PolicyCommand().explain(ctx, argparse.Namespace())

    assert call_order == ['forward', 'explain']
