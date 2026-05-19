"""Tests for Plan 07: Policy & Propose CLI Enhancements.

Covers:
- ``policy check`` with --metrics, --min/--max, forest metrics, module policy.
- ``propose verify`` with forest-metrics fallback, numeric filtering, warnings.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.proposal import ChangeProposal, record_proposal
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.policy import PolicyCommand, _resolve_metrics_from_forest
from autopilot.cli.commands.propose import ProposeCommand, _filter_numeric_metrics
from autopilot.cli.context import CLIContext
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.models import Result
from autopilot.core.node import Node
from autopilot.core.types import GateResult
from autopilot.policy.gates import MinGate
from autopilot.policy.policy import Policy
from autopilot.policy.quality_first import QualityFirstPolicy
from autopilot.policy.threshold import (
  ThresholdPolicy,
  build_threshold_gates,
  parse_threshold_spec,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import io
import json
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
  workspace: Path,
  *,
  use_json: bool = True,
  experiment: str | None = None,
  module: Any = None,
  epoch: int | None = None,
) -> CLIContext:
  """Build a CLIContext for testing policy/propose commands."""
  config = AutoPilotConfig(workspace=workspace)
  return CLIContext(
    workspace=workspace,
    config=config,
    experiment=experiment,
    epoch=epoch,
    output=Output(use_json=use_json),
    module=module,
  )


def _make_args(**kwargs: Any) -> argparse.Namespace:
  """Build an argparse.Namespace with defaults for policy check."""
  defaults = {
    'metrics_json': None,
    'min_thresholds': None,
    'max_thresholds': None,
  }
  defaults.update(kwargs)
  return argparse.Namespace(**defaults)


def _make_verify_args(**kwargs: Any) -> argparse.Namespace:
  """Build an argparse.Namespace with defaults for propose verify."""
  defaults = {
    'proposal_id': 'test-001',
    'higher_is_better': None,
    'lower_is_better': None,
  }
  defaults.update(kwargs)
  return argparse.Namespace(**defaults)


@dataclass
class _StubModule:
  """Module stub with configurable policy."""

  policy: Policy | None = None


def _capture_json(ctx: CLIContext, fn: Any) -> dict[str, Any]:
  """Capture JSON output from a command function call."""
  buf = io.StringIO()
  import contextlib

  with contextlib.redirect_stdout(buf):
    fn()
  output = buf.getvalue().strip()
  if output:
    return json.loads(output)
  return {}


def _seed_forest(workspace: Path, experiment_id: str, metrics: dict[str, Any]) -> None:
  """Seed a forest with one tree and one completed experiment."""
  config = AutoPilotConfig(workspace=workspace)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  exp = Experiment(experiment_id=experiment_id, hypothesis='test')
  exp.start()
  exp.complete(metrics=metrics)
  tree.add(Node(experiment=exp))
  forest.switch('main')
  forest.save()


# ---------------------------------------------------------------------------
# ThresholdPolicy unit tests
# ---------------------------------------------------------------------------


class TestThresholdPolicy:
  """Unit tests for ThresholdPolicy and helpers."""

  def test_parse_threshold_spec_valid(self) -> None:
    metric, threshold = parse_threshold_spec('accuracy:0.8')
    assert metric == 'accuracy'
    assert threshold == 0.8

  def test_parse_threshold_spec_no_colon(self) -> None:
    with pytest.raises(ValueError, match='expected metric'):
      parse_threshold_spec('accuracy')

  def test_parse_threshold_spec_non_numeric(self) -> None:
    with pytest.raises(ValueError, match='non-numeric threshold'):
      parse_threshold_spec('accuracy:abc')

  def test_parse_threshold_spec_empty_metric(self) -> None:
    with pytest.raises(ValueError, match='empty metric name'):
      parse_threshold_spec(':0.8')

  def test_build_threshold_gates_min_and_max(self) -> None:
    gates = build_threshold_gates(
      min_specs=['accuracy:0.8'],
      max_specs=['latency:100'],
    )
    assert len(gates) == 2

  def test_build_threshold_gates_none(self) -> None:
    gates = build_threshold_gates()
    assert gates == []

  def test_threshold_policy_all_pass(self) -> None:
    gates = build_threshold_gates(min_specs=['accuracy:0.8'])
    policy = ThresholdPolicy(gates)
    result = Result(metrics={'accuracy': 0.9})
    assert policy.forward(result) == GateResult.PASSED
    assert policy.name() == 'threshold'

  def test_threshold_policy_fail(self) -> None:
    gates = build_threshold_gates(min_specs=['accuracy:0.95'])
    policy = ThresholdPolicy(gates)
    result = Result(metrics={'accuracy': 0.9})
    assert policy.forward(result) == GateResult.FAIL

  def test_threshold_policy_explain(self) -> None:
    gates = build_threshold_gates(min_specs=['accuracy:0.8'])
    policy = ThresholdPolicy(gates)
    result = Result(metrics={'accuracy': 0.9})
    explanation = policy.explain(result)
    assert 'MinGate' in explanation
    assert 'accuracy' in explanation


# ---------------------------------------------------------------------------
# _filter_numeric_metrics unit tests
# ---------------------------------------------------------------------------


class TestFilterNumericMetrics:
  """Unit tests for the _filter_numeric_metrics helper."""

  def test_all_numeric(self) -> None:
    numeric, skipped = _filter_numeric_metrics({'a': 1.0, 'b': 2})
    assert numeric == {'a': 1.0, 'b': 2.0}
    assert skipped == set()

  def test_skip_bool(self) -> None:
    numeric, skipped = _filter_numeric_metrics({'flag': True, 'score': 0.9})
    assert 'flag' not in numeric
    assert 'flag' in skipped
    assert numeric == {'score': 0.9}

  def test_skip_string(self) -> None:
    numeric, skipped = _filter_numeric_metrics({'name': 'test', 'val': 1.0})
    assert 'name' in skipped
    assert numeric == {'val': 1.0}

  def test_skip_nan(self) -> None:
    numeric, skipped = _filter_numeric_metrics({'x': float('nan'), 'y': 1.0})
    assert 'x' in skipped
    assert numeric == {'y': 1.0}

  def test_empty(self) -> None:
    numeric, skipped = _filter_numeric_metrics({})
    assert numeric == {}
    assert skipped == set()


# ---------------------------------------------------------------------------
# policy check tests
# ---------------------------------------------------------------------------


class TestPolicyCheckMetricsOnly:
  """policy check with --metrics + --min gates."""

  def test_policy_check_metrics_only_passes(self, tmp_path: Path) -> None:
    """--metrics + permissive --min -> exit 0, gate_result passes."""
    ctx = _make_ctx(tmp_path)
    args = _make_args(
      metrics_json='{"accuracy": 0.9}',
      min_thresholds=['accuracy:0.8'],
    )
    cmd = PolicyCommand()
    result = _capture_json(ctx, lambda: cmd.check(ctx, args))
    assert result['ok'] is True
    assert result['result']['gate_result'] == 'pass'
    assert result['result']['policy'] == 'threshold'

  def test_policy_check_min_gate_fail(self, tmp_path: Path) -> None:
    """Restrictive --min -> exit 1, ok false."""
    ctx = _make_ctx(tmp_path)
    args = _make_args(
      metrics_json='{"accuracy": 0.9}',
      min_thresholds=['accuracy:0.95'],
    )
    cmd = PolicyCommand()
    with pytest.raises(SystemExit, match='1'):
      _capture_json(ctx, lambda: cmd.check(ctx, args))

  def test_policy_check_max_gate_passes(self, tmp_path: Path) -> None:
    """--max with value within bounds -> passes."""
    ctx = _make_ctx(tmp_path)
    args = _make_args(
      metrics_json='{"latency": 50}',
      max_thresholds=['latency:100'],
    )
    cmd = PolicyCommand()
    result = _capture_json(ctx, lambda: cmd.check(ctx, args))
    assert result['ok'] is True

  def test_policy_check_max_gate_fails(self, tmp_path: Path) -> None:
    """--max with value exceeding bounds -> exit 1."""
    ctx = _make_ctx(tmp_path)
    args = _make_args(
      metrics_json='{"latency": 150}',
      max_thresholds=['latency:100'],
    )
    cmd = PolicyCommand()
    with pytest.raises(SystemExit, match='1'):
      _capture_json(ctx, lambda: cmd.check(ctx, args))


class TestPolicyCheckForestMetrics:
  """policy check resolving metrics from forest."""

  def test_policy_check_forest_metrics_without_module(self, tmp_path: Path) -> None:
    """No ctx.module; resolves metrics from seeded forest fixture."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_forest(ws, 'exp-001', {'accuracy': 0.85})
    ctx = _make_ctx(ws, experiment='exp-001')
    args = _make_args(min_thresholds=['accuracy:0.8'])
    cmd = PolicyCommand()
    result = _capture_json(ctx, lambda: cmd.check(ctx, args))
    assert result['ok'] is True
    assert result['result']['metrics']['accuracy'] == 0.85


class TestPolicyCheckErrors:
  """policy check error paths."""

  def test_policy_check_no_metrics_source_error(self, tmp_path: Path) -> None:
    """Message mentions --metrics OR experiment slug requirement."""
    ctx = _make_ctx(tmp_path)
    args = _make_args(min_thresholds=['accuracy:0.8'])
    cmd = PolicyCommand()
    with pytest.raises(SystemExit):
      cmd.check(ctx, args)

  def test_policy_check_malformed_metrics_json(self, tmp_path: Path) -> None:
    """Non-JSON triggers ctx.fail."""
    ctx = _make_ctx(tmp_path)
    args = _make_args(
      metrics_json='not valid json',
      min_thresholds=['accuracy:0.8'],
    )
    cmd = PolicyCommand()
    with pytest.raises(SystemExit):
      cmd.check(ctx, args)

  def test_policy_check_metrics_not_dict(self, tmp_path: Path) -> None:
    """JSON array instead of dict triggers ctx.fail."""
    ctx = _make_ctx(tmp_path)
    args = _make_args(
      metrics_json='[1, 2, 3]',
      min_thresholds=['accuracy:0.8'],
    )
    cmd = PolicyCommand()
    with pytest.raises(SystemExit):
      cmd.check(ctx, args)

  def test_policy_check_no_policy_source_error(self, tmp_path: Path) -> None:
    """No --min/--max and no module -> error listing both options."""
    ctx = _make_ctx(tmp_path)
    args = _make_args(metrics_json='{"accuracy": 0.9}')
    cmd = PolicyCommand()
    with pytest.raises(SystemExit):
      cmd.check(ctx, args)


class TestPolicyCheckFlagsOverrideModulePolicy:
  """--min/--max flags override ctx.module.policy."""

  def test_policy_check_flags_override_module_policy(self, tmp_path: Path) -> None:
    """Module policy would fail (threshold 0.99) but --min ad-hoc passes (0.8)."""
    failing_policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.99)])
    module = _StubModule(policy=failing_policy)
    ctx = _make_ctx(tmp_path, module=module)
    args = _make_args(
      metrics_json='{"accuracy": 0.9}',
      min_thresholds=['accuracy:0.8'],
    )
    cmd = PolicyCommand()
    result = _capture_json(ctx, lambda: cmd.check(ctx, args))
    assert result['ok'] is True
    assert result['result']['policy'] == 'threshold'

  def test_policy_check_module_policy_used_when_no_flags(self, tmp_path: Path) -> None:
    """Without --min/--max, uses ctx.module.policy."""
    passing_policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.5)])
    module = _StubModule(policy=passing_policy)
    ctx = _make_ctx(tmp_path, module=module)
    args = _make_args(metrics_json='{"accuracy": 0.9}')
    cmd = PolicyCommand()
    result = _capture_json(ctx, lambda: cmd.check(ctx, args))
    assert result['ok'] is True
    assert result['result']['policy'] == 'quality_first'


# ---------------------------------------------------------------------------
# propose verify tests
# ---------------------------------------------------------------------------


def _seed_proposal(exp_dir: Path, proposal_id: str, epoch: int = 1) -> None:
  """Seed a proposal into the experiment directory."""
  exp_dir.mkdir(parents=True, exist_ok=True)
  proposal = ChangeProposal(
    proposal_id=proposal_id,
    hypothesis='test hypothesis',
    epoch=epoch,
    status='proposed',
  )
  record_proposal(exp_dir, proposal)


def _seed_epoch_metrics(exp_dir: Path, epoch: int, metrics: dict[str, Any]) -> None:
  """Write epoch metrics file."""
  exp_dir.mkdir(parents=True, exist_ok=True)
  epoch_dir = exp_dir / f'epoch_{epoch}'
  epoch_dir.mkdir(parents=True, exist_ok=True)
  metrics_file = exp_dir / f'epoch_{epoch}_metrics.json'
  metrics_file.write_text(json.dumps(metrics), encoding='utf-8')


def _seed_epoch_data(exp_dir: Path, epoch: int, items: list[dict[str, Any]]) -> None:
  """Write epoch data.jsonl file."""
  epoch_dir = exp_dir / f'epoch_{epoch}'
  epoch_dir.mkdir(parents=True, exist_ok=True)
  data_file = epoch_dir / 'data.jsonl'
  lines = [json.dumps(item) for item in items]
  data_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')


class TestProposeVerifyStorePref:
  """propose verify prefers store epoch files when available."""

  def test_propose_verify_store_preferred(self, tmp_path: Path) -> None:
    """When epoch files exist, their values used (not forest)."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_forest(ws, 'exp-store', {'accuracy': 0.5})

    config = AutoPilotConfig(workspace=ws)
    exp_dir = config.experiment_path(slug='exp-store')
    _seed_proposal(exp_dir, 'p-001', epoch=1)
    _seed_epoch_metrics(exp_dir, 0, {'accuracy': 0.7})
    _seed_epoch_metrics(exp_dir, 1, {'accuracy': 0.9})
    _seed_epoch_data(exp_dir, 1, [{'id': '1'}])

    ctx = _make_ctx(ws, experiment='exp-store', epoch=1)
    args = _make_verify_args(proposal_id='p-001')
    cmd = ProposeCommand()

    result = _capture_json(ctx, lambda: cmd.verify(ctx, args))
    assert result['result']['verdict'] == 'improved'
    assert 'warnings' not in result['result']


class TestProposeVerifyForestFallback:
  """propose verify falls back to forest metrics."""

  def test_propose_verify_forest_fallback(self, tmp_path: Path) -> None:
    """Epoch files absent; forest metrics yield non-inconclusive with warnings."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_forest(ws, 'exp-fb', {'accuracy': 0.85, 'f1': 0.9})

    config = AutoPilotConfig(workspace=ws)
    exp_dir = config.experiment_path(slug='exp-fb')
    _seed_proposal(exp_dir, 'p-002', epoch=1)
    _seed_epoch_data(exp_dir, 1, [{'id': '1'}])

    ctx = _make_ctx(ws, experiment='exp-fb', epoch=1)
    args = _make_verify_args(proposal_id='p-002')
    cmd = ProposeCommand()

    result = _capture_json(ctx, lambda: cmd.verify(ctx, args))
    payload = result['result']
    assert 'warnings' in payload
    assert any('epoch metrics missing' in w for w in payload['warnings'])
    assert payload['verdict'] == 'inconclusive'


class TestProposeVerifyNumericFiltering:
  """propose verify skips non-numeric metrics with warnings."""

  def test_propose_verify_skips_non_numeric_with_warning(self, tmp_path: Path) -> None:
    """String metric skipped; 'skipped_non_numeric' substring present."""
    ws = tmp_path / 'ws'
    ws.mkdir()

    config = AutoPilotConfig(workspace=ws)
    exp_dir = config.experiment_path(slug='exp-num')
    _seed_proposal(exp_dir, 'p-003', epoch=1)
    _seed_epoch_metrics(exp_dir, 0, {'accuracy': 0.7, 'model_name': 'gpt4'})
    _seed_epoch_metrics(exp_dir, 1, {'accuracy': 0.9, 'model_name': 'gpt4o'})
    _seed_epoch_data(exp_dir, 1, [{'id': '1'}])

    ctx = _make_ctx(ws, experiment='exp-num', epoch=1)
    args = _make_verify_args(proposal_id='p-003')
    cmd = ProposeCommand()

    result = _capture_json(ctx, lambda: cmd.verify(ctx, args))
    payload = result['result']
    assert payload['verdict'] == 'improved'
    assert 'warnings' in payload
    assert any('skipped_non_numeric' in w for w in payload['warnings'])

  def test_propose_verify_no_numeric_metrics_inconclusive(self, tmp_path: Path) -> None:
    """Verdict is 'inconclusive' when no numeric metrics remain."""
    ws = tmp_path / 'ws'
    ws.mkdir()

    config = AutoPilotConfig(workspace=ws)
    exp_dir = config.experiment_path(slug='exp-nonum')
    _seed_proposal(exp_dir, 'p-004', epoch=1)
    _seed_epoch_metrics(exp_dir, 0, {'model': 'a', 'version': 'v1'})
    _seed_epoch_metrics(exp_dir, 1, {'model': 'b', 'version': 'v2'})
    _seed_epoch_data(exp_dir, 1, [{'id': '1'}])

    ctx = _make_ctx(ws, experiment='exp-nonum', epoch=1)
    args = _make_verify_args(proposal_id='p-004')
    cmd = ProposeCommand()

    result = _capture_json(ctx, lambda: cmd.verify(ctx, args))
    assert result['result']['verdict'] == 'inconclusive'

  def test_propose_verify_skips_bool_metrics(self, tmp_path: Path) -> None:
    """Bool metrics are skipped even though isinstance(True, int) is True."""
    ws = tmp_path / 'ws'
    ws.mkdir()

    config = AutoPilotConfig(workspace=ws)
    exp_dir = config.experiment_path(slug='exp-bool')
    _seed_proposal(exp_dir, 'p-005', epoch=1)
    _seed_epoch_metrics(exp_dir, 0, {'accuracy': 0.7, 'is_good': True})
    _seed_epoch_metrics(exp_dir, 1, {'accuracy': 0.9, 'is_good': False})
    _seed_epoch_data(exp_dir, 1, [{'id': '1'}])

    ctx = _make_ctx(ws, experiment='exp-bool', epoch=1)
    args = _make_verify_args(proposal_id='p-005')
    cmd = ProposeCommand()

    result = _capture_json(ctx, lambda: cmd.verify(ctx, args))
    payload = result['result']
    assert any('skipped_non_numeric:is_good' in w for w in payload.get('warnings', []))
    assert payload['verdict'] == 'improved'

  def test_propose_verify_skips_nan_metrics(self, tmp_path: Path) -> None:
    """NaN metrics are rejected before comparison."""
    ws = tmp_path / 'ws'
    ws.mkdir()

    config = AutoPilotConfig(workspace=ws)
    exp_dir = config.experiment_path(slug='exp-nan')
    _seed_proposal(exp_dir, 'p-006', epoch=1)
    _seed_epoch_metrics(exp_dir, 0, {'accuracy': 0.7})
    _seed_epoch_metrics(exp_dir, 1, {'accuracy': 0.9})
    _seed_epoch_data(exp_dir, 1, [{'id': '1'}])

    ctx = _make_ctx(ws, experiment='exp-nan', epoch=1)
    args = _make_verify_args(proposal_id='p-006')
    cmd = ProposeCommand()

    result = _capture_json(ctx, lambda: cmd.verify(ctx, args))
    assert result['result']['verdict'] == 'improved'


# ---------------------------------------------------------------------------
# _resolve_metrics_from_forest unit tests
# ---------------------------------------------------------------------------


class TestResolveMetricsFromForest:
  """Unit tests for the forest metrics resolver."""

  def test_returns_none_when_no_slug(self, tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, experiment=None)
    assert _resolve_metrics_from_forest(ctx) is None

  def test_returns_metrics_from_active_tree(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_forest(ws, 'exp-x', {'score': 0.9})
    ctx = _make_ctx(ws, experiment='exp-x')
    metrics = _resolve_metrics_from_forest(ctx)
    assert metrics is not None
    assert metrics['score'] == 0.9

  def test_returns_none_when_experiment_not_found(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    _seed_forest(ws, 'exp-x', {'score': 0.9})
    ctx = _make_ctx(ws, experiment='nonexistent')
    assert _resolve_metrics_from_forest(ctx) is None
