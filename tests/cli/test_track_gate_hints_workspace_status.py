"""Tests for dogfood-v3 CLI handler cleanups (plans 01, 03).

Covers:
  - ISSUE-001: ``build_compare_deltas`` returns list, not tuple.
  - ISSUE-009: ``_detect_fingerprint_drift`` explicit None vs empty handling.
  - ISSUE-010: ``Forest.find_experiment`` returns None on missing (plan 02 migration).
  - ISSUE-002: ``_build_status_payload`` surfaces forest_error in payload.
  - ISSUE-003: ``gate_hints()`` typed method on policy classes.
  - ISSUE-008: ``track`` dry-run guard skips subprocess execution.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.experiment.compare import _detect_fingerprint_drift
from autopilot.cli.commands.experiment.verdict import build_compare_deltas
from autopilot.cli.commands.policy import _collect_gate_hints
from autopilot.cli.commands.track import TrackCommand
from autopilot.cli.commands.workspace import _build_status_payload, _workspace_status_overall_ok
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.models import Result
from autopilot.core.node import Node
from autopilot.policy.gates import MinGate, collect_gate_hints
from autopilot.policy.policy import Policy
from autopilot.policy.quality_first import QualityFirstPolicy
from autopilot.policy.threshold import ThresholdPolicy
from pathlib import Path
from tests.cli.conftest import make_ctx
from tests.doubles import make_completed_experiment
from unittest.mock import patch
import argparse
import pytest

# ---------------------------------------------------------------------------
# 4.1 ISSUE-001: build_compare_deltas returns list
# ---------------------------------------------------------------------------


class TestBuildCompareReturnsListIssue001:
  """``build_compare_deltas`` must return a plain list, not a tuple."""

  def test_build_compare_deltas_returns_list(self) -> None:
    metrics_a = {'accuracy': 0.8, 'loss': 1.0}
    metrics_b = {'accuracy': 0.9, 'loss': 0.5}
    result = build_compare_deltas(metrics_a, metrics_b)
    assert isinstance(result, list)
    assert not isinstance(result, tuple)
    for entry in result:
      assert 'metric' in entry
      assert 'delta' in entry
      assert 'type' in entry
      assert 'higher_is_better' in entry


# ---------------------------------------------------------------------------
# 4.2 ISSUE-009: _detect_fingerprint_drift explicit truthiness
# ---------------------------------------------------------------------------


class TestDetectFingerprintDriftIssue009:
  """Explicit None vs empty-dict handling in ``_detect_fingerprint_drift``."""

  def test_detect_fingerprint_drift_none_dataset_meta(self) -> None:
    exp_a = make_completed_experiment('a', 'a hypothesis', {'accuracy': 0.9})
    exp_b = make_completed_experiment('b', 'b hypothesis', {'accuracy': 0.9})
    exp_a.dataset_meta = None  # type: ignore[ty:invalid-assignment]  # intentional: test defensive guard
    exp_b.dataset_meta = {
      'dataset_fingerprint': {
        'paths': ['/x'],
        'hashes': ['h'],
        'bundle_hash': 'b' * 64,
        'timestamp': 't',
      },
    }
    result = _detect_fingerprint_drift(exp_a, exp_b)
    assert result is None

  def test_detect_fingerprint_drift_empty_dataset_meta(self) -> None:
    exp_a = make_completed_experiment('a', 'a hypothesis', {'accuracy': 0.9})
    exp_b = make_completed_experiment('b', 'b hypothesis', {'accuracy': 0.9})
    assert exp_a.dataset_meta == {}
    result = _detect_fingerprint_drift(exp_a, exp_b)
    assert result is None


# ---------------------------------------------------------------------------
# 4.3 ISSUE-010: Forest.find_experiment returns None on missing (plan 02)
# ---------------------------------------------------------------------------


class TestFindExperimentCrossTreeIssue010:
  """``Forest.find_experiment`` returns None for missing experiments."""

  def test_forest_find_experiment_missing_returns_none(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')

    exp = Experiment(experiment_id='exists', hypothesis='h')
    exp.start()
    exp.complete(metrics={})
    tree.add(Node(experiment=exp))
    forest.save()

    assert forest.find_experiment('nonexistent-id') is None
    result = forest.find_experiment('exists')
    assert result is not None
    assert result[0].experiment.id == 'exists'


# ---------------------------------------------------------------------------
# 4.4 ISSUE-002: _build_status_payload surfaces forest_error
# ---------------------------------------------------------------------------


class TestWorkspaceStatusForestErrorIssue002:
  """Workspace status surfaces forest load failures in payload."""

  def test_workspace_status_surfaces_forest_error(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    autopilot_dir = ws / '.autopilot'
    autopilot_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ('experiments', 'records', 'datasets', 'projects'):
      (autopilot_dir / subdir).mkdir(exist_ok=True)

    forest_file = config.forest_file
    forest_file.parent.mkdir(parents=True, exist_ok=True)
    forest_file.write_text('NOT VALID JSON')

    ctx = make_ctx(ws, use_json=True)
    payload = _build_status_payload(ctx)
    assert payload.get('forest_error') is not None
    assert isinstance(payload['forest_error'], str)
    assert len(payload['forest_error']) > 0
    assert _workspace_status_overall_ok(payload) is False


# ---------------------------------------------------------------------------
# 4.5 ISSUE-003: gate_hints typed method
# ---------------------------------------------------------------------------


class TestGateHintsIssue003:
  """Typed ``gate_hints()`` method on policy classes."""

  def test_quality_first_gate_hints_after_missing_metric(self) -> None:
    policy = QualityFirstPolicy(gates=[MinGate('nonexistent', 0.5)])
    policy.forward(Result(metrics={'accuracy': 0.9}))
    hints = policy.gate_hints()
    assert isinstance(hints, dict)
    assert 'nonexistent' in hints
    assert isinstance(hints['nonexistent'], str)

  def test_quality_first_gate_hints_all_present(self) -> None:
    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.5)])
    policy.forward(Result(metrics={'accuracy': 0.9}))
    hints = policy.gate_hints()
    assert hints == {}

  def test_threshold_policy_gate_hints_after_missing_metric(self) -> None:
    policy = ThresholdPolicy([MinGate('nonexistent', 0.5)])
    policy.forward(Result(metrics={'accuracy': 0.9}))
    hints = policy.gate_hints()
    assert isinstance(hints, dict)
    assert 'nonexistent' in hints
    assert isinstance(hints['nonexistent'], str)

  def test_collect_gate_hints_policy_without_method(self) -> None:
    policy = Policy()
    result = _collect_gate_hints(policy)
    assert result == {}

  def test_collect_gate_hints_helper_standalone(self) -> None:
    gate = MinGate('missing_metric', 0.5)
    gate.forward(Result(metrics={'other': 0.9}))
    hints = collect_gate_hints([gate])
    assert 'missing_metric' in hints

  def test_collect_gate_hints_helper_no_hints(self) -> None:
    gate = MinGate('accuracy', 0.5)
    gate.forward(Result(metrics={'accuracy': 0.9}))
    hints = collect_gate_hints([gate])
    assert hints == {}


# ---------------------------------------------------------------------------
# 4.6 ISSUE-008: track dry-run guard
# ---------------------------------------------------------------------------


def _make_track_args(user_argv: list[str]) -> argparse.Namespace:
  """Build a minimal Namespace matching TrackCommand.forward expectations."""
  return argparse.Namespace(user_argv=user_argv)


class TestTrackDryRunIssue008:
  """Track command respects ``ctx.dry_run`` and skips subprocess execution."""

  def test_track_dry_run_json_no_execution(self, tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, use_json=True)
    ctx.dry_run = True
    args = _make_track_args(['--', 'echo', 'hello'])
    cmd = TrackCommand()

    captured_payload: dict | None = None

    original_result = ctx.output.result

    def capture_result(payload: dict, ok: bool = True) -> None:
      nonlocal captured_payload
      captured_payload = payload
      original_result(payload, ok=ok)

    with (
      patch('autopilot.cli.commands.track.subprocess.run') as mock_run,
      patch.object(ctx.output, 'result', side_effect=capture_result),
    ):
      cmd.forward(ctx, args)

    mock_run.assert_not_called()
    assert captured_payload is not None
    assert captured_payload['argv'] == ['echo', 'hello']
    assert captured_payload['dry_run'] is True

  def test_track_dry_run_text_info(self, tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, use_json=False)
    ctx.dry_run = True
    args = _make_track_args(['--', 'echo', 'hello'])
    cmd = TrackCommand()

    info_messages: list[str] = []
    original_info = ctx.output.info

    def capture_info(message: str) -> None:
      info_messages.append(message)
      original_info(message)

    with (
      patch('autopilot.cli.commands.track.subprocess.run') as mock_run,
      patch.object(ctx.output, 'info', side_effect=capture_info),
    ):
      cmd.forward(ctx, args)

    mock_run.assert_not_called()
    assert any('would execute' in msg for msg in info_messages)

  def test_track_normal_still_runs(self, tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, use_json=False)
    ctx.dry_run = False
    args = _make_track_args(['--', 'echo', 'hello'])
    cmd = TrackCommand()

    with patch('autopilot.cli.commands.track.subprocess.run') as mock_run:
      mock_run.return_value.returncode = 0
      with pytest.raises(SystemExit) as exc_info:
        cmd.forward(ctx, args)

    mock_run.assert_called_once_with(['echo', 'hello'], shell=False, check=False)
    assert exc_info.value.code == 0
