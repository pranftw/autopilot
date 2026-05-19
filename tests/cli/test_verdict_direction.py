"""Tests for direction-aware verdict in experiment compare (Plan 01).

Covers:
  - ``infer_direction`` heuristic for lower-is-better pattern matching.
  - ``compute_verdict`` with directional improvement logic.
  - ``build_compare_deltas`` ``higher_is_better`` population.
  - ``--higher-metric`` / ``--lower-metric`` CLI flag validation and wiring.
  - JSON schema: every delta includes ``higher_is_better``.
  - Text vs JSON verdict parity.
  - Error paths: conflicting overrides, unknown override names.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.experiment.verdict import (
  build_compare_deltas,
  compute_verdict,
)
from autopilot.core.config import AutoPilotConfig
from autopilot.core.metric_utils import (
  LOWER_IS_BETTER_PATTERNS,
  LOWER_IS_BETTER_SEGMENT_PATTERNS,
  infer_direction,
)
from pathlib import Path
from tests.cli.conftest import (
  run_cli_no_context,
  run_cli_text,
  seed_tree_with_experiments,
)
from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def _patch_store_checkout():
  """Patch FileStore.checkout for tests that don't create snapshots."""
  with patch('autopilot.ai.store.file_store.FileStore.checkout'):
    yield


@pytest.fixture
def ws_direction(tmp_path: Path) -> Path:
  """Workspace with experiments having directional metrics."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {
        'id': 'base',
        'hypothesis': 'baseline',
        'status': 'completed',
        'metrics': {'accuracy': 0.80, 'loss': 1.0, 'latency_ms': 200.0},
      },
      {
        'id': 'candidate',
        'hypothesis': 'improved',
        'status': 'completed',
        'metrics': {'accuracy': 0.85, 'loss': 0.5, 'latency_ms': 150.0},
        'parent': 'base',
      },
      {
        'id': 'regressed',
        'hypothesis': 'worse',
        'status': 'completed',
        'metrics': {'accuracy': 0.75, 'loss': 1.5, 'latency_ms': 250.0},
        'parent': 'base',
      },
      {
        'id': 'mixed',
        'hypothesis': 'mixed signals',
        'status': 'completed',
        'metrics': {'accuracy': 0.85, 'loss': 1.5, 'latency_ms': 250.0},
        'parent': 'base',
      },
    ],
  )
  return ws


class TestInferDirection:
  """Unit tests for ``infer_direction`` heuristic."""

  def test_accuracy_higher_is_better(self) -> None:
    assert infer_direction('accuracy') is True

  def test_val_loss_lower_is_better(self) -> None:
    assert infer_direction('val_loss') is False

  def test_error_rate_lower_is_better(self) -> None:
    assert infer_direction('error_rate') is False

  def test_latency_ms_lower_is_better(self) -> None:
    assert infer_direction('latency_ms') is False

  def test_cost_usd_lower_is_better(self) -> None:
    assert infer_direction('cost_usd') is False

  def test_perplexity_lower_is_better(self) -> None:
    assert infer_direction('perplexity') is False

  def test_cer_lower_is_better(self) -> None:
    assert infer_direction('cer') is False

  def test_wer_lower_is_better(self) -> None:
    assert infer_direction('wer') is False

  def test_train_loss_lower_is_better(self) -> None:
    assert infer_direction('train_loss') is False

  def test_f1_score_higher_is_better(self) -> None:
    assert infer_direction('f1_score') is True

  def test_pattern_tuple_completeness(self) -> None:
    """All patterns in both constants produce lower-is-better results."""
    for pat in LOWER_IS_BETTER_PATTERNS:
      assert infer_direction(pat) is False, f'{pat} should be lower-is-better'
    for pat in LOWER_IS_BETTER_SEGMENT_PATTERNS:
      assert infer_direction(pat) is False, f'{pat} should be lower-is-better'

  def test_case_insensitive(self) -> None:
    assert infer_direction('Val_Loss') is False
    assert infer_direction('LATENCY') is False


class TestInferDirectionSegmentMatching:
  """Segment matching for short patterns avoids false positives."""

  def test_answer_is_higher_is_better(self) -> None:
    assert infer_direction('answer') is True

  def test_answer_accuracy_is_higher_is_better(self) -> None:
    assert infer_direction('answer_accuracy') is True

  def test_wer_is_lower_is_better(self) -> None:
    assert infer_direction('wer') is False

  def test_val_wer_is_lower_is_better(self) -> None:
    assert infer_direction('val_wer') is False

  def test_train_wer_is_lower_is_better(self) -> None:
    assert infer_direction('train_wer') is False

  def test_cer_is_lower_is_better(self) -> None:
    assert infer_direction('cer') is False

  def test_val_cer_is_lower_is_better(self) -> None:
    assert infer_direction('val_cer') is False

  def test_character_error_rate_is_lower_via_error(self) -> None:
    assert infer_direction('character_error_rate') is False

  def test_recovery_rate_is_higher_is_better(self) -> None:
    assert infer_direction('recovery_rate') is True

  def test_concern_is_higher_is_better(self) -> None:
    assert infer_direction('concern') is True

  def test_flower_count_is_higher_is_better(self) -> None:
    assert infer_direction('flower_count') is True

  def test_werewolf_metric_is_higher_is_better(self) -> None:
    assert infer_direction('werewolf_score') is True


class TestInferDirectionPatternIntegrity:
  """Pattern tuple structure and coverage tests."""

  def test_pattern_tuples_cover_documented_patterns(self) -> None:
    all_patterns = set(LOWER_IS_BETTER_PATTERNS) | set(LOWER_IS_BETTER_SEGMENT_PATTERNS)
    assert 'loss' in all_patterns
    assert 'cer' in all_patterns
    assert 'wer' in all_patterns
    assert len(LOWER_IS_BETTER_PATTERNS) == 5
    assert len(LOWER_IS_BETTER_SEGMENT_PATTERNS) == 2

  def test_substring_patterns_are_long(self) -> None:
    for pat in LOWER_IS_BETTER_PATTERNS:
      assert len(pat) >= 4, f'{pat} too short for substring matching'

  def test_segment_patterns_are_short(self) -> None:
    for pat in LOWER_IS_BETTER_SEGMENT_PATTERNS:
      assert len(pat) <= 3, f'{pat} too long for segment matching'


class TestComputeVerdict:
  """Unit tests for direction-aware ``compute_verdict``."""

  def test_all_improved(self) -> None:
    deltas = [
      {'metric': 'accuracy', 'delta': 0.05, 'type': 'numeric', 'higher_is_better': True},
      {'metric': 'loss', 'delta': -0.5, 'type': 'numeric', 'higher_is_better': False},
    ]
    assert compute_verdict(deltas) == 'improved'

  def test_all_regressed(self) -> None:
    deltas = [
      {'metric': 'accuracy', 'delta': -0.05, 'type': 'numeric', 'higher_is_better': True},
      {'metric': 'loss', 'delta': 0.5, 'type': 'numeric', 'higher_is_better': False},
    ]
    assert compute_verdict(deltas) == 'regressed'

  def test_mixed_inconclusive(self) -> None:
    deltas = [
      {'metric': 'accuracy', 'delta': 0.05, 'type': 'numeric', 'higher_is_better': True},
      {'metric': 'loss', 'delta': 0.5, 'type': 'numeric', 'higher_is_better': False},
    ]
    assert compute_verdict(deltas) == 'inconclusive'

  def test_zero_delta_skipped(self) -> None:
    deltas = [
      {'metric': 'accuracy', 'delta': 0.0, 'type': 'numeric', 'higher_is_better': True},
      {'metric': 'loss', 'delta': -0.5, 'type': 'numeric', 'higher_is_better': False},
    ]
    assert compute_verdict(deltas) == 'improved'

  def test_no_numeric_inconclusive(self) -> None:
    deltas = [
      {'metric': 'mode', 'delta': None, 'type': 'non_numeric', 'higher_is_better': True},
    ]
    assert compute_verdict(deltas) == 'inconclusive'

  def test_higher_override_wins(self) -> None:
    deltas = [
      {'metric': 'custom', 'delta': 0.1, 'type': 'numeric', 'higher_is_better': False},
    ]
    assert compute_verdict(deltas, higher_overrides=['custom']) == 'improved'

  def test_lower_override_wins(self) -> None:
    deltas = [
      {'metric': 'custom', 'delta': -0.1, 'type': 'numeric', 'higher_is_better': True},
    ]
    assert compute_verdict(deltas, lower_overrides=['custom']) == 'improved'


class TestBuildCompareDeltas:
  """Tests for ``build_compare_deltas`` higher_is_better population."""

  def test_higher_is_better_populated(self) -> None:
    metrics_a = {'accuracy': 0.8, 'loss': 1.0}
    metrics_b = {'accuracy': 0.85, 'loss': 0.5}
    deltas = build_compare_deltas(metrics_a, metrics_b)
    for d in deltas:
      assert 'higher_is_better' in d, f'missing higher_is_better in {d}'
    by_name = {d['metric']: d for d in deltas}
    assert by_name['accuracy']['higher_is_better'] is True
    assert by_name['loss']['higher_is_better'] is False

  def test_direction_override_applied(self) -> None:
    from autopilot.core.comparison import ComparatorMetric

    metrics_a = {'custom_metric': 1.0}
    metrics_b = {'custom_metric': 0.5}
    overrides = [ComparatorMetric('custom_metric', higher_is_better=False)]
    deltas = build_compare_deltas(metrics_a, metrics_b, direction_overrides=overrides)
    assert deltas[0]['higher_is_better'] is False
    assert compute_verdict(deltas) == 'improved'


class TestVerdictLowerIsBetter:
  """P0#1 regression: loss-only metrics with candidate loss drop -> improved."""

  def test_loss_drop_is_improved(self, ws_direction: Path) -> None:
    result = run_cli_no_context(ws_direction, ['experiment', 'compare', 'base', 'candidate'])
    assert result['result']['verdict'] == 'improved'

  def test_loss_increase_is_regressed(self, ws_direction: Path) -> None:
    result = run_cli_no_context(ws_direction, ['experiment', 'compare', 'base', 'regressed'])
    assert result['result']['verdict'] == 'regressed'


class TestVerdictMixedDirections:
  """Accuracy up and loss down -> improved."""

  def test_all_favorable_improved(self, ws_direction: Path) -> None:
    result = run_cli_no_context(ws_direction, ['experiment', 'compare', 'base', 'candidate'])
    deltas = result['result']['deltas']
    by_name = {d['metric']: d for d in deltas}
    assert by_name['accuracy']['delta'] > 0
    assert by_name['loss']['delta'] < 0
    assert by_name['latency_ms']['delta'] < 0
    assert result['result']['verdict'] == 'improved'


class TestVerdictMixedConflict:
  """Accuracy up but loss up and latency up -> inconclusive.

  Expected: accuracy improves (higher-is-better, delta > 0), but loss
  regresses (lower-is-better, delta > 0) and latency regresses
  (lower-is-better, delta > 0). Mixed signals yield 'inconclusive'.
  """

  def test_mixed_signals_inconclusive(self, ws_direction: Path) -> None:
    result = run_cli_no_context(ws_direction, ['experiment', 'compare', 'base', 'mixed'])
    assert result['result']['verdict'] == 'inconclusive'


class TestVerdictHeuristicNames:
  """Metrics named val_loss, error_rate, latency_ms classify as lower-is-better."""

  def test_heuristic_names(self) -> None:
    metrics_a = {'val_loss': 1.0, 'error_rate': 0.3, 'latency_ms': 200.0}
    metrics_b = {'val_loss': 0.5, 'error_rate': 0.1, 'latency_ms': 100.0}
    deltas = build_compare_deltas(metrics_a, metrics_b)
    for d in deltas:
      assert d['higher_is_better'] is False
    assert compute_verdict(deltas) == 'improved'


class TestVerdictExplicitFlags:
  """``--higher-metric accuracy --lower-metric loss`` forces directions."""

  def test_flags_override_heuristic(self, ws_direction: Path) -> None:
    result = run_cli_no_context(
      ws_direction,
      [
        'experiment',
        'compare',
        'base',
        'candidate',
        '--higher-metric',
        'accuracy',
        '--lower-metric',
        'loss',
      ],
    )
    assert result['result']['verdict'] == 'improved'


class TestCompareJsonIncludesDirection:
  """Every delta dict in JSON includes boolean ``higher_is_better``."""

  def test_all_deltas_have_higher_is_better(self, ws_direction: Path) -> None:
    result = run_cli_no_context(ws_direction, ['experiment', 'compare', 'base', 'candidate'])
    deltas = result['result']['deltas']
    assert len(deltas) > 0
    for d in deltas:
      assert 'higher_is_better' in d
      assert isinstance(d['higher_is_better'], bool)


class TestVerdictConflictingOverridesError:
  """Same metric in both --higher-metric and --lower-metric is a hard error."""

  def test_conflicting_overrides_fails(self, ws_direction: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(
        ws_direction,
        [
          'experiment',
          'compare',
          'base',
          'candidate',
          '--higher-metric',
          'accuracy',
          '--lower-metric',
          'accuracy',
        ],
      )


class TestVerdictUnknownMetricOverrideError:
  """``--higher-metric totally_missing_metric`` fails."""

  def test_unknown_metric_fails(self, ws_direction: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(
        ws_direction,
        [
          'experiment',
          'compare',
          'base',
          'candidate',
          '--higher-metric',
          'totally_missing_metric',
        ],
      )


class TestCompareExitCodeOnMissingExperiment:
  """Invalid slug resolves non-zero exit."""

  def test_missing_experiment_exits(self, ws_direction: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(ws_direction, ['experiment', 'compare', 'base', 'ghost'])


class TestCompareTextVsJsonParity:
  """Parsed text summary verdict label matches JSON verdict field."""

  def test_parity(self, ws_direction: Path) -> None:
    json_result = run_cli_no_context(ws_direction, ['experiment', 'compare', 'base', 'candidate'])
    json_verdict = json_result['result']['verdict']

    text = run_cli_text(ws_direction, ['experiment', 'compare', 'base', 'candidate'])
    assert f'Verdict: {json_verdict}' in text


@pytest.fixture
def ws_prefix_mismatch(tmp_path: Path) -> Path:
  """Workspace where baseline uses val_loss and candidate uses loss (prefix mismatch)."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {
        'id': 'base',
        'hypothesis': 'baseline',
        'status': 'completed',
        'metrics': {'val_loss': 1.0, 'val_accuracy': 0.80},
      },
      {
        'id': 'candidate',
        'hypothesis': 'improved',
        'status': 'completed',
        'metrics': {'loss': 0.5, 'accuracy': 0.85},
        'parent': 'base',
      },
    ],
  )
  return ws


class TestOverrideUnderPrefixNormalization:
  """Overrides using prefixed names apply correctly after normalization."""

  def test_val_loss_override_applies_to_normalized_key(self, ws_prefix_mismatch: Path) -> None:
    result = run_cli_no_context(
      ws_prefix_mismatch,
      [
        'experiment',
        'compare',
        'base',
        'candidate',
        '--lower-metric',
        'val_loss',
      ],
    )
    deltas = result['result']['deltas']
    loss_delta = next((d for d in deltas if 'loss' in d['metric']), None)
    assert loss_delta is not None
    assert loss_delta['higher_is_better'] is False
    assert result['result']['verdict'] == 'improved'
