"""Tests for experiment compare --weights weighted verdict feature."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.experiment.compare import (
  WEIGHTED_VERDICT_EPSILON,
  _compute_weighted_verdict,
  _parse_weights,
  _validate_weight_metrics,
)
from autopilot.cli.commands.experiment.verdict import build_compare_deltas
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context, run_cli_text
from unittest.mock import MagicMock
import pytest


@pytest.fixture
def compare_workspace(tmp_path: Path) -> tuple[Path, str, str]:
  """Workspace with two completed experiments for compare tests.

  Returns:
    Tuple of (workspace_path, baseline_id, candidate_id).
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  baseline = Experiment(experiment_id='exp-base', hypothesis='baseline')
  baseline.start()
  baseline.complete(metrics={'accuracy': 0.8, 'latency': 50.0, 'loss': 0.5})
  tree.add(Node(experiment=baseline))

  candidate = Experiment(experiment_id='exp-cand', hypothesis='candidate')
  candidate.start()
  candidate.complete(metrics={'accuracy': 0.9, 'latency': 60.0, 'loss': 0.3})
  tree.add(Node(experiment=candidate))

  forest.save()
  return ws, 'exp-base', 'exp-cand'


@pytest.fixture
def mock_ctx() -> MagicMock:
  """Mock CLIContext that captures ctx.fail calls via SystemExit."""
  ctx = MagicMock()

  def fail_side_effect(msg: str) -> None:
    raise SystemExit(msg)

  ctx.fail = MagicMock(side_effect=fail_side_effect)
  return ctx


# ---------------------------------------------------------------------------
# 4.1 Parsing and validation
# ---------------------------------------------------------------------------


class TestWeightsParsing:
  """Tests for _parse_weights parsing logic."""

  def test_compare_weights_parsing(self, mock_ctx: MagicMock) -> None:
    """'a:7,b:3' yields normalized weights {a: 0.7, b: 0.3}."""
    result = _parse_weights(mock_ctx, 'a:7,b:3')
    assert abs(result['a'] - 0.7) < 1e-9
    assert abs(result['b'] - 0.3) < 1e-9
    assert abs(sum(result.values()) - 1.0) < 1e-9

  def test_compare_weights_invalid_format(self, mock_ctx: MagicMock) -> None:
    """'accuracy' (no colon) fails via ctx.fail / SystemExit."""
    with pytest.raises(SystemExit, match='missing colon'):
      _parse_weights(mock_ctx, 'accuracy')

  def test_compare_weights_unknown_metric(self, mock_ctx: MagicMock) -> None:
    """Weight referencing metric absent from deltas fails."""
    weights = {'nonexistent': 1.0}
    deltas = [
      {
        'metric': 'accuracy',
        'baseline': 0.8,
        'candidate': 0.9,
        'delta': 0.1,
        'type': 'numeric',
        'higher_is_better': True,
      }
    ]
    with pytest.raises(SystemExit, match='not found in compared metrics'):
      _validate_weight_metrics(mock_ctx, weights, deltas)

  def test_weights_negative_value_rejected(self, mock_ctx: MagicMock) -> None:
    """Negative weight raises SystemExit with guidance."""
    with pytest.raises(SystemExit, match='non-negative'):
      _parse_weights(mock_ctx, 'accuracy:-1.0')

  def test_weights_zero_sum_rejected(self, mock_ctx: MagicMock) -> None:
    """All-zero weights raises SystemExit."""
    with pytest.raises(SystemExit, match='total weight is zero'):
      _parse_weights(mock_ctx, 'a:0,b:0')

  def test_weights_nan_rejected(self, mock_ctx: MagicMock) -> None:
    """NaN weight raises SystemExit."""
    with pytest.raises(SystemExit, match='NaN'):
      _parse_weights(mock_ctx, 'accuracy:nan')

  def test_weights_inf_rejected(self, mock_ctx: MagicMock) -> None:
    """Infinite weight raises SystemExit."""
    with pytest.raises(SystemExit, match='infinite'):
      _parse_weights(mock_ctx, 'accuracy:inf')

  def test_weights_duplicate_metric_rejected(self, mock_ctx: MagicMock) -> None:
    """Duplicate metric key raises SystemExit."""
    with pytest.raises(SystemExit, match='duplicate metric'):
      _parse_weights(mock_ctx, 'acc:1.0,acc:2.0')

  def test_weights_empty_flag_value(self, mock_ctx: MagicMock) -> None:
    """Empty string flag value fails with guidance."""
    with pytest.raises(SystemExit, match='metric:weight tokens'):
      _parse_weights(mock_ctx, '')

  def test_weights_trailing_commas(self, mock_ctx: MagicMock) -> None:
    """Trailing comma (empty segment) is rejected."""
    with pytest.raises(SystemExit, match='empty segment'):
      _parse_weights(mock_ctx, 'accuracy:7,')

  def test_weights_spaces_around_tokens(self, mock_ctx: MagicMock) -> None:
    """Spaces around metric names and weights are stripped."""
    result = _parse_weights(mock_ctx, ' accuracy : 7 , latency : 3 ')
    assert 'accuracy' in result
    assert 'latency' in result
    assert abs(result['accuracy'] - 0.7) < 1e-9
    assert abs(result['latency'] - 0.3) < 1e-9

  def test_weights_single_metric(self, mock_ctx: MagicMock) -> None:
    """Single metric normalizes to weight 1.0."""
    result = _parse_weights(mock_ctx, 'accuracy:5')
    assert abs(result['accuracy'] - 1.0) < 1e-9

  def test_weights_zero_and_positive(self, mock_ctx: MagicMock) -> None:
    """Zero weight with a positive sibling is accepted (sum > 0)."""
    result = _parse_weights(mock_ctx, 'a:0,b:10')
    assert result['a'] == 0.0
    assert abs(result['b'] - 1.0) < 1e-9

  def test_weights_negative_inf_rejected(self, mock_ctx: MagicMock) -> None:
    """Negative infinity is both infinite and negative; rejected."""
    with pytest.raises(SystemExit, match='infinite'):
      _parse_weights(mock_ctx, 'accuracy:-inf')

  def test_weights_non_numeric_rejected(self, mock_ctx: MagicMock) -> None:
    """Non-numeric weight string is rejected."""
    with pytest.raises(SystemExit, match='not a valid number'):
      _parse_weights(mock_ctx, 'accuracy:abc')


# ---------------------------------------------------------------------------
# 4.2 Weighted verdict and math
# ---------------------------------------------------------------------------


class TestWeightedVerdict:
  """Tests for weighted score computation and verdict logic."""

  def test_compare_weighted_verdict_improved(self) -> None:
    """Positive weighted aggregate yields 'improved'."""
    deltas = [
      {
        'metric': 'accuracy',
        'baseline': 0.8,
        'candidate': 0.9,
        'delta': 0.1,
        'type': 'numeric',
        'higher_is_better': True,
      },
      {
        'metric': 'latency',
        'baseline': 50.0,
        'candidate': 40.0,
        'delta': -10.0,
        'type': 'numeric',
        'higher_is_better': False,
      },
    ]
    weights = {'accuracy': 0.7, 'latency': 0.3}
    score, verdict = _compute_weighted_verdict(deltas, weights)
    assert verdict == 'improved'
    assert score > 0

  def test_compare_weighted_verdict_regressed(self) -> None:
    """Negative weighted aggregate yields 'regressed'."""
    deltas = [
      {
        'metric': 'accuracy',
        'baseline': 0.9,
        'candidate': 0.7,
        'delta': -0.2,
        'type': 'numeric',
        'higher_is_better': True,
      },
    ]
    weights = {'accuracy': 1.0}
    score, verdict = _compute_weighted_verdict(deltas, weights)
    assert verdict == 'regressed'
    assert score < 0

  def test_compare_weighted_verdict_inconclusive(self) -> None:
    """Deltas within epsilon yield 'inconclusive'."""
    deltas = [
      {
        'metric': 'accuracy',
        'baseline': 0.5,
        'candidate': 0.5,
        'delta': 0.0,
        'type': 'numeric',
        'higher_is_better': True,
      },
    ]
    weights = {'accuracy': 1.0}
    score, verdict = _compute_weighted_verdict(deltas, weights)
    assert verdict == 'inconclusive'
    assert abs(score) <= WEIGHTED_VERDICT_EPSILON

  def test_compare_weights_with_direction_overrides(
    self, compare_workspace: tuple[Path, str, str]
  ) -> None:
    """Direction overrides change the sign contribution in weighted verdict.

    accuracy: baseline=0.8, candidate=0.9 (delta=+0.1)
    loss: baseline=0.5, candidate=0.3 (delta=-0.2)

    With --lower-metric loss (lower is better), loss direction is
    higher_is_better=False, so sign=-1: contribution = w * (-1) * (-0.2) > 0.
    Both metrics improve -> verdict 'improved'.
    """
    ws, base_id, cand_id = compare_workspace
    result = run_cli_no_context(
      ws,
      [
        'experiment',
        'compare',
        base_id,
        cand_id,
        '--lower-metric',
        'loss',
        '--weights',
        'accuracy:7,loss:3',
      ],
    )
    data = result['result']
    assert data['weighted_verdict'] == 'improved'
    assert data['weighted_score_delta'] > 0

  def test_compare_without_weights_unchanged(
    self, compare_workspace: tuple[Path, str, str]
  ) -> None:
    """Without --weights, JSON lacks weighted_verdict and weighted_score_delta."""
    ws, base_id, cand_id = compare_workspace
    result = run_cli_no_context(
      ws,
      ['experiment', 'compare', base_id, cand_id],
    )
    data = result['result']
    assert 'weighted_verdict' not in data
    assert 'weighted_score_delta' not in data
    assert 'verdict' in data
    assert 'deltas' in data

  def test_compare_weighted_score_delta_calculation(self) -> None:
    """Hand-computed weighted_score_delta matches expected float sum.

    accuracy: delta=+0.1, higher_is_better=True, weight=0.7
      contribution = 0.7 * (+1) * 0.1 = 0.07
    latency: delta=+10.0, higher_is_better=False, weight=0.3
      contribution = 0.3 * (-1) * 10.0 = -3.0
    total = 0.07 + (-3.0) = -2.93
    """
    deltas = [
      {
        'metric': 'accuracy',
        'baseline': 0.8,
        'candidate': 0.9,
        'delta': 0.1,
        'type': 'numeric',
        'higher_is_better': True,
      },
      {
        'metric': 'latency',
        'baseline': 50.0,
        'candidate': 60.0,
        'delta': 10.0,
        'type': 'numeric',
        'higher_is_better': False,
      },
    ]
    weights = {'accuracy': 0.7, 'latency': 0.3}
    score, verdict = _compute_weighted_verdict(deltas, weights)
    expected = 0.7 * 1.0 * 0.1 + 0.3 * (-1.0) * 10.0
    assert abs(score - expected) < 1e-12
    assert verdict == 'regressed'

  def test_weights_metric_prefix_normalization(self, tmp_path: Path) -> None:
    """Weight key resolves via prefix normalization matching compare deltas.

    Experiment A has val_accuracy, experiment B has accuracy.
    build_compare_deltas normalizes to 'accuracy'. Weight key 'accuracy'
    matches the normalized delta key.
    """
    metrics_a = {'val_accuracy': 0.8}
    metrics_b = {'accuracy': 0.9}
    deltas = build_compare_deltas(metrics_a, metrics_b)

    delta_keys = {d['metric'] for d in deltas}
    assert 'accuracy' in delta_keys

    weights = {'accuracy': 1.0}
    score, verdict = _compute_weighted_verdict(deltas, weights)
    assert score > 0
    assert verdict == 'improved'


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestWeightsCLIIntegration:
  """End-to-end CLI tests for --weights flag."""

  def test_weighted_json_fields_present(self, compare_workspace: tuple[Path, str, str]) -> None:
    """JSON output includes weighted_verdict and weighted_score_delta."""
    ws, base_id, cand_id = compare_workspace
    result = run_cli_no_context(
      ws,
      [
        'experiment',
        'compare',
        base_id,
        cand_id,
        '--weights',
        'accuracy:7,latency:3',
      ],
    )
    data = result['result']
    assert 'weighted_verdict' in data
    assert 'weighted_score_delta' in data
    assert isinstance(data['weighted_score_delta'], float)
    assert data['weighted_verdict'] in {'improved', 'regressed', 'inconclusive'}

  def test_weighted_text_output_contains_verdict(
    self, compare_workspace: tuple[Path, str, str]
  ) -> None:
    """Text mode includes weighted verdict summary line."""
    ws, base_id, cand_id = compare_workspace
    text = run_cli_text(
      ws,
      ['experiment', 'compare', base_id, cand_id, '--weights', 'accuracy:7,latency:3'],
    )
    assert 'Weighted verdict:' in text
    assert 'score delta:' in text

  def test_cli_unknown_metric_fails(self, compare_workspace: tuple[Path, str, str]) -> None:
    """--weights with unknown metric name fails."""
    ws, base_id, cand_id = compare_workspace
    with pytest.raises(SystemExit):
      run_cli_no_context(
        ws,
        [
          'experiment',
          'compare',
          base_id,
          cand_id,
          '--weights',
          'nonexistent:5',
        ],
      )

  def test_cli_invalid_format_fails(self, compare_workspace: tuple[Path, str, str]) -> None:
    """--weights with invalid format (no colon) fails."""
    ws, base_id, cand_id = compare_workspace
    with pytest.raises(SystemExit):
      run_cli_no_context(
        ws,
        ['experiment', 'compare', base_id, cand_id, '--weights', 'accuracy'],
      )

  def test_cli_weighted_with_lower_metric_override(
    self, compare_workspace: tuple[Path, str, str]
  ) -> None:
    """--weights combined with --lower-metric uses override direction.

    latency: baseline=50, candidate=60, delta=+10.
    With --lower-metric latency: higher_is_better=False, sign=-1.
    contribution = 1.0 * (-1) * 10.0 = -10.0 -> regressed.
    """
    ws, base_id, cand_id = compare_workspace
    result = run_cli_no_context(
      ws,
      [
        'experiment',
        'compare',
        base_id,
        cand_id,
        '--lower-metric',
        'latency',
        '--weights',
        'latency:1',
      ],
    )
    data = result['result']
    assert data['weighted_verdict'] == 'regressed'

  def test_cli_weighted_preserves_existing_verdict(
    self, compare_workspace: tuple[Path, str, str]
  ) -> None:
    """Existing verdict field is unaffected by --weights."""
    ws, base_id, cand_id = compare_workspace
    result = run_cli_no_context(
      ws,
      [
        'experiment',
        'compare',
        base_id,
        cand_id,
        '--weights',
        'accuracy:1',
      ],
    )
    data = result['result']
    assert 'verdict' in data
    assert 'weighted_verdict' in data
    assert data['verdict'] in {'improved', 'regressed', 'inconclusive'}

  def test_non_numeric_weighted_metric_rejected(self, tmp_path: Path) -> None:
    """Non-numeric metric in --weights is rejected with error."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')

    exp_a = Experiment(experiment_id='a', hypothesis='a')
    exp_a.start()
    exp_a.complete(metrics={'accuracy': 0.8, 'model': 'gpt-4'})
    tree.add(Node(experiment=exp_a))

    exp_b = Experiment(experiment_id='b', hypothesis='b')
    exp_b.start()
    exp_b.complete(metrics={'accuracy': 0.9, 'model': 'gpt-5'})
    tree.add(Node(experiment=exp_b))

    forest.save()

    with pytest.raises(SystemExit):
      run_cli_no_context(
        ws,
        ['experiment', 'compare', 'a', 'b', '--weights', 'accuracy:1,model:1'],
      )

  def test_missing_metric_weighted_rejected(self, tmp_path: Path) -> None:
    """Weighted metric missing from one experiment is rejected."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')

    exp_a = Experiment(experiment_id='a', hypothesis='a')
    exp_a.start()
    exp_a.complete(metrics={'accuracy': 0.8, 'f1': 0.7})
    tree.add(Node(experiment=exp_a))

    exp_b = Experiment(experiment_id='b', hypothesis='b')
    exp_b.start()
    exp_b.complete(metrics={'accuracy': 0.9})
    tree.add(Node(experiment=exp_b))

    forest.save()

    with pytest.raises(SystemExit):
      run_cli_no_context(
        ws,
        ['experiment', 'compare', 'a', 'b', '--weights', 'accuracy:1,f1:1'],
      )

  def test_all_zero_delta_weighted_inconclusive(self) -> None:
    """When all weighted deltas are zero, verdict is inconclusive."""
    deltas = [
      {
        'metric': 'accuracy',
        'baseline': 0.9,
        'candidate': 0.9,
        'delta': 0.0,
        'type': 'numeric',
        'higher_is_better': True,
      },
      {
        'metric': 'loss',
        'baseline': 0.1,
        'candidate': 0.1,
        'delta': 0.0,
        'type': 'numeric',
        'higher_is_better': False,
      },
    ]
    weights = {'accuracy': 0.5, 'loss': 0.5}
    score, verdict = _compute_weighted_verdict(deltas, weights)
    assert verdict == 'inconclusive'
    assert score == 0.0

  def test_weighted_epsilon_boundary(self) -> None:
    """Score exactly at epsilon boundary is inconclusive."""
    tiny_delta = WEIGHTED_VERDICT_EPSILON * 0.5
    deltas = [
      {
        'metric': 'accuracy',
        'baseline': 0.5,
        'candidate': 0.5 + tiny_delta,
        'delta': tiny_delta,
        'type': 'numeric',
        'higher_is_better': True,
      },
    ]
    weights = {'accuracy': 1.0}
    _score, verdict = _compute_weighted_verdict(deltas, weights)
    assert verdict == 'inconclusive'

  def test_weighted_just_above_epsilon(self) -> None:
    """Score just above epsilon is improved."""
    delta_val = WEIGHTED_VERDICT_EPSILON * 100
    deltas = [
      {
        'metric': 'accuracy',
        'baseline': 0.5,
        'candidate': 0.5 + delta_val,
        'delta': delta_val,
        'type': 'numeric',
        'higher_is_better': True,
      },
    ]
    weights = {'accuracy': 1.0}
    score, verdict = _compute_weighted_verdict(deltas, weights)
    assert verdict == 'improved'
    assert score > WEIGHTED_VERDICT_EPSILON
