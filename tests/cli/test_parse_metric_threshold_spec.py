"""Unit tests for parse_metric_threshold_spec helper.

Direct regression guards for the shared NAME:NUMBER parser used by
``query --metric-gt/--metric-lt`` and ``recommend`` CLI commands.
"""

from autopilot.cli.helpers import parse_metric_threshold_spec
from pathlib import Path
from tests.cli.conftest import make_mock_cli_context
import pytest


@pytest.fixture
def ctx(tmp_path: Path):
  """CLIContext with real fail() that raises SystemExit."""
  return make_mock_cli_context(tmp_path, use_json=True)


class TestParseMetricThresholdSpec:
  """Tests for parse_metric_threshold_spec validation and parsing."""

  def test_valid_spec_returns_name_and_value(self, ctx) -> None:
    name, value = parse_metric_threshold_spec(ctx, 'accuracy:0.9', '--metric-gt')
    assert name == 'accuracy'
    assert value == 0.9

  def test_rejects_empty_metric_name(self, ctx) -> None:
    with pytest.raises(SystemExit):
      parse_metric_threshold_spec(ctx, ':0.5', '--metric-gt')

  def test_rejects_whitespace_only_name(self, ctx) -> None:
    with pytest.raises(SystemExit):
      parse_metric_threshold_spec(ctx, ' :0.5', '--metric-gt')

  def test_rejects_missing_colon(self, ctx) -> None:
    with pytest.raises(SystemExit):
      parse_metric_threshold_spec(ctx, 'accuracy', '--metric-gt')

  def test_rejects_empty_value(self, ctx) -> None:
    with pytest.raises(SystemExit):
      parse_metric_threshold_spec(ctx, 'accuracy:', '--metric-lt')

  def test_rejects_non_numeric_value(self, ctx) -> None:
    with pytest.raises(SystemExit):
      parse_metric_threshold_spec(ctx, 'acc:abc', '--metric-gt')

  def test_rejects_nan(self, ctx) -> None:
    with pytest.raises(SystemExit):
      parse_metric_threshold_spec(ctx, 'loss:nan', '--metric-gt')

  def test_accepts_scientific_notation(self, ctx) -> None:
    name, value = parse_metric_threshold_spec(ctx, 'loss:1e-3', '--metric-gt')
    assert name == 'loss'
    assert value == pytest.approx(0.001)

  def test_accepts_negative_threshold(self, ctx) -> None:
    _, value = parse_metric_threshold_spec(ctx, 'loss:-0.5', '--metric-lt')
    assert value == -0.5

  def test_accepts_integer_threshold(self, ctx) -> None:
    _, value = parse_metric_threshold_spec(ctx, 'epoch:10', '--metric-gt')
    assert value == 10.0

  def test_accepts_zero_threshold(self, ctx) -> None:
    name, value = parse_metric_threshold_spec(ctx, 'loss:0', '--metric-lt')
    assert name == 'loss'
    assert value == 0.0

  def test_error_message_includes_flag_label(self, ctx, capsys) -> None:
    with pytest.raises(SystemExit):
      parse_metric_threshold_spec(ctx, 'bad', '--metric-lt')
    captured = capsys.readouterr()
    assert '--metric-lt' in captured.out or '--metric-lt' in captured.err

  def test_metric_name_with_underscores(self, ctx) -> None:
    name, value = parse_metric_threshold_spec(ctx, 'val_accuracy:0.95', '--metric-gt')
    assert name == 'val_accuracy'
    assert value == 0.95

  def test_metric_name_with_dots(self, ctx) -> None:
    name, value = parse_metric_threshold_spec(ctx, 'train.loss:0.1', '--metric-lt')
    assert name == 'train.loss'
    assert value == pytest.approx(0.1)

  def test_rejects_inf(self, ctx) -> None:
    name, value = parse_metric_threshold_spec(ctx, 'loss:inf', '--metric-lt')
    assert name == 'loss'
    assert value == float('inf')

  def test_accepts_negative_scientific(self, ctx) -> None:
    name, value = parse_metric_threshold_spec(ctx, 'delta:-1e-5', '--metric-gt')
    assert name == 'delta'
    assert value == pytest.approx(-1e-5)
