"""Tests for ``policy check`` gate_hints JSON field (Sub-plan 03, section 2.1).

Verifies that the ``gate_hints`` dict is included in the ``policy check``
JSON result payload -- non-empty when metrics are missing, empty when all
metrics are satisfied.
"""

from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from pathlib import Path
from typing import Any
import contextlib
import io
import json


def _policy_check_payload(
  workspace: Path,
  metrics: dict[str, Any],
  *,
  min_thresholds: list[str] | None = None,
  max_thresholds: list[str] | None = None,
) -> dict[str, Any]:
  """Run policy check with inline metrics, capture JSON even on SystemExit."""
  argv = [
    'policy',
    'check',
    '--metrics',
    json.dumps(metrics),
    '--workspace',
    str(workspace),
    '--json',
  ]
  if min_thresholds:
    for t in min_thresholds:
      argv.extend(['--min', t])
  if max_thresholds:
    for t in max_thresholds:
      argv.extend(['--max', t])

  parser = build_parser()
  parsed = parser.parse_args(argv)
  ctx = build_context(parsed)

  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
    parsed.handler(ctx, parsed)

  output = buf.getvalue().strip()
  if output:
    return json.loads(output)
  return {}


class TestPolicyCheckGateHintsOnMissingMetric:
  """policy check includes gate_hints when a gate references an absent metric."""

  def test_policy_check_includes_gate_hints_on_missing_metric(self, tmp_path: Path) -> None:
    """gate_hints is non-empty when the gate metric is not in the metrics dict."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    envelope = _policy_check_payload(
      ws,
      {'accuracy': 0.9},
      min_thresholds=['nonexistent_metric:0.5'],
    )
    result = envelope.get('result', envelope)
    assert 'gate_hints' in result
    assert isinstance(result['gate_hints'], dict)
    assert len(result['gate_hints']) > 0
    assert 'nonexistent_metric' in result['gate_hints']

  def test_policy_check_gate_result_fail_on_missing(self, tmp_path: Path) -> None:
    """policy check reports FAIL when referencing a missing metric."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    envelope = _policy_check_payload(
      ws,
      {'accuracy': 0.9},
      min_thresholds=['nonexistent_metric:0.5'],
    )
    assert envelope.get('ok') is False


class TestPolicyCheckGateHintsEmptyWhenAllPresent:
  """policy check gate_hints is empty dict when all metrics are present."""

  def test_policy_check_gate_hints_empty_when_all_present(self, tmp_path: Path) -> None:
    """gate_hints == {} when all metrics referenced by gates are satisfied."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    envelope = _policy_check_payload(
      ws,
      {'accuracy': 0.95},
      min_thresholds=['accuracy:0.9'],
    )
    result = envelope.get('result', envelope)
    assert 'gate_hints' in result
    assert result['gate_hints'] == {}
