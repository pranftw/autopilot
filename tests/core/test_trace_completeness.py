"""Tests for trace completeness audit in autopilot.core.trace.

Covers TraceDimension/TraceReport round-trips, verify_trace_completeness
across all four dimensions (policy_gate, gradient_journal, store_context,
cost_attribution), vacuous passes, partial coverage, and edge cases.
"""

from autopilot.core.callbacks.cost import COST_ATTRIBUTION_TYPE
from autopilot.core.context import ContextEntry, ContextLog
from autopilot.core.decision import DecisionEntry
from autopilot.core.trace import (
  TraceDimension,
  TraceReport,
  verify_trace_completeness,
)
from typing import Any
import pytest

# ---------------------------------------------------------------------------
# test helpers (local to this file)
# ---------------------------------------------------------------------------


def _policy_entry(epoch: int) -> ContextEntry:
  """Build a policy gate context entry for a given epoch."""
  return ContextEntry.create(
    f'epoch {epoch} accepted by policy gate',
    source='policy',
    epoch=epoch,
    metadata={'_type': DecisionEntry.POLICY_GATE_TYPE, 'gates': []},
  )


def _gradient_entry(epoch: int, *, source: str = 'trainer') -> ContextEntry:
  """Build a gradient journal context entry."""
  return ContextEntry.create(
    'gradient feedback recorded',
    source=source,
    epoch=epoch,
    metadata={
      'gradient_summaries': [
        {
          'param_name': 'p',
          'param_type': 'ScalarParameter',
          'gradient_type': 'TextGradient',
          'summary': 'improve X',
        },
      ],
    },
  )


def _cost_entry(epoch: int) -> ContextEntry:
  """Build a cost attribution context entry for a given epoch."""
  return ContextEntry.create(
    'cost recorded',
    source='cost',
    epoch=epoch,
    metadata={
      '_type': COST_ATTRIBUTION_TYPE,
      'epoch': epoch,
      'cost_usd': 0.01,
      'cumulative': 0.01 * (epoch + 1),
    },
  )


def _mutating_reflog(operation: str, *, context: str | None) -> dict[str, Any]:
  """Build a synthetic mutating reflog row."""
  return {
    'timestamp': '2025-06-01T12:00:00+00:00',
    'operation': operation,
    'experiment_id': 'exp-001',
    'context': context,
  }


def _build_log(entries: list[ContextEntry]) -> ContextLog:
  """Build a ContextLog from a list of entries."""
  log = ContextLog()
  for entry in entries:
    log.record(entry)
  return log


# ---------------------------------------------------------------------------
# 4.1: round-trips
# ---------------------------------------------------------------------------


class TestTraceDimensionRoundTrip:
  """TraceDimension serialization round-trip."""

  def test_trace_dimension_round_trip(self) -> None:
    """to_dict -> from_dict reproduces the original including nonempty details."""
    dim = TraceDimension(
      name='policy_gate',
      passed=True,
      details=['policy_gate: 3/3 epochs (100%)'],
    )
    data = dim.to_dict()
    restored = TraceDimension.from_dict(data)
    assert restored.name == dim.name
    assert restored.passed == dim.passed
    assert restored.details == dim.details


class TestTraceReportRoundTrip:
  """TraceReport serialization round-trip."""

  def test_trace_report_round_trip(self) -> None:
    """Full TraceReport with multiple dimensions and gaps round-trips."""
    report = TraceReport(
      complete=False,
      dimensions=[
        TraceDimension(name='policy_gate', passed=True, details=['policy_gate: 2/2 epochs (100%)']),
        TraceDimension(
          name='gradient_journal',
          passed=False,
          details=['gradient_journal: 0/1 expected (0%)'],
        ),
      ],
      gaps=['missing gradient_journal context entry'],
    )
    data = report.to_dict()
    restored = TraceReport.from_dict(data)
    assert restored.complete == report.complete
    assert len(restored.dimensions) == len(report.dimensions)
    assert restored.gaps == report.gaps


# ---------------------------------------------------------------------------
# 4.1: negative epochs
# ---------------------------------------------------------------------------


class TestVerifyNegativeEpochsRunRaises:
  """epochs_run=-1 raises ValueError."""

  def test_verify_negative_epochs_run_raises(self) -> None:
    """Negative epochs_run raises ValueError mentioning negative."""
    with pytest.raises(ValueError, match='negative'):
      verify_trace_completeness(ContextLog(), [], epochs_run=-1)


# ---------------------------------------------------------------------------
# 4.2: complete and missing coverage
# ---------------------------------------------------------------------------


class TestCleanRunReportsComplete:
  """A fully covered trace reports complete=True."""

  def test_clean_run_reports_complete(self) -> None:
    """Policy entries for all epochs, gradient journal, reflog with context -> complete."""
    log = _build_log(
      [
        _policy_entry(0),
        _policy_entry(1),
        _policy_entry(2),
        _gradient_entry(2),
      ]
    )
    reflog = [
      _mutating_reflog('snapshot', context='epoch 0 checkpoint'),
      _mutating_reflog('snapshot', context='epoch 1 checkpoint'),
      _mutating_reflog('checkout', context='resume from epoch 0'),
    ]
    report = verify_trace_completeness(log, reflog, epochs_run=3, check_cost=False)
    assert report.complete is True
    assert all(d.passed for d in report.dimensions)
    assert report.gaps == []


class TestMissingPolicyGateEntryReportsGap:
  """Missing a single epoch's policy entry shows as a gap."""

  def test_missing_policy_gate_entry_reports_gap(self) -> None:
    """Omitting epoch 1 policy entry -> gap for epoch 1."""
    log = _build_log(
      [
        _policy_entry(0),
        _policy_entry(2),
        _gradient_entry(2),
      ]
    )
    report = verify_trace_completeness(log, [], epochs_run=3)
    assert report.complete is False
    policy_dim = report.dimensions[0]
    assert policy_dim.name == 'policy_gate'
    assert policy_dim.passed is False
    assert any('epoch 1: missing policy_gate context entry' in g for g in report.gaps)


class TestMissingGradientJournalReportsGap:
  """Missing gradient journal when epochs_run > 0 is a gap."""

  def test_missing_gradient_journal_reports_gap(self) -> None:
    """Policy entries present but no gradient journal -> gap."""
    log = _build_log(
      [
        _policy_entry(0),
        _policy_entry(1),
        _policy_entry(2),
      ]
    )
    report = verify_trace_completeness(log, [], epochs_run=3)
    assert report.complete is False
    grad_dim = report.dimensions[1]
    assert grad_dim.name == 'gradient_journal'
    assert grad_dim.passed is False
    assert 'missing gradient_journal context entry' in report.gaps


class TestNullReflogContextReportsGap:
  """Reflog with None context on mutating op fails store_context."""

  def test_null_reflog_context_reports_gap(self) -> None:
    """Snapshot reflog with context=None -> store_context fails."""
    log = _build_log(
      [
        _policy_entry(0),
        _gradient_entry(0),
      ]
    )
    reflog = [_mutating_reflog('snapshot', context=None)]
    report = verify_trace_completeness(log, reflog, epochs_run=1)
    assert report.complete is False
    store_dim = report.dimensions[2]
    assert store_dim.name == 'store_context'
    assert store_dim.passed is False
    assert any('missing context' in g for g in report.gaps)


class TestPartialEpochsReportsCorrectCoverage:
  """Partial epoch coverage reports correct percentage."""

  def test_partial_epochs_reports_correct_coverage(self) -> None:
    """Only epochs 0,1 covered for policy (epochs_run=3) -> 66% coverage."""
    log = _build_log(
      [
        _policy_entry(0),
        _policy_entry(1),
        _gradient_entry(1),
      ]
    )
    report = verify_trace_completeness(log, [], epochs_run=3)
    policy_dim = report.dimensions[0]
    assert '2/3 epochs (66%)' in policy_dim.details[0]
    assert any('epoch 2' in g for g in report.gaps)


class TestEmptyContextLogReportsAllGaps:
  """Empty log with epochs_run=2 reports gaps for all dimensions."""

  def test_empty_context_log_reports_all_gaps(self) -> None:
    """Empty log, empty reflog, epochs_run=2 -> multiple gaps."""
    report = verify_trace_completeness(ContextLog(), [], epochs_run=2)
    assert report.complete is False
    assert any('epoch 0: missing policy_gate' in g for g in report.gaps)
    assert any('epoch 1: missing policy_gate' in g for g in report.gaps)
    assert 'missing gradient_journal context entry' in report.gaps


class TestZeroEpochsRunVacuousPass:
  """epochs_run=0 with empty inputs is a vacuous pass."""

  def test_zero_epochs_run_vacuous_pass(self) -> None:
    """Zero epochs means nothing to check — all dimensions pass."""
    report = verify_trace_completeness(ContextLog(), [], epochs_run=0)
    assert report.complete is True
    assert all(d.passed for d in report.dimensions)
    assert report.gaps == []


# ---------------------------------------------------------------------------
# 4.3: cost and edge cases
# ---------------------------------------------------------------------------


class TestCheckCostFalseOmitsCostDimension:
  """check_cost=False omits cost_attribution dimension."""

  def test_check_cost_false_omits_cost_dimension(self) -> None:
    """Clean 1-epoch log without cost entries, check_cost=False -> exactly 3 dimensions."""
    log = _build_log(
      [
        _policy_entry(0),
        _gradient_entry(0),
      ]
    )
    report = verify_trace_completeness(log, [], epochs_run=1, check_cost=False)
    dim_names = [d.name for d in report.dimensions]
    assert len(dim_names) == 3
    assert 'cost_attribution' not in dim_names


class TestCheckCostTrueRequiresPerEpochEntries:
  """check_cost=True with missing epoch reports a cost gap."""

  def test_check_cost_true_requires_per_epoch_entries(self) -> None:
    """epochs_run=2, cost entry only for epoch 0, check_cost=True -> cost dimension fails."""
    log = _build_log(
      [
        _policy_entry(0),
        _policy_entry(1),
        _gradient_entry(1),
        _cost_entry(0),
      ]
    )
    report = verify_trace_completeness(log, [], epochs_run=2, check_cost=True)
    cost_dim = next(d for d in report.dimensions if d.name == 'cost_attribution')
    assert cost_dim.passed is False
    assert any('epoch 1: missing cost_attribution' in g for g in report.gaps)


class TestWhitespaceOnlyReflogContextCountsAsMissing:
  """Whitespace-only context on mutating reflog counts as missing."""

  def test_whitespace_only_reflog_context_counts_as_missing(self) -> None:
    """context='   ' on mutating row -> found 0/1."""
    log = _build_log([_policy_entry(0), _gradient_entry(0)])
    reflog = [_mutating_reflog('snapshot', context='   ')]
    report = verify_trace_completeness(log, reflog, epochs_run=1)
    store_dim = report.dimensions[2]
    assert store_dim.passed is False
    assert '0/1' in store_dim.details[0]


class TestNonMutatingReflogIgnoredForStoreContext:
  """Non-mutating reflog operations are not counted for store_context."""

  def test_non_mutating_reflog_ignored_for_store_context(self) -> None:
    """Reflog row with operation='recover' and context=None does not affect expected count."""
    log = _build_log([_policy_entry(0), _gradient_entry(0)])
    reflog = [
      {
        'timestamp': '2025-06-01T12:00:00+00:00',
        'operation': 'recover',
        'experiment_id': 'exp-001',
        'context': None,
      },
    ]
    report = verify_trace_completeness(log, reflog, epochs_run=1)
    store_dim = report.dimensions[2]
    assert store_dim.passed is True
    assert '0/0' in store_dim.details[0]


class TestGradientEntryRequiresNonemptySummaries:
  """Entry with gradient_summaries=[] does not count toward coverage."""

  def test_gradient_entry_requires_nonempty_summaries(self) -> None:
    """Empty summaries list should not qualify as a gradient journal entry."""
    entry = ContextEntry.create(
      'gradient feedback recorded',
      source='trainer',
      epoch=0,
      metadata={'gradient_summaries': []},
    )
    log = _build_log([_policy_entry(0), entry])
    report = verify_trace_completeness(log, [], epochs_run=1)
    grad_dim = report.dimensions[1]
    assert grad_dim.passed is False
    assert 'missing gradient_journal context entry' in report.gaps
