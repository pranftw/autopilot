"""Trace completeness audit for experiment context and reflog coverage.

This module **audits** whether an experiment's trace artifacts cover the
expected dimensions — it does not chronologically merge streams (that is
``build_timeline()`` in ``autopilot.core.timeline``).

Four audit dimensions:
  - **policy_gate**: per-epoch policy gate context entries.
  - **gradient_journal**: at least one gradient summary entry when epochs > 0.
  - **store_context**: every mutating reflog operation has a non-empty context.
  - **cost_attribution** (opt-in): per-epoch cost attribution context entries.

Classes:
  TraceDimension -- single dimension pass/fail with coverage details.
  TraceReport -- aggregate audit result across all checked dimensions.

Functions:
  verify_trace_completeness -- run all dimension checks and return a report.
"""

from autopilot.core.callbacks.cost import COST_ATTRIBUTION_TYPE
from autopilot.core.context import ContextEntry, ContextLog
from autopilot.core.decision import DecisionEntry
from autopilot.core.serialization import DictMixin
from dataclasses import dataclass, field
from typing import Any

GRADIENT_JOURNAL_SOURCES = frozenset({'trainer', 'agent-optimizer'})

MUTATING_REFLOG_OPERATIONS = frozenset(
  {
    'snapshot',
    'checkout',
    'branch',
    'reset_branch',
    'merge_apply',
    'materialize',
    'copy_epoch',
    'tag',
    'stash',
    'stash_pop',
  }
)


@dataclass
class TraceDimension(DictMixin):
  """One completeness dimension result.

  Attributes:
    name: Dimension identifier (``policy_gate``, ``gradient_journal``,
      ``store_context``, ``cost_attribution``).
    passed: True when this dimension meets its coverage requirement.
    details: Human-readable coverage lines (includes percentage when applicable).
  """

  name: str
  passed: bool
  details: list[str] = field(default_factory=list)


@dataclass
class TraceReport(DictMixin):
  """Aggregate trace completeness audit result.

  Attributes:
    complete: True only when every dimension passed and ``gaps`` is empty.
    dimensions: Per-dimension results in stable order (policy, gradient, store,
      cost when checked).
    gaps: Actionable gap messages suitable for agent consumption.
  """

  complete: bool
  dimensions: list[TraceDimension] = field(default_factory=list)
  gaps: list[str] = field(default_factory=list)


def is_policy_gate_entry(entry: ContextEntry) -> bool:
  """True when source is policy and metadata is a typed policy_gate row.

  Args:
    entry: Context entry to test.

  Returns:
    Whether the entry is a policy gate decision record.
  """
  return entry.source == 'policy' and entry.metadata.get('_type') == DecisionEntry.POLICY_GATE_TYPE


def is_gradient_journal_entry(entry: ContextEntry) -> bool:
  """True when entry carries non-empty structured gradient_summaries.

  Args:
    entry: Context entry to test.

  Returns:
    Whether the entry contains a non-empty gradient summaries list.
  """
  if entry.source not in GRADIENT_JOURNAL_SOURCES:
    return False
  summaries = entry.metadata.get('gradient_summaries')
  return isinstance(summaries, list) and len(summaries) > 0


def is_cost_attribution_entry(entry: ContextEntry) -> bool:
  """True when entry is a typed cost attribution row.

  Args:
    entry: Context entry to test.

  Returns:
    Whether the entry is a cost attribution record.
  """
  return entry.source == 'cost' and entry.metadata.get('_type') == COST_ATTRIBUTION_TYPE


def _format_coverage(label: str, found: int, expected: int) -> str:
  """Return e.g. ``'policy_gate: 2/3 epochs (67%)'`` (integer percent, round down).

  Args:
    label: Dimension name for the prefix.
    found: Number of covered items.
    expected: Number of expected items.

  Returns:
    Formatted coverage string with integer percentage.
  """
  pct = 100 if expected == 0 else (found * 100) // expected
  return f'{label}: {found}/{expected} epochs ({pct}%)'


def _missing_epoch_gaps(
  prefix: str,
  expected_epochs: range,
  covered_epochs: set[int],
) -> list[str]:
  """Build gap strings for epochs in expected but not covered.

  Args:
    prefix: Gap message suffix (e.g. ``'policy_gate context entry'``).
    expected_epochs: Range of expected epoch indices.
    covered_epochs: Set of epoch indices that are covered.

  Returns:
    Sorted list of gap messages for missing epochs.
  """
  return [f'epoch {e}: missing {prefix}' for e in sorted(set(expected_epochs) - covered_epochs)]


def _check_policy_gate(
  context_log: ContextLog,
  epochs_run: int,
) -> tuple[TraceDimension, list[str]]:
  """Check policy gate coverage across epochs.

  Args:
    context_log: Experiment context log to scan.
    epochs_run: Number of training epochs executed.

  Returns:
    Tuple of (dimension result, gap messages).
  """
  expected = range(epochs_run)
  covered = {
    entry.epoch for entry in context_log if is_policy_gate_entry(entry) and entry.epoch is not None
  }
  covered_in_range = covered & set(expected)
  passed = covered_in_range == set(expected)
  details = [_format_coverage('policy_gate', len(covered_in_range), epochs_run)]
  gaps = _missing_epoch_gaps('policy_gate context entry', expected, covered)
  return TraceDimension(name='policy_gate', passed=passed, details=details), gaps


def _check_gradient_journal(
  context_log: ContextLog,
  epochs_run: int,
) -> tuple[TraceDimension, list[str]]:
  """Check gradient journal presence.

  Trainer emits gradient journal once at completion, not per-epoch.
  Expected: at least one qualifying entry when ``epochs_run > 0``.

  Args:
    context_log: Experiment context log to scan.
    epochs_run: Number of training epochs executed.

  Returns:
    Tuple of (dimension result, gap messages).
  """
  found = sum(1 for entry in context_log if is_gradient_journal_entry(entry))
  if epochs_run == 0:
    passed = True
    expected_count = 0
  else:
    passed = found > 0
    expected_count = 1
  details = [_format_coverage('gradient_journal', min(found, expected_count), expected_count)]
  gaps: list[str] = []
  if epochs_run > 0 and found == 0:
    gaps.append('missing gradient_journal context entry')
  return TraceDimension(name='gradient_journal', passed=passed, details=details), gaps


def _check_store_context(
  reflog_entries: list[dict[str, Any]],
) -> tuple[TraceDimension, list[str]]:
  """Check that all mutating reflog operations have non-empty context.

  Args:
    reflog_entries: Pre-filtered reflog rows for the experiment branch.

  Returns:
    Tuple of (dimension result, gap messages).
  """
  mutating = [
    entry for entry in reflog_entries if entry.get('operation') in MUTATING_REFLOG_OPERATIONS
  ]
  expected = len(mutating)
  found = 0
  gaps: list[str] = []
  for entry in mutating:
    ctx = entry.get('context')
    if isinstance(ctx, str) and ctx.strip():
      found += 1
    else:
      op = entry.get('operation', 'unknown')
      ts = entry.get('timestamp', 'unknown')
      exp_id = entry.get('experiment_id')
      gap = f'reflog {op} at {ts}: missing context'
      if exp_id:
        gap = f'reflog {op} at {ts} (experiment {exp_id}): missing context'
      gaps.append(gap)

  pct = 100 if expected == 0 else (found * 100) // expected
  details = [f'store_context: {found}/{expected} mutating operations ({pct}%)']
  passed = found == expected
  return TraceDimension(name='store_context', passed=passed, details=details), gaps


def _check_cost_attribution(
  context_log: ContextLog,
  epochs_run: int,
) -> tuple[TraceDimension, list[str]]:
  """Check per-epoch cost attribution coverage.

  Args:
    context_log: Experiment context log to scan.
    epochs_run: Number of training epochs executed.

  Returns:
    Tuple of (dimension result, gap messages).
  """
  expected = range(epochs_run)
  covered = {
    entry.epoch
    for entry in context_log
    if is_cost_attribution_entry(entry) and entry.epoch is not None
  }
  covered_in_range = covered & set(expected)
  passed = covered_in_range == set(expected)
  details = [_format_coverage('cost_attribution', len(covered_in_range), epochs_run)]
  gaps = _missing_epoch_gaps('cost_attribution context entry', expected, covered)
  return TraceDimension(name='cost_attribution', passed=passed, details=details), gaps


def verify_trace_completeness(
  context_log: ContextLog,
  reflog_entries: list[dict[str, Any]],
  epochs_run: int,
  *,
  check_cost: bool = False,
) -> TraceReport:
  """Audit whether experiment trace artifacts cover the expected dimensions.

  Args:
    context_log: Experiment decision journal (append order preserved).
    reflog_entries: Store reflog rows **pre-filtered** to the target
      experiment branch (same contract as ``build_timeline`` callers).
    epochs_run: Number of training epochs executed (0-based indices
      ``0 .. epochs_run - 1`` are expected). Use ``0`` for experiments that
      never entered the epoch loop.
    check_cost: When ``True``, also require per-epoch cost attribution
      entries (``source='cost'``, ``metadata['_type']=='cost_attribution'``).
      Default ``False`` — do not fail when ``CostTrackerCallback`` was not wired.

  Returns:
    TraceReport with per-dimension coverage, percentages in ``details``, and
    specific gap strings (e.g. ``'epoch 2: missing policy_gate context entry'``).

  Raises:
    ValueError: When ``epochs_run`` is negative.
  """
  if epochs_run < 0:
    msg = f'epochs_run must be non-negative, got {epochs_run}'
    raise ValueError(msg)

  dimensions: list[TraceDimension] = []
  all_gaps: list[str] = []

  policy_dim, policy_gaps = _check_policy_gate(context_log, epochs_run)
  dimensions.append(policy_dim)
  all_gaps.extend(policy_gaps)

  grad_dim, grad_gaps = _check_gradient_journal(context_log, epochs_run)
  dimensions.append(grad_dim)
  all_gaps.extend(grad_gaps)

  store_dim, store_gaps = _check_store_context(reflog_entries)
  dimensions.append(store_dim)
  all_gaps.extend(store_gaps)

  if check_cost:
    cost_dim, cost_gaps = _check_cost_attribution(context_log, epochs_run)
    dimensions.append(cost_dim)
    all_gaps.extend(cost_gaps)

  complete = all(d.passed for d in dimensions) and not all_gaps
  return TraceReport(complete=complete, dimensions=dimensions, gaps=all_gaps)
