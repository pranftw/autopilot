"""Propose command -- create, verify, revert, and list proposals.

Verify inputs: ``--proposal-id`` identifies the proposal, ``--epoch``
identifies the candidate epoch. Baseline metrics come from the epoch
preceding the proposal's epoch. Candidate metrics come from the specified
``--epoch``. When baseline metrics are unavailable, the verdict is
``inconclusive``.

Metric direction defaults to ``higher_is_better=True`` for all shared
metrics. Use ``--higher-is-better METRIC`` and ``--lower-is-better METRIC``
(repeatable) on ``verify`` to override per metric.

Forest-metrics fallback: when epoch metrics files are absent for either
baseline or candidate, ``experiment.metrics`` from the forest are used for
**both** sides (never mixed). A ``warnings`` list is included in the JSON
output when the fallback is engaged.

Non-numeric metrics (strings, bools, NaN) are skipped with a warning before
comparison; only ``int``/``float`` values (excluding ``bool``) participate.

JSON result schema for verify::

  {
    "proposal_id": "<id>",
    "verdict": "improved" | "regressed" | "inconclusive",
    "items_tested": <int>,
    "deltas": [{"metric": ..., "baseline": ..., "candidate": ..., "delta": ..., ...}],
    "warnings": ["..."]
  }
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.proposal import (
  ChangeProposal,
  ProposalVerdict,
  read_proposals,
  record_proposal,
  record_verdict,
)
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import CLIError, load_forest, resolve_epoch
from autopilot.cli.messages import MSG_EXPERIMENT_SLUG_REQUIRED
from autopilot.cli.primitives import argument, subcommand
from autopilot.core.artifacts.epoch import DataArtifact
from autopilot.core.comparison import ComparatorMetric, MetricsComparator
from autopilot.core.config import AutoPilotConfig
from autopilot.core.decision import DecisionEntry
from enum import StrEnum
from pathlib import Path
from typing import Any
import argparse
import json
import math
import re
import uuid

PROPOSAL_ID_HEX_LEN = 8


class VerdictKind(StrEnum):
  """Verdict values for proposal verification.

  Members compare equal to plain strings (``VerdictKind.improved == 'improved'``)
  so verdict strings in data models and JSON stay unchanged.
  """

  improved = 'improved'
  regressed = 'regressed'
  inconclusive = 'inconclusive'


def _filter_numeric_metrics(
  metrics: dict[str, Any],
) -> tuple[dict[str, float], set[str]]:
  """Filter metrics to numeric-only values, rejecting bools and NaN.

  Args:
    metrics: Raw metrics dict with mixed-type values.

  Returns:
    Tuple of (numeric-only dict, set of skipped key names).
  """
  numeric: dict[str, float] = {}
  skipped: set[str] = set()
  for key, value in metrics.items():
    if isinstance(value, bool):
      skipped.add(key)
      continue
    if isinstance(value, (int, float)):
      if math.isnan(value):
        skipped.add(key)
        continue
      numeric[key] = float(value)
    else:
      skipped.add(key)
  return numeric, skipped


class ProposeCommand(Command):
  """Manage optimization proposals."""

  name = 'propose'
  help = 'manage proposals'

  @argument('--target', default=None, help='target node for proposal')
  @argument('--hypothesis', default=None, help='hypothesis text')
  @argument('--category', default=None, help='proposal category')
  @subcommand('create', help_text='create a new proposal')
  def create(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Create a new change proposal and record it."""
    exp_dir = ctx.experiment_path()
    proposal = ChangeProposal(
      proposal_id=str(uuid.uuid4())[:PROPOSAL_ID_HEX_LEN],
      hypothesis='no hypothesis provided' if args.hypothesis is None else args.hypothesis,
      target_node=args.target,
      change_type='general' if args.category is None else args.category,
      epoch=ctx.epoch if ctx.epoch is not None else 0,
      status='proposed',
    )
    record_proposal(exp_dir, proposal)
    ctx.output.result({'proposal_id': proposal.proposal_id, 'status': 'created'})

  @argument(
    '--proposal-id',
    required=True,
    help='proposal ID (8 lowercase hex chars; see `propose list` for IDs)',
  )
  @argument(
    '--higher-is-better',
    action='append',
    default=None,
    dest='higher_is_better',
    help='metric where higher values are better (repeatable)',
  )
  @argument(
    '--lower-is-better',
    action='append',
    default=None,
    dest='lower_is_better',
    help='metric where lower values are better (repeatable)',
  )
  @subcommand(
    'verify',
    help_text='compare metrics and record the verdict (mutating; requires --context)',
  )
  def verify(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Verify a proposal by comparing baseline and candidate metrics.

    Compares metrics and records the verdict (mutating). Requires
    ``--context`` because the verdict is persisted to disk via
    ``record_verdict()``.

    Uses ``MetricsComparator`` to produce improve/regress/inconclusive
    verdicts per metric. Baseline is loaded from the epoch preceding the
    proposal's epoch; candidate from the specified ``--epoch``.

    Fallback: when either baseline or candidate epoch metrics files are
    absent, ``experiment.metrics`` from the forest are used for **both**
    sides. A ``warnings`` list is included in JSON output when this
    fallback is engaged.

    Non-numeric metrics are skipped before comparison; skipped keys are
    reported as ``skipped_non_numeric:<key>`` in ``warnings``.

    By default, all shared metrics use ``higher_is_better=True``. Use
    ``--higher-is-better METRIC`` and ``--lower-is-better METRIC``
    (repeatable) to override per metric. A metric appearing in both
    flags is rejected with an error.
    """
    proposal_id = args.proposal_id
    if re.fullmatch(r'\d+', proposal_id):
      ctx.fail(
        f'--proposal-id {proposal_id!r} looks like a numeric index, not a hex ID. '
        'Use the 8-char hex ID from `propose list`.'
      )
    directions = self._parse_metric_directions(ctx, args)

    exp_dir = ctx.experiment_path()
    epoch = ctx.epoch
    if epoch is None:
      epoch = self._resolve_latest_epoch(ctx)

    proposals = read_proposals(exp_dir)
    proposal = next((p for p in proposals if p.proposal_id == proposal_id), None)
    if not proposal:
      ctx.fail(f'proposal {proposal_id!r} not found')

    items_tested = len(DataArtifact().read_raw(exp_dir, epoch=epoch))

    baseline_raw = self._load_epoch_metrics(exp_dir, max(proposal.epoch - 1, 0))
    candidate_raw = self._load_epoch_metrics(exp_dir, epoch)
    verdict_str, deltas_payload, warnings = self._resolve_and_compare(
      ctx, baseline_raw, candidate_raw, directions
    )

    verdict = ProposalVerdict(
      proposal_id=proposal_id,
      verdict=verdict_str,
      items_tested=items_tested,
    )
    record_verdict(exp_dir, epoch, verdict)
    self._emit_proposal_comparison_context(
      ctx,
      proposal_id=proposal_id,
      proposal_epoch=proposal.epoch,
      candidate_epoch=epoch,
      verdict_kind=VerdictKind(verdict_str),
      deltas=deltas_payload,
    )
    result_payload: dict[str, Any] = {
      'proposal_id': proposal_id,
      'verdict': verdict_str,
      'items_tested': items_tested,
      'deltas': deltas_payload,
    }
    if warnings:
      result_payload['warnings'] = warnings
    ctx.output.result(result_payload)

  def _emit_proposal_comparison_context(
    self,
    ctx: CLIContext,
    *,
    proposal_id: str,
    proposal_epoch: int,
    candidate_epoch: int,
    verdict_kind: VerdictKind,
    deltas: list[dict[str, Any]],
  ) -> None:
    """Journal proposal verification comparison to experiment context log.

    Gracefully no-ops when experiment slug is missing or forest lookup fails.
    """
    experiment_id = ctx.experiment
    if not isinstance(experiment_id, str) or not experiment_id:
      return
    try:
      forest = load_forest(ctx)
    except (CLIError, OSError, ValueError, TypeError, KeyError, AttributeError):
      return
    found = forest.find_experiment(experiment_id)
    if found is None:
      return
    node, _tree = found
    baseline_id = experiment_id
    if node.baseline is not None:
      baseline_id = node.baseline.experiment.id
    metadata = DecisionEntry.comparison(
      baseline_id=baseline_id,
      candidate_id=experiment_id,
      verdict=verdict_kind.value,
      deltas=deltas,
      proposal_id=proposal_id,
      baseline_epoch=max(proposal_epoch - 1, 0),
      candidate_epoch=candidate_epoch,
    )
    node.experiment.add_context(
      f'proposal verified: {verdict_kind.value}',
      source='proposal',
      metadata=metadata,
    )
    forest.save()

  def _resolve_and_compare(
    self,
    ctx: CLIContext,
    baseline_raw: dict[str, Any] | None,
    candidate_raw: dict[str, Any] | None,
    directions: dict[str, bool],
  ) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Resolve metrics (with forest fallback), filter numerics, and compare.

    Args:
      ctx: CLI context for forest lookup.
      baseline_raw: Raw baseline metrics or ``None`` when epoch file absent.
      candidate_raw: Raw candidate metrics or ``None`` when epoch file absent.
      directions: Per-metric direction overrides.

    Returns:
      Tuple of (verdict string, deltas payload, warnings list).
    """
    warnings: list[str] = []

    if baseline_raw is None or candidate_raw is None:
      forest_metrics = self._resolve_forest_metrics(ctx)
      if forest_metrics is not None:
        baseline_raw = forest_metrics
        candidate_raw = forest_metrics
        warnings.append('epoch metrics missing; used forest experiment.metrics')

    if baseline_raw is None or candidate_raw is None:
      return VerdictKind.inconclusive, [], warnings

    baseline_numeric, baseline_skipped = _filter_numeric_metrics(baseline_raw)
    candidate_numeric, candidate_skipped = _filter_numeric_metrics(candidate_raw)
    warnings.extend(
      f'skipped_non_numeric:{key}' for key in sorted(baseline_skipped | candidate_skipped)
    )
    verdict_str, deltas_payload = self._compare_metrics(
      baseline_numeric, candidate_numeric, directions
    )
    return verdict_str, deltas_payload, warnings

  def _parse_metric_directions(
    self,
    ctx: CLIContext,
    args: argparse.Namespace,
  ) -> dict[str, bool]:
    """Parse --higher-is-better / --lower-is-better into a direction map.

    Args:
      ctx: CLI context for error reporting.
      args: Parsed arguments with ``higher_is_better`` and ``lower_is_better``.

    Returns:
      Dict mapping metric names to their direction (True = higher is better).
    """
    higher_set = set(args.higher_is_better) if args.higher_is_better else set()
    lower_set = set(args.lower_is_better) if args.lower_is_better else set()
    conflict = higher_set & lower_set
    if conflict:
      conflicting = sorted(conflict)
      ctx.fail(
        f'metric direction conflict: {conflicting!r} appear in both'
        ' --higher-is-better and --lower-is-better; pick one direction per metric'
      )

    directions: dict[str, bool] = {}
    for metric_name in higher_set:
      directions[metric_name] = True
    for metric_name in lower_set:
      directions[metric_name] = False
    return directions

  def _resolve_latest_epoch(self, ctx: CLIContext) -> int:
    """Resolve epoch to latest when ``--epoch`` is omitted.

    Uses ``resolve_epoch('latest', store, experiment_id)`` via the forest
    store. Fails via ``ctx.fail`` when the store has no snapshots or the
    experiment slug is missing.

    Args:
      ctx: CLI context with experiment and workspace config.

    Returns:
      Latest epoch number from the store.
    """
    experiment_slug = ctx.experiment
    if not experiment_slug:
      ctx.fail(MSG_EXPERIMENT_SLUG_REQUIRED)
    try:
      forest = load_forest(ctx)
      return resolve_epoch('latest', forest.store, experiment_slug)
    except CLIError as exc:
      ctx.fail(str(exc))

  def _load_epoch_metrics(self, exp_dir: Path, epoch: int) -> dict[str, Any] | None:
    """Load metrics for a given epoch from the experiment directory.

    Args:
      exp_dir: Experiment artifacts directory.
      epoch: Epoch to load metrics for.

    Returns:
      Metrics dict (raw values, not pre-filtered), or ``None`` when absent.
    """
    metrics_file = exp_dir / f'epoch_{epoch}_metrics.json'
    if not metrics_file.exists():
      return None
    try:
      raw = json.loads(metrics_file.read_text(encoding='utf-8'))
      if isinstance(raw, dict):
        return dict(raw)
    except (json.JSONDecodeError, ValueError, OSError):
      pass
    return None

  def _resolve_forest_metrics(self, ctx: CLIContext) -> dict[str, Any] | None:
    """Load experiment metrics from the forest as a fallback.

    Tries active tree first, then cross-tree search. Returns ``None``
    when the forest cannot be loaded or the experiment is not found.

    Args:
      ctx: CLI context with experiment slug.

    Returns:
      Metrics dict from the forest experiment, or ``None``.
    """
    slug = ctx.experiment
    if not isinstance(slug, str) or not slug:
      return None
    try:
      forest = load_forest(ctx)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
      return None
    tree = forest.active
    if tree is not None:
      node = tree.get(slug)
      if node is not None:
        return node.experiment.metrics
    for t in forest.list_trees():
      if tree is not None and t.name == tree.name:
        continue
      node = t.get(slug)
      if node is not None:
        return node.experiment.metrics
    return None

  def _compare_metrics(
    self,
    baseline: dict[str, float],
    candidate: dict[str, float],
    metric_directions: dict[str, bool] | None = None,
  ) -> tuple[str, list[dict[str, Any]]]:
    """Compare baseline and candidate metrics via MetricsComparator.

    Builds ``ComparatorMetric`` for each shared key. Default direction is
    ``higher_is_better=True`` unless overridden via *metric_directions*.

    Args:
      baseline: Baseline metric values.
      candidate: Candidate metric values.
      metric_directions: Per-metric direction overrides. Keys are metric
        names; ``True`` means higher is better, ``False`` means lower is
        better. Metrics not present default to ``higher_is_better=True``.

    Returns:
      Tuple of (verdict_string, list of delta dicts).
    """
    shared_keys = set(baseline) & set(candidate)
    if not shared_keys:
      return VerdictKind.inconclusive, []

    directions = metric_directions if metric_directions is not None else {}
    metric_stubs = [
      ComparatorMetric(key, higher_is_better=directions.get(key, True))
      for key in sorted(shared_keys)
    ]
    comparator = MetricsComparator(metric_stubs)
    deltas = comparator.compare(baseline, candidate)

    if not deltas:
      return VerdictKind.inconclusive, []

    improved_count = sum(1 for d in deltas if d.significant and comparator.is_improvement(d))
    regressed_count = sum(1 for d in deltas if d.significant and not comparator.is_improvement(d))

    if improved_count > 0 and regressed_count == 0:
      verdict = VerdictKind.improved
    elif regressed_count > 0 and improved_count == 0:
      verdict = VerdictKind.regressed
    elif improved_count == 0 and regressed_count == 0:
      verdict = VerdictKind.inconclusive
    else:
      verdict = VerdictKind.inconclusive

    deltas_payload = [d.to_dict() for d in deltas]
    return verdict, deltas_payload

  @argument('--proposal-id', default=None, help='proposal ID to revert')
  @argument('--source', default=None, help='source directory for store')
  @argument('--store', default=None, help='store root path')
  @argument('--pattern', default='**/*', help='file pattern')
  @subcommand('revert', help_text='revert a proposal via store checkout')
  def revert(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Revert a proposal by restoring the store to a prior epoch.

    Epoch 0 restores are valid (restores to the initial state). Only
    negative epoch values are rejected.
    """
    proposal_id = args.proposal_id
    if not proposal_id:
      ctx.fail('--proposal-id is required')

    exp_dir = ctx.experiment_path()
    proposals = read_proposals(exp_dir)
    proposal = next((p for p in proposals if p.proposal_id == proposal_id), None)
    if not proposal:
      ctx.fail(f'proposal {proposal_id!r} not found')

    restore_epoch = max(proposal.epoch - 1, 0) if ctx.epoch is None else ctx.epoch
    if restore_epoch < 0:
      ctx.fail(
        f'cannot revert: epoch must be non-negative, got {type(restore_epoch).__name__}'
        f' {restore_epoch}'
      )

    source = args.source
    if not source:
      ctx.fail('--source is required for revert')

    config = AutoPilotConfig(workspace=ctx.workspace, project=ctx.project)
    if args.store:
      config.store_path = Path(args.store).resolve()
    pattern = '**/*' if args.pattern is None else args.pattern
    param = PathParameter(source=str(Path(source).expanduser().resolve()), pattern=pattern)

    experiment_slug = ctx.experiment
    if experiment_slug is None:
      ctx.fail(MSG_EXPERIMENT_SLUG_REQUIRED)

    try:
      store = FileStore(config)
      store.register_parameters({'source': param})
      store.checkout(experiment_slug, restore_epoch, context=ctx.context)
    except (FileNotFoundError, OSError) as exc:
      ctx.fail(
        f'revert failed ({type(exc).__name__}): {exc}; verify the proposal epoch with propose list'
      )

    ctx.output.result(
      {
        'proposal_id': proposal_id,
        'status': 'reverted',
        'restored_epoch': restore_epoch,
      }
    )

  @subcommand('list', help_text='list all proposals')
  def list_proposals(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List all recorded proposals for the experiment."""
    exp_dir = ctx.experiment_path()
    proposals = read_proposals(exp_dir)
    ctx.output.result(
      {
        'proposals': [p.to_dict() for p in proposals],
        'count': len(proposals),
      }
    )
