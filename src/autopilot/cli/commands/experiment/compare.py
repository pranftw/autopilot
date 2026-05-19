"""Experiment comparison and impact analysis commands.

Cross-tree experiment resolution, metric prefix normalization, direction-aware
verdict computation, fingerprint drift detection, dependency impact BFS, and
optional weighted multi-metric aggregation.
"""

from autopilot.ai.fingerprint import DatasetFingerprint, detect_drift
from autopilot.cli.command import Command
from autopilot.cli.commands.experiment.verdict import (
  build_compare_deltas,
  compute_verdict,
  validate_direction_overrides,
)
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest, require_active_tree
from autopilot.cli.primitives import Argument
from autopilot.core.experiment import Experiment
from collections import deque
from typing import Any
import argparse
import math

COMPARE_CONTEXT_TAIL_COUNT = 3
CONTEXT_REASON_DISPLAY_LEN = 60
WEIGHTED_VERDICT_EPSILON = 1e-9


class ExperimentCompare(Command):
  """Compare metrics of two experiments side by side.

  Cross-tree lookup: when an experiment id is not found in the active tree,
  all trees in the forest are searched before failing (BUG-019).

  Metric prefix normalization: when both experiments have metrics that differ
  only by prefix (e.g. ``val_accuracy`` vs ``accuracy``), the base names are
  matched to produce meaningful deltas (TS-prefix-mismatch).

  Verdict: the comparison JSON includes an aggregated ``verdict`` field
  (``improved`` / ``regressed`` / ``inconclusive``) summarizing the overall
  direction of change (GAP-018). Direction is resolved via heuristic
  inference (``infer_direction``) or explicit ``--higher-metric`` /
  ``--lower-metric`` CLI flags. Every delta dict includes a boolean
  ``higher_is_better`` field.

  Text mode appends a context summary section with the last 3 journal entries
  per experiment (abbreviated). JSON mode includes full ``context_log`` arrays
  for both experiments.
  """

  name = 'compare'
  help = 'Compare two experiments'
  exp_a = Argument('a', help='first experiment ID (baseline)')
  exp_b = Argument('b', help='second experiment ID (candidate)')
  higher_metric = Argument(
    '--higher-metric',
    action='append',
    default=None,
    dest='higher_metric',
    metavar='NAME',
    help='metric where higher is better (repeatable)',
  )
  lower_metric = Argument(
    '--lower-metric',
    action='append',
    default=None,
    dest='lower_metric',
    metavar='NAME',
    help='metric where lower is better (repeatable)',
  )
  weights = Argument(
    '--weights',
    default=None,
    metavar='SPEC',
    help=(
      'comma-separated metric:weight pairs for weighted verdict '
      '(e.g. accuracy:7,latency:3); weights are normalized internally '
      'and must be non-negative with a positive sum; JSON output adds '
      'weighted_verdict and weighted_score_delta fields'
    ),
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Compare metrics between two experiments (table or JSON).

    Searches all trees in the forest when an experiment is not found in the
    active tree (BUG-019 fix). Normalizes metric prefixes to produce deltas
    even when keys differ by ``val_``/``train_`` prefix. Emits an aggregated
    direction-aware ``verdict`` field (improved/regressed/inconclusive).

    ``--higher-metric`` / ``--lower-metric`` override the heuristic for
    specific metric names. Conflicts (same name in both) or unknown names
    (absent from compared metrics) are hard errors.

    ``dataset_fingerprint_drift`` is tri-state: ``True`` when both
    experiments have non-empty fingerprints and ``detect_drift`` finds
    differences; ``False`` when both are present and match; ``None`` when
    either side has a missing or empty fingerprint (lineage unknown).
    JSON renders ``None`` as ``null``; text mode prints an info line for
    ``True`` (differ) and ``None`` (unknown), silent for ``False``.

    ``--weights`` enables weighted multi-metric aggregation. Weights are
    normalized internally; JSON output adds ``weighted_verdict`` and
    ``weighted_score_delta`` fields only when the flag is provided.
    """
    forest = load_forest(ctx)
    require_active_tree(ctx, forest)

    node_a = _resolve_experiment(ctx, forest, args.a)
    node_b = _resolve_experiment(ctx, forest, args.b)

    higher_list: list[str] = args.higher_metric or []
    lower_list: list[str] = args.lower_metric or []
    direction_overrides = validate_direction_overrides(
      ctx, higher_list, lower_list, node_a.experiment.metrics, node_b.experiment.metrics
    )

    _warn_spec_version_mismatch(ctx, args, node_a, node_b)
    parsed_weights = _parse_weights(ctx, args.weights) if args.weights is not None else None

    deltas = build_compare_deltas(
      node_a.experiment.metrics,
      node_b.experiment.metrics,
      direction_overrides=direction_overrides or None,
    )

    if parsed_weights is not None:
      _validate_weight_metrics(ctx, parsed_weights, deltas)

    verdict = compute_verdict(
      deltas,
      higher_overrides=higher_list or None,
      lower_overrides=lower_list or None,
    )

    comparison = _build_comparison_result(ctx, args, node_a, node_b, deltas, verdict)

    if parsed_weights is not None:
      weighted_score, weighted_verdict = _compute_weighted_verdict(deltas, parsed_weights)
      comparison['weighted_score_delta'] = weighted_score
      comparison['weighted_verdict'] = weighted_verdict
    else:
      weighted_score = None
      weighted_verdict = None

    ctx.output.result(comparison)

    if not ctx.output.use_json:
      _render_compare_table(ctx, args.a, args.b, deltas, verdict)
      if weighted_verdict is not None:
        ctx.output.info(
          f'Weighted verdict: {weighted_verdict} (score delta: {weighted_score:+.6g})'
        )
      drift = comparison['dataset_fingerprint_drift']
      if drift is True:
        ctx.output.info(f'dataset fingerprints differ between {args.a!r} and {args.b!r}')
      elif drift is None:
        ctx.output.info('unknown (fingerprint missing on one side)')
      _render_context_summary(ctx, args.a, node_a.experiment)
      _render_context_summary(ctx, args.b, node_b.experiment)


class ExperimentImpact(Command):
  """Show direct and transitive dependents of an experiment.

  Builds a reverse adjacency graph from all experiment dependencies
  forest-wide and performs BFS to collect every experiment that directly
  or transitively depends on the given id.

  Additionally collects direct tree children via ``Node.parent`` pointers
  across all trees in the forest. Children are orthogonal to dependency
  edges and appear under a separate ``children`` key in the JSON payload.
  """

  name = 'impact'
  help = 'Show experiments that depend on a given experiment'
  experiment_id = Argument('id', help='experiment ID to check impact for')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Compute and emit dependents + tree children for the experiment."""
    forest = load_forest(ctx)

    eid = args.id
    if forest.find_experiment(eid) is None:
      ctx.fail(f'experiment {eid!r} not found in any tree')

    all_deps = collect_all_dependencies(forest)

    reverse: dict[str, list[str]] = {}
    for exp_id, deps in all_deps.items():
      for dep in deps:
        reverse.setdefault(dep, []).append(exp_id)

    direct_dependents = sorted(reverse.get(eid, []))

    all_dependents: set[str] = set()
    queue = deque(reverse.get(eid, []))
    while queue:
      node_id = queue.popleft()
      if node_id in all_dependents:
        continue
      all_dependents.add(node_id)
      queue.extend(reverse.get(node_id, []))

    children = _collect_tree_children(forest, eid)

    ctx.output.result(
      {
        'experiment_id': eid,
        'dependents': sorted(all_dependents),
        'direct_dependents': direct_dependents,
        'children': children,
      }
    )


def _resolve_experiment(ctx: CLIContext, forest: Any, experiment_id: str) -> Any:
  """Resolve an experiment id across all trees in the forest.

  Args:
    ctx: CLI context for error output.
    forest: Forest to search.
    experiment_id: Experiment id to look up.

  Returns:
    The matched Node.
  """
  result = forest.find_experiment(experiment_id)
  if result is None:
    ctx.fail(
      f'experiment {experiment_id!r} not found in any tree; '
      'verify the experiment id or check available experiments with query'
    )
  node, _ = result
  return node


def _warn_spec_version_mismatch(
  ctx: CLIContext,
  args: argparse.Namespace,
  node_a: Any,
  node_b: Any,
) -> None:
  """Emit a warning when compared experiments have mismatched spec versions.

  Args:
    ctx: CLI context for output.
    args: Parsed CLI args (provides experiment id labels).
    node_a: Baseline node.
    node_b: Candidate node.
  """
  va = node_a.experiment.spec_version
  vb = node_b.experiment.spec_version
  if va is not None and vb is not None and va != vb:
    ctx.output.warn(f'spec_version mismatch: {args.a!r} has {va!r}, {args.b!r} has {vb!r}')


def _build_comparison_result(
  ctx: CLIContext,
  args: argparse.Namespace,
  node_a: Any,
  node_b: Any,
  deltas: list[dict[str, Any]],
  verdict: str,
) -> dict[str, Any]:
  """Assemble the base comparison result dict.

  Args:
    ctx: CLI context.
    args: Parsed CLI args.
    node_a: Baseline node.
    node_b: Candidate node.
    deltas: Per-metric delta records.
    verdict: Aggregated verdict string.

  Returns:
    Dict with core comparison fields; caller may extend with weighted keys.
  """
  comparison: dict[str, Any] = {
    'a': args.a,
    'b': args.b,
    'spec_version': {
      'baseline': node_a.experiment.spec_version,
      'candidate': node_b.experiment.spec_version,
    },
    'deltas': deltas,
    'verdict': verdict,
    'dataset_fingerprint_drift': _detect_fingerprint_drift(node_a.experiment, node_b.experiment),
  }
  if ctx.output.use_json:
    comparison['context_log_a'] = node_a.experiment.context_log.to_list()
    comparison['context_log_b'] = node_b.experiment.context_log.to_list()
  return comparison


def _parse_weights(ctx: CLIContext, raw: str) -> dict[str, float]:
  """Parse a ``--weights`` flag value into normalized metric weights.

  Accepts comma-separated ``metric:weight`` tokens. Strips whitespace around
  both metric names and weight values. Rejects: empty input, missing colons,
  non-finite weights, negative weights, zero total weight, and duplicate
  metric names.

  Args:
    ctx: CLI context for error output via ``ctx.fail``.
    raw: Raw flag value (e.g. ``'accuracy:7,latency:3'``).

  Returns:
    Dict mapping metric names to normalized weights that sum to 1.0.
  """
  stripped = raw.strip()
  if not stripped:
    ctx.fail('--weights requires metric:weight tokens (e.g. --weights accuracy:7,latency:3)')

  segments = stripped.split(',')
  parsed: dict[str, float] = {}
  for raw_segment in segments:
    segment = raw_segment.strip()
    if not segment:
      ctx.fail(
        f'--weights contains an empty segment in {raw!r}; remove trailing commas or empty entries'
      )
    parts = segment.split(':', 1)
    if len(parts) != 2:
      ctx.fail(
        f'--weights segment {segment!r} missing colon; '
        'expected metric:weight format (e.g. accuracy:7)'
      )
    name = parts[0].strip()
    weight_str = parts[1].strip()
    if not name:
      ctx.fail(f'--weights segment {segment!r} has empty metric name')
    try:
      weight = float(weight_str)
    except ValueError:
      ctx.fail(f'--weights: {weight_str!r} is not a valid number for metric {name!r}')
    if math.isnan(weight):
      ctx.fail(f'--weights: NaN is not a valid weight for metric {name!r}')
    if math.isinf(weight):
      ctx.fail(f'--weights: infinite value is not a valid weight for metric {name!r}')
    if weight < 0:
      ctx.fail(f'--weights: weight for {name!r} is {weight}; weights must be non-negative')
    if name in parsed:
      ctx.fail(f'--weights: duplicate metric {name!r}; each metric may appear only once')
    parsed[name] = weight

  total = sum(parsed.values())
  if total <= 0:
    ctx.fail('--weights: total weight is zero; at least one weight must be positive')

  return {k: v / total for k, v in parsed.items()}


def _validate_weight_metrics(
  ctx: CLIContext,
  weights: dict[str, float],
  deltas: list[dict[str, Any]],
) -> None:
  """Ensure every weighted metric exists as a numeric delta in both experiments.

  Rejects metrics that are absent from deltas entirely, or present but with
  ``type`` of ``'missing'`` (one side lacks the key) or ``'non_numeric'``
  (at least one side has a non-numeric value).

  Args:
    ctx: CLI context for error output.
    weights: Normalized weight dict from ``_parse_weights``.
    deltas: Delta records produced by ``build_compare_deltas``.
  """
  delta_by_metric = {d['metric']: d for d in deltas}
  delta_metrics = set(delta_by_metric)
  for name in weights:
    if name not in delta_metrics:
      ctx.fail(
        f'--weights: metric {name!r} not found in compared metrics; '
        f'available: {sorted(delta_metrics)!r}; '
        'run experiment show to inspect metric keys'
      )
    entry = delta_by_metric[name]
    if entry['type'] == 'missing':
      ctx.fail(
        f'--weights: metric {name!r} is missing from one experiment; '
        'weighted metrics must exist in both experiments'
      )
    if entry['type'] == 'non_numeric':
      ctx.fail(
        f'--weights: metric {name!r} has non-numeric values; '
        'weighted metrics must be numeric in both experiments'
      )


def _compute_weighted_verdict(
  deltas: list[dict[str, Any]],
  weights: dict[str, float],
) -> tuple[float, str]:
  """Compute weighted score delta and verdict from deltas and normalized weights.

  Callers **must** invoke ``_validate_weight_metrics`` before calling this
  function. The validation step guarantees every key in ``weights`` exists in
  ``deltas`` with ``type == 'numeric'`` and a usable ``delta`` value. This
  function uses strict dict access so violations surface as ``KeyError``
  instead of silent skips.

  Contribution: ``normalized_weight * direction_sign * delta`` where
  ``direction_sign`` is ``+1`` when ``higher_is_better`` is True, ``-1``
  otherwise.

  Verdict: ``'improved'`` if aggregate > epsilon, ``'regressed'`` if
  < -epsilon, ``'inconclusive'`` otherwise.

  Args:
    deltas: Ordered delta records from ``build_compare_deltas``.
    weights: Normalized weight dict (values sum to 1.0).

  Returns:
    Tuple of (weighted_score_delta, weighted_verdict).
  """
  delta_by_metric = {d['metric']: d for d in deltas}
  score = 0.0
  for metric_name, weight in weights.items():
    entry = delta_by_metric[metric_name]
    sign = 1.0 if entry['higher_is_better'] else -1.0
    score += weight * sign * entry['delta']

  if score > WEIGHTED_VERDICT_EPSILON:
    verdict = 'improved'
  elif score < -WEIGHTED_VERDICT_EPSILON:
    verdict = 'regressed'
  else:
    verdict = 'inconclusive'
  return score, verdict


def _render_compare_table(
  ctx: CLIContext,
  label_a: str,
  label_b: str,
  deltas: list[dict[str, Any]],
  verdict: str,
) -> None:
  """Render comparison as a text table with verdict line.

  Args:
    ctx: CLI context for output.
    label_a: First experiment id label.
    label_b: Second experiment id label.
    deltas: Ordered list of per-metric delta records.
    verdict: Aggregated verdict string.
  """
  rows = []
  for entry in deltas:
    va = entry['baseline']
    vb = entry['candidate']
    delta_str = ''
    if entry['type'] == 'numeric':
      delta_str = f'{entry["delta"]:+.6g}'
    elif entry['type'] == 'non_numeric':
      delta_str = '-'
    rows.append(
      {
        'metric': entry['metric'],
        label_a: str(va) if va is not None else '',
        label_b: str(vb) if vb is not None else '',
        'delta': delta_str,
      }
    )
  ctx.output.table(rows, ['metric', label_a, label_b, 'delta'])
  ctx.output.info(f'Verdict: {verdict}')


def _render_context_summary(ctx: CLIContext, label: str, exp: Experiment) -> None:
  """Render abbreviated context summary for one experiment in compare text.

  Shows the last ``COMPARE_CONTEXT_TAIL_COUNT`` entries with truncated reason.

  Args:
    ctx: CLI context for output.
    label: Experiment label (id) used as section header.
    exp: Experiment whose context log to summarize.
  """
  entries = exp.context_log.entries[-COMPARE_CONTEXT_TAIL_COUNT:]
  if not entries:
    return
  ctx.output.info(f'Context ({label}):')
  rows = [
    {
      'timestamp': e.timestamp[:19],
      'source': e.source or '',
      'reason': e.reason[:CONTEXT_REASON_DISPLAY_LEN],
    }
    for e in entries
  ]
  ctx.output.table(rows, ['timestamp', 'source', 'reason'])


def _detect_fingerprint_drift(exp_a: Experiment, exp_b: Experiment) -> bool | None:
  """Detect dataset fingerprint drift between two experiments.

  Tri-state return:

  - ``True`` when both experiments carry non-empty fingerprint dicts and
    ``detect_drift`` reports structural or hash differences.
  - ``False`` when both carry non-empty fingerprint dicts and no drift is
    found.
  - ``None`` when either side has a missing, ``None``, or empty ``{}``
    fingerprint -- lineage cannot be determined.

  Args:
    exp_a: First experiment.
    exp_b: Second experiment.

  Returns:
    ``True`` for drift, ``False`` for confirmed no drift, ``None`` when
    either fingerprint is missing or empty.
  """
  meta_a = exp_a.dataset_meta
  meta_b = exp_b.dataset_meta
  fp_a_dict = meta_a.get('dataset_fingerprint') if meta_a is not None else None
  fp_b_dict = meta_b.get('dataset_fingerprint') if meta_b is not None else None
  if fp_a_dict is None or fp_b_dict is None:
    return None
  if not fp_a_dict or not fp_b_dict:
    return None
  fp_a = DatasetFingerprint.from_dict(fp_a_dict)
  fp_b = DatasetFingerprint.from_dict(fp_b_dict)
  return detect_drift(fp_a, fp_b)


def collect_all_dependencies(forest: Any) -> dict[str, list[str]]:
  """Collect the dependency graph across all trees in the forest.

  Args:
    forest: Forest instance to scan.

  Returns:
    Dict mapping experiment id to its dependency list.
  """
  deps: dict[str, list[str]] = {}
  for t in forest.list_trees():
    for node in t.query().all():
      deps[node.experiment.id] = list(node.experiment.dependencies)
  return deps


def _collect_tree_children(forest: Any, target_id: str) -> list[str]:
  """Collect direct tree children of an experiment via parent pointers.

  Iterates every tree in the forest and checks each node's parent pointer.
  When a node's parent experiment id matches ``target_id``, the node's
  experiment id is added to the result.

  Args:
    forest: Forest instance to scan.
    target_id: Experiment id whose children to find.

  Returns:
    Sorted list of unique child experiment ids (direct only).
  """
  children: set[str] = set()
  for tree in forest.list_trees():
    for node in tree.query().all():
      if node.parent is not None and node.parent.experiment.id == target_id:
        children.add(node.experiment.id)
  return sorted(children)
