"""Delta building and direction-aware verdict computation for experiment compare."""

from autopilot.cli.context import CLIContext
from autopilot.core.comparison import ComparatorMetric
from autopilot.core.metric_utils import infer_direction, metric_base_name
from typing import Any
import math


def validate_direction_overrides(
  ctx: CLIContext,
  higher_list: list[str],
  lower_list: list[str],
  metrics_a: dict[str, Any],
  metrics_b: dict[str, Any],
) -> list[ComparatorMetric]:
  """Validate ``--higher-metric`` / ``--lower-metric`` flags and build overrides.

  Fails via ``ctx.fail`` when the same metric appears in both override lists
  or when an override name is absent from the union of compared metric keys.

  Args:
    ctx: CLI context for error output.
    higher_list: Metric names forced to higher-is-better.
    lower_list: Metric names forced to lower-is-better.
    metrics_a: Baseline experiment metrics.
    metrics_b: Candidate experiment metrics.

  Returns:
    List of ``ComparatorMetric`` direction overrides for ``build_compare_deltas``.
  """
  conflicting = set(higher_list) & set(lower_list)
  if conflicting:
    ctx.fail(
      f'metric(s) {sorted(conflicting)!r} appear in both --higher-metric and '
      f'--lower-metric; each metric must have exactly one direction'
    )

  raw_keys = set(metrics_a) | set(metrics_b)
  normalized_a, normalized_b = normalize_metric_prefixes(metrics_a, metrics_b)
  normalized_keys = set(normalized_a) | set(normalized_b)

  for name in higher_list + lower_list:
    if name not in raw_keys and name not in normalized_keys:
      ctx.fail(
        f'override metric {name!r} not found in compared metrics; '
        f'available: {sorted(normalized_keys)!r}'
      )

  overrides: list[ComparatorMetric] = []
  for name in higher_list:
    overrides.append(ComparatorMetric(name, higher_is_better=True))
    stripped = metric_base_name(name)
    if stripped != name:
      overrides.append(ComparatorMetric(stripped, higher_is_better=True))
  for name in lower_list:
    overrides.append(ComparatorMetric(name, higher_is_better=False))
    stripped = metric_base_name(name)
    if stripped != name:
      overrides.append(ComparatorMetric(stripped, higher_is_better=False))
  return overrides


def is_numeric_metric_value(value: Any) -> bool:
  """Check whether a metric value is a finite number suitable for delta computation.

  Returns ``True`` for ``int`` and ``float`` values that are not ``bool`` and
  not ``NaN``. Booleans are excluded despite ``isinstance(True, int)`` because
  they represent categorical flags, not quantities.

  Args:
    value: Metric value to test.

  Returns:
    Whether the value is a finite numeric quantity.
  """
  if isinstance(value, bool):
    return False
  if isinstance(value, int):
    return True
  if isinstance(value, float):
    return not math.isnan(value)
  return False


def normalize_metric_prefixes(
  metrics_a: dict[str, Any],
  metrics_b: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Normalize metric keys so mismatched prefixes produce meaningful deltas.

  When both sides have a metric with the same base name (e.g. ``val_accuracy``
  in A and ``accuracy`` in B), they are normalized to the base name. Metrics
  already matching exactly are kept as-is.

  Args:
    metrics_a: Baseline metrics dict (may contain non-numeric values).
    metrics_b: Candidate metrics dict (may contain non-numeric values).

  Returns:
    Tuple of (normalized_a, normalized_b) with aligned keys.
  """
  keys_a = set(metrics_a)
  keys_b = set(metrics_b)
  shared = keys_a & keys_b
  only_a = keys_a - shared
  only_b = keys_b - shared

  normalized_a: dict[str, Any] = {}
  normalized_b: dict[str, Any] = {}

  for key in shared:
    normalized_a[key] = metrics_a[key]
    normalized_b[key] = metrics_b[key]

  bases_a = {metric_base_name(k): k for k in only_a}
  bases_b = {metric_base_name(k): k for k in only_b}

  matched_a: set[str] = set()
  matched_b: set[str] = set()

  for base, orig_a in bases_a.items():
    if base in bases_b:
      orig_b = bases_b[base]
      normalized_a[base] = metrics_a[orig_a]
      normalized_b[base] = metrics_b[orig_b]
      matched_a.add(orig_a)
      matched_b.add(orig_b)

  for k in only_a:
    if k not in matched_a:
      normalized_a[k] = metrics_a[k]
  for k in only_b:
    if k not in matched_b:
      normalized_b[k] = metrics_b[k]

  return normalized_a, normalized_b


def build_compare_deltas(
  metrics_a: dict[str, Any],
  metrics_b: dict[str, Any],
  *,
  direction_overrides: list[ComparatorMetric] | None = None,
) -> list[dict[str, Any]]:
  """Normalize metrics and compute per-key deltas.

  Returns a list of delta records (not a keyed dict) with an explicit ``type``
  field: ``'numeric'`` when both values are finite numbers, ``'non_numeric'``
  when both are present but at least one is non-numeric, or ``'missing'`` when
  one side has no value for the key. Every delta dict includes a
  ``higher_is_better`` boolean resolved from explicit overrides or the
  ``infer_direction`` heuristic.

  Args:
    metrics_a: Baseline experiment metrics (may contain non-numeric values).
    metrics_b: Candidate experiment metrics (may contain non-numeric values).
    direction_overrides: Explicit per-metric direction from CLI flags;
      overrides the heuristic for matching metric names.

  Returns:
    Ordered list of per-metric delta records.
  """
  override_map: dict[str, bool] = {}
  if direction_overrides is not None:
    for cm in direction_overrides:
      override_map[cm.metric_name] = cm.higher_is_better is True

  normalized_a, normalized_b = normalize_metric_prefixes(metrics_a, metrics_b)
  all_keys = sorted(set(normalized_a) | set(normalized_b))
  deltas: list[dict[str, Any]] = []
  for k in all_keys:
    va = normalized_a.get(k)
    vb = normalized_b.get(k)
    hib = override_map[k] if k in override_map else infer_direction(k)
    if va is None or vb is None:
      deltas.append(
        {
          'metric': k,
          'baseline': va,
          'candidate': vb,
          'delta': None,
          'type': 'missing',
          'higher_is_better': hib,
        }
      )
    elif is_numeric_metric_value(va) and is_numeric_metric_value(vb):
      deltas.append(
        {
          'metric': k,
          'baseline': va,
          'candidate': vb,
          'delta': vb - va,
          'type': 'numeric',
          'higher_is_better': hib,
        }
      )
    else:
      deltas.append(
        {
          'metric': k,
          'baseline': va,
          'candidate': vb,
          'delta': None,
          'type': 'non_numeric',
          'higher_is_better': hib,
        }
      )
  return deltas


def compute_verdict(
  deltas: list[dict[str, Any]],
  *,
  higher_overrides: list[str] | None = None,
  lower_overrides: list[str] | None = None,
) -> str:
  """Compute a direction-aware verdict from metric deltas.

  Only entries with ``type == 'numeric'`` and non-zero ``delta`` participate.
  Direction precedence: explicit CLI override sets (``higher_overrides`` /
  ``lower_overrides``) > per-delta ``higher_is_better`` key >
  ``infer_direction`` heuristic (defensive guard).

  Returns ``'improved'`` when all directional moves are favorable,
  ``'regressed'`` when all are unfavorable, or ``'inconclusive'`` on mixed
  signals or when no numeric deltas exist.

  Args:
    deltas: Ordered delta records; each must include ``metric``, ``delta``,
      ``type``, and ``higher_is_better`` keys.
    higher_overrides: Metric names forced to higher-is-better by CLI flags.
    lower_overrides: Metric names forced to lower-is-better by CLI flags.

  Returns:
    One of ``'improved'``, ``'regressed'``, or ``'inconclusive'``.
  """
  improved_count = 0
  regressed_count = 0
  higher_set: set[str] = set()
  lower_set: set[str] = set()
  for name in higher_overrides or ():
    higher_set.add(name)
    higher_set.add(metric_base_name(name))
  for name in lower_overrides or ():
    lower_set.add(name)
    lower_set.add(metric_base_name(name))
  for delta_row in deltas:
    if delta_row.get('type') != 'numeric':
      continue
    delta_val = delta_row.get('delta')
    if delta_val is None or delta_val == 0.0:
      continue
    metric = delta_row['metric']
    if metric in higher_set:
      hib = True
    elif metric in lower_set:
      hib = False
    else:
      hib = delta_row.get('higher_is_better', infer_direction(metric))
    is_better = (delta_val > 0 and hib) or (delta_val < 0 and not hib)
    if is_better:
      improved_count += 1
    else:
      regressed_count += 1
  if improved_count > 0 and regressed_count == 0:
    return 'improved'
  if regressed_count > 0 and improved_count == 0:
    return 'regressed'
  return 'inconclusive'
