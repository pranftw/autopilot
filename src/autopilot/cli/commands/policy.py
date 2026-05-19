"""Policy engine: check constraints and explain decisions.

Delegates to the policy implementation for evaluation and explanation.

Metrics source precedence for ``policy check``:

  1. ``--metrics JSON_STRING`` -- inline JSON dict.
  2. Forest experiment metrics (``--experiment`` slug resolves via forest).
  3. Error listing both options.

Gate definition precedence:

  1. ``--min`` / ``--max`` flags -> ad-hoc ``ThresholdPolicy``.
  2. ``ctx.module.policy`` -> live Policy from project CLI.
  3. Error listing both options.

When ``--metrics`` + ``--min``/``--max`` satisfy both chains, ``ctx.module``
may be absent (no project CLI required).
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest
from autopilot.cli.messages import MSG_EXPERIMENT_SLUG_REQUIRED
from autopilot.cli.primitives import argument, subcommand
from autopilot.core.artifacts.experiment import ResultArtifact
from autopilot.core.models import Result
from autopilot.core.types import GateResult
from autopilot.policy.threshold import ThresholdPolicy, build_threshold_gates
from typing import Any
import argparse
import json


def _collect_gate_hints(policy: Any) -> dict[str, str]:
  """Extract metric-mismatch hints from policy after evaluation.

  Calls ``gate_hints()`` on policies that implement it.
  For policy types without ``gate_hints``, returns an empty dict.

  Args:
    policy: Evaluated policy instance.

  Returns:
    Mapping of gate metric name to hint string.
  """
  if hasattr(policy, 'gate_hints') and callable(policy.gate_hints):
    return policy.gate_hints()
  return {}


def _resolve_metrics_from_forest(ctx: CLIContext) -> dict[str, Any] | None:
  """Load experiment metrics from the forest for the current experiment slug.

  Tries the active tree first, then cross-tree search mirroring
  ``experiment compare`` helper patterns.

  Args:
    ctx: CLI context with experiment slug set.

  Returns:
    Metrics dict from the experiment, or ``None`` when not found.
  """
  slug = ctx.experiment
  if not slug:
    return None
  try:
    forest = load_forest(ctx)
  except (OSError, ValueError):
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


class PolicyCommand(Command):
  """``autopilot policy`` group: evaluate and explain policy gate decisions."""

  name = 'policy'
  help = 'Policy checks and explanations'

  @argument(
    '--min',
    action='append',
    default=None,
    dest='min_thresholds',
    metavar='METRIC:THRESHOLD',
    help='metric must be >= threshold (repeatable)',
  )
  @argument(
    '--max',
    action='append',
    default=None,
    dest='max_thresholds',
    metavar='METRIC:THRESHOLD',
    help='metric must be <= threshold (repeatable)',
  )
  @argument(
    '--metrics',
    default=None,
    dest='metrics_json',
    metavar='JSON',
    help='metrics as inline JSON dict',
  )
  @subcommand('check', help_text='Evaluate policies against experiment state')
  def check(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Evaluate policies against metrics with configurable sources and gates.

    Metrics source precedence (strict):

    1. ``--metrics JSON`` -> parsed inline.
    2. Forest experiment metrics (``--experiment`` slug resolved via forest).
    3. Error listing both fix paths.

    Gate definition precedence:

    1. ``--min`` / ``--max`` flags -> ad-hoc ``ThresholdPolicy``.
    2. ``ctx.module.policy`` -> live Policy instance from project CLI.
    3. Error listing both fix paths.

    Exit code contract: ``FAIL`` -> exit 1, ``ok: false``.
    ``PASSED`` / ``WARN`` / ``SKIP`` -> exit 0, ``ok: true``.

    Raises:
      SystemExit: With code 1 when the policy gate returns FAIL.
    """
    metrics = self._resolve_metrics(ctx, args)
    policy = self._resolve_policy(ctx, args)

    eval_result = Result.from_dict({'metrics': metrics})
    policy_key = policy.name()
    policy_out = policy.forward(eval_result)
    gate_hints = _collect_gate_hints(policy)

    ok = policy_out in {GateResult.PASSED, GateResult.WARN, GateResult.SKIP}
    ctx.output.result(
      {
        'policy': policy_key,
        'gate_result': policy_out.value,
        'metrics': metrics,
        'gate_hints': gate_hints,
      },
      ok=ok,
    )
    if not ok:
      raise SystemExit(1)

  def _resolve_metrics(
    self,
    ctx: CLIContext,
    args: argparse.Namespace,
  ) -> dict[str, Any]:
    """Resolve metrics dict following strict precedence.

    Args:
      ctx: CLI context.
      args: Parsed arguments.

    Returns:
      Metrics dict for policy evaluation.
    """
    if args.metrics_json is not None:
      try:
        parsed = json.loads(args.metrics_json)
      except json.JSONDecodeError as exc:
        ctx.fail(
          f'malformed --metrics JSON ({type(exc).__name__}): {exc};'
          ' provide a valid JSON object like \'{"accuracy": 0.9}\''
        )
      if not isinstance(parsed, dict):
        ctx.fail(
          f'--metrics must be a JSON object, got {type(parsed).__name__};'
          ' provide a dict like \'{"accuracy": 0.9}\''
        )
      return parsed

    forest_metrics = _resolve_metrics_from_forest(ctx)
    if forest_metrics is not None:
      return forest_metrics

    ctx.fail('policy check requires --metrics or --experiment with forest-backed metrics')
    return None  # unreachable; ctx.fail raises SystemExit

  def _resolve_policy(
    self,
    ctx: CLIContext,
    args: argparse.Namespace,
  ) -> Any:
    """Resolve the policy to evaluate, following gate definition precedence.

    Args:
      ctx: CLI context.
      args: Parsed arguments.

    Returns:
      Policy instance (ThresholdPolicy or ctx.module.policy).
    """
    has_thresholds = args.min_thresholds is not None or args.max_thresholds is not None
    if has_thresholds:
      try:
        gates = build_threshold_gates(args.min_thresholds, args.max_thresholds)
      except ValueError as exc:
        ctx.fail(str(exc))
      return ThresholdPolicy(gates)

    if ctx.module is not None:
      policy = ctx.module.policy
      if policy is not None:
        return policy

    ctx.fail(
      'no policy source: use --min/--max flags for ad-hoc gates'
      ' or run via project CLI with a module that has a policy configured'
    )
    return None  # unreachable; ctx.fail raises SystemExit

  @subcommand('explain', help_text='Explain policy outcome for experiment')
  def explain(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Explain why the policy produced its gate result.

    Calls ``policy.forward()`` explicitly before ``policy.explain()`` to ensure
    gate state is populated.  Exit code contract matches ``check``: FAIL yields
    exit 1, WARN/SKIP/PASSED yield exit 0.

    Raises:
      SystemExit: With code 1 when the policy gate returns FAIL.
    """
    slug = ctx.experiment
    if not slug:
      ctx.fail(MSG_EXPERIMENT_SLUG_REQUIRED)

    if not ctx.module:
      ctx.fail('no module; run via project CLI with module=')

    policy = ctx.module.policy
    if policy is None:
      ctx.fail('module has no policy configured')

    exp_dir = ctx.experiment_path(slug)
    result_data = ResultArtifact().read_raw(exp_dir)
    if not isinstance(result_data, dict):
      ctx.fail('no result found')

    eval_result = Result.from_dict(result_data)
    policy_key = policy.name()
    policy_out = policy.forward(eval_result)
    explanation = policy.explain(eval_result)

    gate_hints = _collect_gate_hints(policy)

    ok = policy_out in {GateResult.PASSED, GateResult.WARN, GateResult.SKIP}
    ctx.output.result(
      {
        'slug': slug,
        'policy': policy_key,
        'gate_result': policy_out.value,
        'explanation': explanation,
        'gate_hints': gate_hints,
        'result': eval_result.to_dict(),
      },
      ok=ok,
    )
    if not ok:
      raise SystemExit(1)
