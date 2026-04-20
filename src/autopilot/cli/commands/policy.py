"""Policy engine: check constraints and explain decisions.

Delegates to the policy implementation for evaluation and explanation.
"""

from autopilot.cli.command import Command, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.config import resolve_experiment_dir
from autopilot.core.models import Result
from autopilot.core.normalization import load_result
import argparse


class PolicyCommand(Command):
  name = 'policy'
  help = 'Policy checks and explanations'

  @subcommand('check', help='Evaluate policies against experiment state')
  def check(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    slug = ctx.experiment
    if not slug:
      ctx.output.info('Available policies:')
      ctx.output.result({'policies': ['quality_first']})
      return

    if not ctx.module:
      ctx.output.result(
        {
          'slug': slug,
          'error': 'no module; run via project CLI with module=',
        },
        ok=False,
      )
      return

    policy = ctx.module.policy
    if policy is None:
      ctx.output.result(
        {
          'slug': slug,
          'error': 'module has no policy configured',
        },
        ok=False,
      )
      return

    exp_dir = resolve_experiment_dir(ctx.workspace, slug, ctx.project)
    result_data = load_result(exp_dir)
    if not result_data:
      ctx.output.result(
        {
          'slug': slug,
          'error': 'no result found; run validation or produce a result first',
        },
        ok=False,
      )
      return

    eval_result = Result.from_dict(result_data)
    policy_key = policy.name()
    policy_out = policy.forward(eval_result)

    ctx.output.result(
      {
        'slug': slug,
        'policy': policy_key,
        'gate_result': policy_out.value,
        'result': eval_result.to_dict(),
      }
    )

  @subcommand('explain', help='Explain policy outcome for experiment')
  def explain(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    slug = ctx.experiment
    if not slug:
      raise ValueError('experiment slug required (--experiment)')

    if not ctx.module:
      ctx.output.result(
        {
          'slug': slug,
          'error': 'no module; run via project CLI with module=',
        },
        ok=False,
      )
      return

    policy = ctx.module.policy
    if policy is None:
      ctx.output.result(
        {
          'slug': slug,
          'error': 'module has no policy configured',
        },
        ok=False,
      )
      return

    exp_dir = resolve_experiment_dir(ctx.workspace, slug, ctx.project)
    result_data = load_result(exp_dir)
    if not result_data:
      ctx.output.result(
        {
          'slug': slug,
          'error': 'no result found',
        },
        ok=False,
      )
      return

    eval_result = Result.from_dict(result_data)
    policy_key = policy.name()
    explanation = policy.explain(eval_result)

    ctx.output.result(
      {
        'slug': slug,
        'policy': policy_key,
        'explanation': explanation,
        'result': eval_result.to_dict(),
      }
    )
