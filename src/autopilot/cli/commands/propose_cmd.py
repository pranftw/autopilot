"""Propose command -- create, verify, revert, and list proposals."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store import FileStore
from autopilot.cli.command import Command, argument, subcommand
from autopilot.core.proposal import read_proposals, record_proposal, record_verdict
from autopilot.core.stage_io import read_epoch_artifact, read_experiment_artifact
from autopilot.core.stage_models import ChangeProposal, ProposalVerdict
from pathlib import Path
from typing import Any
import argparse
import autopilot.core.paths as paths
import uuid


class ProposeCommand(Command):
  """Manage optimization proposals."""

  name = 'propose'
  help = 'manage proposals'

  @argument('--target', default='', help='target node for proposal')
  @argument('--hypothesis', default='', help='hypothesis text')
  @argument('--category', default='', help='proposal category')
  @subcommand('create', help='create a new proposal')
  def create(self, ctx: Any, args: argparse.Namespace) -> None:
    exp_dir = ctx.experiment_dir()
    proposal = ChangeProposal(
      proposal_id=str(uuid.uuid4())[:8],
      hypothesis=args.hypothesis or 'no hypothesis provided',
      target_node=args.target,
      change_type=args.category or 'general',
      epoch=ctx.epoch,
      status='proposed',
    )
    record_proposal(exp_dir, proposal)
    ctx.output.result({'proposal_id': proposal.proposal_id, 'status': 'created'})

  @argument('--proposal-id', default='', help='proposal ID to verify')
  @subcommand('verify', help='verify a proposal against metrics')
  def verify(self, ctx: Any, args: argparse.Namespace) -> None:
    proposal_id = args.proposal_id
    if not proposal_id:
      ctx.output.error('--proposal-id is required')
      return

    exp_dir = ctx.experiment_dir()
    epoch = ctx.epoch
    if not epoch:
      ctx.output.error('--epoch is required')
      return

    proposals = read_proposals(exp_dir)
    proposal = next((p for p in proposals if p.proposal_id == proposal_id), None)
    if not proposal:
      ctx.output.error(f'proposal {proposal_id!r} not found')
      return

    regression = read_epoch_artifact(exp_dir, epoch, 'regression_analysis.json')
    if regression and regression.get('overall_verdict') in ('net_regression', 'mixed'):
      regressed_categories = [r.get('metric', '') for r in regression.get('regressions', [])]
      target_match = (
        proposal.target_node and proposal.target_node in str(regressed_categories)
      ) or (proposal.change_type and proposal.change_type in str(regressed_categories))
      if target_match or not proposal.target_node:
        verdict = ProposalVerdict(
          proposal_id=proposal_id,
          verdict='regression_after_change',
          items_tested=len(regression.get('regressions', [])),
        )
        record_verdict(exp_dir, epoch, verdict)
        ctx.output.result(
          {
            'proposal_id': proposal_id,
            'verdict': 'regression_after_change',
            'regressed_metrics': regressed_categories,
          }
        )
        return

    baseline = read_experiment_artifact(exp_dir, 'best_baseline.json')
    metrics = read_epoch_artifact(exp_dir, epoch, 'epoch_metrics.json')
    trace_data = read_epoch_artifact(exp_dir, epoch, 'data.jsonl')

    items_tested = 0
    if trace_data and isinstance(trace_data, list):
      items_tested = len(trace_data)
    elif metrics and 'total' in metrics:
      items_tested = metrics['total']

    if baseline and metrics:
      improved = False
      for key in baseline:
        if key in metrics and metrics[key] > baseline[key]:
          improved = True
          break

      if improved:
        verdict_str = 'fix_confirmed'
      else:
        verdict_str = 'inconclusive'
    else:
      verdict_str = 'inconclusive'

    verdict = ProposalVerdict(
      proposal_id=proposal_id,
      verdict=verdict_str,
      items_tested=items_tested,
    )
    record_verdict(exp_dir, epoch, verdict)
    ctx.output.result(
      {
        'proposal_id': proposal_id,
        'verdict': verdict_str,
        'items_tested': items_tested,
      }
    )

  @argument('--proposal-id', default='', help='proposal ID to revert')
  @argument('--source', default='', help='source directory for store')
  @argument('--store', default='', help='store root path')
  @argument('--pattern', default='**/*', help='file pattern')
  @subcommand('revert', help='revert a proposal via store checkout')
  def revert(self, ctx: Any, args: argparse.Namespace) -> None:
    proposal_id = args.proposal_id
    if not proposal_id:
      ctx.output.error('--proposal-id is required')
      return

    exp_dir = ctx.experiment_dir()
    proposals = read_proposals(exp_dir)
    proposal = next((p for p in proposals if p.proposal_id == proposal_id), None)
    if not proposal:
      ctx.output.error(f'proposal {proposal_id!r} not found')
      return

    restore_epoch = ctx.epoch or max(proposal.epoch - 1, 0)
    if restore_epoch <= 0:
      ctx.output.error('cannot revert: no prior epoch to restore')
      return

    source = args.source
    if not source:
      ctx.output.error('--source is required for revert')
      return

    store_root = (
      Path(args.store).resolve()
      if args.store
      else paths.store(
        ctx.workspace,
        ctx.project,
      )
    )
    pattern = args.pattern or '**/*'
    params = [PathParameter(source=str(Path(source).expanduser().resolve()), pattern=pattern)]

    try:
      store = FileStore(store_root, ctx.experiment, params)
      store.checkout(restore_epoch)
    except Exception as exc:
      ctx.output.error(f'revert failed: {exc}')
      return

    ctx.output.result(
      {
        'proposal_id': proposal_id,
        'status': 'reverted',
        'restored_epoch': restore_epoch,
      }
    )

  @subcommand('list', help='list all proposals')
  def list_proposals(self, ctx: Any, args: argparse.Namespace) -> None:
    exp_dir = ctx.experiment_dir()
    proposals = read_proposals(exp_dir)
    ctx.output.result(
      {
        'proposals': [p.to_dict() for p in proposals],
        'count': len(proposals),
      }
    )
