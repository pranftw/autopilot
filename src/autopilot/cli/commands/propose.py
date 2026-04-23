"""Propose command -- create, verify, revert, and list proposals."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store import FileStore
from autopilot.cli.command import Command, argument, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.artifacts.epoch import DataArtifact
from autopilot.core.artifacts.experiment import BaselineArtifact
from autopilot.core.comparison import load_metric_comparison
from autopilot.core.proposal import (
  ChangeProposal,
  ProposalVerdict,
  read_proposals,
  record_proposal,
  record_verdict,
)
from pathlib import Path
import argparse
import autopilot.core.paths as paths
import uuid


class ProposeCommand(Command):
  """Manage optimization proposals."""

  name = 'propose'
  help = 'manage proposals'

  @argument('--target', default=None, help='target node for proposal')
  @argument('--hypothesis', default=None, help='hypothesis text')
  @argument('--category', default=None, help='proposal category')
  @subcommand('create', help='create a new proposal')
  def create(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Create a new change proposal and record it."""
    exp_dir = ctx.experiment_dir()
    proposal = ChangeProposal(
      proposal_id=str(uuid.uuid4())[:8],
      hypothesis=args.hypothesis if args.hypothesis else 'no hypothesis provided',
      target_node=args.target,
      change_type=args.category if args.category else 'general',
      epoch=ctx.epoch,
      status='proposed',
    )
    record_proposal(exp_dir, proposal)
    ctx.output.result({'proposal_id': proposal.proposal_id, 'status': 'created'})

  @argument('--proposal-id', default=None, help='proposal ID to verify')
  @subcommand('verify', help='verify a proposal against metrics')
  def verify(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Verify a proposal against metric comparisons."""
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

    mc = load_metric_comparison(exp_dir, epoch)

    if mc and (mc.regression_detected or mc.is_mixed):
      regressed_categories = [r['metric'] for r in mc.regressions]
      target_match = (
        proposal.target_node and proposal.target_node in str(regressed_categories)
      ) or (proposal.change_type and proposal.change_type in str(regressed_categories))
      if target_match or not proposal.target_node:
        verdict = ProposalVerdict(
          proposal_id=proposal_id,
          verdict='regression_after_change',
          items_tested=len(mc.regressions),
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

    baseline = BaselineArtifact().read_raw(exp_dir)
    trace_data = DataArtifact().read_raw(exp_dir, epoch=epoch)

    items_tested = len(trace_data)

    if mc and mc.improvement_detected:
      verdict_str = 'fix_confirmed'
    elif baseline and mc:
      candidate = mc.candidate_metrics
      improved = any(candidate[key] > baseline[key] for key in baseline if key in candidate)
      verdict_str = 'fix_confirmed' if improved else 'inconclusive'
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

  @argument('--proposal-id', default=None, help='proposal ID to revert')
  @argument('--source', default=None, help='source directory for store')
  @argument('--store', default=None, help='store root path')
  @argument('--pattern', default='**/*', help='file pattern')
  @subcommand('revert', help='revert a proposal via store checkout')
  def revert(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Revert a proposal by restoring the store to a prior epoch."""
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
    except (FileNotFoundError, OSError) as exc:
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
  def list_proposals(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List all recorded proposals for the experiment."""
    exp_dir = ctx.experiment_dir()
    proposals = read_proposals(exp_dir)
    ctx.output.result(
      {
        'proposals': [p.to_dict() for p in proposals],
        'count': len(proposals),
      }
    )
