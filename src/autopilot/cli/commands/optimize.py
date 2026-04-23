"""Optimization pipeline: preflight through deploy, train, validation, and tuning loop.

Command handlers are thin wrappers around Trainer. They resolve CLI arguments,
build params, and delegate to module directly. Execution and callback dispatch
are the Trainer's responsibility.
"""

from autopilot.cli.command import Argument, Command, argument, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.callbacks.cost import CostTrackerCallback
from autopilot.core.callbacks.data_recorder import DataRecorderCallback
from autopilot.core.callbacks.diagnostics import DiagnosticsCallback
from autopilot.core.callbacks.memory import MemoryCallback
from autopilot.core.callbacks.run_state import RunStateCallback
from autopilot.core.config import resolve_experiment_dir
from autopilot.core.diagnostics import Diagnostics
from autopilot.core.errors import PreflightError
from autopilot.core.hyperparams import load_hyperparams, update_hyperparams
from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.memory import FileMemory
from autopilot.core.summary import build_experiment_summary, write_experiment_summary
from autopilot.core.trainer import Trainer
from autopilot.tracking.commands import create_command_record, log_command
from autopilot.tracking.io import read_json
from autopilot.tracking.manifest import load_manifest
from pathlib import Path
from typing import Any
import argparse
import json


class Train(Command):
  name = 'train'
  help = 'Run training'
  limit = Argument(
    '--limit',
    type=int,
    default=0,
    metavar='N',
    help='use at most N items from the train split (0 = all)',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run training on the specified split with optional item limit."""
    split = ctx.split or 'train'
    limit = args.limit
    exp_dir = _resolve_experiment(ctx)
    _get_trainer(ctx)

    _log_optimize_command(exp_dir, 'train', ctx)
    limit_msg = f', limit {limit}' if limit else ''
    ctx.output.info(f'Training on {split} split (epoch {ctx.epoch}{limit_msg})...')

    params: dict[str, Any] = {'split': split, 'command': 'train'}
    if limit:
      params['limit'] = limit
    if ctx.epoch:
      params['epoch'] = ctx.epoch

    if not ctx.module:
      ctx.output.error('no module configured')
      return
    runtime_ctx = _build_runtime_ctx(ctx, exp_dir)
    observation = ctx.module(runtime_ctx, params)
    ctx.output.result({'command': 'train', 'success': observation.success}, ok=observation.success)


class Deploy(Command):
  name = 'deploy'
  help = 'Deploy experiment artifacts'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Deploy experiment artifacts and capture the deploy ID."""
    exp_dir = _resolve_experiment(ctx)
    _get_trainer(ctx)

    _log_optimize_command(exp_dir, 'deploy', ctx)
    ctx.output.info('Deploying...')

    if not ctx.module:
      ctx.output.error('no module configured')
      return
    params: dict[str, Any] = {'command': 'deploy'}
    runtime_ctx = _build_runtime_ctx(ctx, exp_dir)
    observation = ctx.module(runtime_ctx, params)

    if observation.success:
      extracted = observation.metadata.get('extracted_value')
      if extracted:
        ctx.output.info(f'Captured deploy ID: {extracted}')
      elif not ctx.dry_run:
        ctx.output.warn('deploy succeeded but no deploy ID was extracted')

    ctx.output.result(
      {'command': 'deploy', 'success': observation.success},
      ok=observation.success,
    )


class Validate(Command):
  name = 'validate'
  help = 'Run validation'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run validation on the val split."""
    exp_dir = _resolve_experiment(ctx)
    _get_trainer(ctx)

    _log_optimize_command(exp_dir, 'validate', ctx)
    ctx.output.info('Validating on val split...')

    params: dict[str, Any] = {'split': 'val', 'command': 'validate'}
    if ctx.epoch:
      params['epoch'] = ctx.epoch

    if not ctx.module:
      ctx.output.error('no module configured')
      return
    runtime_ctx = _build_runtime_ctx(ctx, exp_dir)
    observation = ctx.module(runtime_ctx, params)
    ctx.output.result(
      {'command': 'validate', 'success': observation.success},
      ok=observation.success,
    )


class Test(Command):
  name = 'test'
  help = 'Run test split'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run the test split and report success."""
    exp_dir = _resolve_experiment(ctx)
    _get_trainer(ctx)

    _log_optimize_command(exp_dir, 'test', ctx)
    ctx.output.info('Running test split...')

    params: dict[str, Any] = {'split': 'test', 'command': 'test'}
    if ctx.epoch:
      params['epoch'] = ctx.epoch

    if not ctx.module:
      ctx.output.error('no module configured')
      return
    runtime_ctx = _build_runtime_ctx(ctx, exp_dir)
    observation = ctx.module(runtime_ctx, params)
    ctx.output.result({'command': 'test', 'success': observation.success}, ok=observation.success)


class OptimizeCommand(Command):
  name = 'optimize'
  help = 'Optimization pipeline'

  def __init__(self) -> None:
    super().__init__()
    self.train = Train()
    self.deploy = Deploy()
    self.validate = Validate()
    self.test = Test()

  @subcommand('preflight', help='Run preflight checks')
  def preflight(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run preflight checks on all module children that support them."""
    exp_dir = _resolve_experiment(ctx)
    _get_trainer(ctx)

    ctx.output.info('Running backend preflight checks...')
    all_errors: list[str] = []
    if ctx.module:
      for name, child in ctx.module.named_children():
        if hasattr(child, 'preflight'):
          errors = child.preflight(_build_runtime_ctx(ctx, exp_dir))
          for err in errors:
            all_errors.append(f'[{name}/{type(child).__name__}] {err}')
            ctx.output.warn(f'[{name}/{type(child).__name__}] {err}')

    passed = len(all_errors) == 0
    _log_optimize_command(exp_dir, 'preflight', ctx)

    ctx.output.result(
      {
        'command': 'preflight',
        'passed': passed,
        'backend_errors': all_errors,
        'total_issues': len(all_errors),
      },
      ok=passed,
    )

    if not passed and not ctx.dry_run:
      raise PreflightError(all_errors)

  @argument(
    '--values',
    default=None,
    metavar='JSON',
    help='JSON string of hyperparameter key=value pairs',
  )
  @subcommand('set-hparams', help='Apply hyperparameter updates')
  def set_hparams(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Apply hyperparameter updates from JSON string or file."""
    exp_dir = _resolve_experiment(ctx)
    ctx.output.info('Setting hyperparameters...')

    updates: dict[str, Any] = {}
    if args.values:
      updates = json.loads(args.values)
    elif ctx.hyperparams_file:
      loaded = read_json(Path(ctx.hyperparams_file))
      if loaded and isinstance(loaded, dict):
        updates = loaded

    if not updates:
      ctx.output.warn('no hyperparameter updates provided')
      ctx.output.result({'updated': False})
      return

    hparams = update_hyperparams(exp_dir, updates)
    ctx.output.result(
      {
        'version': hparams.version,
        'values': hparams.values,
        'locked': hparams.locked,
      }
    )

  @argument('--max-epochs', type=int, default=10, help='maximum training epochs')
  @argument(
    '--strategy',
    default='conservative',
    choices=['conservative', 'aggressive', 'exploratory'],
    help='orchestration strategy',
  )
  @subcommand('loop', help='Run optimization loop')
  def loop(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run the full optimization loop with orchestration and callbacks."""
    exp_dir = _resolve_experiment(ctx)
    trainer = _get_trainer(ctx)

    ctx.output.info('Starting optimization loop...')
    checkpoint = load_manifest(exp_dir, strict=False)
    if not checkpoint:
      ctx.output.warn('no manifest found; create experiment first')
      ctx.output.result({'error': 'no_experiment'}, ok=False)
      return

    if checkpoint.is_decided:
      ctx.output.info(f'experiment already decided: {checkpoint.decision}')
      ctx.output.result(
        {
          'decision': checkpoint.decision,
          'message': 'nothing further to run for this experiment',
        }
      )
      return

    max_epochs = args.max_epochs
    strategy = args.strategy

    orch_config = OrchestratorConfig(strategy=strategy)
    orchestrator = EpochOrchestrator(config=orch_config)

    memory = FileMemory(exp_dir)
    cost_tracker = CostTrackerCallback(exp_dir)
    stage_cbs = [
      DataRecorderCallback(exp_dir),
      DiagnosticsCallback(Diagnostics(exp_dir)),
      MemoryCallback(memory),
      RunStateCallback(exp_dir),
      cost_tracker,
    ]

    all_callbacks = list(trainer.callbacks) + stage_cbs
    loop_trainer = Trainer(
      callbacks=all_callbacks,
      loop=orchestrator,
      dry_run=trainer.dry_run,
      logger=trainer.logger,
      policy=trainer.policy,
      experiment=trainer.experiment,
      accumulate_grad_batches=trainer.accumulate_grad_batches,
    )

    runtime_ctx = _build_runtime_ctx(ctx, exp_dir)
    runtime_ctx['strategy'] = strategy

    datamodule = ctx.datamodule
    result = loop_trainer.fit(
      ctx.module,
      datamodule=datamodule,
      max_epochs=max_epochs,
      ctx=runtime_ctx,
    )

    summary = build_experiment_summary(exp_dir, result, cost_tracker=cost_tracker)
    write_experiment_summary(exp_dir, summary)

    ctx.output.result(
      {
        'total_epochs': summary.total_epochs,
        'final_metrics': summary.final_metrics,
        'best_epoch': summary.best_epoch,
        'stop_reason': result.get('stop_reason'),
        'comparisons': len(summary.comparisons),
        'memory_entries': summary.memory_entries,
      }
    )


def _resolve_experiment(ctx: CLIContext) -> Path:
  slug = ctx.experiment
  if not slug:
    raise ValueError('experiment slug required (--experiment)')
  return resolve_experiment_dir(ctx.workspace, slug, ctx.project)


def _get_trainer(ctx: CLIContext) -> Trainer:
  if ctx.trainer is not None:
    return ctx.trainer
  raise ValueError('no trainer configured; ensure the project passes a Module to run()')


def _build_runtime_ctx(ctx: CLIContext, exp_dir: Path) -> dict[str, Any]:
  """Build runtime context for module forward()."""
  runtime_ctx: dict[str, Any] = {}
  runtime_ctx['workspace'] = str(ctx.workspace)
  runtime_ctx['dry_run'] = ctx.dry_run

  hparams = load_hyperparams(exp_dir)
  runtime_ctx['hyperparams'] = hparams.values

  return runtime_ctx


def _log_optimize_command(
  exp_dir: Path,
  subcommand: str,
  ctx: CLIContext,
) -> None:
  """Log the CLI invocation for this optimize subcommand."""
  record = create_command_record(
    command='autopilot',
    args=['optimize', subcommand, '--experiment', ctx.experiment],
  )
  log_command(exp_dir, record)
