"""Optimization pipeline: preflight through deploy, train, validation, and tuning loop.

Command handlers are thin wrappers around Trainer. They resolve CLI arguments,
build params, and delegate to module directly. Execution and callback dispatch
are the Trainer's responsibility.

JSON result shapes:

  optimize loop (--json):
    {final_metrics, stop_reason, last_good_epoch, total_epochs, epochs}

  optimize train / validate (--json):
    {success, metrics, error_message, feedback}

  optimize set-hparams (--json):
    {experiment_id, hparams}

  set-hparams persists hyperparameters under the ``hparams`` key inside the
  experiment's ``notes`` field (JSON object string). Existing notes content is
  preserved and merged.
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import (
  journal_user_context,
  load_forest,
  require_active_tree,
  require_experiment_node,
)
from autopilot.cli.messages import (
  MSG_EXPERIMENT_SLUG_REQUIRED,
  MSG_NO_MODULE_CONFIGURED,
  MSG_NO_TRAINER_CONFIGURED,
)
from autopilot.cli.primitives import Argument, argument, subcommand
from autopilot.core.callbacks.cost import CostTrackerCallback
from autopilot.core.callbacks.data_recorder import DataRecorderCallback
from autopilot.core.callbacks.diagnostics import DiagnosticsCallback
from autopilot.core.callbacks.run_state import RunStateCallback
from autopilot.core.diagnostics import Diagnostics
from autopilot.core.errors import PreflightError
from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.trainer.trainer import Trainer
from autopilot.tracking.commands import create_command_record, log_command
from autopilot.tracking.io import read_json
from pathlib import Path
from typing import Any
import argparse
import json
import logging

logger = logging.getLogger(__name__)


class Train(Command):
  """Runs training on a split with optional item limit."""

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
    split = 'train' if ctx.split is None else ctx.split
    limit = args.limit
    exp_dir, _ = _prepare_optimize_run(ctx)

    log_optimize_command(exp_dir, 'train', ctx)
    limit_msg = f', limit {limit}' if limit else ''
    ctx.output.info(f'Training on {split} split (epoch {ctx.epoch}{limit_msg})...')

    params: dict[str, Any] = {'split': split, 'command': 'train'}
    if limit:
      params['limit'] = limit
    if ctx.epoch:
      params['epoch'] = ctx.epoch

    if not ctx.module:
      ctx.fail(MSG_NO_MODULE_CONFIGURED)
    runtime_ctx = _build_runtime_ctx(ctx, exp_dir)
    observation = ctx.module(runtime_ctx, params)
    ctx.output.result(
      {
        'command': 'train',
        'success': observation.success,
        'metrics': observation.metrics or {},
        'error_message': observation.error_message if not observation.success else None,
        'feedback': observation.feedback,
      },
      ok=observation.success,
    )


class Deploy(Command):
  """Deploys experiment artifacts and captures the deploy ID."""

  name = 'deploy'
  help = 'Deploy experiment artifacts'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Deploy experiment artifacts and capture the deploy ID."""
    exp_dir, _ = _prepare_optimize_run(ctx)

    log_optimize_command(exp_dir, 'deploy', ctx)
    ctx.output.info('Deploying...')

    if not ctx.module:
      ctx.fail(MSG_NO_MODULE_CONFIGURED)
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
  """Runs validation on the val split."""

  name = 'validate'
  help = 'Run validation'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run validation on the val split."""
    exp_dir, _ = _prepare_optimize_run(ctx)

    log_optimize_command(exp_dir, 'validate', ctx)
    ctx.output.info('Validating on val split...')

    params: dict[str, Any] = {'split': 'val', 'command': 'validate'}
    if ctx.epoch:
      params['epoch'] = ctx.epoch

    if not ctx.module:
      ctx.fail(MSG_NO_MODULE_CONFIGURED)
    runtime_ctx = _build_runtime_ctx(ctx, exp_dir)
    observation = ctx.module(runtime_ctx, params)
    ctx.output.result(
      {
        'command': 'validate',
        'success': observation.success,
        'metrics': observation.metrics or {},
        'error_message': observation.error_message if not observation.success else None,
        'feedback': observation.feedback,
      },
      ok=observation.success,
    )


class Test(Command):
  """Runs the test split and reports success."""

  name = 'test'
  help = 'Run test split'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run the test split and report success."""
    exp_dir, _ = _prepare_optimize_run(ctx)

    log_optimize_command(exp_dir, 'test', ctx)
    ctx.output.info('Running test split...')

    params: dict[str, Any] = {'split': 'test', 'command': 'test'}
    if ctx.epoch:
      params['epoch'] = ctx.epoch

    if not ctx.module:
      ctx.fail(MSG_NO_MODULE_CONFIGURED)
    runtime_ctx = _build_runtime_ctx(ctx, exp_dir)
    observation = ctx.module(runtime_ctx, params)
    ctx.output.result({'command': 'test', 'success': observation.success}, ok=observation.success)


class Resume(Command):
  """Resume training from a checkpoint file.

  Calls ``Trainer.fit(..., ckpt_path=...)`` with the resolved checkpoint path.
  Reuses the same module/experiment/trainer context as ``optimize loop``.

  JSON result shape::

    {resumed_from: str, epochs_run: int, final_epoch: int}
  """

  name = 'resume'
  help = 'Resume training from a checkpoint'
  ckpt = Argument('ckpt', help='path to checkpoint file')
  max_epochs = Argument('--max-epochs', type=int, default=10, help='maximum training epochs')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Resume training from the specified checkpoint.

    Resolves the checkpoint path relative to workspace or cwd, validates
    existence, then delegates to ``Trainer.fit`` with ``ckpt_path``.
    """
    ckpt_raw = args.ckpt
    ckpt_path = Path(ckpt_raw)
    if not ckpt_path.is_absolute():
      ckpt_path = Path.cwd() / ckpt_path
    ckpt_path = ckpt_path.resolve()

    if not ckpt_path.exists():
      ctx.fail(f'checkpoint file not found: {ckpt_path}')

    _, trainer = _prepare_optimize_run(ctx)
    if not ctx.module:
      ctx.fail(MSG_NO_MODULE_CONFIGURED)

    ctx.output.info(f'Resuming from {ckpt_path}...')

    result = trainer.fit(
      ctx.module,
      datamodule=ctx.datamodule,
      max_epochs=args.max_epochs,
      ckpt_path=ckpt_path,
    )

    epochs = result.get('epochs', [])
    total = result.get('total_epochs', len(epochs))
    final_epoch = epochs[-1].get('epoch', total - 1) if epochs else 0

    ctx.output.result(
      {
        'resumed_from': str(ckpt_path),
        'epochs_run': total,
        'final_epoch': final_epoch,
      }
    )


class OptimizeCommand(Command):
  """``autopilot optimize`` group: train, deploy, validate, test, preflight, loop, resume."""

  name = 'optimize'
  help = 'Optimization pipeline'

  def __init__(self) -> None:
    """Wire core optimize subcommands (train, deploy, validate, test, resume)."""
    super().__init__()
    self.train = Train()
    self.deploy = Deploy()
    self.validate = Validate()
    self.test = Test()
    self.resume = Resume()

  @subcommand('preflight', help_text='Run preflight checks')
  def preflight(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run preflight checks on all module children that support them.

    Raises:
      PreflightError: When any check fails and ``ctx.dry_run`` is false.
    """
    exp_dir, _ = _prepare_optimize_run(ctx)

    ctx.output.info('Running backend preflight checks...')
    all_errors: list[str] = []
    if ctx.module:
      for name, child in ctx.module.named_children():
        if hasattr(child, 'preflight'):
          errors = child.preflight(_build_runtime_ctx(ctx, exp_dir))
          for err in errors:
            all_errors.append(f'[{name}/{type(child).__name__}] {err}')
            ctx.output.warn(f'[{name}/{type(child).__name__}] {err}')

    passed = not all_errors
    log_optimize_command(exp_dir, 'preflight', ctx)

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
    help='JSON string of hyperparameter key=value pairs to persist under experiment notes',
  )
  @subcommand('set-hparams', help_text='Apply hyperparameter updates')
  def set_hparams(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Persist hyperparameters into the experiment's notes field.

    Parses --values JSON, merges into the existing notes object under
    the top-level 'hparams' key, and saves the forest. Experiment notes
    are stored as a JSON object string with structure:
    ``{"hparams": {...}, ...other_keys...}``.
    """
    ctx.output.info('Setting hyperparameters...')

    updates: dict[str, Any] = {}
    if args.values:
      try:
        updates = json.loads(args.values)
      except json.JSONDecodeError as e:
        ctx.fail(
          f'invalid JSON for --values ({type(e).__name__}): {e};'
          ' expected a JSON object like {"key": "value"}'
        )
    elif ctx.hyperparams_file:
      loaded = read_json(Path(ctx.hyperparams_file))
      if loaded and isinstance(loaded, dict):
        updates = loaded

    if not updates:
      ctx.output.warn('no hyperparameter updates provided')
      ctx.output.result({'updated': False})
      return

    if not ctx.experiment:
      ctx.fail(MSG_EXPERIMENT_SLUG_REQUIRED)

    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)
    node = require_experiment_node(ctx, tree, ctx.experiment)
    exp = node.experiment
    journal_user_context(ctx, exp, args)

    existing: dict[str, Any] = {}
    if exp.notes is not None:
      try:
        parsed = json.loads(exp.notes)
        if isinstance(parsed, dict):
          existing = parsed
      except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(
          'experiment %r notes are not valid JSON, overwriting: %s',
          exp.id,
          exc,
        )

    current_hparams = existing.get('hparams', {})
    if isinstance(current_hparams, dict):
      current_hparams.update(updates)
    else:
      current_hparams = updates
    existing['hparams'] = current_hparams
    exp.notes = json.dumps(existing)
    forest.save()

    ctx.output.result({'experiment_id': exp.id, 'hparams': current_hparams})

  @argument('--max-epochs', type=int, default=10, help='maximum training epochs')
  @subcommand('loop', help_text='Run optimization loop')
  def loop(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run the full optimization loop with orchestration and callbacks."""
    exp_dir, trainer = _prepare_optimize_run(ctx)

    ctx.output.info('Starting optimization loop...')

    if not ctx.module:
      ctx.fail(MSG_NO_MODULE_CONFIGURED)

    loop_trainer = _build_loop_trainer(trainer, exp_dir)
    runtime_ctx = _build_runtime_ctx(ctx, exp_dir)

    result = loop_trainer.fit(
      ctx.module,
      datamodule=ctx.datamodule,
      max_epochs=args.max_epochs,
      ctx=runtime_ctx,
    )

    epochs = result.get('epochs', [])
    final_metrics = epochs[-1].get('metrics', {}) if epochs else {}

    ctx.output.result(
      {
        'final_metrics': final_metrics,
        'stop_reason': result.get('stop_reason'),
        'last_good_epoch': result.get('last_good_epoch'),
        'total_epochs': result.get('total_epochs', 0),
        'epochs': epochs,
      }
    )


def _build_loop_trainer(trainer: Trainer, exp_dir: Path) -> Trainer:
  """Build a Trainer wired for the orchestrated optimization loop.

  Args:
    trainer: Base trainer from the CLI context.
    exp_dir: Experiment directory for callbacks.

  Returns:
    Configured Trainer with orchestrator and stage callbacks.
  """
  orchestrator = EpochOrchestrator(config=OrchestratorConfig(plateau_window=0))
  stage_cbs = [
    DataRecorderCallback(exp_dir),
    DiagnosticsCallback(Diagnostics(exp_dir)),
    RunStateCallback(exp_dir),
    CostTrackerCallback(exp_dir),
  ]
  return Trainer(
    callbacks=list(trainer.callbacks) + stage_cbs,
    loop=orchestrator,
    dry_run=trainer.dry_run,
    logger=trainer.logger,
    policy=trainer.policy,
    experiment=trainer.experiment,
    config=trainer.config,
    accumulate_grad_batches=trainer.accumulate_grad_batches,
    store=trainer.store,
    tree=trainer.tree,
    forest=trainer.forest,
  )


def _resolve_experiment(ctx: CLIContext) -> Path:
  """Resolve experiment directory from CLI context.

  Returns:
    Experiment directory path.

  Raises:
    ValueError: If no experiment slug is set on the context.
  """
  slug = ctx.experiment
  if not slug:
    raise ValueError(MSG_EXPERIMENT_SLUG_REQUIRED)
  return ctx.experiment_path(slug)


def _get_trainer(ctx: CLIContext) -> Trainer:
  """Return the Trainer from CLI context.

  Returns:
    The Trainer instance.

  Raises:
    ValueError: If no trainer is configured.
  """
  if ctx.trainer is not None:
    return ctx.trainer
  raise ValueError(MSG_NO_TRAINER_CONFIGURED)


def _prepare_optimize_run(ctx: CLIContext) -> tuple[Path, Trainer]:
  """Shared bootstrap for optimize handlers that need experiment + trainer.

  Returns:
    Tuple of (experiment directory, trainer).
  """
  exp_dir = _resolve_experiment(ctx)
  trainer = _get_trainer(ctx)
  return exp_dir, trainer


def _build_runtime_ctx(ctx: CLIContext, _exp_dir: Path) -> dict[str, Any]:
  """Build runtime context for module forward().

  Callers pass resolved ``_exp_dir`` for symmetry with logging helpers even though this
  function only reads workspace flags from ``ctx``.

  Args:
    ctx: CLI context supplying workspace paths and flags.

  Returns:
    Dict passed into module ``forward`` with workspace and ``dry_run``.
  """
  runtime_ctx: dict[str, Any] = {}
  runtime_ctx['workspace'] = str(ctx.workspace)
  runtime_ctx['dry_run'] = ctx.dry_run
  return runtime_ctx


def log_optimize_command(
  exp_dir: Path,
  subcommand: str,
  ctx: CLIContext,
) -> None:
  """Log the CLI invocation for this optimize subcommand."""
  args = ['optimize', subcommand]
  if ctx.experiment is not None:
    args.extend(['--experiment', ctx.experiment])
  record = create_command_record(
    command='autopilot',
    args=args,
  )
  log_command(exp_dir, record)
