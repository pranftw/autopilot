"""CLI for the agent harness optimization project.

Registers ``HarnessCLI`` as the ``harness`` project CLI via
``__init_subclass__(project='harness')``. Environment selected via
``HARNESS_ENV`` (default: dev).

Overrides ``run_direct`` so that ``ctx.trainer`` carries store, policy,
and callbacks needed by ``optimize`` and other commands.

``self.judge`` is a live ``HarnessJudge`` instance enabling ``ai judge``
commands. ``self.generator`` is intentionally ``None`` (tau-bench JSONL
scenarios; no ``GeneratorAgent``).

``HarnessOptimizeCommand`` extends ``OptimizeCommand`` with
``--use-judge`` / ``--no-judge`` flags on the ``loop``, ``train``,
``validate``, ``test``, and ``resume`` subcommands so external agents
can toggle ``JudgeLoss`` vs ``HarnessLoss`` without editing code.
"""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.optimize import OptimizeCommand, Resume, Test, Train, Validate
from autopilot.cli.context import CLIContext
from autopilot.cli.main import AutoPilotCLI
from autopilot.cli.primitives import Argument, argument, subcommand
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.config import AutoPilotConfig
from autopilot.core.trainer.trainer import Trainer
from autopilot.policy.quality_first import QualityFirstPolicy
from harness.callbacks import DeployCallback, MetricsWriterCallback, OptimizerContextCallback
from harness.data import HarnessDataModule
from harness.environments import get_environment_config
from harness.judge import HarnessJudge
from harness.module import HarnessModule
from harness.trainer import next_slug
from pathlib import Path
from typing import Any
import argparse
import os
import sys

_USE_JUDGE_KWARGS: dict[str, Any] = {
  'action': 'store_true',
  'default': False,
  'dest': 'use_judge',
  'help': 'force judge loss path (default when neither flag is passed)',
}
"""Shared kwargs for ``--use-judge`` across Argument descriptors and @argument decorators."""

_NO_JUDGE_KWARGS: dict[str, Any] = {
  'action': 'store_true',
  'default': False,
  'dest': 'no_judge',
  'help': 'use heuristic HarnessLoss path',
}
"""Shared kwargs for ``--no-judge`` across Argument descriptors and @argument decorators."""


def _make_use_judge_arg() -> Argument:
  """Create a ``--use-judge`` argument descriptor."""
  return Argument('--use-judge', **_USE_JUDGE_KWARGS)


def _make_no_judge_arg() -> Argument:
  """Create a ``--no-judge`` argument descriptor."""
  return Argument('--no-judge', **_NO_JUDGE_KWARGS)


def _validate_judge_flags(ctx: CLIContext, args: argparse.Namespace) -> None:
  """Reject conflicting ``--use-judge`` and ``--no-judge`` flags.

  Args:
    ctx: CLI context (used for ``ctx.fail``).
    args: Parsed namespace with ``use_judge`` and ``no_judge`` attributes.
  """
  if args.use_judge and args.no_judge:
    ctx.fail('cannot pass both --use-judge and --no-judge')


class HarnessTrain(Train):
  """Train command with ``--use-judge`` / ``--no-judge`` flags."""

  use_judge_flag = _make_use_judge_arg()
  no_judge_flag = _make_no_judge_arg()

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run training with optional judge-mode override."""
    _validate_judge_flags(ctx, args)
    return super().forward(ctx, args)


class HarnessValidate(Validate):
  """Validate command with ``--use-judge`` / ``--no-judge`` flags."""

  use_judge_flag = _make_use_judge_arg()
  no_judge_flag = _make_no_judge_arg()

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run validation with optional judge-mode override."""
    _validate_judge_flags(ctx, args)
    return super().forward(ctx, args)


class HarnessTest(Test):
  """Test command with ``--use-judge`` / ``--no-judge`` flags."""

  use_judge_flag = _make_use_judge_arg()
  no_judge_flag = _make_no_judge_arg()

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run test split with optional judge-mode override."""
    _validate_judge_flags(ctx, args)
    return super().forward(ctx, args)


class HarnessResume(Resume):
  """Resume command with ``--use-judge`` / ``--no-judge`` flags.

  Inherits checkpoint and max-epochs arguments from ``Resume``.
  """

  use_judge_flag = _make_use_judge_arg()
  no_judge_flag = _make_no_judge_arg()

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Resume training with optional judge-mode override."""
    _validate_judge_flags(ctx, args)
    return super().forward(ctx, args)


class HarnessOptimizeCommand(OptimizeCommand):
  """Harness-specific optimize command with ``--use-judge`` / ``--no-judge`` flags.

  Extends ``OptimizeCommand`` subcommands (``loop``, ``train``, ``validate``,
  ``test``, ``resume``) with a mutually exclusive judge-mode pair.  Default
  when neither flag is passed: ``use_judge=True``.
  """

  def __init__(self) -> None:
    """Wire harness-specific subcommands with judge flags."""
    super().__init__()
    self.train = HarnessTrain()
    self.validate = HarnessValidate()
    self.test = HarnessTest()
    self.resume = HarnessResume()

  @argument('--max-epochs', type=int, default=10, help='maximum training epochs')
  @argument('--use-judge', **_USE_JUDGE_KWARGS)
  @argument('--no-judge', **_NO_JUDGE_KWARGS)
  @subcommand('loop', help_text='Run optimization loop')
  def loop(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run the full optimization loop with optional judge-mode override."""
    if args.use_judge and args.no_judge:
      ctx.fail('cannot pass both --use-judge and --no-judge')
    return super().loop(ctx, args)


def _resolve_use_judge(args: argparse.Namespace) -> bool:
  """Resolve the effective ``use_judge`` boolean from parsed CLI args.

  When ``--no-judge`` is passed, returns ``False``.  Otherwise returns
  ``True`` (the default, and also when ``--use-judge`` is explicit).

  Args:
    args: Parsed argparse Namespace (must have ``use_judge`` and
      ``no_judge`` attributes).

  Returns:
    Resolved judge mode boolean.

  Raises:
    ValueError: When both ``--use-judge`` and ``--no-judge`` are set.
  """
  use = args.use_judge
  no = args.no_judge
  if use and no:
    raise ValueError('cannot pass both --use-judge and --no-judge')
  if no:
    return False
  return True


class HarnessCLI(AutoPilotCLI, project='harness'):
  """CLI for the agent harness optimization project.

  Wires module, datamodule, store, policy, and callbacks. Environment
  selected via HARNESS_ENV (default: dev).

  ``self.judge`` is a live ``HarnessJudge`` enabling ``ai judge`` commands.
  ``self.generator`` is intentionally ``None`` (tau-bench JSONL scenarios;
  no ``GeneratorAgent``).
  """

  def __init__(self) -> None:
    """Build all harness components from environment config."""
    super().__init__()
    root = Path(__file__).resolve().parent.parent
    self._harness_root = str(root / 'harness')
    self._root = root
    env_name = os.environ.get('HARNESS_ENV', 'dev')
    self._env_config = get_environment_config(env_name)

    self.generator = None
    self.judge = HarnessJudge()

    self.optimize = HarnessOptimizeCommand()

    self.module = HarnessModule(
      self._harness_root,
      model=self._env_config.model,
      use_judge=self._env_config.use_judge,
      max_turns=self._env_config.max_turns,
    )
    self.datamodule = HarnessDataModule(str(root / 'harness' / 'scenarios'))

    store_path = root / '.autopilot' / 'store'
    self.config = AutoPilotConfig(workspace=root)
    self.config.store_path = store_path
    self.store = FileStore(self.config)
    self.store.register_parameters(dict(self.module.named_parameters()))

    self.policy = QualityFirstPolicy(gates=self._env_config.gates)

    self.callbacks = [
      StoreCheckpointCallback(),
      MetricsWriterCallback(),
      OptimizerContextCallback(),
      DeployCallback(),
    ]

  def _ensure_module_use_judge(self, use_judge: bool) -> None:
    """Rebuild ``HarnessModule`` if current judge mode differs from target.

    When mode changes, reconstructs the module with the same root and model,
    then re-registers parameters on the store.

    Args:
      use_judge: Desired judge mode.
    """
    if self.module.use_judge == use_judge:
      return
    self.module = HarnessModule(
      self._harness_root,
      model=self._env_config.model,
      use_judge=use_judge,
      max_turns=self._env_config.max_turns,
    )
    self.store.register_parameters(dict(self.module.named_parameters()))

  def run_direct(self, *, argv: list[str] | None = None) -> None:
    """Parse and dispatch with a Trainer wired for store, policy, and callbacks.

    When an optimize subcommand is invoked, resolves ``--use-judge`` /
    ``--no-judge`` and rebuilds the module if needed before building
    the trainer context.
    """
    parser = self.build_parser()
    parser.set_defaults(use_judge=False, no_judge=False)
    args = parser.parse_args(argv)
    if not args.command:
      parser.print_help()
      sys.exit(2)

    if args.command == 'optimize':
      try:
        resolved = _resolve_use_judge(args)
      except ValueError as exc:
        # ctx.fail is unavailable here (context not yet built)
        print(str(exc), file=sys.stderr)
        sys.exit(2)
      self._ensure_module_use_judge(resolved)

    ctx = self.build_context(args)
    ctx.generator = self.generator
    ctx.judge = self.judge
    ctx.module = self.module
    ctx.datamodule = self.datamodule
    if self.module is not None:
      experiment = AutoPilotExperiment(
        experiment_id=ctx.experiment or next_slug(self.config.store_path),
      )
      experiment.store = self.store
      ctx.trainer = Trainer(
        self.callbacks,
        dry_run=ctx.dry_run,
        policy=self.policy,
        store=self.store,
        config=self.config,
        experiment=experiment,
      )
    self.dispatch(ctx, args, argv=argv)
