"""Trainer: orchestrates experiment optimization with callbacks.

See ``EpochLoop`` for per-epoch ordering. Checkpoint resume uses
``CheckpointIO`` / ``LoopConfig.min_epoch``. Constructor kwargs are documented on
:class:`Trainer`.
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.checkpoint import CheckpointIO
from autopilot.core.config import Config
from autopilot.core.experiment import Experiment
from autopilot.core.forest import Forest
from autopilot.core.logger import Logger
from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.loops.loop import Loop
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.module.module import Module
from autopilot.core.optimizer import Optimizer
from autopilot.core.profiler import Profiler
from autopilot.core.scheduler import Scheduler
from autopilot.core.store.base import Store
from autopilot.core.trainer import fit_loop as trainer_fit_loop
from autopilot.core.trainer.callbacks_dispatch import dispatch_callbacks as dispatch_callbacks_fn
from autopilot.core.trainer.callbacks_dispatch import should_stop_at as should_stop_at_fn
from autopilot.core.trainer.delegates import TrainerDelegates
from autopilot.core.tree import Tree
from autopilot.data.datamodule import DataModule
from autopilot.policy.policy import Policy
from pathlib import Path
from typing import Any


class Trainer(TrainerDelegates):
  """Orchestrate optimization: ``fit``, ``validate``, ``test``, ``predict``.

  Coordinates ``AutoPilotModule``, ``Loss``, ``Optimizer``, ``Callback``,
  and optional ``Store`` / ``Experiment`` through the training loop.

  Attributes:
    Trainer mirrors constructor kwargs assigned in ``__init__`` (callbacks,
    loop handle, experiment pointer, datamodule slot, policy, forest, etc.).

  Training uses ``EpochLoop`` ordering; ``fit()`` optionally wraps ``experiment``
  context and persists ``forest`` on exit.

  Callback hooks include ``on_before_backward(trainer, module, loss_fn)`` and
  ``on_after_backward(trainer, module)``.

  Note:
    Epoch callback order differs from PyTorch Lightning: AutoPilot invokes
    ``on_train_epoch_end`` after the validation pass completes (including
    ``on_validation_epoch_end``). Lightning calls ``on_train_epoch_end``
    before validation.

  Example:
    >>> from autopilot.core.trainer.trainer import Trainer  # doctest: +SKIP
    >>>
    >>> trainer = Trainer()  # doctest: +SKIP
    >>> trainer.fit(my_module, max_epochs=3)  # doctest: +SKIP
    >>> trainer.validate(my_module)  # doctest: +SKIP
  """

  # --- initialization ---

  def __init__(
    self,
    callbacks: list[Callback] | None = None,
    *,
    loop: Loop | None = None,
    dry_run: bool = False,
    logger: Logger | None = None,
    policy: Policy | None = None,
    experiment: Experiment | None = None,
    config: Config | None = None,
    forest: Forest | None = None,
    tree: Tree | None = None,
    store: Store | None = None,
    accumulate_grad_batches: int = 1,
    enable_context_log: bool = True,
    num_sanity_val_steps: int = 2,
    profiler: Profiler | None = None,
  ) -> None:
    """Wire trainer dependencies and default epoch loop.

    Args:
      callbacks: Lightning-style hooks invoked during training.
      loop: Loop implementation (defaults to ``EpochLoop``).
      dry_run: When true, skip mutating side effects where honored.
      logger: Optional structured logger.
      policy: Optional epoch acceptance policy.
      experiment: Optional experiment entity updated across ``fit``.
      config: Optional path configuration for integrations.
      forest: Optional multi-tree coordinator.
      tree: Optional experiment DAG for the active project.
      store: Optional persistence layer for snapshots.
      accumulate_grad_batches: Micro-batch accumulation count before ``step``.
      enable_context_log: Auto-attach ``ContextLogCallback`` when allowed.
      num_sanity_val_steps: Capped validation batches before epoch 0; ``0`` disables.
      profiler: Optional wall-clock profiler; writes ``profiler_summary.json`` on fit end.
    """
    self._callbacks = list(callbacks) if callbacks is not None else []
    self._loop = loop or EpochLoop()
    self._dry_run = dry_run
    self._logger = logger
    self._policy = policy
    self._experiment = experiment
    self._config = config
    self._forest = forest
    self._tree = tree
    self._store = store
    self._accumulate_grad_batches = accumulate_grad_batches
    self._enable_context_log = enable_context_log
    self._num_sanity_val_steps = num_sanity_val_steps
    self._profiler = profiler
    self._sanity_checking: bool = False
    self._module: Module | None = None
    self._optimizer: Optimizer | None = None
    self._scheduler: Scheduler | None = None
    self._datamodule: DataModule | None = None
    self._fit_ctx: dict[str, Any] = {}
    self.current_epoch: int = 0
    # structured gradient rows: [{param_name, param_type, gradient_type, summary}]
    self._cached_grad_summaries: list[dict[str, str]] = []

  # --- properties ---

  @property
  def module(self) -> Module | None:
    """Active ``Module`` installed by ``fit``.

    Returns:
      Module passed to ``fit``, or ``None`` before ``fit``.
    """
    return self._module

  @property
  def datamodule(self) -> DataModule | None:
    """Currently active DataModule, if any."""
    return self._datamodule

  @property
  def callbacks(self) -> list[Callback]:
    """Callbacks registered for this trainer."""
    return self._callbacks

  @property
  def loop(self) -> Loop:
    """Active training loop implementation."""
    return self._loop

  @property
  def dry_run(self) -> bool:
    """Whether dry-run mode is enabled."""
    return self._dry_run

  @property
  def logger(self) -> Logger | None:
    """Attached logger, if any."""
    return self._logger

  @property
  def policy(self) -> Policy | None:
    """Epoch gate policy, if configured."""
    return self._policy

  @property
  def experiment(self) -> Experiment | None:
    """Experiment tracked for this run."""
    return self._experiment

  @property
  def config(self) -> Config | None:
    """Path/config object when provided."""
    return self._config

  @property
  def forest(self) -> Forest | None:
    """Forest handle for multi-tree workflows."""
    return self._forest

  @property
  def tree(self) -> Tree | None:
    """Tree handle for DAG updates."""
    return self._tree

  @property
  def store(self) -> Store | None:
    """Store used for checkpoints and artifacts."""
    return self._store

  @property
  def accumulate_grad_batches(self) -> int:
    """Gradient accumulation stride."""
    return self._accumulate_grad_batches

  @property
  def enable_context_log(self) -> bool:
    """Whether automatic context recording is enabled."""
    return self._enable_context_log

  @property
  def num_sanity_val_steps(self) -> int:
    """Sanity-check batch cap before the first training epoch."""
    return self._num_sanity_val_steps

  @property
  def sanity_checking(self) -> bool:
    """Whether a sanity check is currently in progress."""
    return self._sanity_checking

  @property
  def optimizer(self) -> Optimizer | None:
    """Optimizer resolved during ``fit``."""
    return self._optimizer

  @property
  def scheduler(self) -> Scheduler | None:
    """Scheduler resolved during ``fit``, or ``None``."""
    return self._scheduler

  @property
  def profiler(self) -> Profiler | None:
    """Wall-clock profiler, if configured."""
    return self._profiler

  @property
  def fit_context(self) -> dict[str, Any]:
    """Caller-provided context from ``fit(ctx=...)``; read-only dict view."""
    return self._fit_ctx

  # --- fit lifecycle ---
  # full epoch ordering and success/failure paths live in ``fit_loop.execute_fit`` so
  # this class stays a thin facade for dependency wiring and checkpoint entry.

  def fit(
    self,
    module: AutoPilotModule,
    train_dataloaders: Any | None = None,
    val_dataloaders: Any | None = None,
    datamodule: DataModule | None = None,
    max_epochs: int = 10,
    ctx: dict[str, Any] | None = None,
    *,
    ckpt_path: Path | str | None = None,
    checkpoint_io: CheckpointIO | None = None,
    test_dataloaders: Any | None = None,
  ) -> dict[str, Any]:
    """Train ``module``; see ``fit_loop.execute_fit`` for the full lifecycle.

    Args:
      module: ``AutoPilotModule`` to train.
      train_dataloaders: Training data iterable.
      val_dataloaders: Optional validation iterable.
      datamodule: Optional ``DataModule``.
      max_epochs: Epoch budget.
      ctx: Optional context dict exposed as ``fit_context``.
      ckpt_path: Resume path or ``'last'`` / ``'best'`` token.
      checkpoint_io: Checkpoint storage backend for resume.
      test_dataloaders: Optional test iterable (run after success path).

    Each epoch runs validation before ``on_train_epoch_end`` fires (AutoPilot
    ordering mirrors the deliberate Lightning divergence documented in AGENTS.md).

    Returns:
      Loop result dict, optionally including ``test_results``.

    Warning:
      Requires ``AutoPilotModule``, not plain ``Module``. Passing a
      ``Module`` subclass that is not an ``AutoPilotModule`` raises
      ``TypeError``.

    Raises:
      TypeError: If ``module`` is not an ``AutoPilotModule``.

    Loss discovery: first submodule where ``isinstance(m, Loss)`` via
    ``module.modules()`` depth-first; if none, the loop receives ``loss_fn=None``.
    """
    if not isinstance(module, AutoPilotModule):
      msg = (
        f'Trainer.fit() requires AutoPilotModule, got {type(module).__name__}. '
        'Use AutoPilotModule for training_step/configure_optimizers support.'
      )
      raise TypeError(msg)
    return trainer_fit_loop.execute_fit(
      self,
      module,
      train_dataloaders,
      val_dataloaders,
      datamodule,
      max_epochs,
      ctx,
      ckpt_path=ckpt_path,
      checkpoint_io=checkpoint_io,
      test_dataloaders=test_dataloaders,
    )

  # --- callback dispatch ---
  # string hook names are intentional: mirrors Lightning hook names for drop-in callbacks.

  def on_epoch_start(self, epoch: int) -> list[Any]:
    """Invoke ``on_epoch_start`` callbacks."""
    return self.dispatch_callbacks('on_epoch_start', epoch=epoch)

  def on_epoch_end(
    self,
    epoch: int,
    result: Result | dict[str, Any] | None = None,
  ) -> list[Any]:
    """Invoke ``on_epoch_end`` callbacks."""
    return self.dispatch_callbacks('on_epoch_end', epoch=epoch, result=result)

  def on_loop_start(self, max_epochs: int) -> list[Any]:
    """Invoke ``on_loop_start`` callbacks."""
    return self.dispatch_callbacks('on_loop_start', max_epochs=max_epochs)

  def on_loop_end(self, result: dict[str, Any]) -> list[Any]:
    """Invoke ``on_loop_end`` callbacks."""
    return self.dispatch_callbacks('on_loop_end', result=result)

  def should_stop_at(self, hook_method: Any, **kwargs: Any) -> bool:
    """Whether hook results contain ``{'stop': True}``; see ``callbacks_dispatch``."""
    return should_stop_at_fn(hook_method, **kwargs)

  def dispatch_callbacks(self, hook_name: str, **kwargs: Any) -> list[Any]:
    """Dispatch ``hook_name`` to callbacks; see ``callbacks_dispatch``."""
    return dispatch_callbacks_fn(self, hook_name, **kwargs)

  def __repr__(self) -> str:
    """Concise trainer summary."""
    return f'Trainer(dry_run={self._dry_run})'
