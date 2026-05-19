"""Epoch loop: forward -> loss -> backward -> optimizer.step().

Epochs are 0-based (0 to max_epochs-1). Checkpoint resume skips
completed epochs via ``LoopConfig.min_epoch``.

Policy gate runs **after** validation in ``_finalize_epoch`` on merged
train/val metrics (prefixed ``train_*``/``val_*`` when both splits
exist). ``trainer.current_epoch`` is set at the top of each loop
iteration, before hooks or callbacks fire.

Hook ordering note: validation completes before ``on_train_epoch_end``
(differs from Lightning). When ``trainer.logger`` is set, scalars are
logged at natural boundaries after metric computation.
"""

from autopilot.core.decision import DecisionEntry
from autopilot.core.loops.loop import Loop, LoopConfig
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.trainer import journal as journal_mod
from autopilot.core.types import GateResult
from autopilot.data.datamodule import Stage, ensure_stage
from autopilot.data.sampler import BatchSampler, EpochAwareSamplerMixin
from contextlib import contextmanager, nullcontext
from typing import Any
import contextlib
import inspect


def _set_epoch_on_sampler(sampler: Any, epoch: int) -> None:
  """Apply ``set_epoch`` to an epoch-aware sampler, unwrapping ``BatchSampler``.

  Args:
    sampler: ``Sampler``, ``BatchSampler``, or epoch-aware mixin instance.
    epoch: 0-based epoch index.
  """
  if isinstance(sampler, BatchSampler):
    inner = sampler.sampler
    if isinstance(inner, EpochAwareSamplerMixin):
      inner.set_epoch(epoch)
  elif isinstance(sampler, EpochAwareSamplerMixin):
    sampler.set_epoch(epoch)


def _set_sampler_epoch_for_loader(loader: Any, epoch: int) -> None:
  """Set the epoch on a loader's sampler if it implements EpochAwareSamplerMixin.

  Handles direct samplers, ``BatchSampler`` on ``loader.sampler``, and the
  ``DataLoader`` pattern where only ``loader.batch_sampler`` is set.

  Precedence matches ``DataLoader.__iter__``: **``batch_sampler`` wins** when
  present; otherwise ``sampler`` is used.  For ``BatchSampler``, the inner
  sampler is unwrapped.  When ``loader.sampler`` is ``None``, falls back to
  ``loader.batch_sampler`` (the common ``DataLoader(..., batch_sampler=...)``
  construction).

  Args:
    loader: DataLoader (or similar) with optional ``sampler`` / ``batch_sampler``.
    epoch: Current epoch index (0-based).
  """
  batch_sampler = getattr(loader, 'batch_sampler', None)
  if batch_sampler is not None:
    _set_epoch_on_sampler(batch_sampler, epoch)
    return
  sampler = getattr(loader, 'sampler', None)
  if sampler is not None:
    _set_epoch_on_sampler(sampler, epoch)


class EpochLoop(Loop):
  """Epoch-based optimization loop with lifecycle hooks.

  Epochs are 0-based (0 to max_epochs-1), matching Lightning's current_epoch
  convention. max_epochs=3 means epochs 0, 1, 2 (3 iterations total).

  Accesses experiment.store, experiment.rollback(), experiment.last_accepted_epoch
  directly -- all defined on the base Experiment class with sensible defaults
  (None / no-op), so no getattr fallbacks are needed.

  _run_epoch() flow (train path):
    1. Set trainer.current_epoch = epoch
    2. Fire on_train_start, module.train(), on_train_epoch_start
    3. Per train batch: training_step (or module(batch)), loss(data, batch),
       metric.update(data)
    4. When _should_step is true: on_before_backward(loss_fn=loss_fn),
       loss.backward() (graph-based), on_after_backward,
       capture_gradient_summaries, optimizer.step(),
       optimizer.zero_grad(), loss.reset()
    5. Compute metrics, call experiment.on_epoch_complete(epoch, metrics)
    6. _finalize_epoch:
       a. Validation pass: module.eval(), validation_step per batch under
          no_grad(), experiment.on_validation_complete(epoch, val_metrics)
       b. Merge metrics (train_*/val_* when both splits exist)
       c. Policy evaluation on merged metrics (with ``_prev_<key>``
          injection from prior accepted epoch and ``cost_usd`` from
          ``CostTrackerCallback``): on FAIL, rollback and stop
       d. experiment.advance_epoch(metrics) on pass
    7. Reset metrics, fire on_train_epoch_end, on_train_end

  Validates accumulate_grad_batches >= 1 at run() start.

  Gradient accumulation:
    _should_step(batch_idx, is_last_batch, accumulate) returns True when
    (batch_idx + 1) % accumulate == 0 OR is_last_batch. This gates
    backward/step/zero_grad so multiple micro-batches accumulate before
    an optimizer step.

  IterableDataset support:
    Batches are iterated lazily (no list() materialization) to support
    large/infinite IterableDatasets. Accumulation uses a count-based
    approach for last-batch detection.

  Callback Result:
    _build_callback_result() merges val_metrics with 'val_' prefix into
    a Result for on_epoch_end callback dispatch.

  Template hooks for subclass customization (e.g. ``EpochOrchestrator``):
    ``_pre_run``           -- called once before the epoch loop begins.
    ``_should_stop_before_epoch`` -- pre-epoch stop check (callback signal).
    ``_should_stop_after_epoch``  -- post-epoch stop check (policy fail).
    ``_build_run_result``  -- assemble the final dict returned by ``run()``.
  """

  def _profile_section(self, trainer: Any, action: str) -> Any:
    """Return a profiler context manager for the given action, or nullcontext.

    Isolates profiler errors so they never abort training. When the profiler
    raises during start/stop, the exception is silently swallowed.

    Args:
      trainer: Trainer whose profiler attribute is consulted.
      action: Section name to profile.

    Returns:
      Context manager that wraps the section with profiler timing.
    """
    profiler = getattr(trainer, 'profiler', None)
    if profiler is None:
      return nullcontext()

    @contextmanager
    def _safe_profile():
      try:
        profiler.start(action)
      except (ValueError, RuntimeError, OSError):
        yield
        return
      try:
        yield
      finally:
        with contextlib.suppress(ValueError, RuntimeError, OSError):
          profiler.stop(action)

    return _safe_profile()

  def _build_callback_result(self, epoch_result: dict[str, Any]) -> Result:
    """Build a Result for callback dispatch, merging val_metrics with val_ prefix.

    Args:
      epoch_result: Epoch output containing train metrics, optional val_metrics, stopped flag.

    Returns:
      Result whose metrics merge train and ``val_``-prefixed validation scalars.
    """
    merged_metrics = dict(epoch_result.get('metrics', {}))
    val = epoch_result.get('val_metrics', {})
    if val:
      merged_metrics.update({f'val_{k}': v for k, v in val.items()})
    return Result(
      metrics=merged_metrics,
    )

  def _pre_run(self, trainer: Any, config: LoopConfig) -> None:
    """Hook called once before the epoch loop begins.

    Override in subclasses for per-run initialization (e.g. state reset).

    Args:
      trainer: Trainer driving the loop.
      config: Loop configuration for this run.
    """

  def _should_stop_before_epoch(
    self,
    trainer: Any,
    epoch: int,
    config: LoopConfig,
  ) -> bool:
    """Return True to skip ``epoch`` and break the loop before it runs.

    Base implementation delegates to ``trainer.should_stop_at``.

    Args:
      trainer: Trainer providing callback-based stop signals.
      epoch: Epoch index about to start.
      config: Loop configuration for this run.

    Returns:
      Whether to stop before this epoch.
    """
    return trainer.should_stop_at(trainer.on_epoch_start, epoch=epoch)

  def _should_stop_after_epoch(
    self,
    trainer: Any,
    epoch: int,
    epoch_result: dict[str, Any],
    config: LoopConfig,
  ) -> bool:
    """Return True to break the loop after ``epoch`` completes.

    Base implementation checks the ``stopped`` flag set by policy-gate failure.

    Args:
      trainer: Trainer driving the loop.
      epoch: Epoch index that just completed.
      epoch_result: Result dict from ``_run_epoch``.
      config: Loop configuration for this run.

    Returns:
      Whether to stop after this epoch.
    """
    return bool(epoch_result.get('stopped'))

  def _build_run_result(self, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the final dict returned by ``run()``.

    Args:
      results: Per-epoch result dicts collected during the loop.

    Returns:
      Dict with ``epochs`` and ``total_epochs``.
    """
    return {'epochs': results, 'total_epochs': len(results)}

  def run(self, trainer: Any, config: LoopConfig) -> dict[str, Any]:
    """Run up to ``max_epochs`` epochs, invoking ``on_epoch_end`` after each.

    The loop body delegates to overridable template hooks so that subclasses
    (e.g. ``EpochOrchestrator``) can inject behavior without duplicating the
    full iteration skeleton.

    Args:
      trainer: Trainer driving modules, callbacks, and policy.
      config: Loop configuration including loaders, experiment, and max epochs.

    Returns:
      Dict with key ``epochs`` listing per-epoch result dicts and ``total_epochs`` count.

    Raises:
      ValueError: When ``accumulate_grad_batches`` is less than 1.
    """
    if config.accumulate_grad_batches < 1:
      msg = f'accumulate_grad_batches must be >= 1, got {config.accumulate_grad_batches}'
      raise ValueError(msg)

    self._pre_run(trainer, config)
    results: list[dict[str, Any]] = []

    for epoch in range(config.min_epoch, config.max_epochs):
      trainer.current_epoch = epoch
      if config.train_loader is not None:
        _set_sampler_epoch_for_loader(config.train_loader, epoch)
      if config.val_loader is not None:
        _set_sampler_epoch_for_loader(config.val_loader, epoch)
      if self._should_stop_before_epoch(trainer, epoch, config):
        trainer.emit_context(
          f'early stopping triggered before epoch {epoch}',
          source='early-stopping',
          metadata={'epoch': epoch},
        )
        break
      epoch_result = self._run_epoch(trainer, epoch, config)
      results.append(epoch_result)
      cb_result = self._build_callback_result(epoch_result)
      trainer.on_epoch_end(epoch=epoch, result=cb_result)
      if self._should_stop_after_epoch(trainer, epoch, epoch_result, config):
        break

    return self._build_run_result(results)

  def _should_step(
    self,
    batch_idx: int,
    is_last_batch: bool,
    accumulate: int,
  ) -> bool:
    if is_last_batch:
      return True
    return (batch_idx + 1) % accumulate == 0

  def _dry_run_epoch(
    self,
    trainer: Any,
    epoch: int,
    config: LoopConfig,
  ) -> dict[str, Any]:
    """Handle dry-run mode: dispatch callbacks and advance epoch without training.

    Args:
      trainer: Trainer for callback dispatch.
      epoch: Epoch index being planned.
      config: Loop configuration describing planned components.

    Returns:
      Dict describing dry-run metadata and enabled components.
    """
    trainer.dispatch_callbacks('on_train_epoch_start', epoch=epoch)
    trainer.dispatch_callbacks('on_train_epoch_end', epoch=epoch)
    dry_result: dict[str, Any] = {
      'dry_run': True,
      'epoch': epoch,
      'planned_epochs': config.max_epochs,
      'components': {
        'loss': config.loss is not None,
        'optimizer': config.optimizer is not None,
        'store': trainer.store is not None,
        'metrics': bool(config.metrics),
        'train_loader': config.train_loader is not None,
        'val_loader': config.val_loader is not None,
      },
    }
    if config.experiment:
      config.experiment.advance_epoch()
    return dry_result

  def _run_train_batches(
    self,
    trainer: Any,
    module: Any,
    config: LoopConfig,
    loss_fn: Any,
    optimizer: Any,
    metrics: dict[str, Any],
    accumulate: int,
  ) -> None:
    """Iterate training batches with lookahead for last-batch detection."""
    if config.train_loader is None:
      return
    pending_batch: tuple[int, Any] | None = None
    for batch_idx, batch in enumerate(config.train_loader):
      if pending_batch is not None:
        p_idx, p_batch = pending_batch
        self._process_batch(
          trainer,
          module,
          p_idx,
          p_batch,
          is_last=False,
          loss_fn=loss_fn,
          optimizer=optimizer,
          metrics=metrics,
          accumulate=accumulate,
        )
      pending_batch = (batch_idx, batch)
    if pending_batch is not None:
      p_idx, p_batch = pending_batch
      self._process_batch(
        trainer,
        module,
        p_idx,
        p_batch,
        is_last=True,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        accumulate=accumulate,
      )

  def _run_validation_pass(
    self,
    trainer: Any,
    module: Any,
    epoch: int,
    config: LoopConfig,
    metrics: dict[str, Any],
  ) -> dict[str, float] | None:
    """Run validation via ``trainer.run_eval_phase`` and return val_metrics.

    Delegates batch iteration, callbacks (epoch- and batch-level), module
    eval-mode toggling, and metric aggregation to the single canonical
    ``run_eval_phase`` runner. This ensures fit-loop validation dispatches
    the same batch hooks as standalone ``Trainer.validate()``.

    Wraps the validation pass with profiler timing (``validation_step``
    action) when a profiler is configured.

    Args:
      trainer: Trainer providing ``run_eval_phase`` and datamodule ref.
      module: Module evaluated during validation.
      epoch: Current epoch index for lifecycle hooks.
      config: Loop config providing val_loader and experiment hooks.
      metrics: Train metrics dict (unused after consolidation; metrics are
        collected from the module by ``run_eval_phase``).

    Returns:
      Computed validation scalars keyed by metric name, or ``None`` when
      no validation loader is configured. An empty dict ``{}`` means
      validation ran but produced no scalar metrics (distinct from ``None``).
    """
    if not config.val_loader:
      return None
    datamodule = trainer.datamodule
    if datamodule is not None:
      datamodule.setup(ensure_stage(Stage.validate))
    with self._profile_section(trainer, 'validation_step'):
      val_metrics = trainer.run_eval_phase(
        module,
        config.val_loader,
        step_method='validation_step',
        hook_prefix='validation',
        epoch_arg=epoch,
      )
    if config.experiment:
      config.experiment.on_validation_complete(
        epoch,
        val_metrics,
        metric_metadata=config.metric_metadata,
      )
    return val_metrics

  def _inject_cost_usd(
    self,
    trainer: Any,
    metric_values: dict[str, float],
  ) -> None:
    """Merge cumulative ``cost_usd`` from an attached cost-tracker callback.

    Scans ``trainer.callbacks`` for a callback exposing ``cumulative_usd``
    (duck-typed to avoid ``isinstance`` on a concrete leaf). The first
    match wins; its value is injected as ``cost_usd`` into ``metric_values``
    unless the key is already present. This enables ``BudgetGate`` to
    evaluate cumulative cost without explicit plumbing.

    Args:
      trainer: Trainer whose callback list may contain a cost tracker.
      metric_values: Gate metric dict to augment in-place.
    """
    callbacks = getattr(trainer, 'callbacks', None)
    if callbacks is None:
      return
    for cb in callbacks:
      usd = getattr(cb, 'cumulative_usd', None)
      if usd is not None and 'cost_usd' not in metric_values:
        metric_values['cost_usd'] = usd
        break

  def _check_policy_gate(
    self,
    trainer: Any,
    epoch: int,
    metric_values: dict[str, float],
    experiment: Any,
  ) -> dict[str, Any] | None:
    """Apply policy gate and handle rollback on failure.

    When an experiment is present with prior metrics, injects ``_prev_<key>``
    entries into ``metric_values`` so that ``MonotonicGate`` (and any other
    gate reading prior-epoch data) can compare current vs previous values.
    The ``_prev_`` prefix (single leading underscore) is a reserved convention;
    current-epoch keys must not use this prefix.

    Injects ``cost_usd`` from an attached ``CostTrackerCallback`` (via
    ``cumulative_usd``) so that ``BudgetGate`` can evaluate cumulative cost
    without explicit plumbing through the optimizer or Trainer fields.

    Emits context entries via ``trainer.emit_context`` on both accept and
    reject paths so the experiment's decision journal captures policy
    gate outcomes with associated metrics. Metadata includes a ``_type``
    discriminator (``DecisionEntry.POLICY_GATE_TYPE``) and a ``gates`` list
    with per-gate ``ConstraintResult.to_dict()`` payloads. Reject path emits
    ``_type`` + ``gates`` only; accept path adds ``metrics``.

    Returns:
      Early-stop result dict when the gate fails, or ``None`` to continue.
    """
    if not trainer.policy:
      return None
    prior = experiment.metrics if experiment is not None else {}
    if prior:
      for k, v in prior.items():
        prev_key = f'_prev_{k}'
        if prev_key not in metric_values:
          metric_values[prev_key] = v
    self._inject_cost_usd(trainer, metric_values)
    result = Result(metrics=metric_values)
    gate_result = trainer.policy(result)
    gates = [cr.to_dict() for cr in result.gates]
    typed = {'_type': DecisionEntry.POLICY_GATE_TYPE}
    if gate_result == GateResult.FAIL:
      trainer.emit_context(
        f'policy gate rejected epoch {epoch}',
        source='policy',
        metadata={**typed, 'gates': gates},
      )
      if experiment:
        experiment.rollback(experiment.last_accepted_epoch)
      return {'epoch': epoch, 'metrics': metric_values, 'stopped': True}
    trainer.emit_context(
      f'epoch {epoch} accepted by policy gate',
      source='policy',
      metadata={**typed, 'gates': gates, 'metrics': metric_values},
    )
    return None

  def _merge_metrics_for_gate(
    self,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
  ) -> dict[str, float]:
    """Merge train and val metrics with prefixed keys for the policy gate.

    When both train and validation metrics are present, keys become
    ``train_*`` / ``val_*`` (aligned with ``Trainer._complete_experiment_success``
    which uses ``strip_metric_prefix`` from ``core.metric_utils`` to avoid double-prefixing).
    When only train metrics exist (no validation), keys remain unprefixed.

    Args:
      train_metrics: Metrics from the training pass.
      val_metrics: Metrics from the validation pass, or ``None``.

    Returns:
      Merged metric dict suitable for the policy gate.
    """
    if val_metrics is not None and train_metrics:
      return {
        **{f'train_{k}': v for k, v in train_metrics.items()},
        **{f'val_{k}': v for k, v in val_metrics.items()},
      }
    if val_metrics is not None:
      return dict(val_metrics)
    return dict(train_metrics)

  def _finalize_epoch(
    self,
    trainer: Any,
    module: Any,
    epoch: int,
    config: LoopConfig,
    metric_values: dict[str, float],
  ) -> dict[str, Any]:
    """Run validation, apply policy gate on merged metrics, advance experiment.

    Policy gate now runs **after** validation with merged train/val metrics
    so the gate sees the full picture (BUG-010 fix). Metric prefixing aligns
    with ``Trainer._complete_experiment_success`` semantics.

    Returns:
      Epoch result dict with metrics and optional val_metrics.
    """
    experiment = config.experiment
    metrics = config.metrics

    for m in metrics.values():
      m.reset()

    val_metrics = self._run_validation_pass(trainer, module, epoch, config, metrics)

    if trainer.logger is not None and val_metrics is not None:
      trainer.logger.log_metrics(val_metrics, step=epoch)

    for m in metrics.values():
      m.reset()

    gate_metrics = self._merge_metrics_for_gate(metric_values, val_metrics)
    accepted_metrics = dict(gate_metrics)
    stopped = self._check_policy_gate(trainer, epoch, gate_metrics, experiment)
    if stopped is not None:
      if val_metrics is not None:
        stopped['val_metrics'] = val_metrics
      return stopped

    journal_mod.emit_epoch_gradient_journal(trainer, epoch=epoch)

    opt = getattr(trainer, '_optimizer', None)
    if opt is None or not opt.owns_step_gradient_context:
      summaries = journal_mod.capture_param_summaries(module)
      journal_mod.emit_context(
        trainer,
        f'optimizer step completed epoch {epoch}',
        source='trainer',
        metadata=DecisionEntry.optimizer_step(
          epoch=epoch,
          param_summaries=summaries,
        ),
      )

    if experiment:
      experiment.last_accepted_epoch = epoch
      experiment.advance_epoch(accepted_metrics)

    self._last_epoch_metrics = accepted_metrics
    trainer.dispatch_callbacks('on_train_epoch_end', epoch=epoch)
    if isinstance(module, AutoPilotModule):
      module.on_train_end()

    if trainer.scheduler is not None:
      trainer.scheduler.step(epoch)

    result_dict: dict[str, Any] = {'epoch': epoch, 'metrics': accepted_metrics}
    if val_metrics is not None:
      result_dict['val_metrics'] = val_metrics
    return result_dict

  def _run_epoch(
    self,
    trainer: Any,
    epoch: int,
    config: LoopConfig,
  ) -> dict[str, Any]:
    trainer.current_epoch = epoch
    experiment = config.experiment
    if experiment:
      experiment.should_rollback = False

    module = trainer.module
    metrics = config.metrics

    if config.dry_run:
      return self._dry_run_epoch(trainer, epoch, config)

    if isinstance(module, AutoPilotModule):
      module.on_train_start()
    module.train()
    trainer.dispatch_callbacks('on_train_epoch_start', epoch=epoch)

    self._run_train_batches(
      trainer,
      module,
      config,
      config.loss,
      config.optimizer,
      metrics,
      config.accumulate_grad_batches,
    )

    metric_values: dict[str, float] = {}
    for m in metrics.values():
      metric_values.update(m.compute())

    if experiment:
      experiment.on_epoch_complete(epoch, metric_values)

    if trainer.logger is not None:
      trainer.logger.log_metrics(metric_values, step=epoch)

    return self._finalize_epoch(trainer, module, epoch, config, metric_values)

  def _process_batch(
    self,
    trainer: Any,
    module: Any,
    batch_idx: int,
    batch: Any,
    *,
    is_last: bool,
    loss_fn: Any,
    optimizer: Any,
    metrics: dict[str, Any],
    accumulate: int,
  ) -> None:
    """Process a single training batch with optional gradient accumulation.

    Graph-based backward flow:
      1. on_before_backward(loss_fn=loss_fn) -- callbacks receive the loss object
      2. loss_fn.backward() -- seeds gradient, calls graph.backward() internally
      3. on_after_backward -- no loss_fn argument
      4. optimizer.step() -- reads param.grad set by AccumulateGrad leaf nodes
      5. optimizer.zero_grad() -- clears param.grad and param.grad_accumulator
      6. loss_fn.reset() -- clears accumulated feedback state

    The loop does NOT import or invoke Graph APIs directly. loss_fn.backward()
    owns graph traversal and implicit reset (retain_graph=False by default).

    Raises:
      TypeError: When training_step returns ``None`` (must return a ``Datum``),
        or when training_step signature is missing batch_idx (wraps the
        original error with guidance). Other TypeErrors propagate unchanged.
    """
    trainer.dispatch_callbacks('on_train_batch_start', batch_idx=batch_idx)
    if isinstance(module, AutoPilotModule):
      try:
        with self._profile_section(trainer, 'training_step'):
          data = module.training_step(batch, batch_idx)
      except TypeError as exc:
        sig = inspect.signature(type(module).training_step)
        params = list(sig.parameters.keys())
        if len(params) < 3 and 'batch_idx' not in params:
          msg = (
            f'{type(module).__name__}.training_step() signature error: '
            f'expected training_step(self, batch, batch_idx) but got a TypeError. '
            f'Add batch_idx: int as the argument after batch.'
          )
          raise TypeError(msg) from exc
        raise
    else:
      with self._profile_section(trainer, 'training_step'):
        data = module(batch)
    if data is None:
      msg = (
        'training_step() returned None. It must return a loss Datum '
        "for backward. Check your module's training_step implementation."
      )
      raise TypeError(msg)
    if loss_fn:
      loss_fn(data, batch)
    for m in metrics.values():
      m.update(data)
    trainer.dispatch_callbacks('on_train_batch_end', batch_idx=batch_idx, data=data)

    if self._should_step(batch_idx, is_last, accumulate):
      self._backward_and_step(trainer, loss_fn, optimizer)

  def _backward_and_step(
    self,
    trainer: Any,
    loss_fn: Any,
    optimizer: Any,
  ) -> None:
    """Run backward pass, optimizer step, and loss reset.

    Extracted from ``_process_batch`` to keep branch count manageable.
    Wraps ``backward`` and ``optimizer_step`` sections with profiler
    timing when a profiler is configured on the trainer.

    Args:
      trainer: Trainer for callback dispatch and gradient capture.
      loss_fn: Loss function (may be ``None`` for loss-free flows).
      optimizer: Optimizer (may be ``None`` when no step is needed).

    Raises:
      RuntimeError: When ``loss_fn.backward()`` fails because the autograd graph
        was already consumed (e.g. manual loss calls inside training_step).
    """
    if loss_fn:
      trainer.dispatch_callbacks('on_before_backward', loss_fn=loss_fn)
      with self._profile_section(trainer, 'backward'):
        try:
          loss_fn.backward()
        except RuntimeError as exc:
          if 'graph has been freed' in str(exc).lower():
            msg = (
              f'{exc} — If you called loss.forward() or loss.backward() inside '
              f'training_step, remove those calls -- '
              f'the Trainer manages the loss lifecycle.'
            )
            raise RuntimeError(msg) from exc
          raise
      trainer.dispatch_callbacks('on_after_backward')
      trainer.capture_gradient_summaries()
    if optimizer:
      trainer.dispatch_callbacks('on_before_optimizer_step')
      with self._profile_section(trainer, 'optimizer_step'):
        optimizer.step()
      trainer.dispatch_callbacks('on_before_zero_grad')
      optimizer.zero_grad()
    if loss_fn:
      loss_fn.reset()
