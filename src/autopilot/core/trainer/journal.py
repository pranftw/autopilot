"""Context emission, gradient journal, profiler summary, and experiment completion.

Domain module for Trainer context/journal concerns. All functions take
a ``trainer`` instance (typed as ``Any`` to avoid circular imports) as
their first argument.

Gradient journal entries use structured ``list[dict[str, str]]`` with keys
``param_name``, ``param_type``, ``gradient_type``, ``summary``. Both
``Trainer`` completion and ``AgentOptimizer`` step emit the same schema
via :func:`build_gradient_journal_row`.
"""

from autopilot.core.context import ContextEntry
from autopilot.core.errors import ExperimentError
from autopilot.core.metric_utils import strip_metric_prefix
from autopilot.core.module.module import Module
from autopilot.core.parameter import Parameter
from autopilot.tracking.io import atomic_write_json
from itertools import starmap
from typing import Any
import contextlib
import logging
import traceback as traceback_mod

logger = logging.getLogger(__name__)

GRAD_SUMMARY_MAX_CHARS = 200
PARAM_SUMMARY_MAX_CHARS = 200


def build_gradient_journal_row(
  param_name: str,
  param: Parameter,
  max_chars: int = GRAD_SUMMARY_MAX_CHARS,
) -> dict[str, str]:
  """Build a structured gradient journal dict for one parameter.

  Produces the canonical row shape shared by Trainer completion and
  AgentOptimizer step context emission. Callers must pre-check that
  ``param.grad is not None`` before calling.

  Args:
    param_name: Module-relative parameter name from ``named_parameters()``.
    param: Parameter instance with a non-None ``grad``.
    max_chars: Maximum character length for the ``summary`` value.

  Returns:
    Dict with keys ``param_name``, ``param_type``, ``gradient_type``,
    ``summary``.
  """
  grad = param.grad
  assert grad is not None
  return {
    'param_name': param_name,
    'param_type': type(param).__name__,
    'gradient_type': type(grad).__name__,
    'summary': grad.render()[:max_chars],
  }


def build_param_summary_row(
  param_name: str,
  param: Parameter,
  max_chars: int = PARAM_SUMMARY_MAX_CHARS,
) -> dict[str, str]:
  """Build a post-step parameter state summary row for one parameter.

  Args:
    param_name: Module-relative parameter name from ``named_parameters()``.
    param: Parameter instance to summarize.
    max_chars: Maximum character length for the ``value_preview`` value.

  Returns:
    Dict with keys ``param_name``, ``param_type``, ``value_preview``.
  """
  value_preview = str(param.data)[:max_chars] if hasattr(param, 'data') else str(param)[:max_chars]
  return {
    'param_name': param_name,
    'param_type': type(param).__name__,
    'value_preview': value_preview,
  }


def capture_param_summaries(module: Module) -> list[dict[str, str]]:
  """Capture post-step parameter state summaries for all named parameters.

  Args:
    module: Module whose parameters to summarize.

  Returns:
    List of per-parameter summary dicts with keys ``param_name``,
    ``param_type``, ``value_preview``.
  """
  return list(starmap(build_param_summary_row, module.named_parameters()))


def emit_context(
  trainer: Any,
  reason: str,
  *,
  source: str | None = None,
  metadata: dict[str, Any] | None = None,
) -> None:
  """Emit a context entry and broadcast to all callbacks via on_context_emit.

  Builds a :class:`~autopilot.core.context.ContextEntry` using the canonical
  ``ContextEntry.create()`` factory (DRY-01) with ``epoch`` set to
  ``trainer.current_epoch``. Always dispatches to callbacks regardless of
  whether an experiment is present -- ``ContextLogCallback`` silently
  no-ops when ``trainer.experiment`` is ``None``; other callbacks
  (monitoring, logging) still receive the entry.

  Args:
    trainer: ``Trainer`` instance.
    reason: Human- or machine-readable explanation of what happened.
    source: Origin of the entry (e.g. 'trainer', 'policy').
    metadata: Arbitrary key-value context data.
  """
  entry = ContextEntry.create(
    reason,
    source=source,
    epoch=trainer.current_epoch,
    metadata=metadata,
  )
  trainer.dispatch_callbacks('on_context_emit', entry=entry)


def capture_gradient_summaries(trainer: Any) -> None:
  """Snapshot current parameter gradients as structured dicts for later emission.

  Called by the epoch loop after ``loss.backward()`` and before
  ``optimizer.zero_grad()`` so that gradient text is captured while
  still alive. Only captures when the optimizer does not own per-step
  gradient context (``owns_step_gradient_context`` is False).

  Populates ``trainer._cached_grad_summaries`` as ``list[dict[str, str]]``
  with keys ``param_name``, ``param_type``, ``gradient_type``, ``summary``
  via :func:`build_gradient_journal_row`. Parameters without gradients
  are skipped.

  Args:
    trainer: ``Trainer`` instance.
  """
  if trainer._module is None:
    return
  opt = trainer._optimizer
  if opt is not None and opt.owns_step_gradient_context:
    return
  summaries: list[dict[str, str]] = []
  for name, param in trainer._module.named_parameters():
    if param.requires_grad and param.grad is not None:
      summaries.append(build_gradient_journal_row(name, param))
  trainer._cached_grad_summaries = summaries


def emit_epoch_gradient_journal(trainer: Any, *, epoch: int) -> None:
  """Emit cached gradient summaries for one accepted training epoch.

  Called from ``EpochLoop._finalize_epoch`` after the policy gate accepts.
  Skips when the optimizer owns step gradient context (``AgentOptimizer``
  agentic mode). Does **not** clear ``_cached_grad_summaries`` so
  ``_emit_gradient_journal`` at experiment completion can still emit a
  final summary from the last epoch's cache.

  Args:
    trainer: Trainer instance with ``_cached_grad_summaries`` populated
      during the training backward phase.
    epoch: 0-based epoch index for metadata.
  """
  opt = trainer._optimizer
  if opt is not None and opt.owns_step_gradient_context:
    return
  if not trainer._cached_grad_summaries:
    return
  emit_context(
    trainer,
    f'gradient feedback recorded for epoch {epoch}',
    source='trainer',
    metadata={
      'gradient_summaries': list(trainer._cached_grad_summaries),
      'epoch': epoch,
    },
  )


def _emit_gradient_journal(trainer: Any) -> None:
  """Emit cached gradient summaries as a context entry when non-empty.

  Uses structured ``list[dict[str, str]]`` summaries captured by
  :func:`capture_gradient_summaries` during the backward phase.
  Each dict has keys ``param_name``, ``param_type``, ``gradient_type``,
  ``summary`` (truncated per :data:`GRAD_SUMMARY_MAX_CHARS`).
  Emits under metadata key ``gradient_summaries``; omits the key entirely
  when the cache is empty.

  Args:
    trainer: ``Trainer`` instance.
  """
  if not trainer._cached_grad_summaries:
    return
  emit_context(
    trainer,
    'gradient feedback recorded',
    source='trainer',
    metadata={'gradient_summaries': trainer._cached_grad_summaries},
  )
  trainer._cached_grad_summaries = []


def write_profiler_summary(trainer: Any) -> None:
  """Write profiler describe() output to experiment directory.

  Called in fit() finally block (both success and failure paths).
  Writes ``profiler_summary.json`` beside other experiment artifacts.
  No-op when profiler is None or no experiment/config is set. Profiler
  errors are isolated and never propagate.

  Args:
    trainer: ``Trainer`` instance.
  """
  if trainer._profiler is None:
    return
  if trainer._experiment is None or trainer._config is None:
    return
  try:
    summary = trainer._profiler.describe()
    exp_path_fn = getattr(trainer._config, 'experiment_path', None)
    if exp_path_fn is None:
      return
    exp_dir = exp_path_fn(slug=trainer._experiment.id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    summary_path = exp_dir / 'profiler_summary.json'
    atomic_write_json(summary_path, summary)
  except (OSError, ValueError, RuntimeError):
    pass


@contextlib.contextmanager
def profile_store_section(trainer: Any, action: str):
  """Wrap a store operation with profiler timing, isolating errors.

  Args:
    trainer: ``Trainer`` instance.
    action: Section name (e.g. 'store_snapshot', 'store_checkout').

  Yields:
    None -- the body runs between start and stop.
  """
  if trainer._profiler is None:
    yield
    return
  try:
    trainer._profiler.start(action)
  except (ValueError, RuntimeError, OSError):
    yield
    return
  try:
    yield
  finally:
    with contextlib.suppress(ValueError, RuntimeError, OSError):
      trainer._profiler.stop(action)


def complete_experiment_success(trainer: Any, loop_result: dict[str, Any]) -> None:
  """Mark experiment as completed with final metrics from last epoch.

  Emits ``'experiment completed successfully'`` context via
  ``emit_context`` (``source='trainer'``, ``metadata={'final_metrics': ...}``)
  before calling ``experiment.complete()``.

  Metric merge policy: when both train and validation metrics exist,
  keys are prefixed ``train_*`` / ``val_*`` via ``strip_metric_prefix``
  to prevent silent overwrite and avoid double-prefixing (e.g.
  ``train_train_accuracy`` is normalized to ``train_accuracy``).
  An empty dict ``{}`` for val_metrics counts as "validation present"
  (paired with BUG-059 empty-dict fix).

  Args:
    trainer: ``Trainer`` instance.
    loop_result: Dict with per-epoch telemetry from the loop.
  """
  if trainer._experiment is None:
    return
  final_metrics: dict[str, float] | None = None
  if isinstance(loop_result, dict) and loop_result['epochs']:
    last_epoch = loop_result['epochs'][-1]
    val = last_epoch.get('val_metrics')
    train = last_epoch.get('metrics', {})
    if val is not None and train:
      final_metrics = {}
      for key, value in train.items():
        base, _ = strip_metric_prefix(key)
        final_metrics[f'train_{base}'] = value
      for key, value in val.items():
        base, _ = strip_metric_prefix(key)
        final_metrics[f'val_{base}'] = value
    elif val is not None:
      final_metrics = val
    else:
      final_metrics = train
  _attach_dataset_fingerprint(trainer)
  _emit_gradient_journal(trainer)
  emit_context(
    trainer,
    'experiment completed successfully',
    source='trainer',
    metadata={'final_metrics': final_metrics},
  )
  trainer._experiment.complete(final_metrics)


def _attach_dataset_fingerprint(trainer: Any) -> None:
  """Merge DataModule fingerprint into experiment dataset_meta if present.

  Auto-attaches ``dataset_fingerprint`` from ``trainer._datamodule`` when:
  (1) a DataModule is configured, (2) it carries a non-None
  ``dataset_fingerprint``, and (3) the experiment's ``dataset_meta``
  does not already contain a ``'dataset_fingerprint'`` key (avoids
  duplicate writes on checkpoint resume).

  Args:
    trainer: ``Trainer`` instance.
  """
  if trainer._experiment is None:
    return
  if trainer._datamodule is None:
    return
  fp = trainer._datamodule.dataset_fingerprint
  if fp is None:
    return
  if 'dataset_fingerprint' in trainer._experiment.dataset_meta:
    return
  trainer._experiment.dataset_meta['dataset_fingerprint'] = fp.to_dict()


def fit_success_path(trainer: Any, loop_result: dict[str, Any], loop_config: Any) -> dict[str, Any]:
  """Complete experiment after a successful loop: journal, tree update, logger finalize.

  Args:
    trainer: Trainer instance.
    loop_result: Dict with per-epoch telemetry from the loop.
    loop_config: Loop config for determining expected epoch count.

  Returns:
    The loop result dict unchanged.
  """
  epochs = loop_result.get('epochs', [])
  expected_epoch_count = loop_config.max_epochs - loop_config.min_epoch
  ran_all_epochs = len(epochs) == expected_epoch_count
  stopped_by_gate = any(e.get('stopped') for e in epochs)
  if ran_all_epochs and not stopped_by_gate:
    emit_context(
      trainer,
      'training completed: max_epochs reached',
      source='trainer',
    )

  if stopped_by_gate and trainer._experiment is not None:
    if trainer._experiment.last_accepted_epoch is None:
      _emit_gradient_journal(trainer)
      emit_context(
        trainer,
        'policy gate rejected all epochs -- experiment failed',
        source='trainer',
        metadata={'gate_result': 'fail'},
      )
      trainer._experiment.fail('policy gate rejected all epochs')
    else:
      emit_context(
        trainer,
        f'policy gate stopped training after epoch {trainer._experiment.last_accepted_epoch}',
        source='trainer',
      )
      complete_experiment_success(trainer, loop_result)
  else:
    complete_experiment_success(trainer, loop_result)

  trainer.on_loop_end(result=loop_result)
  trainer.dispatch_callbacks('on_fit_end')
  if trainer._tree is not None and trainer._experiment is not None:
    trainer._tree.update(trainer._experiment.id, metrics=trainer._experiment.metrics)
  if trainer._logger is not None:
    trainer._logger.finalize('success')
  return loop_result


def fit_failure_path(trainer: Any, exc: Exception) -> None:
  """Fail experiment after ``fit`` errors: context emission, tree update, logger finalize.

  Args:
    trainer: Trainer instance.
    exc: The exception that aborted training.
  """
  emit_context(
    trainer,
    f'experiment failed: {exc}',
    source='trainer',
    metadata={
      'error': str(exc),
      'traceback': traceback_mod.format_exc(),
      'exception_type': type(exc).__name__,
    },
  )
  if trainer._experiment is not None:
    try:
      trainer._experiment.fail(str(exc))
    except ExperimentError as fail_exc:
      logger.warning(
        'experiment %r fail() raised during cleanup: %s',
        trainer._experiment.id,
        fail_exc,
      )
  if trainer._tree is not None and trainer._experiment is not None:
    try:
      trainer._tree.update(trainer._experiment.id, error=trainer._experiment.error)
    except (TypeError, ValueError) as tree_exc:
      logger.warning(
        'tree update for experiment %r raised during cleanup: %s',
        trainer._experiment.id,
        tree_exc,
      )
  if trainer._logger is not None:
    trainer._logger.finalize('failed')
