"""Standalone evaluation runners: validate, test, predict, run_eval_phase.

Domain module for Trainer evaluation concerns. All functions take
a ``trainer`` instance (typed as ``Any`` to avoid circular imports) as
their first argument.
"""

from autopilot.core.errors import ConfigError
from autopilot.core.graph import no_grad
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.module.module import Module
from autopilot.data.datamodule import DataModule, Stage, ensure_stage
from typing import Any
import logging

logger = logging.getLogger(__name__)


def collect_module_metrics(
  module: Module,
) -> tuple[dict[str, Metric], dict[str, bool]]:
  """Collect flattened metric dict and metadata from a module tree.

  Discovers all ``Metric`` instances (excluding ``Loss``) from the module
  tree via ``named_modules``. Excludes children of ``MetricCollection``
  to prevent double-update.

  Args:
    module: Module whose subtree is scanned for metrics.

  Returns:
    Tuple of ``(metrics, metric_metadata)`` where metrics is a name-to-Metric
    mapping and metric_metadata holds ``higher_is_better`` hints.
  """
  all_metrics = {
    name: m
    for name, m in module.named_modules()
    if isinstance(m, Metric) and not isinstance(m, Loss)
  }
  collection_names = {name for name, m in all_metrics.items() if isinstance(m, MetricCollection)}
  metrics: dict[str, Metric] = {}
  for name, m in all_metrics.items():
    is_child_of_collection = any(name.startswith(cn + '.') for cn in collection_names)
    if not is_child_of_collection:
      metrics[name] = m

  metric_metadata: dict[str, bool] = {}
  for met_name, m in metrics.items():
    if m.higher_is_better is not None:
      metric_metadata[met_name] = m.higher_is_better

  return metrics, metric_metadata


def dispatch_module_eval_hook(module: Module, hook_prefix: str, phase: str) -> None:
  """Fire module-level start/end hooks for eval phases.

  Args:
    module: Module being evaluated.
    hook_prefix: ``'validation'`` or ``'test'``.
    phase: ``'start'`` or ``'end'``.
  """
  if not isinstance(module, AutoPilotModule):
    return
  hook_name = f'on_{hook_prefix}_{phase}'
  hook_fn = getattr(module, hook_name, None)
  if hook_fn is not None:
    hook_fn()


def run_eval_phase(
  trainer: Any,
  module: AutoPilotModule,
  dataloader: Any,
  *,
  step_method: str,
  hook_prefix: str,
  max_batches: int | None = None,
  epoch_arg: int = 0,
) -> dict[str, float]:
  """Single canonical eval runner for validate, test, and sanity check.

  Iterates batches under ``no_grad()``, calling ``step_fn(batch, batch_idx)``
  on the module. Dispatches epoch- and batch-level callback hooks via
  ``dispatch_callbacks``. Fires module-level hooks (``on_validation_start``/
  ``on_validation_end`` or ``on_test_start``/``on_test_end``) when the module
  is an ``AutoPilotModule``.

  Does **not** call ``configure_optimizers``, policy gate, experiment
  lifecycle methods, or logger writes. Returns a plain metrics dict.

  Args:
    trainer: ``Trainer`` instance.
    module: Module being evaluated.
    dataloader: Iterable of batches.
    step_method: Name of the step method on the module (e.g.
      ``'validation_step'``, ``'test_step'``).
    hook_prefix: Callback hook prefix (e.g. ``'validation'``, ``'test'``).
    max_batches: When not ``None``, stop after this many batches.
    epoch_arg: Epoch index passed to epoch-level callbacks (default 0
      for standalone eval).

  Returns:
    Dict of metric name to computed float value.

  Raises:
    ConfigError: When ``step_method`` is not callable on the module.
    TypeError: When the step method signature is missing batch_idx.
  """
  step_fn = getattr(module, step_method, None)
  if step_fn is None or not callable(step_fn):
    msg = (
      f'{type(module).__name__} has no callable {step_method!r}. '
      f'Override {step_method}(batch, batch_idx) in your AutoPilotModule subclass.'
    )
    raise ConfigError(msg)

  metrics, _ = collect_module_metrics(module)

  for m in metrics.values():
    m.reset()

  module.eval()
  try:
    dispatch_module_eval_hook(module, hook_prefix, 'start')
    trainer.dispatch_callbacks(f'on_{hook_prefix}_epoch_start', epoch=epoch_arg)

    batch_idx: int = -1
    for batch_idx, batch in enumerate(dataloader):
      if max_batches is not None and batch_idx >= max_batches:
        break
      trainer.dispatch_callbacks(f'on_{hook_prefix}_batch_start', batch=batch, batch_idx=batch_idx)
      with no_grad():
        try:
          step_output = step_fn(batch, batch_idx)
        except TypeError as exc:
          if 'argument' in str(exc) and 'positional' in str(exc):
            method_name = step_fn.__name__
            msg = (
              f'{type(module).__name__}.{method_name}() signature error: '
              f'expected {method_name}(self, batch, batch_idx) but got a TypeError. '
              f'Add batch_idx: int as the second parameter.'
            )
            raise TypeError(msg) from exc
          raise
      for m in metrics.values():
        m.update(step_output)
      trainer.dispatch_callbacks(f'on_{hook_prefix}_batch_end', batch=batch, batch_idx=batch_idx)

    result: dict[str, float] = {}
    if batch_idx >= 0:
      for m in metrics.values():
        result.update(m.compute())

    trainer.dispatch_callbacks(f'on_{hook_prefix}_epoch_end', epoch=epoch_arg)
    dispatch_module_eval_hook(module, hook_prefix, 'end')
    return result
  finally:
    module.train()


def validate(
  trainer: Any,
  module: AutoPilotModule,
  dataloaders: Any | None = None,
  datamodule: DataModule | None = None,
) -> dict[str, float]:
  """Run standalone validation without training or optimizer setup.

  Resolves a validation dataloader from ``dataloaders`` or
  ``datamodule.val_dataloader()``, runs ``module.validation_step`` per
  batch under ``no_grad()``, and returns aggregated metrics.

  Does **not** call ``configure_optimizers``, modify experiment state,
  or invoke the policy gate.

  Args:
    trainer: ``Trainer`` instance.
    module: ``AutoPilotModule`` to evaluate.
    dataloaders: Explicit validation dataloader. Takes priority over
      ``datamodule.val_dataloader()``.
    datamodule: Optional ``DataModule`` providing ``val_dataloader()``
      and ``setup``/``teardown`` lifecycle hooks.

  Returns:
    Dict of metric name to computed float value.

  Raises:
    TypeError: If ``module`` is not an ``AutoPilotModule``.
    ConfigError: When no validation dataloader can be resolved.
  """
  if not isinstance(module, AutoPilotModule):
    msg = (
      f'Trainer.validate() requires AutoPilotModule, got {type(module).__name__}. '
      'Use AutoPilotModule for validation_step support.'
    )
    raise TypeError(msg)

  prev_module = trainer._module
  prev_datamodule = trainer._datamodule
  prev_trainer = module.trainer
  trainer._module = module
  trainer._datamodule = datamodule
  module.trainer = trainer

  try:
    module.setup()

    if datamodule is not None:
      datamodule.setup(ensure_stage(Stage.validate))

    val_loader = dataloaders
    if val_loader is None and datamodule is not None:
      val_loader = datamodule.val_dataloader()

    if val_loader is None:
      msg = (
        'No validation dataloader available. '
        'Pass dataloaders= to Trainer.validate() or provide a DataModule '
        'with a val_dataloader() method.'
      )
      raise ConfigError(msg)

    trainer.dispatch_callbacks('setup', stage=Stage.validate)
    result = run_eval_phase(
      trainer,
      module,
      val_loader,
      step_method='validation_step',
      hook_prefix='validation',
    )
    return result
  finally:
    if datamodule is not None:
      datamodule.teardown(ensure_stage(Stage.validate))
    trainer.dispatch_callbacks('teardown', stage=Stage.validate)
    trainer._module = prev_module
    trainer._datamodule = prev_datamodule
    module.trainer = prev_trainer


def test(
  trainer: Any,
  module: AutoPilotModule,
  dataloaders: Any | None = None,
  datamodule: DataModule | None = None,
) -> dict[str, float]:
  """Run standalone test without training or optimizer setup.

  Resolves a test dataloader from ``dataloaders`` or
  ``datamodule.test_dataloader()``, runs ``module.test_step`` per batch
  under ``no_grad()``, and returns aggregated metrics.

  Does **not** call ``configure_optimizers``, modify experiment state,
  or invoke the policy gate.

  Args:
    trainer: ``Trainer`` instance.
    module: ``AutoPilotModule`` to test.
    dataloaders: Explicit test dataloader. Takes priority over
      ``datamodule.test_dataloader()``.
    datamodule: Optional ``DataModule`` providing ``test_dataloader()``
      and ``setup``/``teardown`` lifecycle hooks.

  Returns:
    Dict of metric name to computed float value.

  Raises:
    TypeError: If ``module`` is not an ``AutoPilotModule``.
    ConfigError: When no test dataloader can be resolved.
  """
  if not isinstance(module, AutoPilotModule):
    msg = (
      f'Trainer.test() requires AutoPilotModule, got {type(module).__name__}. '
      'Use AutoPilotModule for test_step support.'
    )
    raise TypeError(msg)

  prev_module = trainer._module
  prev_datamodule = trainer._datamodule
  prev_trainer = module.trainer
  trainer._module = module
  trainer._datamodule = datamodule
  module.trainer = trainer

  try:
    module.setup()

    if datamodule is not None:
      datamodule.setup(ensure_stage(Stage.test))

    test_loader = dataloaders
    if test_loader is None and datamodule is not None:
      try:
        test_loader = datamodule.test_dataloader()
      except NotImplementedError:
        test_loader = None

    if test_loader is None:
      msg = (
        'No test dataloader available. '
        'Pass dataloaders= to Trainer.test() or provide a DataModule '
        'with a test_dataloader() method.'
      )
      raise ConfigError(msg)

    trainer.dispatch_callbacks('setup', stage=Stage.test)
    result = run_eval_phase(
      trainer,
      module,
      test_loader,
      step_method='test_step',
      hook_prefix='test',
    )
    return result
  finally:
    if datamodule is not None:
      datamodule.teardown(ensure_stage(Stage.test))
    trainer.dispatch_callbacks('teardown', stage=Stage.test)
    trainer._module = prev_module
    trainer._datamodule = prev_datamodule
    module.trainer = prev_trainer


def predict(
  trainer: Any,
  module: AutoPilotModule,
  dataloaders: Any | None = None,
  datamodule: DataModule | None = None,
) -> list[Any]:
  """Run prediction loop, collecting outputs without metrics or optimizer.

  Iterates the predict dataloader, calling ``module.predict_step(batch,
  batch_idx)`` per batch under ``no_grad()``, and returns all outputs as
  a flat list. No optimizer, no metrics, no loss, no experiment mutation.

  Does **not** call ``configure_optimizers``.

  Args:
    trainer: ``Trainer`` instance.
    module: ``AutoPilotModule`` with ``predict_step`` implemented.
    dataloaders: Explicit predict dataloader. Takes priority over
      ``datamodule.predict_dataloader()``.
    datamodule: Optional ``DataModule`` providing ``predict_dataloader()``
      and ``setup``/``teardown`` lifecycle hooks.

  Returns:
    List of outputs from ``predict_step``, one element per batch.

  Raises:
    TypeError: If ``module`` is not an ``AutoPilotModule``.
    ConfigError: When no predict dataloader can be resolved.
  """
  if not isinstance(module, AutoPilotModule):
    msg = (
      f'Trainer.predict() requires AutoPilotModule, got {type(module).__name__}. '
      'Use AutoPilotModule for predict_step support.'
    )
    raise TypeError(msg)

  prev_module = trainer._module
  prev_datamodule = trainer._datamodule
  prev_trainer = module.trainer
  trainer._module = module
  trainer._datamodule = datamodule
  module.trainer = trainer

  try:
    module.setup()

    if datamodule is not None:
      datamodule.setup(ensure_stage(Stage.predict))

    predict_loader = dataloaders
    if predict_loader is None and datamodule is not None:
      try:
        predict_loader = datamodule.predict_dataloader()
      except NotImplementedError:
        predict_loader = None

    if predict_loader is None:
      msg = (
        'No predict dataloader available. '
        'Pass dataloaders= to Trainer.predict() or provide a DataModule '
        'with a predict_dataloader() method.'
      )
      raise ConfigError(msg)

    trainer.dispatch_callbacks('setup', stage=Stage.predict)
    return predict_loop(trainer, module, predict_loader)
  finally:
    if datamodule is not None:
      datamodule.teardown(ensure_stage(Stage.predict))
    trainer.dispatch_callbacks('teardown', stage=Stage.predict)
    trainer._module = prev_module
    trainer._datamodule = prev_datamodule
    module.trainer = prev_trainer


def predict_loop(trainer: Any, module: AutoPilotModule, predict_loader: Any) -> list[Any]:
  """Execute the predict batch loop under no_grad.

  Args:
    trainer: ``Trainer`` instance.
    module: Module with predict_step implemented.
    predict_loader: Iterable of batches.

  Returns:
    List of outputs from predict_step.

  Raises:
    TypeError: When predict_step signature is missing batch_idx.
  """
  outputs: list[Any] = []
  module.eval()
  try:
    trainer.dispatch_callbacks('on_predict_start')
    for batch_idx, batch in enumerate(predict_loader):
      trainer.dispatch_callbacks(
        'on_predict_batch_start',
        batch=batch,
        batch_idx=batch_idx,
      )
      with no_grad():
        try:
          out = module.predict_step(batch, batch_idx)
        except TypeError as exc:
          if 'argument' in str(exc) and 'positional' in str(exc):
            msg = (
              f'{type(module).__name__}.predict_step() signature error: '
              f'expected predict_step(self, batch, batch_idx) but got a TypeError. '
              f'Add batch_idx: int as the second parameter.'
            )
            raise TypeError(msg) from exc
          raise
      outputs.append(out)
      trainer.dispatch_callbacks(
        'on_predict_batch_end',
        batch=batch,
        batch_idx=batch_idx,
      )
    trainer.dispatch_callbacks('on_predict_end')
  finally:
    module.train()
  return outputs


def run_test_phase(
  trainer: Any,
  module: AutoPilotModule,
  datamodule: DataModule | None,
  test_loader: Any,
) -> dict[str, float] | None:
  """Run the test phase after training completes, if test data is available.

  Follows Lightning convention: ``setup(Stage.test)`` before the loop,
  ``teardown(Stage.test)`` after. Delegates to :func:`run_eval_phase` so
  that ``test_step(batch, batch_idx)`` and test callbacks are invoked.

  Args:
    trainer: ``Trainer`` instance.
    module: Trained module.
    datamodule: Optional ``DataModule`` for lifecycle hooks.
    test_loader: Resolved test dataloader, or ``None``.

  Returns:
    Test metrics dict, or ``None`` when no test loader.
  """
  if test_loader is None:
    return None

  if datamodule is not None:
    datamodule.setup(ensure_stage(Stage.test))

  trainer.dispatch_callbacks('setup', stage=Stage.test)
  result = run_eval_phase(
    trainer,
    module,
    test_loader,
    step_method='test_step',
    hook_prefix='test',
    epoch_arg=trainer.current_epoch,
  )

  if datamodule is not None:
    datamodule.teardown(ensure_stage(Stage.test))
  trainer.dispatch_callbacks('teardown', stage=Stage.test)

  return result
