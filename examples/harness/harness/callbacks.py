"""Harness-specific Trainer callbacks.

MetricsWriterCallback: per-epoch metrics JSON for propose verify compatibility.
OptimizerContextCallback: keeps AgentOptimizer epoch/metrics in sync and emits
context when the primary validation metric improves over the prior best.
DeployCallback: no-op deployment hook for production promotion simulation.
HarnessCostTrackerCallback: populates ``CostEntry.api_calls`` and
``CostEntry.tokens_used`` from harness token-sum metrics in
``result.metrics``.
"""

from autopilot.ai.optimizer import AgentOptimizer
from autopilot.core.callbacks.callback import Callback
from autopilot.core.callbacks.cost import CostEntry, CostTrackerCallback
from autopilot.tracking.io import atomic_write_json
from typing import Any

PRIMARY_METRIC_KEY = 'task_success_rate'


class MetricsWriterCallback(Callback):
  """Write per-epoch metrics JSON for propose verify compatibility.

  Uses on_epoch_end which receives the Result object with metrics dict.
  Writes to {experiment_dir}/epoch_{N}_metrics.json.
  """

  def on_epoch_end(self, trainer, module, epoch, result=None):
    """Write metrics JSON when result and experiment are available.

    Args:
      trainer: Active Trainer instance.
      module: The module being trained.
      epoch: Current epoch number.
      result: Result with metrics dict, or None.
    """
    if result is None:
      return
    experiment = trainer.experiment
    if experiment is None:
      return
    exp_dir = trainer.config.experiment_path(slug=experiment.id)
    path = exp_dir / f'epoch_{epoch}_metrics.json'
    atomic_write_json(path, result.metrics)


class OptimizerContextCallback(Callback):
  """Keep AgentOptimizer context in sync with epoch and metrics.

  Trainer auto-wires initial context (epoch 0, no metrics) but
  never refreshes it. This callback bridges that gap so the
  optimizer sees the current epoch and latest metric values when
  building prompts for the coding agent.

  Additionally emits a context entry via ``trainer.emit_context()``
  when the primary validation metric (``task_success_rate``) improves
  over the prior best observed by this callback. Emission is sparse:
  one entry per improvement, not per epoch.
  """

  def __init__(self) -> None:
    super().__init__()
    self._best: float = float('-inf')

  def on_epoch_start(self, trainer, module, epoch):
    """Update optimizer context with current epoch number.

    Args:
      trainer: Active Trainer instance.
      module: The module being trained.
      epoch: Current epoch number.
    """
    opt = trainer.optimizer
    if isinstance(opt, AgentOptimizer):
      opt.update_context(epoch=epoch)

  def on_epoch_end(self, trainer, module, epoch, result=None):
    """Update optimizer context with latest metrics and emit on improvement.

    Args:
      trainer: Active Trainer instance.
      module: The module being trained.
      epoch: Current epoch number.
      result: Result with metrics dict, or None.
    """
    opt = trainer.optimizer
    if isinstance(opt, AgentOptimizer) and result is not None:
      opt.update_context(metrics=dict(result.metrics))

    if result is None:
      return
    metrics = result.metrics
    val_key = f'val_{PRIMARY_METRIC_KEY}'
    current_metric = metrics.get(val_key) or metrics.get(PRIMARY_METRIC_KEY)
    if current_metric is None:
      return

    current_value = float(current_metric)
    if current_value > self._best:
      prior_best = self._best
      self._best = current_value
      trainer.emit_context(
        'harness optimization decision: val improved vs prior best',
        source='harness',
        metadata={
          'epoch': epoch,
          'metric': current_value,
          'prior_best': prior_best,
        },
      )


class DeployCallback(Callback):
  """No-op deployment hook for production promotion simulation.

  In a real deployment, this would push parameters to serving
  infrastructure. Here it logs what would be deployed.
  """

  def on_fit_end(self, trainer, module):
    """Print parameter names that would be deployed.

    Args:
      trainer: Active Trainer instance.
      module: The module whose parameters would be deployed.
    """
    params = list(module.named_parameters())
    param_names = [name for name, _ in params]
    print(f'[deploy] Would deploy parameters: {param_names}')


class HarnessCostTrackerCallback(CostTrackerCallback):
  """Cost tracker that populates ``api_calls`` and ``tokens_used`` from harness metrics.

  The harness ``HarnessMetrics`` collection includes three sum metrics
  (``total_input_tokens``, ``total_output_tokens``, ``total_api_calls``)
  sourced from ``EvalDatum.metadata``. After the epoch loop merges train
  and validation results, ``result.metrics`` carries these keys (with
  ``val_*`` prefixes for validation-split values when both splits ran).

  ``measure()`` reads those metric keys and sets ``CostEntry.api_calls``
  to the train + val API call sums and ``CostEntry.tokens_used`` to the
  total input + output tokens across both splits.

  Use this callback on standalone ``build_trainer()`` paths where the
  harness owns the full trainer lifecycle. On ``optimize loop`` paths,
  the framework injects its own ``CostTrackerCallback`` -- do **not**
  pre-register this callback there to avoid duplicate cost trackers.
  """

  def measure(self, epoch: int, elapsed: float, result: Any = None) -> CostEntry:
    """Build a ``CostEntry`` with token and API call totals from epoch metrics.

    Reads train-split keys (``total_input_tokens``, ``total_output_tokens``,
    ``total_api_calls``) and, when present, val-split keys (``val_total_*``).
    Sums train + val for each dimension.

    Args:
      epoch: Epoch index.
      elapsed: Wall-clock seconds for the epoch.
      result: Trainer result with ``metrics`` dict.

    Returns:
      A ``CostEntry`` with ``api_calls`` and ``tokens_used`` populated.
    """
    entry = super().measure(epoch, elapsed, result)
    if result is None or not hasattr(result, 'metrics') or not result.metrics:
      return entry

    metrics = result.metrics

    train_input = int(metrics.get('total_input_tokens') or 0)
    train_output = int(metrics.get('total_output_tokens') or 0)
    train_api = int(metrics.get('total_api_calls') or 0)

    val_input = int(metrics.get('val_total_input_tokens') or 0)
    val_output = int(metrics.get('val_total_output_tokens') or 0)
    val_api = int(metrics.get('val_total_api_calls') or 0)

    entry.api_calls = train_api + val_api
    entry.tokens_used = train_input + train_output + val_input + val_output
    return entry
