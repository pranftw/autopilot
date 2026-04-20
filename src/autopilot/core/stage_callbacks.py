"""Stage callbacks for epoch recording, regression detection, judge validation, and memory."""

from autopilot.core.callbacks import Callback
from autopilot.core.memory import Memory
from autopilot.core.models import Datum, Result
from autopilot.core.regression import (
  compare_metrics,
  is_regression,
  read_best_baseline,
  write_best_baseline,
)
from autopilot.core.stage_io import (
  append_epoch_artifact,
  read_epoch_artifact_lines,
  write_epoch_artifact,
  write_experiment_artifact,
)
from autopilot.core.stage_models import EpochMetrics, RegressionAnalysis
from pathlib import Path
from typing import Any
import time


class EpochRecorderCallback(Callback):
  """Records per-batch data and epoch metrics to artifact files."""

  def __init__(
    self,
    experiment_dir: Path,
    data_filename: str = 'data.jsonl',
    metrics_filename: str = 'epoch_metrics.json',
    delta_filename: str = 'delta_metrics.json',
  ) -> None:
    self._dir = experiment_dir
    self._data_filename = data_filename
    self._metrics_filename = metrics_filename
    self._delta_filename = delta_filename
    self._batch_data: list[dict[str, Any]] = []
    self._current_epoch = 0
    self._prev_metrics: dict[str, float] = {}

  def on_train_epoch_start(self, trainer: Any, epoch: int) -> None:
    self._current_epoch = epoch
    self._batch_data = []

  def on_train_batch_end(
    self,
    trainer: Any,
    batch_idx: int = 0,
    data: Any = None,
  ) -> None:
    if data is not None:
      if isinstance(data, Datum):
        self._batch_data.append(data.to_dict())
      elif isinstance(data, dict):
        self._batch_data.append(data)

  def on_train_epoch_end(self, trainer: Any, epoch: int) -> None:
    for item in self._batch_data:
      append_epoch_artifact(self._dir, epoch, self._data_filename, item)

    total = len(self._batch_data)
    passed = sum(1 for d in self._batch_data if d.get('success', True))
    failed = total - passed
    accuracy = passed / total if total > 0 else 0.0

    metrics_data = EpochMetrics(
      epoch=epoch,
      split='train',
      total=total,
      passed=passed,
      failed=failed,
      accuracy=accuracy,
      error_rate=1.0 - accuracy,
    ).to_dict()
    write_epoch_artifact(self._dir, epoch, self._metrics_filename, metrics_data)

    if self._prev_metrics:
      delta = {k: accuracy - self._prev_metrics.get(k, 0.0) for k in ['accuracy']}
      write_epoch_artifact(self._dir, epoch, self._delta_filename, delta)

    self._prev_metrics = {'accuracy': accuracy}

  def state_dict(self) -> dict[str, Any]:
    return {'prev_metrics': self._prev_metrics}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    self._prev_metrics = state_dict.get('prev_metrics', {})


class JudgeValidationCallback(Callback):
  """Cross-validates judge results after backward pass."""

  def __init__(
    self,
    experiment_dir: Path,
    filename: str = 'judge_validation.json',
  ) -> None:
    self._dir = experiment_dir
    self._filename = filename
    self._current_epoch = 0

  def on_train_epoch_start(self, trainer: Any, epoch: int) -> None:
    self._current_epoch = epoch

  def on_after_backward(self, trainer: Any) -> None:
    pass

  def state_dict(self) -> dict[str, Any]:
    return {}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    pass


class RegressionCallback(Callback):
  """Detects regression by comparing validation metrics to best baseline.

  Signals regression by setting trainer.regression_detected = True.
  Does NOT call store.checkout() -- callbacks observe, loops control.
  """

  def __init__(
    self,
    experiment_dir: Path,
    threshold_pct: float = 0.0,
    analysis_filename: str = 'regression_analysis.json',
  ) -> None:
    self._dir = experiment_dir
    self._threshold_pct = threshold_pct
    self._analysis_filename = analysis_filename
    self._current_epoch = 0

  def on_train_epoch_start(self, trainer: Any, epoch: int) -> None:
    self._current_epoch = epoch

  def on_validation_epoch_end(self, trainer: Any, epoch: int) -> None:
    val_metrics = getattr(trainer, '_last_val_metrics', None)
    if val_metrics is None:
      return

    baseline = read_best_baseline(self._dir)
    if baseline is None:
      write_best_baseline(self._dir, epoch, val_metrics)
      return

    analysis = compare_metrics(
      baseline,
      val_metrics,
      threshold_pct=self._threshold_pct,
    )
    analysis_with_epoch = RegressionAnalysis(
      epoch=epoch,
      overall_verdict=analysis.overall_verdict,
      per_category_deltas=analysis.per_category_deltas,
      regressions=analysis.regressions,
      improvements=analysis.improvements,
    )
    write_epoch_artifact(
      self._dir,
      epoch,
      self._analysis_filename,
      analysis_with_epoch.to_dict(),
    )

    if is_regression(analysis):
      trainer.regression_detected = True
    else:
      write_best_baseline(self._dir, epoch, val_metrics)

  def state_dict(self) -> dict[str, Any]:
    return {}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    pass


class MemoryCallback(Callback):
  """Auto-records structured learnings and syncs blocked strategies."""

  def __init__(self, memory: Memory, default_category: str = 'epoch_result') -> None:
    self._memory = memory
    self._default_category = default_category

  def on_epoch_end(
    self,
    trainer: Any,
    epoch: int,
    result: Result | None = None,
  ) -> None:
    metrics = result.metrics if result is not None else {}
    outcome = 'worked' if result is not None and result.passed else 'failed'

    val_metrics = getattr(trainer, '_last_val_metrics', None)
    if isinstance(val_metrics, dict) and val_metrics:
      metrics = {**metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}

    regression = getattr(trainer, 'regression_detected', False)
    if regression is True:
      outcome = 'regression'

    ctx = getattr(trainer, '_fit_ctx', None)
    strategy = ctx.get('strategy', '') if isinstance(ctx, dict) else ''

    self._memory.learn(
      epoch=epoch,
      outcome=outcome,
      category=self._default_category,
      strategy=strategy,
      metrics=metrics,
    )

  def on_before_optimizer_step(self, trainer: Any) -> None:
    if trainer.optimizer:
      for strategy in self._memory.blocked_strategies():
        trainer.optimizer.block_strategy(strategy)

  def state_dict(self) -> dict[str, Any]:
    return self._memory.state_dict()

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    self._memory.load_state_dict(state_dict)


class DiagnoseCallback(Callback):
  """Produces trace_diagnoses.jsonl and node_heatmap.json from batch data.

  Runs on_train_epoch_end: reads the epoch's data.jsonl, categorizes failures,
  and writes diagnostic artifacts for the diagnose command.
  """

  def __init__(self, experiment_dir: Path) -> None:
    self._dir = experiment_dir

  def on_train_epoch_end(self, trainer: Any, epoch: int) -> None:
    data = read_epoch_artifact_lines(self._dir, epoch, 'data.jsonl')
    if not data:
      return

    categories: dict[str, list[str]] = {}
    heatmap: dict[str, dict[str, int]] = {}

    for item in data:
      node = item.get('item_id') or item.get('metadata', {}).get('node', 'unknown')
      category = item.get('metadata', {}).get('category', 'uncategorized')
      success = item.get('success', True)
      error_msg = item.get('error_message', '')

      if node not in heatmap:
        heatmap[node] = {'total': 0, 'failed': 0}
      heatmap[node]['total'] += 1

      if not success or error_msg:
        heatmap[node]['failed'] += 1
        if category not in categories:
          categories[category] = []
        if len(categories[category]) < 5:
          categories[category].append(error_msg or 'failed (no message)')

    for diagnosis_cat, sample_errors in categories.items():
      append_epoch_artifact(
        self._dir,
        epoch,
        'trace_diagnoses.jsonl',
        {
          'category': diagnosis_cat,
          'count': len(sample_errors),
          'sample_errors': sample_errors,
        },
      )

    heatmap_out: dict[str, Any] = {}
    for node, counts in heatmap.items():
      total = counts['total']
      failed = counts['failed']
      heatmap_out[node] = {
        'total': total,
        'failed': failed,
        'error_rate': round(failed / total, 4) if total > 0 else 0.0,
      }
    write_epoch_artifact(self._dir, epoch, 'node_heatmap.json', heatmap_out)

  def state_dict(self) -> dict[str, Any]:
    return {}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    pass


class RunStateCallback(Callback):
  """Persists run state for crash detection and stop-reason forensics.

  Writes run_state.json with status='running' on each epoch end.
  On loop end, updates with status='completed' and stop_reason.
  If the process dies, run_state.json will still say 'running'.
  """

  def __init__(self, experiment_dir: Path) -> None:
    self._dir = experiment_dir

  def on_epoch_end(
    self,
    trainer: Any,
    epoch: int,
    result: Result | None = None,
  ) -> None:
    write_experiment_artifact(
      self._dir,
      'run_state.json',
      {
        'epoch': epoch,
        'timestamp': time.time(),
        'status': 'running',
      },
    )

  def on_loop_end(self, trainer: Any, result: dict[str, Any]) -> None:
    write_experiment_artifact(
      self._dir,
      'run_state.json',
      {
        'epoch': result.get('total_epochs', 0),
        'timestamp': time.time(),
        'status': 'completed',
        'stop_reason': result.get('stop_reason'),
        'last_good_epoch': result.get('last_good_epoch', 0),
      },
    )

  def state_dict(self) -> dict[str, Any]:
    return {}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    pass
