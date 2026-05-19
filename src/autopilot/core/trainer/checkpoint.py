"""Checkpoint save, restore, and resume token resolution.

Domain module for Trainer checkpoint concerns. All functions take
a ``trainer`` instance (typed as ``Any`` to avoid circular imports) as
their first argument.

Resolution order for ``'last'``/``'best'`` resume tokens:
  1. ``CheckpointCallback`` in-memory properties (set during same process).
  2. Disk scan of ``epoch-NNNN.json`` files under the callback's directory
     (only when the in-memory path is ``None`` -- e.g., fresh process after
     prior crash). Corrupt files are skipped silently.
  3. ``ConfigError`` when neither path resolves.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.core.callbacks.checkpoint import CheckpointCallback
from autopilot.core.checkpoint import CheckpointIO, JSONCheckpointIO
from autopilot.core.errors import ConfigError
from autopilot.core.trainer.journal import profile_store_section
from pathlib import Path
from typing import Any
import json
import re

_EPOCH_CKPT_PATTERN = re.compile(r'^epoch-(\d+)\.json$')


def _epoch_key(item: tuple[int, Path]) -> int:
  """Extract epoch integer for sorting checkpoint candidates.

  Returns:
    The epoch number from the tuple.
  """
  return item[0]


def save_checkpoint(
  trainer: Any,
  path: Path,
  checkpoint_io: CheckpointIO | None = None,
) -> None:
  """Assemble full training state and persist via checkpoint IO.

  State includes: experiment, module, optimizer, scheduler, callbacks (when
  present). Like Lightning's ``Trainer.save_checkpoint``. Callbacks receive
  the assembled dict via ``on_save_checkpoint`` and may mutate it in-place
  before the dict is written to storage.

  Args:
    trainer: ``Trainer`` instance.
    path: File path for the checkpoint.
    checkpoint_io: Storage backend. Defaults to ``JSONCheckpointIO``.
  """
  io = checkpoint_io or JSONCheckpointIO()
  state = build_checkpoint_state(trainer)
  trainer.dispatch_callbacks('on_save_checkpoint', checkpoint=state)
  io.save(state, path)


def build_checkpoint_state(trainer: Any) -> dict[str, Any]:
  """Assemble checkpoint dict from all components.

  When a ``DataModule`` is configured, its ``state_dict()`` is always
  included under the ``'datamodule'`` key (even when empty) so that the
  resume path is uniform.

  Args:
    trainer: ``Trainer`` instance.

  Returns:
    JSON-serializable dict with keys for each present component.
  """
  state: dict[str, Any] = {}
  if trainer._experiment is not None:
    state['experiment'] = trainer._experiment.state_dict()
  if trainer._module is not None:
    state['module'] = trainer._module.state_dict()
  if trainer._optimizer is not None:
    state['optimizer'] = trainer._optimizer.state_dict()
  if trainer._scheduler is not None:
    state['scheduler'] = trainer._scheduler.state_dict()
  if trainer._datamodule is not None:
    state['datamodule'] = trainer._datamodule.state_dict()
  if trainer._callbacks:
    state['callbacks'] = {
      f'{type(cb).__name__}_{idx}': cb.state_dict() for idx, cb in enumerate(trainer._callbacks)
    }
  return state


def restore_from_checkpoint(
  trainer: Any,
  state: dict[str, Any],
  module: Any,
) -> None:
  """Restore component state from a loaded checkpoint dict.

  Datamodule state is restored before module/optimizer so that iterator
  recreation can depend on data-module state if needed.

  When a Store is configured and the checkpoint contains experiment state
  with an epoch, triggers ``store.checkout`` to restore PathParameter
  backing files from the content-addressed store (BUG-056).

  Args:
    trainer: ``Trainer`` instance.
    state: Checkpoint dict as returned by :func:`build_checkpoint_state`.
    module: Module instance being restored (same as ``trainer._module``).
  """
  if 'datamodule' in state and trainer._datamodule is not None:
    trainer._datamodule.load_state_dict(state['datamodule'])
  if 'experiment' in state and trainer._experiment is not None:
    trainer._experiment.load_state_dict(state['experiment'])
  if 'module' in state:
    module.load_state_dict(state['module'])
  if 'optimizer' in state and trainer._optimizer is not None:
    trainer._optimizer.load_state_dict(state['optimizer'])
  if 'scheduler' in state and trainer._scheduler is not None:
    trainer._scheduler.load_state_dict(state['scheduler'])
  if 'callbacks' in state:
    for idx, cb in enumerate(trainer._callbacks):
      key = f'{type(cb).__name__}_{idx}'
      if key in state['callbacks']:
        cb.load_state_dict(state['callbacks'][key])
  restore_path_parameter_files(trainer, state, module)


def restore_path_parameter_files(
  trainer: Any,
  state: dict[str, Any],
  module: Any,
) -> None:
  """Restore PathParameter backing files from Store on checkpoint resume.

  When both a Store and an experiment with a valid epoch are available,
  triggers ``store.checkout`` to restore the content-addressed file blobs
  for PathParameter instances. Raises on failure so checkpoint resume
  does not proceed with stale files.

  Args:
    trainer: ``Trainer`` instance.
    state: Checkpoint state dict.
    module: Module whose PathParameters may need file restoration.
  """
  if trainer.store is None or trainer._experiment is None:
    return
  exp_state = state.get('experiment')
  if exp_state is None:
    return
  epoch = exp_state.get('epoch')
  if epoch is None or epoch < 0:
    return
  has_path_params = any(isinstance(p, PathParameter) for p in module.parameters())
  if not has_path_params:
    return
  checkout_context = f'checkpoint resume epoch {epoch}'
  with profile_store_section(trainer, 'store_checkout'):
    trainer.store.checkout(trainer._experiment.id, epoch, context=checkout_context)


def resolve_checkpoint_resume(
  trainer: Any,
  ckpt_path: Path | None,
  checkpoint_io: CheckpointIO | None,
  module: Any,
) -> int:
  """Load checkpoint and compute min_epoch for resume.

  Args:
    trainer: ``Trainer`` instance.
    ckpt_path: Resolved checkpoint path or ``None``.
    checkpoint_io: Storage backend for loading.
    module: Module instance being restored.

  Returns:
    min_epoch value (0 if no checkpoint).
  """
  if ckpt_path is not None:
    io = checkpoint_io or JSONCheckpointIO()
    ckpt_state = io.load(ckpt_path)
    trainer.dispatch_callbacks('on_load_checkpoint', checkpoint=ckpt_state)
    trainer._restore_from_checkpoint(ckpt_state, module)
  min_epoch = 0
  if ckpt_path is not None and trainer._experiment is not None:
    min_epoch = trainer._experiment.epoch + 1
  return min_epoch


def _scan_checkpoint_directory(directory: Path) -> list[Path]:
  """List epoch checkpoint paths under ``directory`` sorted by ascending epoch integer.

  Filename filter: only ``epoch-<int>.json`` matching ``CheckpointCallback``'s
  writer format ``f'epoch-{epoch:04d}.json'`` (zero-padded width >= 4 for typical
  epochs; wider digit strings when epoch >= 10000).

  Sort by parsed integer epoch (not lexicographic filename sort).
  Listing is filename-driven; JSON validity is enforced when loading checkpoint
  bodies in the ``last``/``best`` resolvers that call this function.

  Args:
    directory: Checkpoint directory to scan.

  Returns:
    List of matching paths sorted by ascending epoch number. Empty when
    directory does not exist or contains no matching files.
  """
  if not directory.is_dir():
    return []
  candidates: list[tuple[int, Path]] = []
  for entry in directory.iterdir():
    match = _EPOCH_CKPT_PATTERN.match(entry.name)
    if match is not None:
      epoch_num = int(match.group(1))
      candidates.append((epoch_num, entry))
  candidates.sort(key=_epoch_key)
  return [path for _, path in candidates]


def _resolve_last_from_disk(directory: Path) -> Path | None:
  """Return highest-epoch readable checkpoint path, or None.

  Among files matching the epoch filename pattern, ignores checkpoints whose
  JSON cannot be read/parsed (silent skip). If none remain, returns ``None``.

  Args:
    directory: Checkpoint directory to scan.

  Returns:
    Path to the highest-epoch valid checkpoint, or ``None``.
  """
  paths = _scan_checkpoint_directory(directory)
  for path in reversed(paths):
    try:
      text = path.read_text(encoding='utf-8')
      data = json.loads(text)
      if isinstance(data, dict):
        return path
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
      continue
  return None


def _resolve_best_from_disk(directory: Path, monitor: str) -> Path | None:
  """Load each candidate checkpoint, read metrics, pick highest monitor value.

  Skips corrupt checkpoints silently without logging (consistent with last
  resolver). Returns None when no valid checkpoints carry parseable metrics
  for the monitor key. Ties (same metric value) are broken by epoch index
  (later epoch wins).

  Args:
    directory: Checkpoint directory to scan.
    monitor: Metric key to maximize.

  Returns:
    Path to the checkpoint with the highest monitored metric, or ``None``.
  """
  paths = _scan_checkpoint_directory(directory)
  best_path: Path | None = None
  best_value: float | None = None
  for path in paths:
    try:
      text = path.read_text(encoding='utf-8')
      data = json.loads(text)
      if not isinstance(data, dict):
        continue
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
      continue
    metrics = _extract_metrics_from_checkpoint(data)
    if metrics is None or monitor not in metrics:
      continue
    try:
      value = float(metrics[monitor])
    except (TypeError, ValueError):
      continue
    if best_value is None or value >= best_value:
      best_value = value
      best_path = path
  return best_path


def _extract_metrics_from_checkpoint(data: dict[str, Any]) -> dict[str, Any] | None:
  """Extract metrics dict from a checkpoint state dict.

  Looks for metrics in the experiment state (the same location Trainer
  persists them during checkpoint assembly).

  Args:
    data: Loaded checkpoint dict.

  Returns:
    Metrics dict or None if not found.
  """
  exp = data.get('experiment')
  if not isinstance(exp, dict):
    return None
  metrics = exp.get('metrics')
  if isinstance(metrics, dict):
    return metrics
  return None


def resolve_ckpt_path_token(trainer: Any, ckpt_path: Path | str | None) -> Path | None:
  """Resolve string resume tokens to concrete checkpoint paths.

  Handles ``'last'`` and ``'best'`` tokens by delegating to
  :func:`_resolve_last_checkpoint` and :func:`_resolve_best_checkpoint`.
  Unknown string tokens raise ``ConfigError`` with valid options.
  ``Path`` and ``None`` values pass through unchanged.

  Args:
    trainer: ``Trainer`` instance.
    ckpt_path: Raw ckpt_path from ``fit()`` call.

  Returns:
    Resolved ``Path`` or ``None``.

  Raises:
    ConfigError: When the string token is unknown, or when 'last'/'best'
      cannot be resolved due to missing callbacks or checkpoints.
  """
  if ckpt_path is None:
    return None
  if isinstance(ckpt_path, Path):
    return ckpt_path
  if isinstance(ckpt_path, str):
    if ckpt_path == 'last':
      return _resolve_last_checkpoint(trainer)
    if ckpt_path == 'best':
      return _resolve_best_checkpoint(trainer)
    msg = (
      f"Unknown resume token {ckpt_path!r}. Valid tokens: 'last', 'best'. "
      f'Or pass a Path to a specific checkpoint file.'
    )
    raise ConfigError(msg)
  return ckpt_path


def find_checkpoint_callbacks(trainer: Any) -> list[CheckpointCallback]:
  """Find all CheckpointCallback instances in registered callbacks.

  Args:
    trainer: ``Trainer`` instance.

  Returns:
    List of CheckpointCallback instances.
  """
  return [cb for cb in trainer._callbacks if isinstance(cb, CheckpointCallback)]


def _resolve_last_checkpoint(trainer: Any) -> Path:
  """Resolve 'last' token to the most recently saved checkpoint path.

  Resolution order:
    1. In-memory ``CheckpointCallback.last_checkpoint_path`` (primary).
    2. Disk scan of ``epoch-NNNN.json`` under the callback's directory
       (fallback for crash recovery / fresh process).
    3. ``ConfigError`` when neither resolves.

  Args:
    trainer: ``Trainer`` instance.

  Returns:
    Path to the last checkpoint file.

  Raises:
    ConfigError: When no CheckpointCallback is registered, multiple are
      registered (ambiguous), or no checkpoint can be found (memory or disk).
  """
  cbs = find_checkpoint_callbacks(trainer)
  if not cbs:
    msg = (
      "Cannot resolve ckpt_path='last': no CheckpointCallback registered. "
      'Add CheckpointCallback(directory=...) to Trainer callbacks.'
    )
    raise ConfigError(msg)
  if len(cbs) > 1:
    msg = (
      'Multiple CheckpointCallbacks registered; resume token resolution is ambiguous. '
      'Use an explicit Path instead.'
    )
    raise ConfigError(msg)
  cb = cbs[0]
  path = cb.last_checkpoint_path
  if path is not None:
    return path
  directory = cb.directory
  disk_path = _resolve_last_from_disk(directory)
  if disk_path is not None:
    return disk_path
  msg = f'no checkpoints found in {directory}'
  raise ConfigError(msg)


def _resolve_best_checkpoint(trainer: Any) -> Path:
  """Resolve 'best' token to the checkpoint with the best monitored metric.

  Resolution order:
    1. In-memory ``CheckpointCallback.best_checkpoint_path`` (primary).
    2. Disk scan of ``epoch-NNNN.json`` under the callback's directory,
       reading metrics from each checkpoint to pick the highest monitored
       value (fallback for crash recovery / fresh process).
    3. ``ConfigError`` when neither resolves.

  Args:
    trainer: ``Trainer`` instance.

  Returns:
    Path to the best checkpoint file.

  Raises:
    ConfigError: When no CheckpointCallback with monitor is registered,
      multiple callbacks are registered (ambiguous), or no best checkpoint
      can be found (memory or disk).
  """
  cbs = find_checkpoint_callbacks(trainer)
  if not cbs:
    msg = (
      "Cannot resolve ckpt_path='best': no CheckpointCallback registered. "
      'Add CheckpointCallback(directory=..., monitor=...) to Trainer callbacks.'
    )
    raise ConfigError(msg)
  if len(cbs) > 1:
    msg = (
      'Multiple CheckpointCallbacks registered; resume token resolution is ambiguous. '
      'Use an explicit Path instead.'
    )
    raise ConfigError(msg)
  cb = cbs[0]
  if cb.monitor is None:
    msg = (
      "Cannot resolve ckpt_path='best': CheckpointCallback has no monitor set. "
      "Use CheckpointCallback(directory=..., monitor='<metric>') or use ckpt_path='last'."
    )
    raise ConfigError(msg)
  path = cb.best_checkpoint_path
  if path is not None:
    return path
  directory = cb.directory
  disk_path = _resolve_best_from_disk(directory, cb.monitor)
  if disk_path is not None:
    return disk_path
  msg = f'no checkpoints found in {directory}'
  raise ConfigError(msg)
