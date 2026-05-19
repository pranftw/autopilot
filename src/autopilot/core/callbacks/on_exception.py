"""OnExceptionCallback: best-effort crash checkpoint and store snapshot.

Saves a crash checkpoint and optional store snapshot when an unhandled
exception occurs during ``Trainer.fit``. On clean teardown (no exception
fired), removes any stale crash checkpoint from a prior run. After an
exception, teardown preserves the crash file for recovery.

Best-effort semantics: save failures are swallowed (narrow exceptions)
and must not mask the original exception that triggered the callback.
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.errors import StoreError
from autopilot.data.datamodule import Stage
from pathlib import Path
from typing import Any
import contextlib


class OnExceptionCallback(Callback):
  """Save crash checkpoint and store snapshot on exception during fit.

  On clean teardown (no exception fired), removes any stale crash checkpoint.
  After an exception, teardown preserves the crash file for recovery.

  Args:
    directory: Optional directory for the crash checkpoint file.
      When omitted, uses ``trainer.config.root`` (if experiment present)
      or process CWD as fallback.
  """

  def __init__(self, directory: Path | None = None) -> None:
    """Initialize the on-exception callback.

    Args:
      directory: Optional explicit directory for crash checkpoint output.
        When ``None``, path is resolved from trainer config at exception time.
    """
    self._directory = directory
    self._crash_path: Path | None = None
    self._exception_fired: bool = False

  def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
    """Resolve crash checkpoint path at fit start for stale cleanup.

    When a prior crashed run left a crash checkpoint on disk, this
    allows ``teardown()`` to remove it if the current run succeeds.

    Args:
      trainer: The Trainer instance.
      module: The module being trained.
      stage: Current stage (fit/validate/test/predict).
    """
    self._crash_path = self._resolve_path(trainer)
    self._exception_fired = False

  def on_exception(self, trainer: Any, module: Any, exception: BaseException) -> None:
    """Save crash checkpoint and store snapshot on exception.

    Best-effort: failures in save_checkpoint or store.snapshot are caught
    (narrow exception types) so the original exception always propagates.

    Args:
      trainer: Active Trainer instance with ``save_checkpoint`` and ``store``.
      module: Module being trained when the error occurred.
      exception: The exception that was raised.
    """
    self._exception_fired = True
    path = self._resolve_path(trainer)
    with contextlib.suppress(OSError, RuntimeError):
      trainer.save_checkpoint(path)
    if trainer.store is not None and trainer.experiment is not None:
      with contextlib.suppress(OSError, RuntimeError, StoreError):
        trainer.store.snapshot(
          trainer.experiment.id,
          trainer.current_epoch,
          context=f'crash: {type(exception).__name__}',
        )
    self._crash_path = path

  def teardown(self, trainer: Any, module: Any, stage: Stage) -> None:
    """Remove stale crash checkpoint on clean exit; preserve on exception.

    If ``on_exception`` fired, the crash file is preserved for recovery.
    If no exception occurred, stale crash files from prior runs are cleaned up.

    Args:
      trainer: Active Trainer instance.
      module: Module that was trained.
      stage: Lifecycle stage that is ending.
    """
    if self._crash_path is not None and self._crash_path.exists() and not self._exception_fired:
      self._crash_path.unlink()
    self._crash_path = None

  def _resolve_path(self, trainer: Any) -> Path:
    """Determine the crash checkpoint file path.

    Resolution order:
      1. Explicit ``directory`` from constructor -> ``directory/crash_checkpoint.json``
      2. ``trainer.config.root`` when experiment is set -> ``{root}/crash_checkpoint.json``
      3. Fallback: ``crash_checkpoint.json`` in process CWD.

    Args:
      trainer: Active Trainer instance.

    Returns:
      Resolved path for the crash checkpoint file.
    """
    if self._directory is not None:
      return self._directory / 'crash_checkpoint.json'
    if trainer.experiment is not None:
      return Path(trainer.config.root) / 'crash_checkpoint.json'
    return Path('crash_checkpoint.json')
