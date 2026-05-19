"""ContextLogCallback: wires context entries to experiment.context_log.

Attached by Trainer when ``enable_context_log=True`` (default) and an
experiment is present. When a user provides their own callback with
``_is_context_log_callback = True``, the default is skipped (replacement
pattern matching PyTorch Lightning's ``_CallbackConnector``).

Detection uses the ``_is_context_log_callback`` class-level flag (DRY-06),
not ``isinstance``, so Trainer never couples to this concrete type.

Recording uses ``ContextLog.record(entry)``, which calls ``accept()`` on the
log before appending -- entries rejected by the log's accept gate are silently
dropped even if ``should_record`` returned True.

Override points:
  should_record(entry) -- return False to suppress specific entries.
  on_context_emit(trainer, module, entry) -- full custom recording logic.
"""

from autopilot.core.callbacks.callback import Callback
from typing import Any


class ContextLogCallback(Callback):
  """Writes context entries to experiment.context_log.

  Attached by Trainer when ``enable_context_log=True`` (default) and an
  experiment is present. Users replace this by providing their own callback
  with ``_is_context_log_callback = True``; the default is then skipped.
  To disable context recording entirely, set ``enable_context_log=False``
  on the Trainer constructor.

  Uses ``context_log.record(entry)`` which respects ``ContextLog.accept()``.
  Override ``should_record`` to filter entries before recording.
  Override ``on_context_emit`` for fully custom recording logic;
  call ``super().on_context_emit(...)`` to preserve the default path.
  """

  _is_context_log_callback = True

  def should_record(self, entry: Any) -> bool:
    """Pre-record filter hook. Override to reject specific entries.

    Args:
      entry: The ContextEntry about to be recorded.

    Returns:
      True to record, False to skip.
    """
    return True

  def on_context_emit(self, trainer: Any, module: Any, entry: Any) -> None:
    """Record entry to experiment context log if accepted.

    Silently no-ops when ``trainer.experiment`` is None (no experiment
    active). Other callbacks still receive the entry for monitoring.

    Args:
      trainer: The active Trainer instance.
      module: The module being trained.
      entry: The ContextEntry emitted by Trainer.emit_context().
    """
    if trainer.experiment is not None and self.should_record(entry):
      trainer.experiment.context_log.record(entry)
