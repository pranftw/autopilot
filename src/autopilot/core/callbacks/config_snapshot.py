"""Optional callback that captures module state at fit start for diffing.

Opt-in only: add ``ConfigSnapshotCallback()`` to ``Trainer(callbacks=[...])``.
Not auto-attached. Emits a context entry with the module's ``state_dict()``
at the start of ``fit()``, enabling post-training comparison of initial vs
final parameter state.

Warning: large ``PathParameter`` entries may bloat the context log. Consider
truncating or filtering when modules have very large file-based parameters.
"""

from autopilot.core.callbacks.callback import Callback
from typing import Any


class ConfigSnapshotCallback(Callback):
  """Captures ``module.state_dict()`` at fit start via ``emit_context``.

  On ``on_fit_start``, emits a context entry with ``source='trainer'`` and
  the module state dict in metadata under ``'module_state_dict'``. This
  enables diffing the initial parameter state against final state.

  Warning: modules with large file-based parameters may produce very large
  context entries. Consider truncation or selective use.
  """

  def on_fit_start(self, trainer: Any, module: Any) -> None:
    """Emit module state snapshot as a context entry.

    Args:
      trainer: Active Trainer instance with ``emit_context``.
      module: Module whose ``state_dict()`` is captured.
    """
    state = module.state_dict()
    trainer.emit_context(
      'config snapshot at fit start',
      source='trainer',
      metadata={'module_state_dict': state},
    )
