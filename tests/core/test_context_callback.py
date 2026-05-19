"""Tests for the on_context_emit callback hook and ContextLogCallback.

Covers:
  - on_context_emit hook existence and no-op default on Callback.
  - ContextLogCallback sentinel flag, recording, filtering, override, and
    multi-callback dispatch.
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.callbacks.context import ContextLogCallback
from autopilot.core.context import ContextEntry, ContextLog
from types import SimpleNamespace


def _make_entry(reason: str = 'test reason') -> ContextEntry:
  """Build a ContextEntry via the canonical factory."""
  return ContextEntry.create(reason, source='test')


def _make_trainer(*, with_experiment: bool = True) -> SimpleNamespace:
  """Build a minimal trainer stub with optional experiment.

  When with_experiment is True, the stub has an experiment namespace with
  a real ContextLog. When False, experiment is None.
  """
  if with_experiment:
    experiment = SimpleNamespace(context_log=ContextLog())
    return SimpleNamespace(experiment=experiment)
  return SimpleNamespace(experiment=None)


# -- 2.1 tests: on_context_emit hook on Callback base class --


def test_on_context_emit_hook_exists_on_callback():
  """Callback defines on_context_emit as a callable method."""
  cb = Callback()
  assert hasattr(cb, 'on_context_emit')
  assert callable(cb.on_context_emit)


def test_on_context_emit_default_is_noop():
  """Default on_context_emit returns None and mutates nothing."""
  cb = Callback()
  trainer = _make_trainer()
  entry = _make_entry()
  result = cb.on_context_emit(trainer, None, entry)
  assert result is None
  assert len(trainer.experiment.context_log) == 0


# -- 2.2 tests: ContextLogCallback --


def test_context_log_callback_has_flag():
  """ContextLogCallback._is_context_log_callback is True."""
  assert ContextLogCallback._is_context_log_callback is True


def test_context_log_callback_flag_on_instance():
  """Instance-level access to sentinel flag works."""
  cb = ContextLogCallback()
  assert cb._is_context_log_callback is True


def test_context_log_callback_records_to_experiment():
  """on_context_emit records entry into experiment.context_log."""
  cb = ContextLogCallback()
  trainer = _make_trainer(with_experiment=True)
  entry = _make_entry('optimizer applied changes')

  cb.on_context_emit(trainer, None, entry)

  log = trainer.experiment.context_log
  assert len(log) == 1
  assert log.entries[0].reason == 'optimizer applied changes'
  assert log.entries[0] is entry


def test_context_log_callback_no_experiment_noop():
  """on_context_emit silently no-ops when trainer.experiment is None."""
  cb = ContextLogCallback()
  trainer = _make_trainer(with_experiment=False)
  entry = _make_entry()

  cb.on_context_emit(trainer, None, entry)


def test_context_log_callback_should_record_default_true():
  """Default should_record returns True for any entry."""
  cb = ContextLogCallback()
  entry = _make_entry()
  assert cb.should_record(entry) is True


def test_context_log_callback_should_record_override():
  """Subclass overriding should_record=False prevents recording."""

  class RejectAll(ContextLogCallback):
    """Rejects every entry."""

    def should_record(self, entry):
      """Reject all entries."""
      return False

  cb = RejectAll()
  trainer = _make_trainer(with_experiment=True)
  entry = _make_entry('should be rejected')

  cb.on_context_emit(trainer, None, entry)
  assert len(trainer.experiment.context_log) == 0


def test_context_log_callback_should_record_selective_filter():
  """Subclass can selectively filter by entry content."""

  class OnlyPolicy(ContextLogCallback):
    """Only records entries from the policy source."""

    def should_record(self, entry):
      """Accept only policy-sourced entries."""
      return entry.source == 'policy'

  cb = OnlyPolicy()
  trainer = _make_trainer(with_experiment=True)

  policy_entry = ContextEntry.create('gate rejected', source='policy')
  trainer_entry = ContextEntry.create('epoch start', source='trainer')

  cb.on_context_emit(trainer, None, policy_entry)
  cb.on_context_emit(trainer, None, trainer_entry)

  log = trainer.experiment.context_log
  assert len(log) == 1
  assert log.entries[0].source == 'policy'


def test_context_log_callback_on_context_emit_override():
  """Subclass can replace on_context_emit with custom side effect."""
  recorded = []

  class CustomRecorder(ContextLogCallback):
    """Records entries to an external list instead of context_log."""

    def on_context_emit(self, trainer, module, entry):
      """Custom recording to external list."""
      recorded.append(entry)

  cb = CustomRecorder()
  trainer = _make_trainer(with_experiment=True)
  entry = _make_entry('custom path')

  cb.on_context_emit(trainer, None, entry)

  assert len(recorded) == 1
  assert recorded[0].reason == 'custom path'
  assert len(trainer.experiment.context_log) == 0


def test_context_log_callback_on_context_emit_override_with_super():
  """Subclass can augment recording by calling super()."""
  side_effects = []

  class AugmentedRecorder(ContextLogCallback):
    """Records to both context_log (via super) and external list."""

    def on_context_emit(self, trainer, module, entry):
      """Augmented recording: side effect + default path."""
      side_effects.append(entry.reason)
      super().on_context_emit(trainer, module, entry)

  cb = AugmentedRecorder()
  trainer = _make_trainer(with_experiment=True)
  entry = _make_entry('augmented')

  cb.on_context_emit(trainer, None, entry)

  assert side_effects == ['augmented']
  assert len(trainer.experiment.context_log) == 1


def test_multiple_callbacks_all_fire():
  """Multiple callbacks all receive on_context_emit when dispatched in a loop."""
  callbacks = [ContextLogCallback(), ContextLogCallback()]
  trainer = _make_trainer(with_experiment=True)
  entry = _make_entry('broadcast')

  for cb in callbacks:
    cb.on_context_emit(trainer, None, entry)

  assert len(trainer.experiment.context_log) == 2


def test_multiple_mixed_callbacks_all_fire():
  """Mixed Callback and ContextLogCallback both receive the hook."""
  base_called = []

  class TrackingCallback(Callback):
    """Tracks on_context_emit invocations."""

    def on_context_emit(self, trainer, module, entry):
      """Track call."""
      base_called.append(entry)

  callbacks = [TrackingCallback(), ContextLogCallback()]
  trainer = _make_trainer(with_experiment=True)
  entry = _make_entry('mixed dispatch')

  for cb in callbacks:
    cb.on_context_emit(trainer, None, entry)

  assert len(base_called) == 1
  assert len(trainer.experiment.context_log) == 1


def test_context_log_callback_inherits_callback():
  """ContextLogCallback is a proper Callback subclass."""
  cb = ContextLogCallback()
  assert isinstance(cb, Callback)


def test_context_log_callback_respects_accept_gate():
  """ContextLog.accept() gate is respected during recording."""

  class RejectingLog(ContextLog):
    """Log that rejects all entries."""

    def accept(self, entry):
      """Reject all entries."""
      return False

  cb = ContextLogCallback()
  experiment = SimpleNamespace(context_log=RejectingLog())
  trainer = SimpleNamespace(experiment=experiment)
  entry = _make_entry('will be rejected by log')

  cb.on_context_emit(trainer, None, entry)

  assert len(trainer.experiment.context_log) == 0


def test_dispatch_callbacks_pattern():
  """Simulates Trainer.dispatch_callbacks('on_context_emit', entry=entry)."""
  callbacks = [ContextLogCallback(), Callback()]
  trainer = _make_trainer(with_experiment=True)
  entry = _make_entry('dispatch pattern')

  for cb in callbacks:
    hook = getattr(cb, 'on_context_emit', None)
    if hook is not None and callable(hook):
      hook(trainer, None, entry=entry)

  assert len(trainer.experiment.context_log) == 1


def test_base_callback_flag_absent():
  """Base Callback does not have the _is_context_log_callback flag."""
  cb = Callback()
  assert not getattr(cb, '_is_context_log_callback', False)
