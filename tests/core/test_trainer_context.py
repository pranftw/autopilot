"""Tests for Trainer.emit_context() and _attach_default_callbacks().

Covers:
  - emit_context dispatches on_context_emit to callbacks (2.1)
  - emit_context entry fields match arguments and trainer state (2.1)
  - emit_context epoch tracks current_epoch (2.1)
  - emit_context with no callbacks is a no-op (2.1)
  - emit_context passes metadata through (2.1)
  - _attach_default_callbacks with experiment appends ContextLogCallback (2.2)
  - _attach_default_callbacks without experiment does nothing (2.2)
  - _attach_default_callbacks is idempotent (2.2)
  - _attach_default_callbacks respects user callback with flag (2.2)
  - fit() attaches context callback when experiment is present (2.2)
  - end-to-end: emit_context -> ContextLogCallback -> experiment.context_log (2.2)
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.callbacks.context import ContextLogCallback
from autopilot.core.context import ContextEntry
from autopilot.core.errors import ConfigError
from autopilot.core.experiment import Experiment
from autopilot.core.trainer.trainer import Trainer
from tests.doubles import NoopEvalModule
import pytest

# -- helpers --


class SpyCallback(Callback):
  """Callback that records on_context_emit calls."""

  def __init__(self) -> None:
    super().__init__()
    self.entries: list[ContextEntry] = []

  def on_context_emit(self, trainer, module, entry) -> None:
    """Record emitted entry."""
    self.entries.append(entry)


class UserContextLogCallback(Callback):
  """User-provided callback with _is_context_log_callback flag set."""

  _is_context_log_callback = True

  def __init__(self) -> None:
    super().__init__()
    self.entries: list[ContextEntry] = []

  def on_context_emit(self, trainer, module, entry) -> None:
    """Record emitted entry to user-controlled list."""
    self.entries.append(entry)


# -- 2.1 tests: emit_context --


def test_emit_context_dispatches_to_callbacks():
  """emit_context triggers on_context_emit on all registered callbacks."""
  spy = SpyCallback()
  trainer = Trainer(callbacks=[spy])

  trainer.emit_context('test reason', source='trainer')

  assert len(spy.entries) == 1
  assert spy.entries[0].reason == 'test reason'


def test_emit_context_entry_has_correct_fields():
  """Emitted entry carries the supplied reason, source, and current epoch."""
  spy = SpyCallback()
  trainer = Trainer(callbacks=[spy])
  trainer.current_epoch = 5

  trainer.emit_context('epoch started', source='trainer')

  entry = spy.entries[0]
  assert entry.reason == 'epoch started'
  assert entry.source == 'trainer'
  assert entry.epoch == 5
  assert entry.timestamp


def test_emit_context_epoch_from_current_epoch():
  """Emitted entry epoch tracks trainer.current_epoch at call time."""
  spy = SpyCallback()
  trainer = Trainer(callbacks=[spy])

  trainer.current_epoch = 0
  trainer.emit_context('epoch 0')
  trainer.current_epoch = 3
  trainer.emit_context('epoch 3')
  trainer.current_epoch = 7
  trainer.emit_context('epoch 7')

  assert spy.entries[0].epoch == 0
  assert spy.entries[1].epoch == 3
  assert spy.entries[2].epoch == 7


def test_emit_context_no_callbacks_noop():
  """emit_context with no callbacks does not raise."""
  trainer = Trainer(callbacks=[])
  trainer.emit_context('no one listening')


def test_emit_context_metadata_passed_through():
  """Metadata dict appears on the emitted entry."""
  spy = SpyCallback()
  trainer = Trainer(callbacks=[spy])

  trainer.emit_context(
    'policy gate',
    source='policy',
    metadata={'accuracy': 0.85, 'threshold': 0.8},
  )

  entry = spy.entries[0]
  assert entry.metadata == {'accuracy': 0.85, 'threshold': 0.8}


def test_emit_context_metadata_none_defaults_to_empty_dict():
  """When metadata is None, the entry gets an empty dict."""
  spy = SpyCallback()
  trainer = Trainer(callbacks=[spy])

  trainer.emit_context('no metadata')

  assert spy.entries[0].metadata == {}


def test_emit_context_uses_context_entry_create():
  """Entry is built via ContextEntry.create() -- has auto-generated timestamp."""
  spy = SpyCallback()
  trainer = Trainer(callbacks=[spy])

  trainer.emit_context('factory check')

  entry = spy.entries[0]
  assert isinstance(entry, ContextEntry)
  assert entry.timestamp
  assert len(entry.timestamp) > 0


def test_emit_context_multiple_callbacks():
  """All callbacks receive the same entry reference."""
  spy1 = SpyCallback()
  spy2 = SpyCallback()
  trainer = Trainer(callbacks=[spy1, spy2])

  trainer.emit_context('broadcast', source='trainer')

  assert len(spy1.entries) == 1
  assert len(spy2.entries) == 1
  assert spy1.entries[0] is spy2.entries[0]


def test_emit_context_source_none_default():
  """Source defaults to None when omitted."""
  spy = SpyCallback()
  trainer = Trainer(callbacks=[spy])

  trainer.emit_context('no source')

  assert spy.entries[0].source is None


# -- 2.2 tests: _attach_default_callbacks --


def test_attach_default_callbacks_with_experiment():
  """With experiment set, _attach_default_callbacks appends ContextLogCallback."""
  exp = Experiment('exp-1')
  trainer = Trainer(experiment=exp)

  trainer._attach_default_callbacks()

  context_cbs = [cb for cb in trainer.callbacks if getattr(cb, '_is_context_log_callback', False)]
  assert len(context_cbs) == 1


def test_attach_default_callbacks_no_experiment():
  """Without experiment, _attach_default_callbacks appends nothing."""
  trainer = Trainer()
  initial_count = len(trainer.callbacks)

  trainer._attach_default_callbacks()

  assert len(trainer.callbacks) == initial_count


def test_attach_default_callbacks_idempotent():
  """Multiple calls do not duplicate the context log callback."""
  exp = Experiment('exp-1')
  trainer = Trainer(experiment=exp)

  trainer._attach_default_callbacks()
  trainer._attach_default_callbacks()
  trainer._attach_default_callbacks()

  context_cbs = [cb for cb in trainer.callbacks if getattr(cb, '_is_context_log_callback', False)]
  assert len(context_cbs) == 1


def test_attach_default_callbacks_user_callback_present():
  """User callback with _is_context_log_callback suppresses auto-attach."""
  exp = Experiment('exp-1')
  user_cb = UserContextLogCallback()
  trainer = Trainer(callbacks=[user_cb], experiment=exp)

  trainer._attach_default_callbacks()

  context_cbs = [cb for cb in trainer.callbacks if getattr(cb, '_is_context_log_callback', False)]
  assert len(context_cbs) == 1
  assert context_cbs[0] is user_cb


def test_attach_default_callbacks_preserves_existing():
  """Auto-attach appends to existing callbacks without replacing them."""
  exp = Experiment('exp-1')
  spy = SpyCallback()
  trainer = Trainer(callbacks=[spy], experiment=exp)

  trainer._attach_default_callbacks()

  assert trainer.callbacks[0] is spy
  assert len(trainer.callbacks) == 2


def test_fit_attaches_context_callback():
  """fit() auto-attaches ContextLogCallback when experiment is present."""
  exp = Experiment('fit-test')
  module = NoopEvalModule()
  trainer = Trainer(experiment=exp)

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  context_cbs = [cb for cb in trainer.callbacks if getattr(cb, '_is_context_log_callback', False)]
  assert len(context_cbs) == 1


def test_fit_no_experiment_no_attach():
  """fit() without experiment does not attach ContextLogCallback."""
  module = NoopEvalModule()
  trainer = Trainer()

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  context_cbs = [cb for cb in trainer.callbacks if getattr(cb, '_is_context_log_callback', False)]
  assert len(context_cbs) == 0


def test_emit_context_end_to_end():
  """emit_context -> ContextLogCallback -> experiment.context_log pipeline."""
  exp = Experiment('e2e-test')
  exp.start()
  trainer = Trainer(experiment=exp)
  trainer._attach_default_callbacks()

  trainer.emit_context('optimizer applied changes', source='agent-optimizer')
  trainer.emit_context('policy gate passed', source='policy', metadata={'score': 0.9})

  log = exp.context_log
  assert len(log) == 2
  assert log.entries[0].reason == 'optimizer applied changes'
  assert log.entries[0].source == 'agent-optimizer'
  assert log.entries[1].reason == 'policy gate passed'
  assert log.entries[1].metadata == {'score': 0.9}


def test_emit_context_end_to_end_via_fit():
  """Full integration: fit() attaches callback, emit_context records to log."""
  recorded = []

  class EmittingCallback(Callback):
    """Callback that emits context during on_fit_start."""

    def on_fit_start(self, trainer, module) -> None:
      """Emit context at fit start."""
      trainer.emit_context('fit started', source='trainer')

    def on_context_emit(self, trainer, module, entry) -> None:
      """Track emitted entries."""
      recorded.append(entry)

  exp = Experiment('e2e-fit')
  module = NoopEvalModule()
  trainer = Trainer(callbacks=[EmittingCallback()], experiment=exp)

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  fit_entries = [e for e in recorded if e.reason == 'fit started']
  assert len(fit_entries) == 1
  assert fit_entries[0].reason == 'fit started'
  log_fit = exp.context_log.search('fit started')
  assert len(log_fit) == 1
  assert log_fit[0].reason == 'fit started'


def test_emit_context_no_experiment_still_dispatches():
  """emit_context dispatches to callbacks even without an experiment."""
  spy = SpyCallback()
  context_cb = ContextLogCallback()
  trainer = Trainer(callbacks=[spy, context_cb])

  trainer.emit_context('no experiment', source='trainer')

  assert len(spy.entries) == 1
  assert spy.entries[0].reason == 'no experiment'


# -- enable_context_log flag tests --


def test_enable_context_log_false_no_callback():
  """enable_context_log=False with no user callback: no context cb attached."""
  exp = Experiment('ecl-1')
  trainer = Trainer(experiment=exp, enable_context_log=False)

  trainer._attach_default_callbacks()

  context_cbs = [cb for cb in trainer.callbacks if getattr(cb, '_is_context_log_callback', False)]
  assert len(context_cbs) == 0


def test_enable_context_log_false_with_user_callback_raises():
  """enable_context_log=False + user context callback raises ConfigError."""
  exp = Experiment('ecl-2')
  user_cb = UserContextLogCallback()
  trainer = Trainer(callbacks=[user_cb], experiment=exp, enable_context_log=False)

  with pytest.raises(ConfigError, match='enable_context_log=False'):
    trainer._attach_default_callbacks()


def test_enable_context_log_false_no_experiment_clean():
  """enable_context_log=False without experiment: no error, no callbacks added."""
  trainer = Trainer(enable_context_log=False)
  initial_count = len(trainer.callbacks)

  trainer._attach_default_callbacks()

  assert len(trainer.callbacks) == initial_count


def test_enable_context_log_true_user_callback_no_experiment():
  """enable_context_log=True + user callback + no experiment: skip, no error."""
  user_cb = UserContextLogCallback()
  trainer = Trainer(callbacks=[user_cb], enable_context_log=True)

  trainer._attach_default_callbacks()

  context_cbs = [cb for cb in trainer.callbacks if getattr(cb, '_is_context_log_callback', False)]
  assert len(context_cbs) == 1
  assert context_cbs[0] is user_cb


def test_enable_context_log_property_true():
  """Default Trainer has enable_context_log=True."""
  trainer = Trainer()
  assert trainer.enable_context_log is True


def test_enable_context_log_property_false():
  """Trainer(enable_context_log=False) property returns False."""
  trainer = Trainer(enable_context_log=False)
  assert trainer.enable_context_log is False


def test_fit_with_enable_context_log_false():
  """fit() with enable_context_log=False: training completes, context_log empty."""
  exp = Experiment('ecl-fit')
  module = NoopEvalModule()
  trainer = Trainer(experiment=exp, enable_context_log=False)

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  assert len(exp.context_log) == 0
  context_cbs = [cb for cb in trainer.callbacks if getattr(cb, '_is_context_log_callback', False)]
  assert len(context_cbs) == 0


def test_enable_context_log_false_conflict_error_type():
  """ConfigError is the exact exception type for conflicting config."""
  user_cb = UserContextLogCallback()
  exp = Experiment('ecl-type')
  trainer = Trainer(callbacks=[user_cb], experiment=exp, enable_context_log=False)

  with pytest.raises(ConfigError) as exc_info:
    trainer._attach_default_callbacks()

  assert type(exc_info.value) is ConfigError
