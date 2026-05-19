"""Tests for structured gradient journal emission (sub-plan 13).

Covers:
  4.1 Trainer capture and journal emission
    - test_gradient_journal_includes_param_names
    - test_gradient_journal_includes_param_type
    - test_gradient_journal_includes_gradient_type
    - test_gradient_journal_truncation
    - test_gradient_journal_skips_no_grad
    - test_gradient_journal_empty_when_no_grads

  4.2 AgentOptimizer parity
    - test_agent_optimizer_context_named
"""

from autopilot.ai.agents.agent import AgentResult
from autopilot.ai.gradient import TextGradient
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.core.callbacks.callback import Callback
from autopilot.core.callbacks.context import ContextLogCallback
from autopilot.core.context import ContextEntry
from autopilot.core.experiment import Experiment
from autopilot.core.gradient import NumericGradient
from autopilot.core.parameter import Parameter, ScalarParameter
from autopilot.core.trainer.journal import (
  GRAD_SUMMARY_MAX_CHARS,
  build_gradient_journal_row,
  emit_epoch_gradient_journal,
)
from autopilot.core.trainer.trainer import Trainer
from tests.doubles import DirectNumericLoss, NoopEvalModule, NoOpOptimizer
from unittest.mock import MagicMock

# -- shared helpers --


class SpyCallback(Callback):
  """Records on_context_emit calls for gradient journal assertions."""

  def __init__(self) -> None:
    super().__init__()
    self.entries: list[ContextEntry] = []

  def on_context_emit(self, trainer, module, entry) -> None:
    """Capture emitted context entries."""
    self.entries.append(entry)


class ModuleWithNamedParams(NoopEvalModule):
  """Module with two named parameters and a gradient-producing loss."""

  def __init__(self) -> None:
    super().__init__()
    self.rules = Parameter(requires_grad=True)
    self.prompt = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.rules, self.prompt])

  def configure_optimizers(self):
    """Return a no-op optimizer over both parameters."""
    return NoOpOptimizer([self.rules, self.prompt])


class ModuleWithMixedParams(NoopEvalModule):
  """Module with mixed parameter types for type/gradient-type assertions."""

  def __init__(self) -> None:
    super().__init__()
    self.rules = Parameter(requires_grad=True)
    self.score = ScalarParameter(value=0.5, requires_grad=True)
    self.loss = DirectNumericLoss([self.rules, self.score])

  def configure_optimizers(self):
    """Return a no-op optimizer over both parameters."""
    return NoOpOptimizer([self.rules, self.score])


class ModuleNoGrad(NoopEvalModule):
  """Module with parameters that never receive gradients."""

  def __init__(self) -> None:
    super().__init__()
    self.frozen = Parameter(requires_grad=False)

  def configure_optimizers(self):
    """Return no optimizer."""
    return


def _mock_agent(output: str = 'done') -> MagicMock:
  """Create a mock agent returning a successful AgentResult."""
  agent = MagicMock()
  agent.run.return_value = AgentResult(output=output)
  agent.limiter = None
  return agent


# -- 4.1 Trainer capture and journal emission --


def test_gradient_journal_includes_param_names():
  """Emitted journal dicts contain param_name matching named_parameters()."""
  exp = Experiment('param-names-test')
  spy = SpyCallback()
  module = ModuleWithNamedParams()
  trainer = Trainer(callbacks=[spy], experiment=exp)

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  completion_entries = [e for e in spy.entries if e.reason == 'gradient feedback recorded']
  assert len(completion_entries) == 1
  summaries = completion_entries[0].metadata['gradient_summaries']
  assert isinstance(summaries, list)
  assert all(isinstance(row, dict) for row in summaries)
  param_names = {row['param_name'] for row in summaries}
  assert 'rules' in param_names
  assert 'prompt' in param_names


def test_gradient_journal_includes_param_type():
  """Each dict param_type equals type(Parameter).__name__."""
  exp = Experiment('param-type-test')
  spy = SpyCallback()
  module = ModuleWithMixedParams()
  trainer = Trainer(callbacks=[spy], experiment=exp)

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  completion_entries = [e for e in spy.entries if e.reason == 'gradient feedback recorded']
  assert len(completion_entries) == 1
  summaries = completion_entries[0].metadata['gradient_summaries']
  type_map = {row['param_name']: row['param_type'] for row in summaries}
  assert type_map['rules'] == 'Parameter'
  assert type_map['score'] == 'ScalarParameter'


def test_gradient_journal_includes_gradient_type():
  """Each dict gradient_type equals type(Gradient).__name__."""
  exp = Experiment('grad-type-test')
  spy = SpyCallback()
  module = ModuleWithMixedParams()
  trainer = Trainer(callbacks=[spy], experiment=exp)

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  completion_entries = [e for e in spy.entries if e.reason == 'gradient feedback recorded']
  assert len(completion_entries) == 1
  summaries = completion_entries[0].metadata['gradient_summaries']
  for row in summaries:
    assert row['gradient_type'] == 'NumericGradient'


def test_gradient_journal_truncation():
  """Every summary length <= GRAD_SUMMARY_MAX_CHARS even when render exceeds."""
  long_text = 'x' * 500
  param = Parameter(requires_grad=True)
  param.grad = TextGradient(attribution=long_text)

  row = build_gradient_journal_row('test_param', param)
  assert len(row['summary']) <= GRAD_SUMMARY_MAX_CHARS
  assert row['param_name'] == 'test_param'
  assert row['param_type'] == 'Parameter'
  assert row['gradient_type'] == 'TextGradient'


def test_gradient_journal_skips_no_grad():
  """Parameters with grad is None never appear as dict rows."""
  exp = Experiment('skip-no-grad-test')
  spy = SpyCallback()
  module = ModuleWithNamedParams()
  module.prompt.requires_grad = False
  trainer = Trainer(callbacks=[spy], experiment=exp)

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  completion_entries = [e for e in spy.entries if e.reason == 'gradient feedback recorded']
  assert len(completion_entries) == 1
  summaries = completion_entries[0].metadata['gradient_summaries']
  param_names = {row['param_name'] for row in summaries}
  assert 'prompt' not in param_names
  assert 'rules' in param_names


def test_gradient_journal_empty_when_no_grads():
  """When nothing to capture, no gradient_summaries key in metadata."""
  exp = Experiment('empty-grads-test')
  spy = SpyCallback()
  module = ModuleNoGrad()
  trainer = Trainer(callbacks=[spy], experiment=exp)

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  grad_entries = [e for e in spy.entries if 'gradient feedback recorded' in e.reason]
  assert len(grad_entries) == 0

  success_entries = [e for e in spy.entries if 'experiment completed successfully' in e.reason]
  assert len(success_entries) == 1
  assert 'gradient_summaries' not in success_entries[0].metadata


# -- emit_epoch_gradient_journal unit tests (dogfood v8, CQ-004) --


def test_emit_epoch_gradient_journal_no_optimizer():
  """Trainer with _optimizer=None still emits journal when cache is populated."""
  trainer = MagicMock()
  trainer._optimizer = None
  trainer._cached_grad_summaries = [
    {
      'param_name': 'p',
      'param_type': 'Parameter',
      'gradient_type': 'TextGradient',
      'summary': 'x',
    },
  ]
  trainer.current_epoch = 0

  emit_epoch_gradient_journal(trainer, epoch=0)

  trainer.dispatch_callbacks.assert_called_once()
  call_kwargs = trainer.dispatch_callbacks.call_args
  assert call_kwargs[0][0] == 'on_context_emit'
  entry = call_kwargs[1]['entry']
  assert entry.metadata['epoch'] == 0
  assert len(entry.metadata['gradient_summaries']) == 1


def test_emit_epoch_gradient_journal_empty_cache():
  """Trainer with _cached_grad_summaries=[] returns without emitting."""
  trainer = MagicMock()
  trainer._optimizer = None
  trainer._cached_grad_summaries = []

  emit_epoch_gradient_journal(trainer, epoch=0)

  trainer.dispatch_callbacks.assert_not_called()


# -- 4.1b Per-epoch gradient journal emission (sub-plan 10) --


class _OwnsStepOptimizer(NoOpOptimizer):
  """Optimizer that claims gradient context ownership."""

  @property
  def owns_step_gradient_context(self) -> bool:
    return True


class ModuleWithOwnsStepOpt(NoopEvalModule):
  """Module with a gradient-producing loss and an optimizer that owns step context."""

  def __init__(self) -> None:
    super().__init__()
    self.rules = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.rules])

  def configure_optimizers(self):
    return _OwnsStepOptimizer([self.rules])


def test_gradient_journal_emitted_per_accepted_epoch():
  """3 accepted epochs produce 3 per-epoch gradient entries with correct metadata."""
  exp = Experiment('per-epoch-test')
  spy = SpyCallback()
  module = ModuleWithNamedParams()
  trainer = Trainer(callbacks=[spy], experiment=exp)

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=3)

  per_epoch = [
    e for e in spy.entries if e.reason.startswith('gradient feedback recorded for epoch')
  ]
  assert len(per_epoch) == 3
  epochs_seen = {e.metadata['epoch'] for e in per_epoch}
  assert epochs_seen == {0, 1, 2}
  for entry in per_epoch:
    assert 'gradient_summaries' in entry.metadata
    assert isinstance(entry.metadata['gradient_summaries'], list)


def test_gradient_journal_not_emitted_on_rejected_epoch():
  """Rejected epoch has no per-epoch gradient entry; accepted epoch does."""
  from autopilot.core.models import Result
  from autopilot.core.types import GateResult
  from autopilot.policy.policy import Policy

  class _AcceptReject(Policy):
    def __init__(self):
      super().__init__()
      self._idx = 0

    def forward(self, result: Result) -> GateResult:
      out = GateResult.PASSED if self._idx == 0 else GateResult.FAIL
      self._idx += 1
      return out

  exp = Experiment('mixed-gate-test')
  exp.start()
  spy = SpyCallback()
  module = ModuleWithNamedParams()
  trainer = Trainer(callbacks=[spy], experiment=exp, policy=_AcceptReject())

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=2)

  per_epoch = [
    e for e in spy.entries if e.reason.startswith('gradient feedback recorded for epoch')
  ]
  assert len(per_epoch) == 1
  assert per_epoch[0].metadata['epoch'] == 0


def test_gradient_journal_on_all_rejected_path():
  """All-rejected path emits one gradient journal before fail context."""
  from autopilot.core.models import Result
  from autopilot.core.types import GateResult
  from autopilot.policy.policy import Policy

  class _AlwaysReject(Policy):
    def forward(self, result: Result) -> GateResult:
      return GateResult.FAIL

  exp = Experiment('all-reject-grad-test')
  exp.start()
  spy = SpyCallback()
  module = ModuleWithNamedParams()
  trainer = Trainer(callbacks=[spy], experiment=exp, policy=_AlwaysReject())

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  per_epoch = [
    e for e in spy.entries if e.reason.startswith('gradient feedback recorded for epoch')
  ]
  assert len(per_epoch) == 0

  completion_grad = [e for e in spy.entries if e.reason == 'gradient feedback recorded']
  assert len(completion_grad) == 1
  assert 'gradient_summaries' in completion_grad[0].metadata

  fail_entries = [e for e in spy.entries if 'policy gate rejected all epochs' in e.reason]
  assert len(fail_entries) >= 1
  grad_idx = spy.entries.index(completion_grad[0])
  fail_idx = spy.entries.index(fail_entries[0])
  assert grad_idx < fail_idx


def test_agent_optimizer_skips_trainer_per_epoch_emission():
  """Optimizer with owns_step_gradient_context=True produces zero trainer gradient entries."""
  exp = Experiment('agent-skip-test')
  spy = SpyCallback()
  module = ModuleWithOwnsStepOpt()
  trainer = Trainer(callbacks=[spy], experiment=exp)

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=2)

  grad_entries = [
    e for e in spy.entries if 'gradient feedback recorded' in e.reason and e.source == 'trainer'
  ]
  assert len(grad_entries) == 0


# -- 4.2 AgentOptimizer parity --


def test_agent_optimizer_context_named(tmp_path):
  """After agentic step, context metadata uses same list[dict] schema as Trainer."""
  agent = _mock_agent()
  p1 = Parameter(requires_grad=True)
  p1.grad = TextGradient(attribution='improve A')
  p2 = ScalarParameter(value=1.0, requires_grad=True)
  p2.grad = NumericGradient(value=0.5)

  module = NoopEvalModule()
  module.rules = p1
  module.score = p2

  opt = AgentOptimizer(agent, [p1, p2], agentic=True, feedback_dir=str(tmp_path))

  exp = Experiment('agent-named-test')
  exp.start()
  cb = ContextLogCallback()
  trainer = Trainer(callbacks=[cb])
  trainer._experiment = exp
  trainer._module = module

  opt._context = {'epoch': 0, 'trainer': trainer}
  opt.step()

  entries = exp.context_log.filter_by_source('agent-optimizer')
  assert len(entries) == 1
  summaries = entries[0].metadata['gradient_summaries']
  assert isinstance(summaries, list)
  assert len(summaries) == 2
  for row in summaries:
    assert isinstance(row, dict)
    assert set(row.keys()) == {'param_name', 'param_type', 'gradient_type', 'summary'}

  type_map = {row['param_name']: row for row in summaries}
  assert 'rules' in type_map
  assert 'score' in type_map
  assert type_map['rules']['param_type'] == 'Parameter'
  assert type_map['rules']['gradient_type'] == 'TextGradient'
  assert type_map['score']['param_type'] == 'ScalarParameter'
  assert type_map['score']['gradient_type'] == 'NumericGradient'
  assert len(type_map['rules']['summary']) <= GRAD_SUMMARY_MAX_CHARS
