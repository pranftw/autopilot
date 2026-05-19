"""Tests for AgentOptimizer context emission (sub-plan 08).

Covers:
  - test_trainer_wires_trainer_ref (2.1)
  - test_optimizer_step_emits_context_on_success (2.2)
  - test_optimizer_step_no_emit_when_no_grads (2.2)
  - test_optimizer_step_no_emit_when_no_trainer_ref (2.2)
  - test_gradient_summary_truncation (2.2)
  - test_context_entry_source_is_agent_optimizer (2.2)
  - test_context_metadata_has_gradient_summaries (2.2)
  - test_end_to_end_optimizer_to_experiment (2.2)
"""

from autopilot.ai.agents.agent import AgentResult
from autopilot.ai.gradient import TextGradient
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.core.callbacks.context import ContextLogCallback
from autopilot.core.context import ContextEntry
from autopilot.core.experiment import Experiment
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.journal import GRAD_SUMMARY_MAX_CHARS
from autopilot.core.trainer.trainer import Trainer
from tests.doubles import NoopEvalModule
from unittest.mock import MagicMock


def _mock_agent(output: str = 'done') -> MagicMock:
  """Create a mock agent that returns a successful AgentResult."""
  agent = MagicMock()
  agent.run.return_value = AgentResult(output=output)
  agent.limiter = None
  return agent


# -- 2.1 tests: Trainer wiring --


def test_trainer_wires_trainer_ref():
  """After _ensure_agent_optimizer_context, optimizer._context['trainer'] is the trainer."""
  agent = _mock_agent()
  param = Parameter(requires_grad=True)
  opt = AgentOptimizer(agent, [param])

  module = NoopEvalModule()

  trainer = Trainer()
  trainer._experiment = Experiment('test-wiring')
  trainer._optimizer = opt
  trainer._module = module
  trainer._ensure_agent_optimizer_context(module)

  assert 'trainer' in opt._context
  assert opt._context['trainer'] is trainer


# -- 2.2 tests: context emission --


def test_optimizer_step_emits_context_on_success(tmp_path):
  """After step with grads and trainer wired, experiment.context_log gains an entry."""
  agent = _mock_agent()
  param = Parameter(requires_grad=True)
  param.grad = TextGradient(attribution='improve accuracy')
  module = NoopEvalModule()
  module.my_param = param
  opt = AgentOptimizer(agent, [param], agentic=True, feedback_dir=str(tmp_path))

  exp = Experiment('test-emit')
  exp.start()
  cb = ContextLogCallback()
  trainer = Trainer(callbacks=[cb])
  trainer._experiment = exp
  trainer._module = module

  opt._context = {'epoch': 0, 'trainer': trainer}
  opt.step()

  assert len(exp.context_log) >= 1
  reasons = [e.reason for e in exp.context_log]
  assert 'optimizer applied changes based on gradient feedback' in reasons


def test_optimizer_step_no_emit_when_no_grads(tmp_path):
  """No parameters have grad; no new context entry."""
  agent = _mock_agent()
  param = Parameter(requires_grad=True)
  param.grad = None
  opt = AgentOptimizer(agent, [param], agentic=True, feedback_dir=str(tmp_path))

  exp = Experiment('test-no-grads')
  exp.start()
  cb = ContextLogCallback()
  trainer = Trainer(callbacks=[cb])
  trainer._experiment = exp
  trainer._module = NoopEvalModule()

  opt._context = {'epoch': 0, 'trainer': trainer}
  opt.step()

  assert len(exp.context_log) == 0


def test_optimizer_step_no_emit_when_no_trainer_ref(tmp_path):
  """Context dict without 'trainer'; step completes without error; no emission."""
  agent = _mock_agent()
  param = Parameter(requires_grad=True)
  param.grad = TextGradient(attribution='improve')
  opt = AgentOptimizer(agent, [param], agentic=True, feedback_dir=str(tmp_path))

  opt._context = {'epoch': 0}
  opt.step()

  agent.run.assert_called_once()


def test_gradient_summary_truncation(tmp_path):
  """Synthetic long grad.render(); stored summary length <= GRAD_SUMMARY_MAX_CHARS."""
  agent = _mock_agent()
  param = Parameter(requires_grad=True)
  long_text = 'x' * 500
  param.grad = TextGradient(attribution=long_text)
  module = NoopEvalModule()
  module.my_param = param
  opt = AgentOptimizer(agent, [param], agentic=True, feedback_dir=str(tmp_path))

  exp = Experiment('test-truncation')
  exp.start()
  cb = ContextLogCallback()
  trainer = Trainer(callbacks=[cb])
  trainer._experiment = exp
  trainer._module = module

  opt._context = {'epoch': 0, 'trainer': trainer}
  opt.step()

  entries = [e for e in exp.context_log if e.source == 'agent-optimizer']
  assert len(entries) == 1
  summaries = entries[0].metadata['gradient_summaries']
  for row in summaries:
    assert isinstance(row, dict)
    assert len(row['summary']) <= GRAD_SUMMARY_MAX_CHARS


def test_context_entry_source_is_agent_optimizer(tmp_path):
  """Last relevant entry has source == 'agent-optimizer'."""
  agent = _mock_agent()
  param = Parameter(requires_grad=True)
  param.grad = TextGradient(attribution='fix bug')
  module = NoopEvalModule()
  module.my_param = param
  opt = AgentOptimizer(agent, [param], agentic=True, feedback_dir=str(tmp_path))

  exp = Experiment('test-source')
  exp.start()
  cb = ContextLogCallback()
  trainer = Trainer(callbacks=[cb])
  trainer._experiment = exp
  trainer._module = module

  opt._context = {'epoch': 0, 'trainer': trainer}
  opt.step()

  entries = exp.context_log.filter_by_source('agent-optimizer')
  assert len(entries) == 1
  assert entries[0].source == 'agent-optimizer'


def test_context_metadata_has_gradient_summaries(tmp_path):
  """metadata['gradient_summaries'] is list[dict] with expected length."""
  agent = _mock_agent()
  p1 = Parameter(requires_grad=True)
  p1.grad = TextGradient(attribution='improve A')
  p2 = Parameter(requires_grad=True)
  p2.grad = TextGradient(attribution='improve B')
  p3 = Parameter(requires_grad=False)
  p3.grad = TextGradient(attribution='should be skipped')

  module = NoopEvalModule()
  module.alpha = p1
  module.beta = p2
  module.gamma = p3
  opt = AgentOptimizer(agent, [p1, p2, p3], agentic=True, feedback_dir=str(tmp_path))

  exp = Experiment('test-metadata')
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
  assert all(isinstance(row, dict) for row in summaries)
  assert all('param_name' in row for row in summaries)


def test_end_to_end_optimizer_to_experiment(tmp_path):
  """Full path: optimizer step -> emit_context -> callback -> experiment.context_log."""
  agent = _mock_agent()
  param = Parameter(requires_grad=True)
  param.grad = TextGradient(attribution='end-to-end test')
  module = NoopEvalModule()
  module.my_param = param
  opt = AgentOptimizer(agent, [param], agentic=True, feedback_dir=str(tmp_path))

  exp = Experiment('test-e2e')
  exp.start()
  cb = ContextLogCallback()
  trainer = Trainer(callbacks=[cb])
  trainer._experiment = exp
  trainer._module = module

  opt._context = {'epoch': 0, 'trainer': trainer}
  opt.step()

  assert len(exp.context_log) >= 1
  entry = exp.context_log.filter_by_source('agent-optimizer')[0]
  assert isinstance(entry, ContextEntry)
  assert entry.reason == 'optimizer applied changes based on gradient feedback'
  assert entry.source == 'agent-optimizer'
  assert entry.epoch == 0
  assert 'gradient_summaries' in entry.metadata
  summaries = entry.metadata['gradient_summaries']
  assert len(summaries) == 1
  assert isinstance(summaries[0], dict)
  assert 'end-to-end test' in summaries[0]['summary']
  assert 'param_name' in summaries[0]
