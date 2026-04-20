"""Tests for autopilot.ai.optimizer.AgentOptimizer."""

from autopilot.ai.optimizer import AgentOptimizer
from autopilot.ai.parameter import PathParameter
from autopilot.core.parameter import Parameter
from unittest.mock import MagicMock


def test_step_no_gradients_no_forward():
  agent = MagicMock()
  p = Parameter(requires_grad=True)
  p.grad = None
  opt = AgentOptimizer(agent, [p])
  opt.step()
  agent.forward.assert_not_called()


def test_step_with_gradients_calls_forward():
  agent = MagicMock()
  p = Parameter(requires_grad=True)
  p.grad = 'fix this'
  opt = AgentOptimizer(agent, [p])
  opt.step()
  agent.forward.assert_called_once()
  prompt = agent.forward.call_args[0][0]
  assert 'fix this' in prompt


def test_zero_grad_clears_grad():
  agent = MagicMock()
  p = Parameter(requires_grad=True)
  p.grad = 'g'
  opt = AgentOptimizer(agent, [p])
  opt.zero_grad()
  assert p.grad is None


def test_prompt_includes_parameter_sections():
  agent = MagicMock()
  p = Parameter(item_id='param-a', requires_grad=True)
  p.grad = 'feedback line'
  opt = AgentOptimizer(agent, [p])
  opt.step()
  prompt = agent.forward.call_args[0][0]
  assert '--- Parameter param-a ---' in prompt
  assert 'feedback line' in prompt


def test_context_passed_to_agent():
  agent = MagicMock()
  p = Parameter(requires_grad=True)
  p.grad = 'improve'
  ctx = {'epoch': 3, 'metrics': {'accuracy': 0.7}}
  opt = AgentOptimizer(agent, [p], context=ctx)
  opt.step()
  call_kwargs = agent.forward.call_args
  passed_ctx = call_kwargs[1]['context'] if 'context' in call_kwargs[1] else call_kwargs[0][1]
  assert passed_ctx['epoch'] == 3
  assert passed_ctx['metrics'] == {'accuracy': 0.7}


def test_prompt_includes_epoch_and_metrics():
  agent = MagicMock()
  p = Parameter(requires_grad=True)
  p.grad = 'fix'
  opt = AgentOptimizer(agent, [p], context={'epoch': 5, 'metrics': {'accuracy': 0.6}})
  opt.step()
  prompt = agent.forward.call_args[0][0]
  assert 'Current epoch: 5' in prompt
  assert 'accuracy' in prompt


def test_prompt_includes_path_parameter_files(tmp_path):
  source = tmp_path / 'src'
  source.mkdir()
  (source / 'rules.json').write_text('[]')
  (source / 'config.yaml').write_text('')

  agent = MagicMock()
  p = PathParameter(source=str(source), pattern='**/*', requires_grad=True)
  p.grad = 'improve rules'
  opt = AgentOptimizer(agent, [p])
  opt.step()
  prompt = agent.forward.call_args[0][0]
  assert 'rules.json' in prompt
  assert 'config.yaml' in prompt


def test_path_parameter_grad_cleared_after_step(tmp_path):
  source = tmp_path / 'src'
  source.mkdir()
  (source / 'a.txt').write_text('hello')

  agent = MagicMock()
  p = PathParameter(source=str(source), pattern='**/*', requires_grad=True)
  p.grad = 'rewrite a.txt'
  opt = AgentOptimizer(agent, [p])
  opt.step()
  assert p.grad is None


def test_update_context():
  agent = MagicMock()
  p = Parameter(requires_grad=True)
  opt = AgentOptimizer(agent, [p], context={'epoch': 1})
  opt.update_context(epoch=2, metrics={'accuracy': 0.9})
  assert opt._context['epoch'] == 2
  assert opt._context['metrics'] == {'accuracy': 0.9}


def test_context_includes_cwd_from_path_parameter(tmp_path):
  source = tmp_path / 'project'
  source.mkdir()
  (source / 'f.py').write_text('')

  agent = MagicMock()
  p = PathParameter(source=str(source), pattern='**/*', requires_grad=True)
  p.grad = 'fix'
  opt = AgentOptimizer(agent, [p])
  opt.step()
  call_kwargs = agent.forward.call_args
  passed_ctx = call_kwargs[1]['context'] if 'context' in call_kwargs[1] else call_kwargs[0][1]
  assert passed_ctx['cwd'] == str(source)
