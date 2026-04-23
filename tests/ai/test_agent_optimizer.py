"""Tests for autopilot.ai.optimizer.AgentOptimizer."""

from autopilot.ai.agents.agent import AgentResult
from autopilot.ai.gradient import TextGradient
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.ai.parameter import PathParameter
from autopilot.core.parameter import Parameter
from unittest.mock import MagicMock


def _mock_agent(output: str = 'done') -> MagicMock:
  agent = MagicMock()
  agent.run.return_value = AgentResult(output=output)
  agent.limiter = None
  return agent


def test_step_no_gradients_no_run():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  p.grad = None
  opt = AgentOptimizer(agent, [p])
  opt.step()
  agent.run.assert_not_called()


def test_step_with_gradients_calls_run():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  p.grad = TextGradient(attribution='fix this')
  opt = AgentOptimizer(agent, [p])
  opt.step()
  agent.run.assert_called_once()
  prompt = agent.run.call_args[0][0]
  assert 'What to change: fix this' in prompt


def test_zero_grad_clears_grad():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  p.grad = TextGradient(attribution='g')
  opt = AgentOptimizer(agent, [p])
  opt.zero_grad()
  assert p.grad is None


def test_prompt_includes_parameter_sections():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  p.grad = TextGradient(attribution='feedback line')
  opt = AgentOptimizer(agent, [p])
  opt.step()
  prompt = agent.run.call_args[0][0]
  assert f'--- Parameter {p.id} ---' in prompt
  assert 'What to change: feedback line' in prompt


def test_context_passed_to_agent():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  p.grad = TextGradient(attribution='improve')
  ctx = {'epoch': 3, 'metrics': {'accuracy': 0.7}}
  opt = AgentOptimizer(agent, [p], context=ctx)
  opt.step()
  call_kwargs = agent.run.call_args
  passed_ctx = call_kwargs[1]['context'] if 'context' in call_kwargs[1] else call_kwargs[0][1]
  assert passed_ctx['epoch'] == 3
  assert passed_ctx['metrics'] == {'accuracy': 0.7}


def test_prompt_includes_epoch_and_metrics():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  p.grad = TextGradient(attribution='fix')
  opt = AgentOptimizer(agent, [p], context={'epoch': 5, 'metrics': {'accuracy': 0.6}})
  opt.step()
  prompt = agent.run.call_args[0][0]
  assert 'Current epoch: 5' in prompt
  assert 'accuracy' in prompt


def test_prompt_includes_path_parameter_render(tmp_path):
  source = tmp_path / 'src'
  source.mkdir()
  (source / 'rules.json').write_text('[]')
  (source / 'config.yaml').write_text('')

  agent = _mock_agent()
  p = PathParameter(source=str(source), pattern='**/*', requires_grad=True)
  p.grad = TextGradient(attribution='improve rules')
  opt = AgentOptimizer(agent, [p])
  opt.step()
  prompt = agent.run.call_args[0][0]
  assert 'rules.json' in prompt
  assert 'config.yaml' in prompt


def test_all_params_grad_cleared_after_step(tmp_path):
  source = tmp_path / 'src'
  source.mkdir()
  (source / 'a.txt').write_text('hello')

  agent = _mock_agent()
  p = PathParameter(source=str(source), pattern='**/*', requires_grad=True)
  p.grad = TextGradient(attribution='rewrite a.txt')
  p2 = Parameter(requires_grad=True)
  p2.grad = TextGradient(attribution='also fix')
  opt = AgentOptimizer(agent, [p, p2])
  opt.step()
  assert p.grad is None
  assert p2.grad is None


def test_update_context():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  opt = AgentOptimizer(agent, [p], context={'epoch': 1})
  opt.update_context(epoch=2, metrics={'accuracy': 0.9})
  assert opt._context['epoch'] == 2
  assert opt._context['metrics'] == {'accuracy': 0.9}


def test_collation_context_in_prompt():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  p.grad = TextGradient(attribution='fix')
  opt = AgentOptimizer(agent, [p], context={'collation_context': 'improve error handling'})
  opt.step()
  prompt = agent.run.call_args[0][0]
  assert '## Overall Direction' in prompt
  assert 'improve error handling' in prompt


def test_render_called_on_grad():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  mock_grad = MagicMock()
  mock_grad.render.return_value = 'rendered gradient'
  p.grad = mock_grad
  opt = AgentOptimizer(agent, [p])
  opt.step()
  mock_grad.render.assert_called_once()
  prompt = agent.run.call_args[0][0]
  assert 'rendered gradient' in prompt


def test_render_called_on_param():
  agent = _mock_agent()

  class _DescParam(Parameter):
    def render(self) -> str:
      return 'my custom scope description'

  p = _DescParam(requires_grad=True)
  p.grad = TextGradient(attribution='fix')
  opt = AgentOptimizer(agent, [p])
  opt.step()
  prompt = agent.run.call_args[0][0]
  assert 'my custom scope description' in prompt


def test_no_pathparameter_import():
  import autopilot.ai.optimizer as mod

  source = open(mod.__file__).read()
  assert 'PathParameter' not in source


def test_build_prompt_is_overrideable():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  p.grad = TextGradient(attribution='x')

  class _Custom(AgentOptimizer):
    def build_prompt(self) -> str:
      return 'custom prompt'

  opt = _Custom(agent, [p])
  opt.step()
  assert agent.run.call_args[0][0] == 'custom prompt'


def test_build_context_is_overrideable():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  p.grad = TextGradient(attribution='x')

  class _Custom(AgentOptimizer):
    def build_context(self) -> dict:
      return {'custom_key': 'custom_val'}

  opt = _Custom(agent, [p])
  opt.step()
  call_kwargs = agent.run.call_args
  passed_ctx = call_kwargs[1]['context'] if 'context' in call_kwargs[1] else call_kwargs[0][1]
  assert passed_ctx == {'custom_key': 'custom_val'}


def test_step_empty_agent_output_does_not_clear_grads():
  agent = MagicMock()
  agent.limiter = None
  agent.run.return_value = AgentResult(output='')
  p = Parameter(requires_grad=True)
  p.grad = TextGradient(attribution='fix')
  opt = AgentOptimizer(agent, [p])
  opt.step()
  assert p.grad is not None


def test_step_agent_returns_none_does_not_clear_grads():
  agent = MagicMock()
  agent.limiter = None
  agent.run.return_value = None
  p = Parameter(requires_grad=True)
  p.grad = TextGradient(attribution='fix')
  opt = AgentOptimizer(agent, [p])
  opt.step()
  assert p.grad is not None


def test_prompt_mixed_params_with_and_without_grads():
  agent = _mock_agent()
  p1 = Parameter(requires_grad=True)
  p1.grad = TextGradient(attribution='fix this')
  p2 = Parameter(requires_grad=True)
  p2.grad = None
  opt = AgentOptimizer(agent, [p1, p2])
  opt.step()
  prompt = agent.run.call_args[0][0]
  assert f'--- Parameter {p1.id} ---' in prompt
  assert f'--- Parameter {p2.id} ---' not in prompt


def test_prompt_no_collation_context():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  p.grad = TextGradient(attribution='fix')
  opt = AgentOptimizer(agent, [p])
  opt.step()
  prompt = agent.run.call_args[0][0]
  assert '## Overall Direction' not in prompt


def test_cwd_not_auto_populated_from_params(tmp_path):
  source = tmp_path / 'proj'
  source.mkdir()
  (source / 'f.py').write_text('')

  agent = _mock_agent()
  p = PathParameter(source=str(source), pattern='**/*', requires_grad=True)
  p.grad = TextGradient(attribution='fix')
  opt = AgentOptimizer(agent, [p])
  opt.step()
  call_kwargs = agent.run.call_args
  passed_ctx = call_kwargs[1]['context'] if 'context' in call_kwargs[1] else call_kwargs[0][1]
  assert 'cwd' not in passed_ctx
  assert 'allowed_files' not in passed_ctx


def test_build_prompt_uses_auto_generated_id():
  agent = _mock_agent()
  p = Parameter(requires_grad=True)
  p.grad = TextGradient(attribution='fix')
  opt = AgentOptimizer(agent, [p])
  prompt = opt.build_prompt()
  assert f'--- Parameter {p.id} ---' in prompt
