"""Tests for HarnessModule (plan 05, section 4.1)."""

from autopilot.ai.gradient import AgentCollator
from autopilot.ai.loss import JudgeLoss
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.ai.parameter import PathParameter
from autopilot.core.types import Datum, EvalDatum
from harness.agent import ConversationResult
from harness.agents import PydanticAgent
from harness.evaluator import EvaluationResult
from harness.judge import HarnessJudge
from harness.loss import HarnessLoss
from harness.module import HarnessModule
from unittest.mock import MagicMock, patch
import json
import pytest

STUB_TOOLS_CODE = '''\
def calculate(ctx, expression):
  """A stub tool."""
  return str(expression)
'''


@pytest.fixture
def harness_tree(tmp_path):
  """Build a minimal harness package tree under tmp_path."""
  pkg = tmp_path / 'harness'
  prompts = pkg / 'prompts'
  prompts.mkdir(parents=True)
  tools = pkg / 'tools'
  tools.mkdir(parents=True)
  db_dir = pkg / 'db'
  db_dir.mkdir(parents=True)

  (prompts / 'system_prompt.md').write_text('You are a test agent.', encoding='utf-8')
  (prompts / 'policies.md').write_text('Be polite.', encoding='utf-8')
  (tools / 'retail_tools.py').write_text(STUB_TOOLS_CODE, encoding='utf-8')
  (db_dir / 'retail.json').write_text(
    json.dumps({'products': {}, 'users': {}, 'orders': {}}),
    encoding='utf-8',
  )
  return pkg


@pytest.fixture
def module(harness_tree):
  """Create a HarnessModule pointing at the temp tree."""
  return HarnessModule(str(harness_tree))


def test_module_has_three_parameters(module):
  """named_parameters() exposes exactly three PathParameter instances."""
  params = list(module.parameters())
  assert len(params) == 3
  for p in params:
    assert isinstance(p, PathParameter)


def test_module_parameter_names(module):
  """Parameter names are system_prompt, policies, tools_code."""
  names = {name for name, _ in module.named_parameters()}
  assert names == {'system_prompt', 'policies', 'tools_code'}


def test_read_instructions(module):
  """_read_instructions concatenates system prompt and policies with separator."""
  instructions = module._read_instructions()
  assert 'You are a test agent.' in instructions
  assert 'Be polite.' in instructions
  assert '---' in instructions


def test_load_tools(module):
  """_load_tools returns a callable map containing the stub tool."""
  tools = module._load_tools()
  assert 'calculate' in tools
  assert callable(tools['calculate'])


def test_forward_success(module):
  """forward returns EvalDatum with success=True on successful conversation."""
  scenario = {'task_id': 'test-1', 'initial_message': 'hi'}
  item = EvalDatum(metadata=scenario)

  conv_result = ConversationResult(
    trajectory=[{'role': 'assistant', 'content': 'hello', 'turn': 0}],
    tool_calls=[],
    turns=1,
  )
  eval_result = EvaluationResult(
    task_success=True,
    tool_recall=1.0,
    tool_precision=1.0,
    tool_argument_accuracy=1.0,
    communication_recall=1.0,
    policy_compliance=1.0,
    turns=1,
    errored=False,
  )

  with (
    patch.object(module._agent, 'run_conversation', return_value=conv_result),
    patch('harness.module.ConversationEvaluator.evaluate', return_value=eval_result),
  ):
    result = module.forward(item)

  assert isinstance(result, EvalDatum)
  assert result.success is True
  assert 'eval_result' in result.metadata
  assert 'scenario' in result.metadata


def test_forward_error(module):
  """forward returns EvalDatum with success=False when agent raises."""
  scenario = {'task_id': 'test-err', 'initial_message': 'crash'}
  item = EvalDatum(metadata=scenario)

  with patch.object(module._agent, 'run_conversation', side_effect=RuntimeError('boom')):
    result = module.forward(item)

  assert isinstance(result, EvalDatum)
  assert result.success is False
  assert result.error_message is not None
  assert 'boom' in result.error_message
  assert result.metadata.get('errored') is True
  eval_dict = result.metadata['eval_result']
  assert eval_dict['errored'] is True


def test_training_step_delegates(module):
  """training_step delegates to forward (via __call__)."""
  scenario = {'task_id': 'test-ts', 'initial_message': 'test'}
  item = EvalDatum(metadata=scenario)

  conv_result = ConversationResult(
    trajectory=[{'role': 'assistant', 'content': 'ok', 'turn': 0}],
    tool_calls=[],
    turns=1,
  )
  eval_result = EvaluationResult(
    task_success=True,
    tool_recall=1.0,
    tool_precision=1.0,
    tool_argument_accuracy=1.0,
    communication_recall=1.0,
    policy_compliance=1.0,
    turns=1,
    errored=False,
  )

  with (
    patch.object(module._agent, 'run_conversation', return_value=conv_result),
    patch('harness.module.ConversationEvaluator.evaluate', return_value=eval_result),
  ):
    result = module.training_step(item)

  assert isinstance(result, EvalDatum)
  assert result.success is True


def test_configure_optimizers(module):
  """configure_optimizers returns an AgentOptimizer whose agent is ClaudeCodeAgent."""
  with patch('harness.module.ClaudeCodeAgent') as mock_agent_cls:
    sentinel = MagicMock()
    mock_agent_cls.return_value = sentinel
    optimizer = module.configure_optimizers()

  assert isinstance(optimizer, AgentOptimizer)
  assert optimizer._agent is sentinel
  mock_agent_cls.assert_called_once()


def test_unwrap_datum(module):
  """_unwrap extracts EvalDatum from Datum(items=[eval_datum])."""
  inner = EvalDatum(metadata={'task_id': 'u1'})
  wrapper = Datum(items=[inner])
  result = module._unwrap(wrapper)
  assert result is inner


def test_unwrap_eval_datum(module):
  """_unwrap returns EvalDatum directly when passed as-is."""
  item = EvalDatum(metadata={'task_id': 'u2'})
  result = module._unwrap(item)
  assert result is item


def test_module_use_judge_true(harness_tree):
  """use_judge=True wires JudgeLoss with AgentCollator and PydanticAgent."""
  module = HarnessModule(str(harness_tree), use_judge=True)
  assert isinstance(module.loss_fn, JudgeLoss)
  assert module.loss_fn.judge is module._judge
  assert isinstance(module._judge, HarnessJudge)
  assert isinstance(module.loss_fn._collator, AgentCollator)
  assert isinstance(module.loss_fn._collator._agent, PydanticAgent)


def test_module_use_judge_false(harness_tree):
  """use_judge=False wires HarnessLoss instead of JudgeLoss."""
  module = HarnessModule(str(harness_tree), use_judge=False)
  assert isinstance(module.loss_fn, HarnessLoss)


def test_module_judge_attribute(harness_tree):
  """_judge is HarnessJudge when use_judge=True, None when False."""
  module_judge = HarnessModule(str(harness_tree), use_judge=True)
  assert isinstance(module_judge._judge, HarnessJudge)

  module_no_judge = HarnessModule(str(harness_tree), use_judge=False)
  assert module_no_judge._judge is None


def test_module_default_is_judge(harness_tree):
  """Default (no use_judge arg) uses judge mode."""
  module = HarnessModule(str(harness_tree))
  assert isinstance(module.loss_fn, JudgeLoss)
  assert module._judge is not None


def test_forward_error_has_traceback(module):
  """Error metadata includes 'traceback' key with a non-empty string."""
  scenario = {'task_id': 'test-tb', 'initial_message': 'crash'}
  item = EvalDatum(metadata=scenario)

  with patch.object(module._agent, 'run_conversation', side_effect=ValueError('oops')):
    result = module.forward(item)

  assert result.success is False
  assert 'traceback' in result.metadata
  assert 'Traceback' in result.metadata['traceback']
  assert 'ValueError' in result.metadata['traceback']
  assert 'ValueError: oops' in result.error_message


def test_unwrap_unsupported_type(module):
  """_unwrap raises TypeError for string input."""
  with pytest.raises(TypeError, match='got str'):
    module._unwrap('not an EvalDatum')


def test_unwrap_unsupported_type_list(module):
  """_unwrap raises TypeError for list input."""
  with pytest.raises(TypeError, match='got list'):
    module._unwrap([1, 2, 3])
