"""Tests for GradientCollator, ConcatCollator, AgentCollator, and CollationResult."""

from autopilot.ai.agents.agent import AgentResult
from autopilot.ai.gradient import (
  AgentCollator,
  CollationResult,
  ConcatCollator,
  GradientCollator,
  TextGradient,
)
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from unittest.mock import MagicMock
import json
import pytest


def _feedback(n: int) -> list[dict]:
  return [
    {'data': Datum(feedback=f'feedback_{i}', error_message=f'err_{i}'), 'targets': None}
    for i in range(n)
  ]


def _params(n: int) -> list[Parameter]:
  return [Parameter(requires_grad=True) for _ in range(n)]


class _DescParam(Parameter):
  def render(self) -> str:
    return 'custom description for param'


class TestBaseCollator:
  def test_base_collate_raises(self):
    with pytest.raises(NotImplementedError):
      GradientCollator().collate([], [])


class TestCollationResult:
  def test_holds_data(self):
    g = TextGradient(attribution='fix')
    r = CollationResult(context='dir', gradients={'p1': g})
    assert r.context == 'dir'
    assert r.gradients['p1'] is g

  def test_gradients_keyed_by_id(self):
    params = _params(2)
    gradients = {p.id: TextGradient(attribution=f'for {p.id}') for p in params}
    r = CollationResult(context='x', gradients=gradients)
    for p in params:
      assert p.id in r.gradients


class TestConcatCollator:
  def test_creates_text_gradients(self):
    params = _params(1)
    result = ConcatCollator().collate(_feedback(2), params)
    assert isinstance(result.gradients[params[0].id], TextGradient)

  def test_collects_evidence(self):
    params = _params(1)
    result = ConcatCollator().collate(_feedback(3), params)
    grad = result.gradients[params[0].id]
    assert len(grad.items) == 3

  def test_same_grad_all_params(self):
    params = _params(2)
    result = ConcatCollator().collate(_feedback(2), params)
    g1 = result.gradients[params[0].id]
    g2 = result.gradients[params[1].id]
    assert len(g1.items) == len(g2.items)

  def test_empty_feedback(self):
    params = _params(1)
    result = ConcatCollator().collate([], params)
    grad = result.gradients[params[0].id]
    assert len(grad.items) == 0

  def test_empty_parameters(self):
    result = ConcatCollator().collate(_feedback(2), [])
    assert result.gradients == {}
    assert result.context

  def test_all_feedback_empty(self):
    fb = [{'data': Datum(), 'targets': None} for _ in range(3)]
    params = _params(1)
    result = ConcatCollator().collate(fb, params)
    grad = result.gradients[params[0].id]
    assert len(grad.items) == 0

  def test_ids_always_unique(self):
    params = _params(2)
    assert params[0].id != params[1].id
    result = ConcatCollator().collate(_feedback(1), params)
    assert len(result.gradients) == 2


def _mock_agent_json(response_dict: dict) -> MagicMock:
  agent = MagicMock()
  agent.run.return_value = AgentResult(output=json.dumps(response_dict))
  return agent


class TestAgentCollator:
  def test_calls_agent(self):
    params = _params(1)
    response = {
      'direction': 'fix errors',
      'parameters': {
        params[0].id: {
          'attribution': 'add examples',
          'severity': 0.6,
          'evidence': ['missing examples'],
        }
      },
    }
    agent = _mock_agent_json(response)
    collator = AgentCollator(agent)
    collator.collate(_feedback(1), params)
    agent.run.assert_called_once()

  def test_prompt_includes_feedback(self):
    params = _params(1)
    agent = _mock_agent_json({'direction': 'd', 'parameters': {}})
    collator = AgentCollator(agent)
    collator.collate(_feedback(2), params)
    prompt = agent.run.call_args[0][0]
    assert 'feedback_0' in prompt
    assert 'feedback_1' in prompt

  def test_prompt_includes_param_render(self):
    p = _DescParam(requires_grad=True)
    agent = _mock_agent_json({'direction': 'd', 'parameters': {}})
    collator = AgentCollator(agent)
    collator.collate(_feedback(1), [p])
    prompt = agent.run.call_args[0][0]
    assert 'custom description for param' in prompt

  def test_parses_valid_json(self):
    params = _params(1)
    response = {
      'direction': 'improve prompts',
      'parameters': {
        params[0].id: {
          'attribution': 'add examples',
          'severity': 0.6,
          'evidence': ['missing examples in 3 items'],
        }
      },
    }
    agent = _mock_agent_json(response)
    result = AgentCollator(agent).collate(_feedback(1), params)
    assert result.context == 'improve prompts'

  def test_creates_text_gradients(self):
    params = _params(1)
    response = {
      'direction': 'improve',
      'parameters': {
        params[0].id: {
          'attribution': 'add examples',
          'severity': 0.7,
          'evidence': ['point 1'],
        }
      },
    }
    agent = _mock_agent_json(response)
    result = AgentCollator(agent).collate(_feedback(1), params)
    grad = result.gradients[params[0].id]
    assert isinstance(grad, TextGradient)
    assert grad.direction == 'improve'
    assert grad.attribution == 'add examples'
    assert grad.severity == 0.7
    assert len(grad.items) == 1
    assert grad.items[0].feedback == 'point 1'

  def test_invalid_json_raises(self):
    agent = MagicMock()
    agent.run.return_value = AgentResult(output='not json at all')
    with pytest.raises(RuntimeError, match='failed to parse'):
      AgentCollator(agent).collate(_feedback(1), _params(1))

  def test_missing_param_in_response(self):
    params = _params(2)
    response = {
      'direction': 'd',
      'parameters': {
        params[0].id: {'attribution': 'x', 'severity': 0.5, 'evidence': []},
      },
    }
    agent = _mock_agent_json(response)
    result = AgentCollator(agent).collate(_feedback(1), params)
    assert params[0].id in result.gradients
    assert params[1].id not in result.gradients

  def test_multiple_params(self):
    params = _params(2)
    response = {
      'direction': 'd',
      'parameters': {
        params[0].id: {'attribution': 'x', 'severity': 0.5, 'evidence': ['e1']},
        params[1].id: {'attribution': 'y', 'severity': 0.3, 'evidence': ['e2']},
      },
    }
    agent = _mock_agent_json(response)
    result = AgentCollator(agent).collate(_feedback(1), params)
    assert result.gradients[params[0].id].attribution == 'x'
    assert result.gradients[params[1].id].attribution == 'y'

  def test_build_prompt_is_overrideable(self):
    class _Custom(AgentCollator):
      def build_prompt(self, feedback, parameters):
        return 'custom prompt'

    agent = _mock_agent_json({'direction': 'd', 'parameters': {}})
    _Custom(agent).collate(_feedback(1), _params(1))
    assert agent.run.call_args[0][0] == 'custom prompt'

  def test_parse_result_is_overrideable(self):
    class _Custom(AgentCollator):
      def parse_result(self, output, parameters):
        return CollationResult(context='custom', gradients={})

    agent = _mock_agent_json({'direction': 'd', 'parameters': {}})
    result = _Custom(agent).collate(_feedback(1), _params(1))
    assert result.context == 'custom'

  def test_agent_raises_propagates(self):
    agent = MagicMock()
    agent.run.side_effect = RuntimeError('agent broke')
    with pytest.raises(RuntimeError, match='agent broke'):
      AgentCollator(agent).collate(_feedback(1), _params(1))

  def test_agent_empty_output(self):
    agent = MagicMock()
    agent.run.return_value = AgentResult(output='')
    with pytest.raises(RuntimeError):
      AgentCollator(agent).collate(_feedback(1), _params(1))

  def test_json_wrong_types(self):
    agent = _mock_agent_json({'direction': 123, 'parameters': 'bad'})
    with pytest.raises(RuntimeError):
      AgentCollator(agent).collate(_feedback(1), _params(1))

  def test_all_params_missing_from_json(self):
    agent = _mock_agent_json({'direction': 'd', 'parameters': {}})
    params = _params(2)
    result = AgentCollator(agent).collate(_feedback(1), params)
    assert result.gradients == {}

  def test_extra_keys_in_json_ignored(self):
    params = _params(1)
    response = {
      'direction': 'd',
      'parameters': {
        params[0].id: {'attribution': 'x', 'severity': 0.1, 'evidence': [], 'extra': 'ignored'}
      },
      'bonus_key': 'also ignored',
    }
    agent = _mock_agent_json(response)
    result = AgentCollator(agent).collate(_feedback(1), params)
    assert params[0].id in result.gradients

  def test_prompt_uses_render_from_base(self):
    p_empty = Parameter(requires_grad=True)
    p_desc = _DescParam(requires_grad=True)
    agent = _mock_agent_json({'direction': 'd', 'parameters': {}})
    collator = AgentCollator(agent)
    collator.collate(_feedback(1), [p_empty, p_desc])
    prompt = agent.run.call_args[0][0]
    assert 'custom description for param' in prompt

  def test_direction_key_missing(self):
    agent = _mock_agent_json({'parameters': {}})
    with pytest.raises(RuntimeError, match='direction'):
      AgentCollator(agent).collate(_feedback(1), _params(1))

  def test_parameters_key_missing(self):
    agent = _mock_agent_json({'direction': 'foo'})
    with pytest.raises(RuntimeError, match='parameters'):
      AgentCollator(agent).collate(_feedback(1), _params(1))

  def test_direction_null_value(self):
    agent = _mock_agent_json({'direction': None, 'parameters': {}})
    with pytest.raises(RuntimeError, match='direction'):
      AgentCollator(agent).collate(_feedback(1), _params(1))
