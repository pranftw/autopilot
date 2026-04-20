"""Tests for autopilot.ai.agent."""

from autopilot.ai.agent import Agent, AgentResult
import pytest


def test_agent_forward_raises_not_implemented():
  agent = Agent()
  with pytest.raises(NotImplementedError):
    agent.forward('x')


def test_agent_result_defaults():
  r = AgentResult(output='hello')
  assert r.output == 'hello'
  assert r.session_id is None
  assert r.metadata == {}


def test_agent_result_all_fields():
  r = AgentResult(
    output='out',
    session_id='sid',
    metadata={'k': 1},
  )
  assert r.output == 'out'
  assert r.session_id == 'sid'
  assert r.metadata == {'k': 1}


def test_agent_result_model_dump():
  r = AgentResult(output='a', session_id='b', metadata={'c': 3})
  d = r.model_dump()
  assert d == {'output': 'a', 'session_id': 'b', 'metadata': {'c': 3}}


def test_agent_result_model_validate():
  r = AgentResult.model_validate({'output': 'x', 'session_id': None, 'metadata': {}})
  assert r.output == 'x'


def test_agent_result_dump_validate_round_trip():
  original = AgentResult(output='o', session_id='s', metadata={'m': True})
  restored = AgentResult.model_validate(original.model_dump())
  assert restored == original
