"""Tests for harness environment presets."""

from harness import DEFAULT_MODEL
from harness.environments import DEV_CONFIG, PROD_CONFIG, EnvironmentConfig, get_environment_config
import pytest


class TestDevConfig:
  """Verify dev environment preset values."""

  def test_max_epochs(self):
    assert DEV_CONFIG.max_epochs == 5

  def test_max_turns(self):
    assert DEV_CONFIG.max_turns == 15

  def test_model(self):
    assert DEV_CONFIG.model == DEFAULT_MODEL

  def test_gate_count(self):
    assert len(DEV_CONFIG.gates) == 2

  def test_task_success_threshold(self):
    gate = DEV_CONFIG.gates[0]
    assert gate.metric == 'task_success_rate'
    assert gate.threshold == 0.3

  def test_tool_recall_threshold(self):
    gate = DEV_CONFIG.gates[1]
    assert gate.metric == 'tool_recall'
    assert gate.threshold == 0.4

  def test_use_judge(self):
    assert DEV_CONFIG.use_judge is True


class TestProdConfig:
  """Verify prod environment preset values."""

  def test_max_epochs(self):
    assert PROD_CONFIG.max_epochs == 10

  def test_max_turns(self):
    assert PROD_CONFIG.max_turns == 10

  def test_model(self):
    assert PROD_CONFIG.model == DEFAULT_MODEL

  def test_gate_count(self):
    assert len(PROD_CONFIG.gates) == 4

  def test_stricter_task_success(self):
    gate = PROD_CONFIG.gates[0]
    assert gate.metric == 'task_success_rate'
    assert gate.threshold == 0.7

  def test_tool_recall_threshold(self):
    gate = PROD_CONFIG.gates[1]
    assert gate.metric == 'tool_recall'
    assert gate.threshold == 0.8

  def test_tool_precision_gate(self):
    gate = PROD_CONFIG.gates[2]
    assert gate.metric == 'tool_precision'
    assert gate.threshold == 0.7

  def test_policy_compliance_gate(self):
    gate = PROD_CONFIG.gates[3]
    assert gate.metric == 'policy_compliance'
    assert gate.threshold == 0.8

  def test_use_judge(self):
    assert PROD_CONFIG.use_judge is True


class TestEnvironmentConfigUseJudge:
  """Verify use_judge field construction."""

  def test_explicit_false(self):
    env = EnvironmentConfig(
      model='test',
      max_epochs=1,
      max_turns=5,
      use_judge=False,
    )
    assert env.use_judge is False

  def test_default_true(self):
    env = EnvironmentConfig(
      model='test',
      max_epochs=1,
      max_turns=5,
    )
    assert env.use_judge is True


class TestGetEnvironmentConfig:
  """Test get_environment_config lookup function."""

  def test_get_dev(self):
    assert get_environment_config('dev') is DEV_CONFIG

  def test_get_prod(self):
    assert get_environment_config('prod') is PROD_CONFIG

  def test_invalid_raises(self):
    with pytest.raises(ValueError, match="Unknown environment 'staging'"):
      get_environment_config('staging')

  def test_error_message_actionable(self):
    with pytest.raises(ValueError, match="Use 'dev' or 'prod'"):
      get_environment_config('test')
