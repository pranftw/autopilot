"""Tests for AI, CLI, and policy docstrings (sub-plan 11)."""

from autopilot.ai.environment import IsolatedEnvironment
from autopilot.ai.evaluation.generator import GeneratorAgent
from autopilot.ai.evaluation.judge import JudgeAgent
from autopilot.ai.evaluation.schemas import (
  CheckpointEvent,
  CheckpointHeader,
  ConversationTurn,
  GeneratorConfig,
  JudgeConfig,
  RetryConfig,
)
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.ai import AICommand
from autopilot.core.models import Result
from autopilot.core.tree import Tree
from autopilot.policy.gates import CustomGate, MaxGate, MinGate, RangeGate
from io import StringIO
import inspect
import pydoc


def _help_text(obj: type) -> str:
  """Capture help() output as a string."""
  buf = StringIO()
  pydoc.doc(obj, output=buf)
  return buf.getvalue()


class TestHelpAIClasses:
  """Section 4.1: help() runs without raising on AI classes."""

  def test_help_file_store(self) -> None:
    text = _help_text(FileStore)
    assert isinstance(text, str)
    assert 'FileStore' in text

  def test_help_generator_agent(self) -> None:
    text = _help_text(GeneratorAgent)
    assert isinstance(text, str)
    assert 'GeneratorAgent' in text

  def test_help_judge_agent(self) -> None:
    text = _help_text(JudgeAgent)
    assert isinstance(text, str)
    assert 'JudgeAgent' in text

  def test_help_agent_optimizer(self) -> None:
    text = _help_text(AgentOptimizer)
    assert isinstance(text, str)
    assert 'AgentOptimizer' in text

  def test_help_isolated_environment(self) -> None:
    text = _help_text(IsolatedEnvironment)
    assert isinstance(text, str)
    assert 'IsolatedEnvironment' in text


class TestDocstringContent:
  """Verify key docstrings exist and follow conventions."""

  def test_agent_optimizer_step_has_raises(self) -> None:
    doc = inspect.getdoc(AgentOptimizer.step)
    assert doc is not None
    assert 'Raises' in doc or 'zero_grad' in doc

  def test_agent_optimizer_build_prompt_has_returns(self) -> None:
    doc = inspect.getdoc(AgentOptimizer.build_prompt)
    assert doc is not None
    assert 'Returns' in doc

  def test_generator_run_has_args(self) -> None:
    doc = inspect.getdoc(GeneratorAgent.run)
    assert doc is not None
    assert 'Args' in doc

  def test_judge_run_has_args(self) -> None:
    doc = inspect.getdoc(JudgeAgent.run)
    assert doc is not None
    assert 'Args' in doc

  def test_file_store_snapshot_has_args(self) -> None:
    doc = inspect.getdoc(FileStore.snapshot)
    assert doc is not None
    assert 'Args' in doc
    assert 'Returns' in doc
    assert 'Raises' in doc


class TestGateExplainReturnsString:
  """Section 4.2: gate.explain(result) returns str for all gate types."""

  def test_min_gate_explain_pass(self) -> None:
    gate = MinGate('accuracy', 0.8)
    result = Result(metrics={'accuracy': 0.9})
    explanation = gate.explain(result)
    assert isinstance(explanation, str)
    assert 'PASS' in explanation

  def test_min_gate_explain_fail(self) -> None:
    gate = MinGate('accuracy', 0.8)
    result = Result(metrics={'accuracy': 0.5})
    explanation = gate.explain(result)
    assert isinstance(explanation, str)
    assert 'FAIL' in explanation

  def test_min_gate_explain_missing(self) -> None:
    gate = MinGate('accuracy', 0.8)
    result = Result(metrics={})
    explanation = gate.explain(result)
    assert isinstance(explanation, str)
    assert 'missing' in explanation
    assert 'FAIL' in explanation

  def test_max_gate_explain_pass(self) -> None:
    gate = MaxGate('loss', 1.0)
    result = Result(metrics={'loss': 0.5})
    explanation = gate.explain(result)
    assert isinstance(explanation, str)
    assert 'PASS' in explanation

  def test_max_gate_explain_fail(self) -> None:
    gate = MaxGate('loss', 1.0)
    result = Result(metrics={'loss': 1.5})
    explanation = gate.explain(result)
    assert isinstance(explanation, str)
    assert 'FAIL' in explanation

  def test_max_gate_explain_missing(self) -> None:
    gate = MaxGate('loss', 1.0)
    result = Result(metrics={})
    explanation = gate.explain(result)
    assert isinstance(explanation, str)
    assert 'missing' in explanation
    assert 'FAIL' in explanation

  def test_range_gate_explain_pass(self) -> None:
    gate = RangeGate('score', 0.0, 1.0)
    result = Result(metrics={'score': 0.5})
    explanation = gate.explain(result)
    assert isinstance(explanation, str)
    assert 'PASS' in explanation

  def test_range_gate_explain_fail(self) -> None:
    gate = RangeGate('score', 0.2, 0.8)
    result = Result(metrics={'score': 0.1})
    explanation = gate.explain(result)
    assert isinstance(explanation, str)
    assert 'FAIL' in explanation

  def test_range_gate_explain_missing(self) -> None:
    gate = RangeGate('score', 0.0, 1.0)
    result = Result(metrics={})
    explanation = gate.explain(result)
    assert isinstance(explanation, str)
    assert 'missing' in explanation
    assert 'FAIL' in explanation

  def test_custom_gate_explain_pass(self) -> None:
    gate = CustomGate('x', lambda v: v > 0.0)
    result = Result(metrics={'x': 1.0})
    explanation = gate.explain(result)
    assert isinstance(explanation, str)
    assert 'PASS' in explanation

  def test_custom_gate_explain_fail(self) -> None:
    gate = CustomGate('x', lambda v: v > 10.0)
    result = Result(metrics={'x': 1.0})
    explanation = gate.explain(result)
    assert isinstance(explanation, str)
    assert 'FAIL' in explanation

  def test_custom_gate_explain_missing(self) -> None:
    gate = CustomGate('x', lambda v: True)
    result = Result(metrics={})
    explanation = gate.explain(result)
    assert isinstance(explanation, str)
    assert 'missing' in explanation
    assert 'FAIL' in explanation


class TestGateExplainDocstrings:
  """Verify explain() docstrings follow third-person convention."""

  def test_min_gate_explain_docstring_starts_with_returns(self) -> None:
    doc = inspect.getdoc(MinGate.explain)
    assert doc is not None
    assert doc.startswith('Return')

  def test_max_gate_explain_docstring_starts_with_returns(self) -> None:
    doc = inspect.getdoc(MaxGate.explain)
    assert doc is not None
    assert doc.startswith('Return')

  def test_range_gate_explain_docstring_starts_with_returns(self) -> None:
    doc = inspect.getdoc(RangeGate.explain)
    assert doc is not None
    assert doc.startswith('Return')

  def test_custom_gate_explain_docstring_starts_with_returns(self) -> None:
    doc = inspect.getdoc(CustomGate.explain)
    assert doc is not None
    assert doc.startswith('Return')


class TestSchemaDocstrings:
  """Verify Pydantic schema classes have non-empty docstrings."""

  def test_conversation_turn_has_docstring(self) -> None:
    assert ConversationTurn.__doc__ is not None
    assert len(ConversationTurn.__doc__.strip()) > 0

  def test_retry_config_has_docstring(self) -> None:
    assert RetryConfig.__doc__ is not None

  def test_generator_config_has_docstring(self) -> None:
    assert GeneratorConfig.__doc__ is not None

  def test_judge_config_has_docstring(self) -> None:
    assert JudgeConfig.__doc__ is not None

  def test_checkpoint_header_has_docstring(self) -> None:
    assert CheckpointHeader.__doc__ is not None

  def test_checkpoint_event_has_docstring(self) -> None:
    assert CheckpointEvent.__doc__ is not None


class TestCLICommandDocstrings:
  """Spot-check: CLI command classes have docstrings."""

  def test_ai_command_has_docstring(self) -> None:
    assert AICommand.__doc__ is not None
    assert len(AICommand.__doc__.strip()) > 0


class TestTreeDisambiguation:
  """Verify Tree class docstring disambiguates from Git terminology."""

  def test_tree_doc_mentions_git(self) -> None:
    doc = Tree.__doc__ or ''
    assert 'Git' in doc or 'git' in doc

  def test_tree_doc_mentions_worktree(self) -> None:
    doc = Tree.__doc__ or ''
    assert 'worktree' in doc.lower()

  def test_tree_doc_mentions_filestore(self) -> None:
    doc = Tree.__doc__ or ''
    assert 'FileStore' in doc or 'Store' in doc
