"""Tests for harness judge pipeline: HarnessVerdict, HarnessJudge, and helpers."""

from autopilot.ai.evaluation.schemas import JudgeConfig, JudgeInput, RetryConfig, RunConfig
from autopilot.ai.evaluation.steps import LLMStep, PythonStep, collect_steps
from harness.agent import ConversationResult
from harness.evaluator import ConversationEvaluator, EvaluationResult
from harness.judge import (
  HarnessJudge,
  HarnessJudgeCustom,
  HarnessVerdict,
  _deserialize_conversation_result,
)
from pydantic import BaseModel
from unittest.mock import patch
import pytest

# -- fixtures ----------------------------------------------------------------


def _make_verdict(
  score: float = 0.8,
  critique: str = 'Good conversation overall.',
  dimension_scores: dict | None = None,
  recommendations: list | None = None,
) -> HarnessVerdict:
  """Build a HarnessVerdict with sensible defaults."""
  return HarnessVerdict(
    score=score,
    critique=critique,
    dimension_scores=dimension_scores or {'accuracy': 0.9, 'tone': 0.7},
    recommendations=recommendations or ['improve greeting'],
  )


def _make_score_dict(
  task_success: bool = True,
  tool_recall: float = 1.0,
  tool_precision: float = 1.0,
  tool_argument_accuracy: float = 1.0,
  communication_recall: float = 1.0,
  policy_compliance: float = 1.0,
  turns: int = 5,
  errored: bool = False,
) -> dict:
  """Build a score dict matching EvaluationResult.to_dict() shape."""
  return {
    'task_success': task_success,
    'tool_recall': tool_recall,
    'tool_precision': tool_precision,
    'tool_argument_accuracy': tool_argument_accuracy,
    'communication_recall': communication_recall,
    'policy_compliance': policy_compliance,
    'turns': turns,
    'errored': errored,
    'details': {},
  }


class _EmptyCustom(BaseModel):
  """Minimal concrete model for JudgeInput.custom in tests."""


def _make_judge_input(item_id: str = 'test-001') -> JudgeInput:
  """Build a minimal JudgeInput for testing."""
  return JudgeInput(
    id=item_id,
    turns=[],
    custom=_EmptyCustom(),
  )


def _make_run_config() -> RunConfig:
  """Build a minimal RunConfig for step collection tests."""
  return RunConfig(
    model='test-model',
    num_parallel=1,
    max_rpm=10,
    rpm_safety_margin=0.1,
    retry=RetryConfig(
      max_retries=1,
      min_timeout_ms=100,
      max_timeout_ms=1000,
      backoff_factor=2,
    ),
    max_tool_steps=5,
    max_output_tokens=1024,
  )


def _make_judge_config() -> JudgeConfig:
  """Build a minimal JudgeConfig for define_steps tests."""
  return JudgeConfig(run=_make_run_config())


# -- 2.1: HarnessVerdict model tests -----------------------------------------


class TestHarnessVerdictModel:
  """Tests for HarnessVerdict Pydantic model."""

  def test_round_trip(self):
    """model_dump -> model_validate round-trip preserves all fields."""
    verdict = _make_verdict()
    dumped = verdict.model_dump()
    restored = HarnessVerdict.model_validate(dumped)
    assert restored == verdict

  def test_all_fields_present(self):
    """All declared fields appear in model_dump output."""
    verdict = _make_verdict(
      score=0.95,
      critique='excellent',
      dimension_scores={'accuracy': 1.0},
      recommendations=['none'],
    )
    dumped = verdict.model_dump()
    assert dumped['score'] == 0.95
    assert dumped['critique'] == 'excellent'
    assert dumped['dimension_scores'] == {'accuracy': 1.0}
    assert dumped['recommendations'] == ['none']

  def test_empty_dimension_scores(self):
    """Empty dimension_scores dict is valid."""
    verdict = HarnessVerdict(
      score=0.5,
      critique='ok',
      dimension_scores={},
      recommendations=[],
    )
    assert verdict.dimension_scores == {}

  def test_empty_recommendations(self):
    """Empty recommendations list is valid."""
    verdict = HarnessVerdict(
      score=0.5,
      critique='ok',
      dimension_scores={'a': 0.1},
      recommendations=[],
    )
    assert verdict.recommendations == []


# -- 2.1: HarnessJudgeCustom model tests ------------------------------------


class TestHarnessJudgeCustomModel:
  """Tests for HarnessJudgeCustom Pydantic model."""

  def test_round_trip(self):
    """model_dump -> model_validate round-trip preserves nested structure."""
    custom = HarnessJudgeCustom(
      scores=_make_score_dict(),
      critique=_make_verdict(),
    )
    dumped = custom.model_dump()
    restored = HarnessJudgeCustom.model_validate(dumped)
    assert restored.scores == custom.scores
    assert restored.critique == custom.critique

  def test_scores_field_is_dict(self):
    """scores field accepts arbitrary dict structure."""
    custom = HarnessJudgeCustom(
      scores={'custom_metric': 42, 'nested': {'a': 1}},
      critique=_make_verdict(),
    )
    assert custom.scores['custom_metric'] == 42


# -- 2.2: HarnessJudge score step tests --------------------------------------


class TestHarnessJudgeScoreStep:
  """Tests for @python_step('score') on HarnessJudge."""

  def test_score_step_returns_evaluator_dict(self):
    """score_conversation calls ConversationEvaluator.evaluate and returns to_dict."""
    judge = HarnessJudge()
    fake_eval_result = EvaluationResult(
      task_success=True,
      tool_recall=1.0,
      tool_precision=0.9,
      tool_argument_accuracy=0.8,
      communication_recall=1.0,
      policy_compliance=1.0,
      turns=3,
      errored=False,
    )

    context = {
      'item': {
        'scenario': {'evaluation_criteria': {}},
        'conversation_result': {
          'trajectory': [],
          'tool_calls': [],
          'turns': 3,
          'error': None,
        },
        'db': None,
      },
    }

    with patch.object(
      ConversationEvaluator,
      'evaluate',
      return_value=fake_eval_result,
    ) as mock_eval:
      result = judge.score_conversation(context)

    mock_eval.assert_called_once()
    call_args = mock_eval.call_args
    assert call_args[0][0] == context['item']['scenario']
    assert isinstance(call_args[0][1], ConversationResult)
    assert call_args[0][2] is None

    assert result['task_success'] is True
    assert result['tool_recall'] == 1.0
    assert result['tool_precision'] == 0.9
    assert result['tool_argument_accuracy'] == 0.8

  def test_score_step_correct_evaluator_arg_order(self):
    """evaluate receives (scenario, conv_result, db) in correct order."""
    judge = HarnessJudge()
    fake_result = EvaluationResult(
      task_success=False,
      tool_recall=0.0,
      tool_precision=0.0,
      tool_argument_accuracy=0.0,
      communication_recall=0.0,
      policy_compliance=0.0,
      turns=0,
      errored=True,
    )

    scenario = {'evaluation_criteria': {'expected_actions': [{'tool': 'find', 'args': {}}]}}
    db_value = 'mock-db'
    context = {
      'item': {
        'scenario': scenario,
        'conversation_result': {'trajectory': [], 'tool_calls': [], 'turns': 1},
        'db': db_value,
      },
    }

    with patch.object(
      ConversationEvaluator,
      'evaluate',
      return_value=fake_result,
    ) as mock_eval:
      judge.score_conversation(context)

    call_args = mock_eval.call_args[0]
    assert call_args[0] is scenario
    assert call_args[2] == db_value

  def test_score_step_without_db(self):
    """score_conversation handles missing db key gracefully."""
    judge = HarnessJudge()
    fake_result = EvaluationResult.error()

    context = {
      'item': {
        'scenario': {},
        'conversation_result': {'trajectory': [], 'tool_calls': [], 'turns': 0},
      },
    }

    with patch.object(
      ConversationEvaluator,
      'evaluate',
      return_value=fake_result,
    ):
      result = judge.score_conversation(context)

    assert result['errored'] is True

  def test_score_conversation_missing_item_raises(self):
    """score_conversation raises KeyError when context is missing 'item'."""
    judge = HarnessJudge()
    with pytest.raises(KeyError):
      judge.score_conversation({})


# -- 2.2: LLM step (critique_conversation) tests ----------------------------


class TestHarnessJudgeCritiqueStep:
  """Tests for @llm_step('critique') on HarnessJudge."""

  def test_critique_returns_instruction_string(self):
    """critique_conversation returns a non-empty instruction string."""
    judge = HarnessJudge()
    context = {
      'score': _make_score_dict(),
      'item': {'scenario': {'task_id': 'test'}},
    }
    instructions = judge.critique_conversation(context)
    assert isinstance(instructions, str)
    assert len(instructions) > 0

  def test_critique_includes_score_data(self):
    """Instruction text references deterministic scores from context."""
    judge = HarnessJudge()
    score_data = _make_score_dict(tool_recall=0.5)
    context = {
      'score': score_data,
      'item': {'scenario': {}},
    }
    instructions = judge.critique_conversation(context)
    assert '0.5' in instructions

  def test_critique_strict_access_missing_score(self):
    """critique_conversation raises KeyError when 'score' is missing."""
    judge = HarnessJudge()
    context: dict = {'item': {'scenario': {}}}
    with pytest.raises(KeyError, match='score'):
      judge.critique_conversation(context)

  def test_critique_strict_access_missing_item(self):
    """critique_conversation raises KeyError when 'item' is missing."""
    judge = HarnessJudge()
    context: dict = {'score': {}}
    with pytest.raises(KeyError, match='item'):
      judge.critique_conversation(context)

  def test_critique_strict_access_empty_context(self):
    """critique_conversation raises KeyError on empty context."""
    judge = HarnessJudge()
    context: dict = {}
    with pytest.raises(KeyError):
      judge.critique_conversation(context)


# -- 2.3: assemble_result tests ---------------------------------------------


class TestHarnessJudgeAssembleResult:
  """Tests for HarnessJudge.assemble_result."""

  def test_basic_assembly(self):
    """assemble_result builds JudgeResult with correct id and custom fields."""
    judge = HarnessJudge()
    item = _make_judge_input('item-42')
    verdict = _make_verdict()
    scores = _make_score_dict()
    step_results = {'score': scores, 'critique': verdict}

    result = judge.assemble_result(item, step_results)

    assert result.item_id == 'item-42'
    assert result.verdict is None
    assert isinstance(result.custom, HarnessJudgeCustom)
    assert result.custom.scores == scores
    assert result.custom.critique == verdict

  def test_custom_model_dump(self):
    """model_dump produces JSON-serializable nested structure."""
    judge = HarnessJudge()
    item = _make_judge_input('dump-test')
    step_results = {
      'score': _make_score_dict(task_success=False),
      'critique': _make_verdict(score=0.3, critique='needs work'),
    }

    result = judge.assemble_result(item, step_results)
    dumped = result.model_dump()

    assert dumped['item_id'] == 'dump-test'
    assert dumped['verdict'] is None
    assert dumped['custom']['scores']['task_success'] is False
    assert dumped['custom']['critique']['score'] == 0.3


# -- 2.3: build_summary tests -----------------------------------------------


class TestHarnessJudgeBuildSummary:
  """Tests for HarnessJudge.build_summary."""

  def _make_judge_result(
    self,
    item_id: str,
    scores: dict,
    verdict: HarnessVerdict,
  ):
    """Helper to build a JudgeResult with HarnessJudgeCustom."""
    from autopilot.ai.evaluation.schemas import JudgeResult

    return JudgeResult(
      id=item_id,
      verdict=None,
      custom=HarnessJudgeCustom(scores=scores, critique=verdict),
    )

  def test_empty_results(self):
    """build_summary returns count_items=0 for empty list."""
    judge = HarnessJudge()
    summary = judge.build_summary([])
    assert summary == {'count_items': 0}

  def test_single_result(self):
    """build_summary with one result produces correct means (== raw values)."""
    judge = HarnessJudge()
    scores = _make_score_dict(
      task_success=True,
      tool_recall=0.8,
      tool_precision=0.9,
      tool_argument_accuracy=0.7,
      communication_recall=1.0,
      policy_compliance=0.6,
    )
    verdict = _make_verdict(
      score=0.75,
      dimension_scores={'accuracy': 0.8, 'tone': 0.9},
    )
    results = [self._make_judge_result('r1', scores, verdict)]

    summary = judge.build_summary(results)

    assert summary['count_items'] == 1
    assert summary['mean_tool_recall'] == 0.8
    assert summary['mean_tool_precision'] == 0.9
    assert summary['mean_tool_argument_accuracy'] == 0.7
    assert summary['mean_communication_recall'] == 1.0
    assert summary['mean_policy_compliance'] == 0.6
    assert summary['task_success_rate'] == 1.0
    assert summary['mean_critique_score'] == 0.75
    assert summary['dimension_means']['accuracy'] == 0.8
    assert summary['dimension_means']['tone'] == 0.9

  def test_two_results_averaging(self):
    """build_summary computes correct means across two items."""
    judge = HarnessJudge()
    scores_a = _make_score_dict(
      task_success=True,
      tool_recall=1.0,
      tool_precision=0.8,
      tool_argument_accuracy=1.0,
      communication_recall=1.0,
      policy_compliance=1.0,
    )
    verdict_a = _make_verdict(
      score=0.9,
      dimension_scores={'accuracy': 1.0, 'tone': 0.8},
    )

    scores_b = _make_score_dict(
      task_success=False,
      tool_recall=0.5,
      tool_precision=0.6,
      tool_argument_accuracy=0.4,
      communication_recall=0.5,
      policy_compliance=0.0,
    )
    verdict_b = _make_verdict(
      score=0.3,
      dimension_scores={'accuracy': 0.4, 'tone': 0.6, 'helpfulness': 0.5},
    )

    results = [
      self._make_judge_result('a', scores_a, verdict_a),
      self._make_judge_result('b', scores_b, verdict_b),
    ]
    summary = judge.build_summary(results)

    assert summary['count_items'] == 2
    assert summary['mean_tool_recall'] == pytest.approx(0.75)
    assert summary['mean_tool_precision'] == pytest.approx(0.7)
    assert summary['mean_tool_argument_accuracy'] == pytest.approx(0.7)
    assert summary['mean_communication_recall'] == pytest.approx(0.75)
    assert summary['mean_policy_compliance'] == pytest.approx(0.5)
    assert summary['task_success_rate'] == pytest.approx(0.5)
    assert summary['mean_critique_score'] == pytest.approx(0.6)
    assert summary['dimension_means']['accuracy'] == pytest.approx(0.7)
    assert summary['dimension_means']['tone'] == pytest.approx(0.7)
    # helpfulness only in one result, mean is just that value
    assert summary['dimension_means']['helpfulness'] == pytest.approx(0.5)

  def test_dimension_union_keys(self):
    """build_summary unions dimension keys across results."""
    judge = HarnessJudge()
    v1 = _make_verdict(score=0.5, dimension_scores={'a': 0.2})
    v2 = _make_verdict(score=0.5, dimension_scores={'b': 0.4})
    results = [
      self._make_judge_result('r1', _make_score_dict(), v1),
      self._make_judge_result('r2', _make_score_dict(), v2),
    ]
    summary = judge.build_summary(results)
    dims = summary['dimension_means']
    assert 'a' in dims
    assert 'b' in dims
    assert dims['a'] == pytest.approx(0.2)
    assert dims['b'] == pytest.approx(0.4)

  def test_build_summary_missing_key(self):
    """Mean is computed from only items that have the key (absent != zero)."""
    judge = HarnessJudge()
    scores_with_recall = {
      'tool_recall': 0.8,
      'tool_precision': 1.0,
      'tool_argument_accuracy': 1.0,
      'communication_recall': 1.0,
      'policy_compliance': 1.0,
      'task_success': True,
    }
    scores_without_recall = {
      'tool_precision': 0.6,
      'tool_argument_accuracy': 0.4,
      'communication_recall': 0.5,
      'policy_compliance': 0.5,
      'task_success': False,
    }
    v1 = _make_verdict(score=0.7, dimension_scores={'a': 0.5})
    v2 = _make_verdict(score=0.3, dimension_scores={'a': 0.3})
    results = [
      self._make_judge_result('r1', scores_with_recall, v1),
      self._make_judge_result('r2', scores_without_recall, v2),
    ]
    summary = judge.build_summary(results)

    assert summary['mean_tool_recall'] == pytest.approx(0.8)
    assert summary['mean_tool_precision'] == pytest.approx(0.8)
    assert summary['mean_tool_argument_accuracy'] == pytest.approx(0.7)
    assert summary['mean_communication_recall'] == pytest.approx(0.75)
    assert summary['mean_policy_compliance'] == pytest.approx(0.75)


# -- 2.4: step collection tests ---------------------------------------------


class TestHarnessJudgeStepCollection:
  """Tests for define_steps / collect_steps on HarnessJudge."""

  def test_step_names_order(self):
    """define_steps yields ['score', 'critique'] in definition order."""
    judge = HarnessJudge()
    config = _make_judge_config()
    steps = judge.define_steps(config)
    names = [s.name for s in steps]
    assert names == ['score', 'critique']

  def test_step_types(self):
    """First step is PythonStep, second is LLMStep."""
    judge = HarnessJudge()
    config = _make_judge_config()
    steps = judge.define_steps(config)
    assert isinstance(steps[0], PythonStep)
    assert isinstance(steps[1], LLMStep)

  def test_llm_step_output_type(self):
    """LLM step is configured with HarnessVerdict as output_type."""
    judge = HarnessJudge()
    config = _make_judge_config()
    steps = judge.define_steps(config)
    llm = steps[1]
    assert isinstance(llm, LLMStep)
    assert llm.output_type is HarnessVerdict

  def test_collect_steps_matches_define_steps(self):
    """collect_steps(instance) produces the same steps as define_steps."""
    judge = HarnessJudge()
    collected = collect_steps(judge)
    config_steps = judge.define_steps(_make_judge_config())
    assert [s.name for s in collected] == [s.name for s in config_steps]

  def test_exactly_two_steps(self):
    """HarnessJudge has exactly two steps."""
    judge = HarnessJudge()
    steps = judge.define_steps(_make_judge_config())
    assert len(steps) == 2


# -- helper tests -----------------------------------------------------------


class TestDeserializeConversationResult:
  """Tests for _deserialize_conversation_result helper."""

  def test_passthrough_conversation_result(self):
    """Already a ConversationResult is returned as-is."""
    cr = ConversationResult(trajectory=[], tool_calls=[], turns=3)
    assert _deserialize_conversation_result(cr) is cr

  def test_dict_to_conversation_result(self):
    """Dict is converted to ConversationResult with correct fields."""
    data = {
      'trajectory': [{'role': 'user', 'content': 'hi'}],
      'tool_calls': [{'name': 'find', 'arguments': {}}],
      'turns': 2,
      'error': None,
      'input_tokens': 10,
      'output_tokens': 20,
      'api_calls': 1,
    }
    result = _deserialize_conversation_result(data)
    assert isinstance(result, ConversationResult)
    assert result.turns == 2
    assert result.input_tokens == 10
    assert result.api_calls == 1
    assert len(result.trajectory) == 1

  def test_dict_with_missing_keys(self):
    """Dict with missing optional keys uses defaults."""
    data = {'trajectory': [], 'tool_calls': []}
    result = _deserialize_conversation_result(data)
    assert isinstance(result, ConversationResult)
    assert result.turns == 0
    assert result.error is None

  def test_non_dict_non_cr_passthrough(self):
    """Non-dict, non-ConversationResult values pass through unchanged."""
    sentinel = object()
    assert _deserialize_conversation_result(sentinel) is sentinel
