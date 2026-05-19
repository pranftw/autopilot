"""Tests for harness.evaluator (EvaluationResult and ConversationEvaluator)."""

from harness.agent import ConversationResult
from harness.database import RetailDB
from harness.evaluator import ConversationEvaluator, EvaluationResult


def _make_conv(
  trajectory: list[dict] | None = None,
  tool_calls: list[dict] | None = None,
  turns: int = 1,
  error: str | None = None,
) -> ConversationResult:
  """Build a ConversationResult for testing."""
  return ConversationResult(
    trajectory=trajectory or [],
    tool_calls=tool_calls or [],
    turns=turns,
    error=error,
  )


def _make_scenario(
  expected_actions: list[dict] | None = None,
  communicate_info: list[str] | None = None,
  nl_assertions: list[str] | None = None,
) -> dict:
  """Build a scenario dict with evaluation_criteria."""
  return {
    'evaluation_criteria': {
      'expected_actions': expected_actions or [],
      'communicate_info': communicate_info or [],
      'nl_assertions': nl_assertions or [],
    },
  }


class TestEvaluationResultError:
  def test_error_result(self) -> None:
    """error() has all metric fields 0.0, turns==0, errored=True, task_success=False."""
    result = EvaluationResult.error()
    assert result.task_success is False
    assert result.tool_recall == 0.0
    assert result.tool_precision == 0.0
    assert result.tool_argument_accuracy == 0.0
    assert result.communication_recall == 0.0
    assert result.policy_compliance == 0.0
    assert result.turns == 0
    assert result.errored is True


class TestEvaluationResultSerialization:
  def test_to_dict_from_dict_roundtrip(self) -> None:
    """to_dict then from_dict equals original."""
    original = EvaluationResult(
      task_success=True,
      tool_recall=0.8,
      tool_precision=0.9,
      tool_argument_accuracy=0.7,
      communication_recall=1.0,
      policy_compliance=0.5,
      turns=5,
      errored=False,
      details={'matched': ['a', 'b']},
    )
    roundtripped = EvaluationResult.from_dict(original.to_dict())
    assert roundtripped.task_success == original.task_success
    assert roundtripped.tool_recall == original.tool_recall
    assert roundtripped.tool_precision == original.tool_precision
    assert roundtripped.tool_argument_accuracy == original.tool_argument_accuracy
    assert roundtripped.communication_recall == original.communication_recall
    assert roundtripped.policy_compliance == original.policy_compliance
    assert roundtripped.turns == original.turns
    assert roundtripped.errored == original.errored
    assert roundtripped.details == original.details

  def test_from_dict_empty_returns_error(self) -> None:
    """Empty dict maps to error result."""
    result = EvaluationResult.from_dict({})
    assert result.errored is True
    assert result.task_success is False

  def test_from_metadata(self) -> None:
    """metadata={'eval_result': {...}} matches from_dict on inner dict."""
    inner = {
      'task_success': True,
      'tool_recall': 1.0,
      'tool_precision': 1.0,
      'tool_argument_accuracy': 1.0,
      'communication_recall': 1.0,
      'policy_compliance': 1.0,
      'turns': 3,
      'errored': False,
      'details': {},
    }
    metadata = {'eval_result': inner}
    from_meta = EvaluationResult.from_metadata(metadata)
    from_dict = EvaluationResult.from_dict(inner)
    assert from_meta.task_success == from_dict.task_success
    assert from_meta.tool_recall == from_dict.tool_recall
    assert from_meta.turns == from_dict.turns

  def test_from_metadata_missing_key(self) -> None:
    """Missing eval_result key in metadata returns error result."""
    result = EvaluationResult.from_metadata({})
    assert result.errored is True


class TestToolRecall:
  def test_perfect_tool_recall(self) -> None:
    """All expected tools present with matching args."""
    scenario = _make_scenario(
      expected_actions=[
        {'tool': 'find_user', 'args': {'name': 'Jane'}},
        {'tool': 'get_order', 'args': {'order_id': '#W1'}},
      ],
    )
    conv = _make_conv(
      tool_calls=[
        {'name': 'find_user', 'arguments': {'name': 'Jane'}},
        {'name': 'get_order', 'arguments': {'order_id': '#W1'}},
      ]
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.tool_recall == 1.0

  def test_partial_tool_recall(self) -> None:
    """Three expected, two matched."""
    scenario = _make_scenario(
      expected_actions=[
        {'tool': 'a', 'args': {}},
        {'tool': 'b', 'args': {}},
        {'tool': 'c', 'args': {}},
      ],
    )
    conv = _make_conv(
      tool_calls=[
        {'name': 'a', 'arguments': {}},
        {'name': 'b', 'arguments': {}},
      ]
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert abs(result.tool_recall - 2 / 3) < 1e-9

  def test_zero_tool_recall(self) -> None:
    """No expected tools called."""
    scenario = _make_scenario(
      expected_actions=[
        {'tool': 'a', 'args': {}},
        {'tool': 'b', 'args': {}},
      ],
    )
    conv = _make_conv(
      tool_calls=[
        {'name': 'x', 'arguments': {}},
      ]
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.tool_recall == 0.0


class TestToolPrecision:
  def test_tool_precision_with_spurious(self) -> None:
    """Extra non-skip tool call drops precision below 1.0."""
    scenario = _make_scenario(
      expected_actions=[{'tool': 'find_user', 'args': {}}],
    )
    conv_baseline = _make_conv(
      tool_calls=[
        {'name': 'find_user', 'arguments': {}},
      ]
    )
    conv_spurious = _make_conv(
      tool_calls=[
        {'name': 'find_user', 'arguments': {}},
        {'name': 'unknown_tool', 'arguments': {}},
      ]
    )
    baseline = ConversationEvaluator.evaluate(scenario, conv_baseline, RetailDB())
    spurious = ConversationEvaluator.evaluate(scenario, conv_spurious, RetailDB())
    assert baseline.tool_precision == 1.0
    assert spurious.tool_precision < 1.0

  def test_skip_tools_excluded_from_precision(self) -> None:
    """think and calculate are not counted in precision denominator."""
    scenario = _make_scenario(
      expected_actions=[{'tool': 'find_user', 'args': {}}],
    )
    conv = _make_conv(
      tool_calls=[
        {'name': 'find_user', 'arguments': {}},
        {'name': 'think', 'arguments': {'thought': 'hmm'}},
        {'name': 'calculate', 'arguments': {'expr': '1+1'}},
      ]
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.tool_precision == 1.0


class TestToolArgumentAccuracy:
  def test_tool_argument_accuracy_wrong(self) -> None:
    """Wrong args give tool_argument_accuracy == 0.0."""
    scenario = _make_scenario(
      expected_actions=[{'tool': 'find_user', 'args': {'name': 'Jane'}}],
    )
    conv = _make_conv(
      tool_calls=[
        {'name': 'find_user', 'arguments': {'name': 'Bob'}},
      ]
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.tool_argument_accuracy == 0.0

  def test_tool_argument_accuracy_correct(self) -> None:
    """Correct args give tool_argument_accuracy == 1.0."""
    scenario = _make_scenario(
      expected_actions=[{'tool': 'find_user', 'args': {'name': 'Jane'}}],
    )
    conv = _make_conv(
      tool_calls=[
        {'name': 'find_user', 'arguments': {'name': 'Jane', 'extra': 'ok'}},
      ]
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.tool_argument_accuracy == 1.0

  def test_subset_semantics(self) -> None:
    """Expected args are a subset: extra actual keys are allowed."""
    scenario = _make_scenario(
      expected_actions=[{'tool': 'a', 'args': {'x': 1}}],
    )
    conv = _make_conv(
      tool_calls=[
        {'name': 'a', 'arguments': {'x': 1, 'y': 2, 'z': 3}},
      ]
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.tool_argument_accuracy == 1.0


class TestCommunicationRecall:
  def test_communication_recall_full(self) -> None:
    """All communicate_info strings in assistant messages."""
    scenario = _make_scenario(communicate_info=['refund policy', 'order number'])
    conv = _make_conv(
      trajectory=[
        {'role': 'assistant', 'content': 'Our refund policy covers returns within 30 days.'},
        {'role': 'assistant', 'content': 'Your order number is #W1234.'},
      ]
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.communication_recall == 1.0

  def test_communication_recall_partial(self) -> None:
    """Two required strings, only one present."""
    scenario = _make_scenario(communicate_info=['refund policy', 'delivery date'])
    conv = _make_conv(
      trajectory=[
        {'role': 'assistant', 'content': 'Our refund policy is flexible.'},
      ]
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.communication_recall == 0.5

  def test_communication_case_insensitive(self) -> None:
    """Mixed-case assistant text matches lowercase requirement."""
    scenario = _make_scenario(communicate_info=['refund policy'])
    conv = _make_conv(
      trajectory=[
        {'role': 'assistant', 'content': 'Our REFUND POLICY is very generous.'},
      ]
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.communication_recall == 1.0


class TestNLAssertions:
  def test_nl_assertion_satisfied(self) -> None:
    """Assertion keywords found in evidence; policy_compliance == 1.0."""
    scenario = _make_scenario(nl_assertions=['agent was polite and helpful'])
    conv = _make_conv(
      trajectory=[
        {'role': 'assistant', 'content': 'Thank you for calling. I am polite and very helpful!'},
      ]
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.policy_compliance == 1.0

  def test_nl_assertion_negative_satisfied(self) -> None:
    """Negative assertion (not...) satisfied when keywords absent."""
    scenario = _make_scenario(nl_assertions=['not rude or dismissive'])
    conv = _make_conv(
      trajectory=[
        {'role': 'assistant', 'content': 'I am happy to help you with your order!'},
      ]
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.policy_compliance == 1.0


class TestTaskSuccess:
  def test_task_success_all_pass(self) -> None:
    """All dimensions perfect and not errored."""
    scenario = _make_scenario(
      expected_actions=[{'tool': 'find_user', 'args': {'name': 'Jane'}}],
      communicate_info=['found user'],
      nl_assertions=['agent was helpful'],
    )
    conv = _make_conv(
      trajectory=[
        {'role': 'assistant', 'content': 'I found user Jane. Agent was helpful.'},
      ],
      tool_calls=[{'name': 'find_user', 'arguments': {'name': 'Jane'}}],
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.task_success is True

  def test_task_success_one_fail(self) -> None:
    """Drop one expected tool call; task_success is False."""
    scenario = _make_scenario(
      expected_actions=[
        {'tool': 'find_user', 'args': {'name': 'Jane'}},
        {'tool': 'get_order', 'args': {'order_id': '#W1'}},
      ],
      communicate_info=['found user'],
    )
    conv = _make_conv(
      trajectory=[
        {'role': 'assistant', 'content': 'I found user Jane.'},
      ],
      tool_calls=[{'name': 'find_user', 'arguments': {'name': 'Jane'}}],
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.task_success is False

  def test_error_conversation_returns_error_result(self) -> None:
    """ConversationResult with error returns EvaluationResult.error()."""
    scenario = _make_scenario(
      expected_actions=[{'tool': 'a', 'args': {}}],
    )
    conv = _make_conv(error='timeout')
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.errored is True
    assert result.task_success is False
    assert result.tool_recall == 0.0


class TestFromDictPartialPayload:
  def test_partial_payload_missing_tool_recall(self) -> None:
    """from_dict with task_success but missing tool_recall returns error()."""
    result = EvaluationResult.from_dict({'task_success': True})
    assert result.errored is True
    assert result.task_success is False

  def test_partial_payload_missing_turns(self) -> None:
    """from_dict with most keys but missing turns returns error()."""
    result = EvaluationResult.from_dict(
      {
        'task_success': True,
        'tool_recall': 0.5,
        'tool_precision': 1.0,
        'tool_argument_accuracy': 1.0,
        'communication_recall': 1.0,
        'policy_compliance': 1.0,
        'errored': False,
      }
    )
    assert result.errored is True
    assert result.task_success is False

  def test_partial_payload_missing_errored(self) -> None:
    """from_dict missing errored returns error()."""
    result = EvaluationResult.from_dict(
      {
        'task_success': True,
        'tool_recall': 1.0,
        'tool_precision': 1.0,
        'tool_argument_accuracy': 1.0,
        'communication_recall': 1.0,
        'policy_compliance': 1.0,
        'turns': 3,
      }
    )
    assert result.errored is True


class TestDetailsPopulation:
  def test_communication_found_populated(self) -> None:
    """details['communication_found'] contains matched strings."""
    scenario = _make_scenario(communicate_info=['order shipped', 'tracking number'])
    conv = _make_conv(
      trajectory=[
        {'role': 'assistant', 'content': 'Your order shipped yesterday.'},
      ],
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.details['communication_found'] == ['order shipped']
    assert result.communication_recall == 0.5

  def test_nl_satisfied_populated(self) -> None:
    """details['nl_satisfied'] contains satisfied assertion strings."""
    scenario = _make_scenario(nl_assertions=['agent was polite and helpful'])
    conv = _make_conv(
      trajectory=[
        {'role': 'assistant', 'content': 'I am polite and very helpful!'},
      ],
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.details['nl_satisfied'] == ['agent was polite and helpful']

  def test_empty_details_on_vacuous(self) -> None:
    """Vacuous dimensions produce empty lists in details."""
    scenario = _make_scenario()
    conv = _make_conv()
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.details['communication_found'] == []
    assert result.details['nl_satisfied'] == []


class TestTaskSuccessToolPrecision:
  def test_task_success_false_on_spurious_tools(self) -> None:
    """Spurious non-skip tool calls cause task_success=False via precision."""
    scenario = _make_scenario(
      expected_actions=[{'tool': 'find_user', 'args': {'name': 'Jane'}}],
    )
    conv = _make_conv(
      tool_calls=[
        {'name': 'find_user', 'arguments': {'name': 'Jane'}},
        {'name': 'spurious_tool', 'arguments': {}},
      ],
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.tool_recall == 1.0
    assert result.tool_precision == 0.5
    assert result.tool_argument_accuracy == 1.0
    assert result.task_success is False

  def test_task_success_true_when_skip_tools_only(self) -> None:
    """Skip tools (think, calculate) don't count against precision."""
    scenario = _make_scenario(
      expected_actions=[{'tool': 'find_user', 'args': {'name': 'Jane'}}],
      communicate_info=['found user'],
    )
    conv = _make_conv(
      trajectory=[
        {'role': 'assistant', 'content': 'I found user Jane.'},
      ],
      tool_calls=[
        {'name': 'find_user', 'arguments': {'name': 'Jane'}},
        {'name': 'think', 'arguments': {'thought': 'hmm'}},
      ],
    )
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.tool_precision == 1.0
    assert result.task_success is True


class TestVacuousDimensions:
  def test_no_expected_actions(self) -> None:
    """Empty expected_actions: tool metrics all 1.0."""
    scenario = _make_scenario(expected_actions=[])
    conv = _make_conv(tool_calls=[])
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.tool_recall == 1.0
    assert result.tool_precision == 1.0
    assert result.tool_argument_accuracy == 1.0

  def test_no_communicate_info(self) -> None:
    """Empty communicate_info: communication_recall == 1.0."""
    scenario = _make_scenario(communicate_info=[])
    conv = _make_conv()
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.communication_recall == 1.0

  def test_no_nl_assertions(self) -> None:
    """Empty nl_assertions: policy_compliance == 1.0."""
    scenario = _make_scenario(nl_assertions=[])
    conv = _make_conv()
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.policy_compliance == 1.0

  def test_all_vacuous_is_success(self) -> None:
    """All dimensions vacuous => task_success is True."""
    scenario = _make_scenario()
    conv = _make_conv()
    result = ConversationEvaluator.evaluate(scenario, conv, RetailDB())
    assert result.task_success is True
