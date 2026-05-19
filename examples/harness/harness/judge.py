"""Hybrid deterministic + LLM judge for harness conversations.

Pipeline steps:
  1. ``score`` (PythonStep): runs ``ConversationEvaluator.evaluate`` for
     deterministic tau-bench scoring (tool recall, precision, communication,
     policy compliance).
  2. ``critique`` (LLMStep): structured critique via ``HarnessVerdict``
     Pydantic model (score, critique text, dimension_scores, recommendations).

Item payload shape (placed in workflow context as ``context['item']``):
  - ``scenario``: scenario dict with ``evaluation_criteria``
  - ``conversation_result``: ``ConversationResult.to_dict()`` payload
  - ``db``: optional ``RetailDB`` serialized state (may be ``None``)

Naming: ``HarnessVerdict`` is distinct from the framework's ``JudgeVerdict``
(which carries ``category`` / ``subcategory`` / ``rationale`` / ``confidence``).
"""

from autopilot.ai.evaluation.judge import JudgeAgent
from autopilot.ai.evaluation.schemas import JudgeInput, JudgeResult
from autopilot.ai.evaluation.steps import llm_step, python_step
from harness.agent import ConversationResult
from harness.evaluator import ConversationEvaluator
from pydantic import BaseModel
from typing import Any


class HarnessVerdict(BaseModel):
  """Structured LLM critique for harness conversations.

  Fields:
    score: Holistic model-reported quality score (0.0--1.0).
    critique: Qualitative analysis of the conversation.
    dimension_scores: Per-dimension rubric scores (e.g. ``accuracy``,
      ``tone``, ``policy_compliance``).
    recommendations: Actionable next steps for improvement.
  """

  score: float
  critique: str
  dimension_scores: dict[str, float]
  recommendations: list[str]


class HarnessJudgeCustom(BaseModel):
  """Payload for ``JudgeResult.custom`` combining scores and critique.

  Attributes:
    scores: Deterministic evaluation metrics from ``EvaluationResult.to_dict()``.
    critique: Structured LLM critique as ``HarnessVerdict``.
  """

  scores: dict[str, Any]
  critique: HarnessVerdict


class HarnessJudge(JudgeAgent):
  """Hybrid deterministic + LLM judge for retail customer service conversations.

  Uses ``@python_step('score')`` for deterministic ``ConversationEvaluator``
  scoring and ``@llm_step('critique')`` for structured Gemma-4 critique via
  ``HarnessVerdict``.

  Step collection via ``define_steps`` (default ``collect_steps``) yields
  steps ``['score', 'critique']`` in definition order.
  """

  @python_step('score')
  def score_conversation(self, context: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic tau-bench scoring on the conversation.

    Args:
      context: Workflow context with ``context['item']`` containing
        ``scenario``, ``conversation_result``, and optional ``db``.

    Returns:
      Dict from ``EvaluationResult.to_dict()`` with all metric fields.
    """
    item = context['item']
    evaluator = ConversationEvaluator()
    scenario = item['scenario']
    conv_result = _deserialize_conversation_result(item['conversation_result'])
    db = item.get('db')
    result = evaluator.evaluate(scenario, conv_result, db)
    return result.to_dict()

  @llm_step('critique', output_type=HarnessVerdict)
  def critique_conversation(self, context: dict[str, Any]) -> str:
    """Build LLM instructions for structured conversation critique.

    The prompt references deterministic scores from the prior ``score`` step
    and the conversation scenario so the model critiques conditionally on
    real performance data.

    Uses strict access for ``context['score']`` and ``context['item']`` since
    these are guaranteed by the pipeline (``@python_step('score')`` and
    ``JudgeAgent._process_item_ckpt``). Mis-wired pipelines fail fast with
    ``KeyError``.

    Args:
      context: Workflow context with ``context['score']`` (prior step output)
        and ``context['item']`` (conversation payload).

    Returns:
      Instruction string for the LLM agent.
    """
    score_data = context['score']
    item = context['item']
    scenario = item.get('scenario', {})
    return (
      'You are evaluating a customer service conversation against '
      'tau-bench criteria. Analyze the deterministic scores and '
      'conversation trajectory, then provide a qualitative critique '
      'with per-dimension scores and actionable recommendations.\n\n'
      f'Deterministic scores: {score_data!r}\n\n'
      f'Scenario: {scenario!r}\n\n'
      'Provide your assessment as a HarnessVerdict with:\n'
      '- score (0.0-1.0): overall quality\n'
      '- critique: qualitative analysis\n'
      '- dimension_scores: dict of dimension name to score (0.0-1.0)\n'
      '- recommendations: list of actionable improvements'
    )

  def assemble_result(
    self,
    item: JudgeInput,
    step_results: dict[str, Any],
  ) -> JudgeResult[HarnessJudgeCustom]:
    """Build a structured judge result from step outputs.

    Args:
      item: The judge input item with ``item_id``.
      step_results: Dict with ``'score'`` (dict) and ``'critique'``
        (``HarnessVerdict`` instance from LLMStep).

    Returns:
      ``JudgeResult`` with ``verdict=None`` and ``custom`` containing
      both deterministic scores and structured critique.
    """
    score_data = step_results['score']
    critique_data = step_results['critique']
    return JudgeResult(
      id=item.item_id,
      verdict=None,
      custom=HarnessJudgeCustom(scores=score_data, critique=critique_data),
    )

  def build_summary(
    self,
    results: list[JudgeResult[HarnessJudgeCustom]],
  ) -> dict[str, Any]:
    """Aggregate per-item judge results into a summary dict.

    Computes means across items for deterministic metrics and LLM critique
    dimensions.  Absent metric keys are excluded from the mean (not treated
    as zero), so means reflect only items that actually report each key.

    Args:
      results: List of ``JudgeResult`` instances from ``assemble_result``.

    Returns:
      JSON-serializable dict with ``count_items``, mean deterministic
      metrics, mean ``critique_score``, and per-dimension means.
    """
    count = len(results)
    if count == 0:
      return {'count_items': 0}

    metric_keys = [
      'tool_recall',
      'tool_precision',
      'tool_argument_accuracy',
      'communication_recall',
      'policy_compliance',
    ]

    metric_sums: dict[str, float] = {k: 0.0 for k in metric_keys}
    metric_counts: dict[str, int] = {k: 0 for k in metric_keys}
    task_success_sum = 0.0
    critique_score_sum = 0.0
    dimension_sums: dict[str, float] = {}
    dimension_counts: dict[str, int] = {}

    for r in results:
      custom = r.custom
      scores = custom.scores
      for key in metric_keys:
        if key in scores:
          metric_sums[key] += float(scores[key])
          metric_counts[key] += 1
      task_success_sum += float(bool(scores.get('task_success', False)))

      critique = custom.critique
      critique_score_sum += critique.score
      for dim_key, dim_val in critique.dimension_scores.items():
        dimension_sums[dim_key] = dimension_sums.get(dim_key, 0.0) + dim_val
        dimension_counts[dim_key] = dimension_counts.get(dim_key, 0) + 1

    summary: dict[str, Any] = {'count_items': count}
    for key in metric_keys:
      if metric_counts[key] > 0:
        summary[f'mean_{key}'] = metric_sums[key] / metric_counts[key]
    summary['task_success_rate'] = task_success_sum / count
    summary['mean_critique_score'] = critique_score_sum / count

    dimension_means: dict[str, float] = {}
    for dim_key in sorted(dimension_sums):
      dimension_means[dim_key] = dimension_sums[dim_key] / dimension_counts[dim_key]
    summary['dimension_means'] = dimension_means

    return summary


def _deserialize_conversation_result(data: Any) -> ConversationResult | Any:
  """Convert a serialized conversation result back to a usable object.

  When the item payload contains a dict (from ``model_dump()``), wraps it
  in a ``ConversationResult`` that ``ConversationEvaluator.evaluate`` can
  consume.  When the payload is already a ``ConversationResult``, returns
  it as-is.

  Args:
    data: Either a ``ConversationResult`` instance or a dict from serialization.

  Returns:
    Object compatible with ``ConversationEvaluator.evaluate``.
  """
  if isinstance(data, ConversationResult):
    return data
  if isinstance(data, dict):
    return ConversationResult(
      trajectory=data.get('trajectory', []),
      tool_calls=data.get('tool_calls', []),
      turns=data.get('turns', 0),
      error=data.get('error'),
      input_tokens=data.get('input_tokens', 0),
      output_tokens=data.get('output_tokens', 0),
      api_calls=data.get('api_calls', 0),
    )
  return data
