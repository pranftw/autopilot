"""AI-layer gradient types and collation: TextGradient, GradientCollator, and built-ins.

TextGradient is the LLM-oriented Gradient with text, attribution, severity,
and evidence items. GradientCollator aggregates per-item feedback into per-parameter
gradients. CollationResult carries context (overall direction) and gradients
(keyed by Parameter.id).
"""

from autopilot.ai.agents.agent import Agent
from autopilot.core.gradient import Gradient
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum, EvalDatum, _hydrate_datum_base
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
import json

AGENT_OUTPUT_ERROR_SNIPPET_LEN = 500


@dataclass
class TextGradient(Gradient):
  """LLM-oriented gradient with collated text and per-parameter attribution.

  Args:
    text: The gradient text content (renamed from direction for clarity).
    attribution: What specifically needs to change for this parameter.
    severity: 0.0-1.0 indicating how strongly this parameter needs to change.

  Evidence is stored in inherited Datum.items as child Datum objects.
  Legacy kwarg ``direction`` is rejected with a migration hint.

  Note:
    The primary content field is ``text``, not the legacy ``direction`` name.
    Passing ``direction`` as a keyword argument raises ``TypeError`` with
    migration guidance.

  Example:
    >>> from autopilot.ai.gradient import TextGradient
    >>>
    >>> grad = TextGradient(
    ...   text='tighten constraints',
    ...   attribution='drop vague rows',
    ...   severity=0.7,
    ... )
    >>> grad.text
    'tighten constraints'
  """

  text: str | None = None
  attribution: str | None = None
  severity: float = 0.0

  def __init__(
    self,
    text: str | None = None,
    attribution: str | None = None,
    severity: float = 0.0,
    *,
    items: list[Any] | None = None,
    **kwargs: Any,
  ) -> None:
    """Initialize a TextGradient.

    Args:
      text: High-level gradient content (shared across parameters).
      attribution: What specifically needs to change for this parameter.
      severity: 0.0-1.0 indicating change urgency.
      items: Child Datum evidence items (inherited from Datum).
      **kwargs: Catches legacy ``direction`` kwarg with migration error.

    Raises:
      TypeError: When ``direction`` is passed (renamed to ``text``).
    """
    if 'direction' in kwargs:
      msg = (
        "TextGradient parameter 'direction' has been renamed to 'text'. "
        "Use TextGradient(text='...') instead of TextGradient(direction='...')."
      )
      raise TypeError(msg)
    if kwargs:
      bad = ', '.join(sorted(kwargs))
      msg = f'TextGradient() got unexpected keyword arguments: {bad}'
      raise TypeError(msg)
    super().__init__(items=items if items is not None else [])
    self.text = text
    self.attribution = attribution
    self.severity = severity

  def accumulate(self, other: 'TextGradient') -> 'TextGradient':  # ty: ignore[invalid-method-override]  # intentional homogeneous fan-in (CLAUDE.md: "fan-in uses homogeneous gradient types")
    """Merge two TextGradients.

    Merges text, attribution (joined with '; ' when both present),
    items (concatenated), and severity (max). Cross-type accumulate raises
    TypeError.

    Args:
      other: Another :class:`TextGradient` to merge into this one.

    Returns:
      New :class:`TextGradient` combining both operands.

    Raises:
      TypeError: If ``other`` is not a :class:`TextGradient`.
    """
    if not isinstance(other, TextGradient):
      msg = (
        f'Cannot accumulate TextGradient with {type(other).__name__}. '
        f'Insert a conversion operator to coerce types before fan-in.'
      )
      raise TypeError(msg)
    merged_text = self.text
    if other.text:
      merged_text = f'{self.text}; {other.text}' if self.text else other.text
    merged_attribution = self.attribution
    if other.attribution:
      merged_attribution = (
        f'{self.attribution}; {other.attribution}' if self.attribution else other.attribution
      )
    return TextGradient(
      text=merged_text,
      attribution=merged_attribution,
      severity=max(self.severity, other.severity),
      items=self.items + other.items,
    )

  def todo_items(self) -> list[str]:
    """Return attribution as the actionable todo item when present.

    When attribution is ``None``, delegates to the base class default
    extraction from ``render()``.

    Returns:
      List containing the attribution string, or base class fallback.
    """
    if self.attribution is not None:
      return [self.attribution]
    return super().todo_items()

  def render(self) -> str:
    """Render text, attribution, evidence lines, and severity as readable output.

    Returns:
      Multi-line human-readable summary for prompts or logs.
    """
    parts: list[str] = []
    if self.text:
      parts.append(f'Text: {self.text}')
    if self.attribution:
      parts.append(f'What to change: {self.attribution}')
    if self.items:
      parts.append('Supporting evidence:')
      for item in self.items:
        if isinstance(item, EvalDatum):
          line = item.feedback or item.error_message
          if line:
            parts.append(f'  - {line}')
    if self.severity > 0:
      parts.append(f'Severity: {self.severity:.2f}')
    return '\n'.join(parts)

  def to_dict(self) -> dict[str, Any]:
    """Serialize this gradient including text fields.

    Returns:
      Dict suitable for :meth:`from_dict` round-trips.
    """
    payload = super().to_dict()
    payload['text'] = self.text
    payload['attribution'] = self.attribution
    payload['severity'] = self.severity
    return payload

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'TextGradient':
    """Deserialize a :class:`TextGradient` from :meth:`to_dict` output.

    Args:
      data: Mapping produced by :meth:`to_dict` or compatible callers.

    Returns:
      Hydrated :class:`TextGradient` instance.
    """
    return _hydrate_datum_base(
      cls,
      data,
      hydrate_child=Datum.from_dict,
      pop_type=True,
    )


@dataclass
class CollationResult:
  """Output of a GradientCollator.

  context: high-level direction string, rendered once at top of optimizer prompt.
            AgentOptimizer receives this via update_context(collation_context=...).
  gradients: per-parameter Gradient instances, keyed by Parameter.id (auto-generated
             12-char hex). Used by Loss.compute_seed_gradient() to build the seed
             gradient that enters the computation graph via graph.backward().
  """

  context: str
  gradients: dict[str, Gradient] = field(default_factory=dict)


def assert_gradients_match_parameters(
  gradients: Mapping[str, Gradient],
  parameters: list[Parameter],
) -> None:
  """Verify gradient keys exactly match parameter ids.

  Args:
    gradients: Mapping of parameter id to gradient.
    parameters: Declared parameters to match against.

  Raises:
    ValueError: If gradient keys differ from parameter ids.
  """
  allowed_ids = {p.id for p in parameters}
  if set(gradients.keys()) != allowed_ids:
    missing = sorted(allowed_ids - set(gradients))
    extra = sorted(set(gradients) - allowed_ids)
    msg = (
      f'collator produced gradients that do not match declared parameters. '
      f'missing={missing!r} extra={extra!r}'
    )
    raise ValueError(msg)


class GradientCollator:
  """Base collator. Subclass and override collate().

  collate(feedback, parameters) -> CollationResult
    feedback: list of {'data': Datum, 'targets': Any} dicts from Loss.forward()
    parameters: list of Parameter instances to attribute gradients to

  Built-ins:
    ConcatCollator -- joins feedback without an LLM call
    AgentCollator  -- uses a read-only Agent to synthesize per-parameter attributions
                      with build_prompt() / parse_result() as public extension methods
  """

  def collate(
    self,
    feedback: list[dict[str, Any]],
    parameters: list[Parameter],
  ) -> CollationResult:
    """Collate per-item feedback into per-parameter gradients.

    Args:
      feedback: Entries shaped like loss feedback dicts with ``data`` / ``targets``.
      parameters: Parameters that must receive exactly one gradient each.

    Returns:
      Collation context string and gradients keyed by parameter id.

    Raises:
      NotImplementedError: Subclasses must implement collation.
    """
    raise NotImplementedError


def _extract_eval_datum(entry: dict[str, Any]) -> EvalDatum | None:
  """Extract the first EvalDatum from a loss feedback entry.

  Handles both direct EvalDatum values and Datum wrappers produced by
  ``_default_collate`` (which always returns ``Datum(items=[...])``) by
  checking ``data``, then ``targets``, unwrapping Datum.items when needed.

  Args:
    entry: Loss feedback dict with ``data`` and optional ``targets`` keys.

  Returns:
    First EvalDatum found, or ``None`` when neither key yields one.
  """
  for key in ('data', 'targets'):
    source = entry.get(key)
    if isinstance(source, EvalDatum):
      return source
    if isinstance(source, Datum) and source.items:
      for item in source.items:
        if isinstance(item, EvalDatum):
          return item
  return None


class ConcatCollator(GradientCollator):
  """Joins all feedback into a single TextGradient per parameter. No LLM required."""

  def collate(
    self,
    feedback: list[dict[str, Any]],
    parameters: list[Parameter],
  ) -> CollationResult:
    """Join feedback into identical :class:`TextGradient` instances per parameter.

    Args:
      feedback: Eval datum entries with feedback or error messages.
      parameters: Parameters to attribute (each gets the same evidence bundle).

    Returns:
      Collation result with shared context and per-parameter gradients.
    """
    evidence_items: list[EvalDatum] = []
    for entry in feedback:
      source = _extract_eval_datum(entry)
      if source is not None and (source.feedback or source.error_message):
        evidence_items.append(
          EvalDatum(
            feedback=source.feedback,
            error_message=source.error_message,
          )
        )

    context = f'{len(feedback)} items evaluated, {len(evidence_items)} with feedback'
    gradients: dict[str, Gradient] = {}
    for param in parameters:
      gradients[param.id] = TextGradient(
        text=context,
        items=list(evidence_items),
      )
    assert_gradients_match_parameters(gradients, parameters)
    return CollationResult(context=context, gradients=gradients)


class AgentCollator(GradientCollator):
  """Uses a read-only Agent to collate feedback into per-parameter gradients."""

  def __init__(self, agent: Agent) -> None:
    """Create a collator backed by the given agent implementation.

    Args:
      agent: Agent whose :meth:`~Agent.run` returns JSON in the expected shape.
    """
    self._agent = agent

  def collate(
    self,
    feedback: list[dict[str, Any]],
    parameters: list[Parameter],
  ) -> CollationResult:
    """Build a prompt, run the agent, and parse structured per-parameter gradients.

    Args:
      feedback: Loss feedback entries with :class:`EvalDatum` payloads.
      parameters: Parameters that must appear in the agent JSON response.

    Returns:
      Parsed :class:`CollationResult` from the agent output.
    """
    prompt = self.build_prompt(feedback, parameters)
    result = self._agent.run(prompt)
    return self.parse_result(result.output, parameters)

  def build_prompt(
    self,
    feedback: list[dict[str, Any]],
    parameters: list[Parameter],
  ) -> str:
    """Construct the collator prompt describing feedback and parameter ids.

    Args:
      feedback: Per-item evaluation records to summarize.
      parameters: Parameters the agent must attribute in its JSON reply.

    Returns:
      Full prompt string passed to :meth:`~Agent.run`.
    """
    parts: list[str] = []
    parts.extend(
      [
        (
          'You are a gradient collator. Analyze the following feedback from evaluating '
          'data points and produce a coherent summary with per-parameter attributions.'
        ),
        '\n## Feedback from evaluated data points\n',
      ]
    )
    for i, entry in enumerate(feedback):
      data = entry.get('data')
      eval_d = _extract_eval_datum(entry)
      if eval_d is not None:
        parts.append(f'### Item {i + 1} (id: {eval_d.id}, success: {eval_d.success})')
        if eval_d.feedback:
          parts.append(f'Feedback: {eval_d.feedback}')
        if eval_d.error_message:
          parts.append(f'Error: {eval_d.error_message}')
        if eval_d.metadata:
          parts.append(f'Metadata: {json.dumps(eval_d.metadata)}')
      elif isinstance(data, Datum):
        parts.append(f'### Item {i + 1} (id: {data.id})')
      else:
        parts.append(f'### Item {i + 1} (missing data field)')
      parts.append('')

    parts.append('## Parameters to attribute feedback to\n')
    for param in parameters:
      parts.append(f'- {param.id}')
      desc = param.render()
      if desc:
        parts.append(desc)

    parts.extend(
      [
        '\n## Required JSON response format\n',
        (
          'Respond with ONLY valid JSON (no markdown fencing):\n'
          '{\n'
          '  "direction": "<1-3 sentence high-level summary of what needs to change>",\n'
          '  "parameters": {\n'
          '    "<param_id>": {\n'
          '      "attribution": "<what specifically needs to change for this parameter>",\n'
          '      "severity": <0.0-1.0>,\n'
          '      "evidence": ["<key feedback point 1>", "<key feedback point 2>"]\n'
          '    }\n'
          '  }\n'
          '}'
        ),
      ]
    )
    return '\n'.join(parts)

  def parse_result(
    self,
    output: str,
    parameters: list[Parameter],
  ) -> CollationResult:
    """Parse the agent's JSON response into :class:`CollationResult`.

    Args:
      output: Raw agent output expected to be JSON with ``direction`` and
        ``parameters`` keys.
      parameters: Declared parameters; ids must match the JSON object exactly.

    Returns:
      Collation with context string and per-parameter :class:`TextGradient`
      instances.

    Raises:
      RuntimeError: If JSON parsing fails or required keys/shapes are invalid.
      TypeError: If a parameter entry under ``parameters`` is not a JSON object.
      ValueError: If the ``parameters`` object keys do not match ``parameters``.
    """
    try:
      data = json.loads(output)
    except json.JSONDecodeError as exc:
      snippet = output[:AGENT_OUTPUT_ERROR_SNIPPET_LEN]
      msg = f'AgentCollator: failed to parse agent response as JSON: {snippet}'
      raise RuntimeError(
        msg,
      ) from exc

    if 'direction' not in data or not isinstance(data['direction'], str):
      snippet = output[:AGENT_OUTPUT_ERROR_SNIPPET_LEN]
      msg = f'AgentCollator: agent response missing or invalid "direction" key: {snippet}'
      raise RuntimeError(msg)
    if 'parameters' not in data or not isinstance(data['parameters'], dict):
      snippet = output[:AGENT_OUTPUT_ERROR_SNIPPET_LEN]
      msg = f'AgentCollator: agent response missing or invalid "parameters" key: {snippet}'
      raise RuntimeError(msg)
    direction = data['direction']
    param_data = data['parameters']

    allowed_ids = {p.id for p in parameters}
    parsed_ids = set(param_data.keys())
    if parsed_ids != allowed_ids:
      missing = sorted(allowed_ids - parsed_ids)
      extra = sorted(parsed_ids - allowed_ids)
      msg = (
        f'AgentCollator: parameters object must contain exactly the declared parameter ids. '
        f'missing={missing!r} extra={extra!r}'
      )
      raise ValueError(msg)

    gradients: dict[str, Gradient] = {}
    for param in parameters:
      pid = param.id
      p_info = param_data[pid]
      if not isinstance(p_info, dict):
        msg = f'AgentCollator: parameter entry for {pid} is not a dict: {type(p_info).__name__}'
        raise TypeError(msg)
      evidence_items: list[Datum] = [EvalDatum(feedback=e) for e in p_info.get('evidence', [])]
      gradients[pid] = TextGradient(
        text=direction,
        attribution=p_info.get('attribution'),
        severity=p_info.get('severity', 0.0),
        items=evidence_items,
      )

    return CollationResult(context=direction, gradients=gradients)
