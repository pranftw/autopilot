"""AI-layer gradient types and collation: TextGradient, GradientCollator, and built-ins.

TextGradient is the LLM-oriented Gradient with direction, attribution, severity,
and evidence items. GradientCollator aggregates per-item feedback into per-parameter
gradients. CollationResult carries context (overall direction) and gradients
(keyed by Parameter.id).
"""

from autopilot.ai.agents.agent import Agent
from autopilot.core.gradient import Gradient
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from typing import Any
import json


@dataclass
class TextGradient(Gradient):
  """LLM-oriented gradient with collated direction and per-parameter attribution.

  direction: high-level collated direction (shared across parameters)
  attribution: what specifically needs to change for this parameter
  severity: 0.0-1.0 indicating how strongly this parameter needs to change
  Evidence is stored in inherited Datum.items as child Datum objects.
  """

  direction: str | None = None
  attribution: str | None = None
  severity: float = 0.0

  def accumulate(self, other: 'TextGradient') -> 'TextGradient':
    return TextGradient(
      direction=self.direction,
      attribution=self.attribution,
      items=self.items + other.items,
      severity=max(self.severity, other.severity),
      metadata={**self.metadata, **other.metadata},
    )

  def render(self) -> str:
    parts: list[str] = []
    if self.attribution:
      parts.append(f'What to change: {self.attribution}')
    if self.items:
      parts.append('Supporting evidence:')
      for item in self.items:
        line = item.feedback or item.error_message
        if line:
          parts.append(f'  - {line}')
    if self.severity > 0:
      parts.append(f'Severity: {self.severity:.2f}')
    return '\n'.join(parts)

  def to_dict(self) -> dict[str, Any]:
    d = super().to_dict()
    d['direction'] = self.direction
    d['attribution'] = self.attribution
    d['severity'] = self.severity
    return d

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'TextGradient':
    data = dict(data)
    stored_id = data.pop('id', None)
    direction = data.pop('direction', None)
    attribution = data.pop('attribution', None)
    severity = data.pop('severity', 0.0)
    items_raw = data.pop('items', [])
    items = [Datum.from_dict(item) for item in items_raw]
    names = {f.name for f in dc_fields(cls)}
    filtered = {k: v for k, v in data.items() if k in names}
    instance = cls(
      direction=direction,
      attribution=attribution,
      severity=severity,
      items=items,
      **filtered,
    )
    if stored_id:
      object.__setattr__(instance, '_id', stored_id)
    return instance


@dataclass
class CollationResult:
  """Output of a GradientCollator.

  context: high-level direction string, rendered once at top of optimizer prompt.
            AgentOptimizer receives this via update_context(collation_context=...).
  gradients: per-parameter Gradient instances, keyed by Parameter.id (auto-generated
             12-char hex). Loss.backward() assigns each to the corresponding param.grad.
  """

  context: str
  gradients: dict[str, Gradient] = field(default_factory=dict)


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
    raise NotImplementedError


class ConcatCollator(GradientCollator):
  """Joins all feedback into a single TextGradient per parameter. No LLM required."""

  def collate(
    self,
    feedback: list[dict[str, Any]],
    parameters: list[Parameter],
  ) -> CollationResult:
    evidence_items: list[Datum] = []
    for entry in feedback:
      data = entry['data']
      if data.feedback or data.error_message:
        evidence_items.append(
          Datum(
            feedback=data.feedback,
            error_message=data.error_message,
          )
        )

    context = f'{len(feedback)} items evaluated, {len(evidence_items)} with feedback'
    gradients: dict[str, Gradient] = {}
    for param in parameters:
      gradients[param.id] = TextGradient(
        direction=context,
        items=list(evidence_items),
      )
    return CollationResult(context=context, gradients=gradients)


class AgentCollator(GradientCollator):
  """Uses a read-only Agent to collate feedback into per-parameter gradients."""

  def __init__(self, agent: Agent) -> None:
    self._agent = agent

  def collate(
    self,
    feedback: list[dict[str, Any]],
    parameters: list[Parameter],
  ) -> CollationResult:
    prompt = self.build_prompt(feedback, parameters)
    result = self._agent.run(prompt)
    return self.parse_result(result.output, parameters)

  def build_prompt(
    self,
    feedback: list[dict[str, Any]],
    parameters: list[Parameter],
  ) -> str:
    parts: list[str] = []
    parts.append(
      'You are a gradient collator. Analyze the following feedback from evaluating '
      'data points and produce a coherent summary with per-parameter attributions.'
    )

    parts.append('\n## Feedback from evaluated data points\n')
    for i, entry in enumerate(feedback):
      data = entry['data']
      parts.append(f'### Item {i + 1} (id: {data.id}, success: {data.success})')
      if data.feedback:
        parts.append(f'Feedback: {data.feedback}')
      if data.error_message:
        parts.append(f'Error: {data.error_message}')
      if data.metadata:
        parts.append(f'Metadata: {json.dumps(data.metadata)}')
      parts.append('')

    parts.append('## Parameters to attribute feedback to\n')
    for param in parameters:
      parts.append(f'- {param.id}')
      desc = param.render()
      if desc:
        parts.append(desc)

    parts.append('\n## Required JSON response format\n')
    parts.append(
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
    )
    return '\n'.join(parts)

  def parse_result(
    self,
    output: str,
    parameters: list[Parameter],
  ) -> CollationResult:
    try:
      data = json.loads(output)
    except json.JSONDecodeError:
      raise RuntimeError(f'AgentCollator: failed to parse agent response as JSON: {output[:500]}')

    if 'direction' not in data or not isinstance(data['direction'], str):
      raise RuntimeError(
        f'AgentCollator: agent response missing or invalid "direction" key: {output[:500]}'
      )
    if 'parameters' not in data or not isinstance(data['parameters'], dict):
      raise RuntimeError(
        f'AgentCollator: agent response missing or invalid "parameters" key: {output[:500]}'
      )
    direction = data['direction']
    param_data = data['parameters']

    gradients: dict[str, Gradient] = {}
    for param in parameters:
      pid = param.id
      if pid in param_data:
        p_info = param_data[pid]
        if not isinstance(p_info, dict):
          raise RuntimeError(
            f'AgentCollator: parameter entry for {pid} is not a dict: {type(p_info)}'
          )
        evidence_items = [Datum(feedback=e) for e in p_info.get('evidence', [])]
        gradients[pid] = TextGradient(
          direction=direction,
          attribution=p_info.get('attribution'),
          severity=p_info.get('severity', 0.0),
          items=evidence_items,
        )

    return CollationResult(context=direction, gradients=gradients)
