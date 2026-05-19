from autopilot.ai.agents.claude_code import ClaudeCodeAgent
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.ai.parameter import PathParameter
from autopilot.core.gradient import Gradient
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum, EvalDatum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PromptGradient(Gradient):
  failures: list[dict[str, str]] = field(default_factory=list)
  metadata: dict[str, Any] = field(default_factory=dict)

  def accumulate(self, other: 'PromptGradient') -> 'PromptGradient':
    return PromptGradient(
      failures=self.failures + other.failures,
      metadata={**self.metadata, **other.metadata},
    )

  def render(self) -> str:
    if not self.failures:
      return 'All questions answered correctly. No changes needed.'
    lines = [f'{len(self.failures)} questions answered incorrectly:\n']
    for f in self.failures:
      lines.append(
        f'- Question: {f["question"]}\n  Expected: {f["expected"]}\n  Got: {f["actual"]}'
      )
    lines.append(
      '\nUpdate the system prompt to help answer these questions correctly. '
      'Add specific knowledge or instructions that address these failures.'
    )
    return '\n'.join(lines)


class PromptLoss(Loss):
  """Accumulates QA failures; graph backward distributes gradients to parameters.

  forward() tracks per-item failures and delegates to Loss.forward() for
  _last_data / _accumulated bookkeeping. compute_seed_gradient() produces
  a PromptGradient seed. backward() is inherited from Loss and drives
  graph.backward() -- no direct param.grad assignment.
  """

  def __init__(self, parameters: list[Parameter] | None = None):
    super().__init__(parameters)
    self._failures: list[dict[str, str]] = []

  def forward(self, data: EvalDatum, targets: Any = None) -> None:
    super().forward(data, targets)
    if not data.success:
      self._failures.append(
        {
          'id': data.id,
          'question': data.metadata.get('question', ''),
          'expected': data.metadata.get('expected', ''),
          'actual': data.metadata.get('actual', ''),
        }
      )

  def compute_seed_gradient(self) -> PromptGradient:
    return PromptGradient(failures=list(self._failures))

  def reset(self) -> None:
    super().reset()
    self._failures = []


class QAAccuracyMetric(Metric):
  higher_is_better = True

  def __init__(self):
    super().__init__()
    self.add_state('_correct', 0)
    self.add_state('_total', 0)

  def update(self, datum: EvalDatum) -> None:
    self._total += 1
    if datum.success:
      self._correct += 1

  def compute(self) -> dict[str, float]:
    accuracy = self._correct / self._total if self._total > 0 else 0.0
    return {
      'accuracy': accuracy,
      'total': float(self._total),
      'correct': float(self._correct),
    }


class PromptModule(AutoPilotModule):
  """QA module: uses ClaudeCodeAgent for inference with the system prompt as parameter."""

  def __init__(self, prompts_dir: str):
    super().__init__()
    self.prompt = PathParameter(source=prompts_dir, pattern='*.txt')
    self.loss = PromptLoss([self.prompt])
    self.accuracy = QAAccuracyMetric()
    self._prompts_dir = prompts_dir
    self._infer_agent = ClaudeCodeAgent(allowed_tools=[], model='haiku')

  def _read_prompt(self) -> str:
    prompt_path = Path(self._prompts_dir) / 'system.txt'
    return prompt_path.read_text(encoding='utf-8').strip()

  def _unwrap_single(self, batch: Any) -> EvalDatum:
    """Extract the single EvalDatum from a collated Datum(items=[...]) wrapper."""
    if isinstance(batch, Datum) and batch.items and isinstance(batch.items[0], EvalDatum):
      return batch.items[0]
    return batch

  def forward(self, batch: Any) -> EvalDatum:
    item = self._unwrap_single(batch)
    question = item.metadata.get('question', '')
    expected = item.metadata.get('expected', '')

    system_prompt = self._read_prompt()
    full_prompt = (
      f'{system_prompt}\n\nAnswer this question with ONLY the answer, no explanation:\n{question}'
    )

    try:
      result = self._infer_agent.run(full_prompt)
      actual = result.output.strip()
    except Exception as exc:
      return EvalDatum(
        success=False,
        error_message=str(exc),
        metadata={'question': question, 'expected': expected, 'actual': ''},
      )

    success = expected.lower() in actual.lower()
    return EvalDatum(
      success=success,
      metadata={
        'question': question,
        'expected': expected,
        'actual': actual,
      },
    )

  def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    return self(batch)

  def validation_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    return self(batch)

  def configure_optimizers(self):
    optimizer_agent = ClaudeCodeAgent(
      allowed_tools=['Edit', 'Write', 'Read'],
      cwd=self._prompts_dir,
      model='haiku',
    )
    return AgentOptimizer(
      agent=optimizer_agent,
      params=list(self.parameters()),
    )
