from autopilot.ai.coding import ClaudeCodeAgent
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.ai.parameter import PathParameter
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.models import Datum
from autopilot.core.module import AutoPilotModule
from autopilot.core.parameter import Parameter
from pathlib import Path
from typing import Any


class PromptLoss(Loss):
  """Accumulates QA failures, backward() builds text gradient for the prompt parameter."""

  def __init__(self, parameters: list[Parameter] | None = None):
    super().__init__(parameters)
    self._failures: list[dict[str, str]] = []

  def forward(self, data: Datum, targets: Any = None) -> None:
    if not data.success:
      self._failures.append(
        {
          'item_id': data.item_id,
          'question': data.metadata.get('question', ''),
          'expected': data.metadata.get('expected', ''),
          'actual': data.metadata.get('actual', ''),
        }
      )

  def backward(self) -> None:
    if not self._failures:
      grad = 'All questions answered correctly. No changes needed.'
    else:
      lines = [f'{len(self._failures)} questions answered incorrectly:\n']
      for f in self._failures:
        lines.append(
          f'- Question: {f["question"]}\n  Expected: {f["expected"]}\n  Got: {f["actual"]}'
        )
      lines.append(
        '\nUpdate the system prompt to help answer these questions correctly. '
        'Add specific knowledge or instructions that address these failures.'
      )
      grad = '\n'.join(lines)
    for param in self._loss_parameters:
      if param.requires_grad:
        param.grad = grad

  def reset(self) -> None:
    self._failures = []


class QAAccuracyMetric(Metric):
  def __init__(self):
    super().__init__()
    self._correct = 0
    self._total = 0

  def update(self, datum: Datum) -> None:
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

  def reset(self) -> None:
    self._correct = 0
    self._total = 0


class PromptModule(AutoPilotModule):
  """QA module: uses ClaudeCodeAgent for inference with the system prompt as parameter."""

  def __init__(self, prompts_dir: str):
    super().__init__()
    self.prompt = PathParameter(source=prompts_dir, pattern='*.txt')
    self.loss = PromptLoss([self.prompt])
    self.accuracy = QAAccuracyMetric()
    self._prompts_dir = prompts_dir
    self._infer_agent = ClaudeCodeAgent(allowed_tools=[])

  def _read_prompt(self) -> str:
    prompt_path = Path(self._prompts_dir) / 'system.txt'
    return prompt_path.read_text(encoding='utf-8').strip()

  def forward(self, batch: Datum) -> Datum:
    question = batch.metadata.get('question', '')
    expected = batch.metadata.get('expected', '')
    item_id = batch.item_id

    system_prompt = self._read_prompt()
    full_prompt = (
      f'{system_prompt}\n\nAnswer this question with ONLY the answer, no explanation:\n{question}'
    )

    try:
      result = self._infer_agent.forward(full_prompt)
      actual = result.output.strip()
    except Exception as exc:
      return Datum(
        success=False,
        item_id=item_id,
        error_message=str(exc),
        metadata={'question': question, 'expected': expected, 'actual': ''},
      )

    success = expected.lower() in actual.lower()
    return Datum(
      success=success,
      item_id=item_id,
      metadata={
        'question': question,
        'expected': expected,
        'actual': actual,
      },
    )

  def training_step(self, batch: Any) -> Datum:
    return self.forward(batch)

  def validation_step(self, batch: Any) -> Datum:
    return self.forward(batch)

  def configure_optimizers(self):
    optimizer_agent = ClaudeCodeAgent(
      allowed_tools=['Edit', 'Write', 'Read'],
      cwd=self._prompts_dir,
    )
    return AgentOptimizer(
      agent=optimizer_agent,
      parameters=list(self.parameters()),
    )
