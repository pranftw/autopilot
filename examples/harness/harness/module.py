"""HarnessModule: multi-turn agent harness optimizable via autopilot.

Integrates a Pydantic AI inference agent with autopilot optimization.
Three PathParameters control the agent's behavior:

- ``system_prompt`` -- markdown system prompt text
- ``policies`` -- markdown policy rules
- ``tools_code`` -- Python tool code (exec'd each forward pass)

BUG-001: ``PathParameter(source=...)`` must receive a ``str``, not a
``pathlib.Path``, due to PosixPath serialization issues in
``schema_entry()`` (see AUTOPILOT_LEARNINGS.md).

Default model: ``DEFAULT_MODEL`` (cheap, tool-capable).

When ``use_judge=True`` (default), the module uses ``JudgeLoss`` with
``AgentCollator`` backed by a ``PydanticAgent`` collator and
``HarnessJudge`` for semantic gradient seeding. When ``use_judge=False``,
falls back to the heuristic ``HarnessLoss`` path.
"""

from autopilot.ai.agents.claude_code import ClaudeCodeAgent
from autopilot.ai.gradient import AgentCollator
from autopilot.ai.loss import JudgeLoss
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.ai.parameter import PathParameter
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.types import Datum, EvalDatum
from harness import DEFAULT_MODEL
from harness.agent import HarnessAgent
from harness.agents import PydanticAgent
from harness.database import RetailDB
from harness.evaluator import ConversationEvaluator, EvaluationResult
from harness.judge import HarnessJudge
from harness.loss import HarnessLoss
from harness.metrics import HarnessMetrics
from harness.simulator import UserSimulator
from harness.tool_loader import load_tools
from pathlib import Path
from typing import Any, Callable
import traceback

INSTRUCTION_SEPARATOR = '\n\n---\n\n'
OPTIMIZER_MODEL = 'haiku'


class HarnessModule(AutoPilotModule):
  """Multi-turn agent harness optimizable via autopilot.

  Integrates Pydantic AI inference agent with autopilot optimization.
  Three PathParameters: system prompt (text), policies (text), tool code (Python).

  The ``loss_fn`` attribute may be either ``JudgeLoss`` (when ``use_judge=True``,
  the default) or ``HarnessLoss`` (when ``use_judge=False``). In judge mode,
  ``AgentCollator`` backed by a ``PydanticAgent`` collator produces per-parameter
  ``TextGradient`` instances from evaluation feedback; ``_judge`` holds the
  ``HarnessJudge`` instance referenced by ``JudgeLoss.judge``. In heuristic mode,
  ``HarnessLoss`` uses failure-bucket categorization and ``_judge`` is ``None``.

  Attributes:
    system_prompt: PathParameter for the system prompt markdown.
    policies: PathParameter for the policy rules markdown.
    tools_code: PathParameter for the retail_tools.py code file.
    use_judge: Whether this module uses JudgeLoss (True) or HarnessLoss (False).
    loss_fn: JudgeLoss (judge mode) or HarnessLoss (heuristic mode).
    metrics: HarnessMetrics MetricCollection for epoch aggregation.
  """

  def __init__(
    self,
    root: str,
    model: str = DEFAULT_MODEL,
    use_judge: bool = True,
    max_turns: int | None = None,
  ) -> None:
    """Initialize with three PathParameters and agent runtime.

    Args:
      root: Path to the harness package directory (contains prompts/, tools/, db/).
      model: Model identifier for Pydantic AI (inference agent and, when
        ``use_judge`` is True, the collator agent's model).
      use_judge: When True, use :class:`~autopilot.ai.loss.JudgeLoss` with
        :class:`~autopilot.ai.gradient.AgentCollator` and :class:`HarnessJudge`.
        When False, use :class:`HarnessLoss` (heuristic gradient).
      max_turns: Maximum conversation turns per scenario.  When ``None``,
        uses ``DEFAULT_MAX_CONVERSATION_TURNS`` (15) from ``harness.agent``.
    """
    super().__init__()
    self.system_prompt = PathParameter(source=f'{root}/prompts', pattern='system_prompt.md')
    self.policies = PathParameter(source=f'{root}/prompts', pattern='policies.md')
    self.tools_code = PathParameter(source=f'{root}/tools', pattern='retail_tools.py')

    self._max_turns = max_turns
    self.use_judge = use_judge
    if use_judge:
      collator_agent = PydanticAgent(model=model)
      collator = AgentCollator(collator_agent)
      self._judge = HarnessJudge()
      self.loss_fn = JudgeLoss(
        judge=self._judge,
        collator=collator,
        parameters=list(self.parameters()),
      )
    else:
      self._judge = None
      self.loss_fn = HarnessLoss([self.system_prompt, self.policies, self.tools_code])

    self.metrics = HarnessMetrics()
    self._root = root
    self._agent = HarnessAgent(model=model)
    self._simulator = UserSimulator()
    self._db = RetailDB.from_file(Path(root) / 'db' / 'retail.json')

  def _read_instructions(self) -> str:
    """Assemble system prompt from text PathParameters.

    Reads via ``working_root`` so worktree/checkout semantics stay correct.

    Returns:
      Concatenated system prompt and policies separated by ``---``.
    """
    sys_dir = Path(self.system_prompt.working_root)
    system = (sys_dir / 'system_prompt.md').read_text(encoding='utf-8')
    pol_dir = Path(self.policies.working_root)
    policies = (pol_dir / 'policies.md').read_text(encoding='utf-8')
    return f'{system}{INSTRUCTION_SEPARATOR}{policies}'

  def _load_tools(self) -> dict[str, Callable[..., Any]]:
    """Load tool functions from the code PathParameter.

    Returns:
      Dict mapping tool name to callable.
    """
    tools_dir = Path(self.tools_code.working_root)
    return load_tools(tools_dir / 'retail_tools.py')

  def forward(self, batch: Any) -> EvalDatum:
    """Run conversation for one scenario, evaluate, return EvalDatum.

    Args:
      batch: A single scenario batch (EvalDatum or Datum wrapping one).

    Returns:
      EvalDatum with success status, metadata, and optional error_message.
    """
    item = self._unwrap(batch)
    scenario = item.metadata
    db = self._db.clone()
    try:
      instructions = self._read_instructions()
      tools = self._load_tools()
      conv_result = self._agent.run_conversation(
        instructions,
        tools,
        scenario,
        db,
        simulator=self._simulator,
        max_turns=self._max_turns,
      )
      eval_result = ConversationEvaluator.evaluate(scenario, conv_result, db)
      return EvalDatum(
        success=eval_result.task_success,
        metadata={
          'scenario': scenario,
          'eval_result': eval_result.to_dict(),
          'trajectory': conv_result.trajectory,
          'tool_calls': conv_result.tool_calls,
          'turns': conv_result.turns,
          'input_tokens': conv_result.input_tokens,
          'output_tokens': conv_result.output_tokens,
          'api_calls': conv_result.api_calls,
        },
      )
    except Exception as exc:
      tb = traceback.format_exc()
      error_msg = f'{type(exc).__name__}: {exc}'
      return EvalDatum(
        success=False,
        error_message=error_msg,
        metadata={
          'scenario': scenario,
          'eval_result': EvaluationResult.error().to_dict(),
          'errored': True,
          'traceback': tb,
        },
      )

  def _unwrap(self, batch: Any) -> EvalDatum:
    """Extract single EvalDatum from batch (DataLoader wraps in Datum).

    Args:
      batch: Raw batch from DataLoader or direct EvalDatum.

    Returns:
      The inner EvalDatum.

    Raises:
      TypeError: When ``batch`` is neither a ``Datum`` wrapping an
        ``EvalDatum`` nor a bare ``EvalDatum``.
    """
    if isinstance(batch, Datum) and batch.items and isinstance(batch.items[0], EvalDatum):
      return batch.items[0]
    if isinstance(batch, EvalDatum):
      return batch
    raise TypeError(
      f'expected Datum wrapping EvalDatum or bare EvalDatum, got {type(batch).__name__}'
    )

  def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    """Run forward pass for one training batch.

    Args:
      batch: A single scenario batch.

    Returns:
      EvalDatum from forward().
    """
    return self(batch)

  def validation_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    """Run forward pass for one validation batch.

    Args:
      batch: A single scenario batch.

    Returns:
      EvalDatum from forward().
    """
    return self(batch)

  def configure_optimizers(self) -> AgentOptimizer:
    """Configure the AgentOptimizer with a ClaudeCodeAgent.

    The agent edits prompts/tools in the harness package directory.

    Returns:
      AgentOptimizer wrapping a ClaudeCodeAgent with Edit/Write/Read tools.
    """
    optimizer_agent = ClaudeCodeAgent(
      allowed_tools=['Edit', 'Write', 'Read'],
      cwd=self._root,
      model=OPTIMIZER_MODEL,
    )
    return AgentOptimizer(
      agent=optimizer_agent,
      params=list(self.parameters()),
    )
