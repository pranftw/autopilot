"""Pydantic schemas for AI eval generation and judging."""

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Generic, TypeVar

T = TypeVar('T', bound=BaseModel)
C = TypeVar('C', bound=BaseModel)
IT = TypeVar('IT', bound=BaseModel)
JC = TypeVar('JC', bound=BaseModel)
JI = TypeVar('JI', bound=BaseModel)
JR = TypeVar('JR', bound=BaseModel)


class ConversationTurn(BaseModel):
  """One message in a multi-turn conversation (OpenAI-style shape)."""

  role: str
  content: str
  name: str | None = None
  tool_call_id: str | None = None
  tool_calls: list[dict[str, Any]] | None = None


class DataItem(BaseModel, Generic[T]):
  """Generated eval dataset item.

  Base: item_id, turns, split. Custom: ground_truth, metadata, domain, etc.
  """

  model_config = ConfigDict(populate_by_name=True)

  item_id: str = Field(alias='id')
  turns: list[ConversationTurn]
  split: str | None = None
  custom: T


class JudgeInput(BaseModel, Generic[T]):
  """Input to judge -- model output with optional traces.

  Base: identity, response, error state, traces.
  Custom: ground_truth, benchmark_metadata, query, session context.
  """

  model_config = ConfigDict(populate_by_name=True)

  item_id: str = Field(alias='id')
  turns: list[ConversationTurn]
  response: str | None = None
  is_error: bool = False
  error_message: str | None = None
  trace_present: bool = False
  trace_summary: str | None = None
  custom: T


class JudgeVerdict(BaseModel):
  """Universal judge classification output."""

  category: str
  subcategory: str | None = None
  rationale: str
  confidence: float = Field(ge=0.0, le=1.0)


class JudgeResult(BaseModel, Generic[T]):
  """Output from judge.

  Base: item_id, verdict. Custom: project-specific result data.
  """

  model_config = ConfigDict(populate_by_name=True)

  item_id: str = Field(alias='id')
  verdict: JudgeVerdict | None = None
  custom: T


class RetryConfig(BaseModel):
  """Retry policy for LLM API calls (backoff, limits, timeouts)."""

  max_retries: int
  min_timeout_ms: int
  max_timeout_ms: int
  backoff_factor: int


class RunConfig(BaseModel):
  """Shared run parameters for generator and judge."""

  model: str
  num_parallel: int
  max_rpm: int
  rpm_safety_margin: float
  retry: RetryConfig
  max_tool_steps: int
  max_output_tokens: int


class GeneratorConfig(BaseModel, Generic[T]):
  """Full configuration for a generator pipeline run."""

  run: RunConfig
  dataset_id: str
  seed: int
  total_count: int
  split_ratios: dict[str, float]
  system_prompt: str | None = None
  custom: T | None = None


class JudgeConfig(BaseModel, Generic[T]):
  """Full configuration for a judge pipeline run."""

  run: RunConfig
  system_prompt: str | None = None
  custom: T | None = None


class CheckpointHeader(BaseModel):
  """First line of a ``checkpoint.jsonl`` file identifying the run."""

  model_config = ConfigDict(populate_by_name=True)

  checkpoint_type: str = Field(default='header', alias='type')
  subsystem: str
  config_hash: str
  created_at: str
  args: dict[str, Any] = Field(default_factory=dict)


class CheckpointEvent(BaseModel):
  """Single event record in a ``checkpoint.jsonl`` file (result, skip, or error)."""

  model_config = ConfigDict(populate_by_name=True)

  event_kind: str = Field(alias='type')
  item_id: str = Field(alias='id')
  timestamp: str
  payload: dict[str, Any] = Field(default_factory=dict)


class VarDef(BaseModel):
  """Variable definition with weighted choices for slot generation."""

  choices: list[str]
  distribution: list[float]
  metadata: list[dict[str, Any]] | None = None
