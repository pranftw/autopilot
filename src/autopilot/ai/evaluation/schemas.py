"""Pydantic schemas for AI eval generation and judging."""

from pydantic import BaseModel, Field
from typing import Any, Generic, TypeVar

T = TypeVar('T', bound=BaseModel)
C = TypeVar('C', bound=BaseModel)
IT = TypeVar('IT', bound=BaseModel)
JC = TypeVar('JC', bound=BaseModel)
JI = TypeVar('JI', bound=BaseModel)
JR = TypeVar('JR', bound=BaseModel)


class ConversationTurn(BaseModel):
  role: str
  content: str
  name: str | None = None
  tool_call_id: str | None = None
  tool_calls: list[dict[str, Any]] | None = None


class DataItem(BaseModel, Generic[T]):
  """Generated eval dataset item.

  Base: id, turns, split. Custom: ground_truth, metadata, domain, etc.
  """

  id: str
  turns: list[ConversationTurn]
  split: str | None = None
  custom: T


class JudgeInput(BaseModel, Generic[T]):
  """Input to judge -- model output with optional traces.

  Base: identity, response, error state, traces.
  Custom: ground_truth, benchmark_metadata, query, session context.
  """

  id: str
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

  Base: id, verdict. Custom: project-specific result data.
  """

  id: str
  verdict: JudgeVerdict | None = None
  custom: T


class RetryConfig(BaseModel):
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
  run: RunConfig
  dataset_id: str
  seed: int
  total_count: int
  split_ratios: dict[str, float]
  system_prompt: str | None = None
  custom: T | None = None


class JudgeConfig(BaseModel, Generic[T]):
  run: RunConfig
  system_prompt: str | None = None
  custom: T | None = None


class CheckpointHeader(BaseModel):
  type: str = 'header'
  subsystem: str
  config_hash: str
  created_at: str
  args: dict[str, Any] = {}


class CheckpointEvent(BaseModel):
  type: str
  id: str
  timestamp: str
  payload: dict[str, Any] = {}


class VarDef(BaseModel):
  """Variable definition with weighted choices for slot generation."""

  choices: list[str]
  distribution: list[float]
  metadata: list[dict[str, Any]] | None = None
