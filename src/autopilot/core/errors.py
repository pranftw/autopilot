"""Explicit exception hierarchy for AutoPilot."""


class AutoPilotError(Exception):
  """Base exception for all AutoPilot errors."""


class ConfigError(AutoPilotError):
  """Configuration loading or resolution failure."""


class ExperimentError(AutoPilotError):
  """Experiment lifecycle or state transition failure."""


class TrackingError(AutoPilotError):
  """Manifest, event, or artifact tracking failure."""


class StoreError(AutoPilotError):
  """Store operation failure (snapshot, checkout, merge, etc.)."""


class PreflightError(AutoPilotError):
  """Preflight check failure preventing operation."""

  def __init__(self, failures: list[str]) -> None:
    self.failures = failures
    super().__init__(f'preflight failed with {len(failures)} issue(s): ' + '; '.join(failures))


class AIError(AutoPilotError):
  """AI generation, judging, or checkpoint failure."""


class AgentError(AutoPilotError):
  """Agent invocation failure."""


class OrchestratorError(AutoPilotError):
  """Orchestration loop failure."""
