"""Environment presets for dev and prod harness configurations.

Provides ``EnvironmentConfig`` dataclass and ``get_environment_config()``
lookup function. Dev relaxes quality gates for rapid iteration; prod
enforces strict thresholds before promotion.
"""

from autopilot.policy.gates import Gate, MinGate
from dataclasses import dataclass, field
from harness import DEFAULT_MODEL


@dataclass
class EnvironmentConfig:
  """Configuration preset for dev or prod environment.

  Attributes:
    model: Model identifier string for the inference agent.
    max_epochs: Maximum training epochs for the environment.
    max_turns: Maximum conversation turns per scenario.
    use_judge: Whether to use JudgeLoss (True) or heuristic HarnessLoss (False).
    gates: Quality gates for QualityFirstPolicy.
  """

  model: str
  max_epochs: int
  max_turns: int
  use_judge: bool = True
  gates: list[Gate] = field(default_factory=list)


DEV_CONFIG = EnvironmentConfig(
  model=DEFAULT_MODEL,
  max_epochs=5,
  max_turns=15,
  use_judge=True,
  gates=[
    MinGate('task_success_rate', threshold=0.3, required=True),
    MinGate('tool_recall', threshold=0.4, required=True),
  ],
)

PROD_CONFIG = EnvironmentConfig(
  model=DEFAULT_MODEL,
  max_epochs=10,
  max_turns=10,
  use_judge=True,
  gates=[
    MinGate('task_success_rate', threshold=0.7, required=True),
    MinGate('tool_recall', threshold=0.8, required=True),
    MinGate('tool_precision', threshold=0.7, required=True),
    MinGate('policy_compliance', threshold=0.8, required=True),
  ],
)


def get_environment_config(name: str) -> EnvironmentConfig:
  """Get environment config by name.

  Args:
    name: 'dev' or 'prod'.

  Returns:
    EnvironmentConfig for the named environment.

  Raises:
    ValueError: If name is not 'dev' or 'prod'.
  """
  configs = {'dev': DEV_CONFIG, 'prod': PROD_CONFIG}
  if name not in configs:
    raise ValueError(f"Unknown environment '{name}'. Use 'dev' or 'prod'.")
  return configs[name]
