"""High-level experiment services.

Policy evaluation is a pure function: takes Result + Policy, returns gate results.
Experiment creation and lifecycle are handled by Experiment class.
"""

from autopilot.core.models import Result
from autopilot.core.normalization import load_result
from autopilot.policy.policy import Policy
from autopilot.tracking.events import append_event, create_event
from pathlib import Path
from typing import Any


def evaluate_experiment_policy(
  experiment_dir: Path,
  policy: Policy,
) -> dict[str, Any]:
  """Run policy evaluation on experiment results. Pure function -- no side effects."""
  result_data = load_result(experiment_dir)
  if not result_data:
    return {
      'policy': policy.name(),
      'gate_result': 'skip',
      'explanation': 'no result available',
    }

  eval_result = Result.from_dict(result_data)
  gate_result = policy.forward(eval_result)
  explanation = policy.explain(eval_result)

  event = create_event(
    event_type='policy_evaluated',
    message=explanation,
    metadata={
      'policy': policy.name(),
      'gate_result': gate_result.value,
    },
  )
  append_event(experiment_dir, event)

  return {
    'policy': policy.name(),
    'gate_result': gate_result.value,
    'explanation': explanation,
    'result': eval_result.to_dict(),
  }
