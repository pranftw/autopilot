"""Build results from observations using Gate objects."""

from autopilot.core.models import Datum, GateResult, Result
from autopilot.policy.gates import Gate


def compute_result(observation: Datum, gates: list[Gate] | None = None) -> Result:
  """Copy observation metrics into a result and run configured gates."""
  result = Result(metrics=dict(observation.metrics))
  for gate in gates or []:
    gate_out = gate(result)
    result.gates[gate.metric] = gate_out.value
  result.passed = all(gate(result) != GateResult.FAIL for gate in (gates or []) if gate.required)
  return result
