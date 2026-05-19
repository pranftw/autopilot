"""Ad-hoc threshold policy built from CLI --min/--max flags.

Wraps ``MinGate`` / ``MaxGate`` instances into a single ``Policy`` subclass
for CLI-only evaluation paths where no project module provides a live policy.

Usage from CLI internals::

    gates = build_threshold_gates(
      min_specs=[('accuracy', 0.8)], max_specs=[('latency', 100)]
    )
    policy = ThresholdPolicy(gates)
    result = policy.forward(Result(metrics={...}))
"""

from autopilot.core.models import Result
from autopilot.core.types import GateResult
from autopilot.policy.gates import Gate, MaxGate, MinGate, collect_gate_hints
from autopilot.policy.policy import Policy

THRESHOLD_SPEC_SEPARATOR = ':'


class ThresholdPolicy(Policy):
  """Policy composed from ad-hoc MinGate/MaxGate instances.

  Used by ``policy check --min metric:threshold --max metric:threshold``
  to evaluate metrics without a project module. All gates are required.

  Attributes:
    gates: Ordered list of ``Gate`` instances to evaluate.
  """

  def __init__(self, gates: list[Gate]) -> None:
    """Create a threshold policy from a list of gates.

    Args:
      gates: Gate instances (typically ``MinGate`` / ``MaxGate``).
    """
    self._gates = gates

  @property
  def gates(self) -> list[Gate]:
    """Return the ordered list of gates for inspection.

    Returns:
      Gates evaluated on each ``Result``.
    """
    return self._gates

  def gate_hints(self) -> dict[str, str]:
    """Return metric-mismatch hints from gates after evaluation."""
    return collect_gate_hints(self._gates)

  def name(self) -> str:
    """Return ``'threshold'`` as the policy identifier."""
    return 'threshold'

  def forward(self, result: Result) -> GateResult:
    """Evaluate all gates; any FAIL means overall FAIL.

    Returns:
      ``GateResult.PASSED`` when all gates pass, ``GateResult.FAIL`` otherwise.
    """
    for gate in self._gates:
      outcome = gate.forward(result)
      if outcome == GateResult.FAIL:
        return GateResult.FAIL
    return GateResult.PASSED

  def explain(self, result: Result) -> str:
    """Return per-gate explanations joined by semicolons.

    Returns:
      Combined explanation string from all gates.
    """
    parts = [gate.explain(result) for gate in self._gates]
    return '; '.join(parts) if parts else 'no gates configured'


def parse_threshold_spec(spec: str) -> tuple[str, float]:
  """Parse a ``metric:threshold`` string into (metric_name, threshold_float).

  Args:
    spec: String in ``metric:threshold`` format (e.g. ``'accuracy:0.8'``).

  Returns:
    Tuple of (metric name, threshold value).

  Raises:
    ValueError: When the spec is malformed (missing colon, non-numeric threshold).
  """
  if THRESHOLD_SPEC_SEPARATOR not in spec:
    msg = (
      f'invalid threshold spec {spec!r}: expected metric{THRESHOLD_SPEC_SEPARATOR}threshold'
      f' (e.g. accuracy{THRESHOLD_SPEC_SEPARATOR}0.8)'
    )
    raise ValueError(msg)
  parts = spec.split(THRESHOLD_SPEC_SEPARATOR, 1)
  metric_name = parts[0].strip()
  if not metric_name:
    msg = f'empty metric name in threshold spec {spec!r}'
    raise ValueError(msg)
  try:
    threshold = float(parts[1].strip())
  except ValueError:
    msg = f'non-numeric threshold in spec {spec!r}: {parts[1].strip()!r} is not a valid number'
    raise ValueError(msg) from None
  return metric_name, threshold


def build_threshold_gates(
  min_specs: list[str] | None = None,
  max_specs: list[str] | None = None,
) -> list[Gate]:
  """Build MinGate/MaxGate instances from CLI ``--min`` / ``--max`` specs.

  Args:
    min_specs: List of ``metric:threshold`` strings for >= checks.
    max_specs: List of ``metric:threshold`` strings for <= checks.

  Returns:
    Ordered list of Gate instances (MinGate first, then MaxGate).
  """
  gates: list[Gate] = []
  for spec in min_specs or []:
    metric_name, threshold = parse_threshold_spec(spec)
    gates.append(MinGate(metric_name, threshold))
  for spec in max_specs or []:
    metric_name, threshold = parse_threshold_spec(spec)
    gates.append(MaxGate(metric_name, threshold))
  return gates
