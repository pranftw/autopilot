# Policies

## Policy protocol

`Policy` (`autopilot.policy.policy`) defines:

- `name() -> str`
- `forward(scorecard: Scorecard, phase: str) -> GateResult`
- `explain(scorecard: Scorecard, phase: str) -> str`

Use policies for phase-level accept, reject, or warn semantics and human-readable rationale beyond raw metrics.

## ScoringRule protocol

`ScoringRule` turns a single `Observation` into a `Scorecard`:

- `name() -> str`
- `forward(observation: Observation) -> Scorecard`

Typically copies metrics, runs gates, and sets `scorecard.passed` and per-metric gate labels.

## Gates

Gates are small objects in a class hierarchy (`autopilot.policy.gates`). Each gate implements `forward(scorecard: Scorecard) -> GateResult`:

- `Gate` — base type; subclass for custom behavior
- `MinGate` — metric must be `>=` a threshold
- `MaxGate` — metric must be `<=` a threshold
- `RangeGate` — metric must lie between min and max
- `CustomGate` — metric must satisfy a custom predicate function

If a metric is **missing** from `scorecard.metrics`, built-in gates return `GateResult.SKIP` (not a failure). Failures apply when the value is present but outside the allowed range.

Compose gates as a Python list and pass that list into your policy and scoring rule constructors. Example:

```python
from autopilot.policy.gates import MinGate, RangeGate
from autopilot.policy.policy import Policy
from autopilot.policy.quality_first import QualityFirstPolicy, QualityFirstScoring

gates = [
  MinGate('accuracy', 0.9),
  RangeGate('latency_p99', min=0.0, max=500.0),
]

policy = QualityFirstPolicy(gates=gates, human_review_on_warn=True)
scoring = QualityFirstScoring(gates=gates)
```

## Quality-first constructors

- `QualityFirstPolicy(gates: list[Gate] | None = None, human_review_on_warn: bool = True)` — required gates must not fail; optional gates can yield `WARN` and optionally trigger human review.
- `QualityFirstScoring(gates: list[Gate] | None = None)` — copies observation metrics into a `Scorecard`, runs each gate, records string outcomes on `scorecard.gates`, and sets `passed` from required gate results.

Policies and scoring rules that accept gates should receive the **same** `list[Gate]` instance when you want policy decisions and scorecard annotations to stay aligned.

## GateResult

`GateResult` values used at the policy and gate layer include `PASS`, `FAIL`, `WARN`, and `SKIP`. Policies interpret stacked gate results (for example, required vs optional gates) and scoring rules persist per-metric labels on the `Scorecard` for manifests and summaries.
