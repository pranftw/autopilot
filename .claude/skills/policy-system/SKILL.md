---
name: policy-system
description: Policy and Metric base classes, Gate hierarchy, and result computation for gating experiment progression. Use when implementing new policies, metrics, or modifying gate evaluation logic.
---

## Policy and Metric

**Policy** is defined in `src/autopilot/policy/policy.py`. **`Metric`** lives in `src/autopilot/core/metric.py` as **`Metric(Module)`** (torchmetrics-style metrics are child modules, registered in **`Module._modules`**). `Policy` is a **concrete base class** with sensible default `forward()` behavior. Subclass it and override `forward()` (and `explain()` when needed). `PolicyProtocol` in `policy.py` supports structural typing. **`Policy.__call__(result)`** forwards to **`forward(result)`** and returns **`GateResult`**.

**Metric** follows a torchmetrics-style lifecycle: `update(datum)`, `compute() -> dict[str, float]`, `reset()`, with **`forward(datum)`** delegating to **`update`** + **`compute`**. Subclass **`Metric`** for accumulation across datums; see **`QualityFirstMetric`** for optional gate wiring on top.

**Policy** -- evaluates experiment results after metrics exist:

- `name() -> str` -- stable identifier
- `forward(result: Result) -> GateResult` -- returns PASS, FAIL, WARN, or SKIP
- `explain(result: Result) -> str` -- human-readable explanation

**Metric** (core) -- accumulates per-datum state and exposes aggregated numbers:

- `name() -> str` -- stable identifier
- `update(datum: Datum) -> None` -- incorporate one execution outcome
- `compute() -> dict[str, float]` -- finalize metrics for the current window
- `reset() -> None` -- clear internal state between windows

**QualityFirstMetric** additionally provides `to_result(metrics=None) -> Result` to apply gates and populate `Result.gates` / `passed`.

## GateResult enum

- `PASS` -- condition met
- `FAIL` -- condition not met for a gate that evaluated
- `WARN` -- used at policy level (for example optional gates failed)
- `SKIP` -- gate did not apply (for example metric missing from result)

## Gate class hierarchy

Defined in `src/autopilot/policy/gates.py`. Gates are objects with `forward(result) -> GateResult`, not dict-driven helpers. Base type is `Gate(metric, *, required=True)`.

**MinGate** -- `forward(result)` returns PASS if `result.metrics[metric] >= threshold`, FAIL if present and below threshold, **SKIP if the metric is missing**.

**MaxGate** -- same pattern with `<= threshold`.

**RangeGate** -- `__init__(metric, min, max, *, required=True)`; `forward(result)` returns PASS if `min <= value <= max`, FAIL if out of range, **SKIP if missing**.

**CustomGate** -- `__init__(metric, fn, *, required=True)` where `fn(value) -> bool`; **SKIP if metric missing**, otherwise PASS or FAIL from `fn`.

Missing metrics: built-in gate classes return `GateResult.SKIP` for a missing metric, **not** FAIL.

## Result computation

`compute_result(datum: Datum, gates: list[Gate] | None = None) -> Result` in `src/autopilot/policy/scoring.py`:

- Copies datum metrics into a `Result`
- Runs each gate, recording string outcomes in `result.gates`
- Sets `result.passed` from required gates only (optional gates that SKIP do not fail the result by themselves)

## Quality-first builtin

`src/autopilot/policy/quality_first.py`:

- **`QualityFirstPolicy(gates: list[Gate] | None = None, human_review_on_warn: bool = True)`** -- optional gate list; when empty, `forward()` returns PASS (nothing to evaluate).
- **`QualityFirstMetric(gates: list[Gate] | None = None)`** -- optional gate list; `update()` accumulates metric values; `to_result()` applies the same gate recording logic as `compute_result`.

Wire gates inside **`Policy`** subclasses (for example **`QualityFirstPolicy(gates=[...])`**) and pass a **`Policy`** instance into **`Trainer(..., policy=...)`** when you want epoch-level gating during **`Trainer.fit()`**. The Trainer builds a fresh **`Result(metrics=...)`** each epoch from aggregated **`Metric.compute()`** outputs and calls **`policy(result)`**; on **`GateResult.FAIL`** it stops early and may invoke **`Store.checkout`** when a **`Store`** is configured.

For offline manifest workflows, **`evaluate_experiment_policy()`** in **`services.py`** still loads persisted **`Result`** JSON and runs the same **`Policy`** protocol independently of **`Trainer`**.

## Adding a new policy

1. Subclass **`Policy`** and/or **`Metric`**, or implement **`PolicyProtocol`** for duck typing.
2. Instantiate **`Gate`** subclasses with explicit thresholds or callables where needed.
3. Pass the composed **`Policy`** into **`Trainer(policy=...)`** (or evaluate it manually via **`evaluate_experiment_policy()`**) -- pure Python wiring, not string-key lookup in core.

## Offline policy evaluation

In `src/autopilot/core/services.py`, **`evaluate_experiment_policy(experiment_dir, policy)`**:

1. Loads persisted **`Result`** data for the experiment (returns a skip payload if none).
2. Calls **`policy.forward(result)`** and **`policy.explain(result)`**.
3. Appends a **`policy_evaluated`** event with the gate outcome and explanation.

Human-review transitions on WARN are **not** implemented inside this helper; handle them in project overlays if needed.

## Key files

- `src/autopilot/policy/policy.py` -- `Policy`, `PolicyProtocol`
- `src/autopilot/core/metric.py` -- `Metric(Module)`, `CompositeMetric`
- `src/autopilot/policy/gates.py` -- `Gate`, `MinGate`, `MaxGate`, `RangeGate`, `CustomGate`
- `src/autopilot/policy/scoring.py` -- `compute_result()`
- `src/autopilot/policy/quality_first.py` -- reference policy and metric with optional gate lists

## Gotchas

- Gate failures on the result do not by themselves change experiment status unless your command layer applies them. **`Trainer.fit()`** stops training on **`GateResult.FAIL`** but does not mutate **`Manifest`** status by itself.
- `compute_result()` and **`QualityFirstMetric.to_result()`** populate `result.gates` and `passed` from **`Gate`** instances; gates do not rely on a separate dict-based rule shape.
- Missing metrics yield **SKIP** from built-in gates, not automatic FAIL. Policy logic (for example **`QualityFirstPolicy`**) treats FAIL only when a gate returns FAIL; SKIP is not FAIL.
