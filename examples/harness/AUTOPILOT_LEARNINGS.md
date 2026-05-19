# AutoPilot Learnings

Framework friction discovered while building the agent harness example.
This file is **not** user-facing product docs -- it captures implementation
notes, bugs, and feature requests for the autopilot core team.

---

## Bugs

### BUG-001: PathParameter source must be str, not Path

`PathParameter.schema_entry()` passes `source` through to JSON
serialization. When `source` is a `pathlib.PosixPath`, the snapshot
payload fails with `StoreError: payload is not JSON-serializable`.

**Workaround:** Always pass `str()` as the `source` argument:

```python
# correct
PathParameter(source=f'{root}/prompts', pattern='system_prompt.md')

# broken -- causes StoreError on snapshot
PathParameter(source=Path(root) / 'prompts', pattern='system_prompt.md')
```

**Status:** Workaround applied in `HarnessModule.__init__`.

---

## Friction Points

- `build_trainer()` requires assembling Config, Store, Policy, Experiment,
  and Callbacks manually. A higher-level builder or preset pattern would
  reduce boilerplate for new examples.

---

## Feature Requests

- Per-parameter gradient routing: currently all gradients broadcast to all
  parameters. The harness would benefit from directing tool-failure gradients
  only to `tools_code` and communication-gap gradients only to
  `system_prompt` / `policies`.

---

## API Ergonomics

- `QualityFirstPolicy(gates=[...])` API is clean.
- `MetricCollection.compute()` returning a flat dict is convenient for
  policy gates that look up metrics by string key.
- `StoreCheckpointCallback` import path (`core.callbacks.store`) is
  non-obvious; easy to guess `core.callbacks.checkpoint` instead.

---

## Verification Pass Findings (2026-05-05)

### Import Verification

All public entrypoints import cleanly in a single process. No circular
import issues across the 15 module entrypoints tested. The harness package
layout with a single `__init__.py` and flat module structure works well.

### NL Assertion Evaluation Sensitivity

The keyword-heuristic NL assertion evaluator (`_check_assertion`) is sensitive
to the specific wording of assertions vs evidence. An assertion like
`'agent was polite'` only passes if the keyword `'polite'` appears in the
assistant's response text. This is a known limitation documented as future
work (LLM-as-judge upgrade). When writing test scenarios, ensure the mock
response text contains keywords that match the NL assertions.

### Metric Names as Policy Gate Keys

Policy gates reference metrics by string key. The nine canonical metric
keys produced by `HarnessMetrics.compute()` are:

```
task_success_rate, tool_recall, tool_precision, tool_argument_accuracy,
communication_recall, policy_compliance, avg_turns, error_rate, tau_reward
```

These must match exactly in `EnvironmentConfig.gates` and `build_trainer`
gate lists. A typo will silently pass (gate never triggers) rather than
raising an error. Consider adding validation that gate metric names
reference keys actually produced by the metrics collection.

### Callback Hook Naming

`MetricsWriterCallback` uses `on_epoch_end` (not `on_train_epoch_end`).
This is an intentional AutoPilot divergence from Lightning where
`on_train_epoch_end` fires before validation. Using `on_epoch_end`
ensures metrics include both train and val data when available.

### Integration Testing Pattern

Mocking `module._agent.run_conversation` with a pre-built
`ConversationResult` is the cleanest way to test the full pipeline
without API calls. The evaluator, loss, and metrics all operate on the
`EvalDatum` metadata dict, so the mock only needs to produce consistent
trajectory and tool_calls fields.

---

## JudgeLoss / AgentCollator Integration

### Mode split: `use_judge=True` vs `False`

`HarnessModule` accepts a `use_judge` boolean (default `True`) that
switches the loss function:

- **`use_judge=True`:** `JudgeLoss` + `AgentCollator` backed by a
  `PydanticAgent` (Gemma-4 on OpenRouter). Produces per-parameter
  `TextGradient` instances with direction, attribution, severity, and
  evidence fields. Richer feedback for the optimizer but requires LLM
  API calls during `backward()`.

- **`use_judge=False`:** `HarnessLoss` + `HarnessGradient`. Deterministic
  failure bucketing (tool failures, communication gaps, policy violations,
  efficiency issues). Fast, local, no API cost. Best for rapid iteration
  and debugging the pipeline itself.

The same switch is exposed on all entry points: `run.py --use-judge /
--no-judge`, `run_trainer.py --use-judge / --no-judge`, and
`autopilot -p harness optimize loop --use-judge / --no-judge`.
Default is judge mode when neither flag is passed.

### Training vs offline judge

**Training hot path (`JudgeLoss`):** The collator runs during
`loss.backward()` -- it batches evaluation feedback across the epoch's
failed scenarios and calls the `PydanticAgent` once to produce attributed
gradients. This is the **training-time** judge cost.

**Offline pipeline (`ai judge`):** `HarnessJudge` runs as a standalone
pipeline via `autopilot -p harness ai judge run --input <path>`. It
executes two steps per item: `PythonStep` (deterministic
`ConversationEvaluator`) then `LLMStep` (structured `HarnessVerdict`
critique). This is **evaluation-time** cost, independent of training.

Both paths share `ConversationEvaluator` for deterministic scoring.
The LLM critique step exists only in the offline pipeline; the training
path uses `AgentCollator` for gradient attribution instead.

### Structured output / parsing

`AgentCollator` expects the backing agent (`PydanticAgent`) to return
JSON-structured output with per-parameter attribution. The expected
shape is:

```json
{
  "direction": "...",
  "parameters": {
    "<param_id>": {
      "attribution": "...",
      "severity": 0.0,
      "evidence": ["..."]
    }
  }
}
```

When the agent returns malformed JSON, `AgentCollator` falls back to a
direction-only `TextGradient` broadcast to all parameters. This is
graceful degradation, not a hard failure. The fallback gradient has
less routing specificity but still drives optimizer edits.

`HarnessVerdict` (used by `HarnessJudge` in the offline pipeline) is
a Pydantic `BaseModel` with `score`, `critique`, `dimension_scores`,
and `recommendations`. The `LLMStep` enforces this schema via
`output_type=HarnessVerdict`.

### Parameters list alignment

`JudgeLoss` is constructed with `parameters=list(self.parameters())`.
This list must match the `PathParameter` instances registered on the
module. If parameters are added or removed, `JudgeLoss` must be
reconstructed. The current three-parameter layout (`system_prompt`,
`policies`, `tools_code`) is stable.

When `store.register_parameters(dict(module.named_parameters()))` runs,
the keys must match between store and loss. Mismatches cause gradients
to target non-existent parameter keys (silently dropped by the
optimizer).

### Cross-links

- **CLI workflows:** See `AGENT_GUIDE.md` for the full experiment
  lifecycle, operational recipes (rollback, collaboration, drift
  detection), and the external agent decision rubric.
- **Primitives table:** See `README.md` for the framework primitives
  table mapping AutoPilot concepts to harness roles.
- **Cost attribution:** Token usage flows from `HarnessAgent` through
  `EvalDatum.metadata` to `HarnessCostTrackerCallback`. See the
  cost attribution section in `AGENT_GUIDE.md`.
