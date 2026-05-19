# Agent Guide: Harness Optimization

This guide is for **external coding agents** (Claude Code, Codex, etc.) operating the harness example project.

## One-Time Setup

```bash
cd examples/harness
uv sync --extra dev
```

**API keys:** Set `OPENROUTER_API_KEY` in your environment (not committed to repo).

**Environment:** Set `HARNESS_ENV=dev` (default) or `HARNESS_ENV=prod` for stricter gates.

**Workspace init** (first time only):

```bash
autopilot -p harness --workspace . workspace init --context 'bootstrap harness workspace'
autopilot -p harness --workspace . tree init dev --context 'create dev tree for iteration'
```

## Mutating commands require `--context`

Every mutating autopilot command **must** include `--context '<reason>'`. This is
enforced at the CLI layer -- omitting it on a mutating command will fail with an
error. Read-only commands (`query`, `debug`, `tree list`, `experiment show`, etc.)
are exempt.

The `--context` value flows to both the execution log and the experiment's
decision journal. Always provide a concise, specific reason explaining *why* the
command is being run.

## CLI Quick Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `experiment add` | Create new experiment | `autopilot -p harness experiment add --hypothesis '<hypothesis>' --json --context 'testing prompt structure changes'` |
| `optimize loop` | Run training loop | `autopilot -p harness optimize loop --max-epochs 5 --json --context 'initial optimization'` |
| `optimize train` | Single epoch train | `autopilot -p harness optimize train --json --context 'single epoch after prompt edit'` |
| `propose verify` | Check improvement | `autopilot -p harness --experiment <slug> --epoch <N> propose verify --proposal-id <id> --json` |
| `tree list` | Show experiment tree | `autopilot -p harness tree list --json` |
| `stabilize` | Lock winning params | `autopilot -p harness stabilize <exp_id> --json --context 'locking best prompt params'` |
| `report compare` | Compare experiments | `autopilot -p harness report compare exp-1 exp-2 --json` |
| `debug executions list` | Show execution history | `autopilot -p harness debug executions list --json` |
| `execute` | Run arbitrary code | `autopilot -p harness execute -c "print('hi')" --context 'quick inspection'` |

All commands accept `--workspace .` (resolved from project root).

## Optimization Workflow

### Judge mode (default)

The harness uses `JudgeLoss` + `AgentCollator` + `TextGradient` when
`--use-judge` is passed (or by default). For the heuristic `HarnessLoss`
path, pass `--no-judge`.

### Experiment lifecycle (phases A-F)

**A. Create experiment and initialize tree:**

```bash
autopilot -p harness --workspace . workspace init --context 'bootstrap harness workspace'
autopilot -p harness --workspace . tree init dev --context 'create dev tree'
autopilot -p harness --workspace . experiment add --hypothesis '<specific, falsifiable hypothesis>' --json --context 'testing hypothesis about prompt structure'
```

**B. Run optimization (judge mode by default):**

```bash
autopilot -p harness --workspace . optimize loop --max-epochs 5 --json --context 'initial optimization run'
```

Or with explicit judge control:

```bash
autopilot -p harness --workspace . optimize loop --max-epochs 5 --use-judge --json --context 'judge-mode optimization'
autopilot -p harness --workspace . optimize loop --max-epochs 5 --no-judge --json --context 'heuristic loss baseline comparison'
```

The loop runs epochs, applies gradients via `AgentOptimizer`, and checks
policy gates. `EpochOrchestrator` detects plateaus and auto-rolls back on
regression.

**C. Offline evaluation with judge pipeline:**

```bash
autopilot -p harness ai judge run --input harness/scenarios/val.jsonl
```

The judge pipeline runs `PythonStep` (deterministic `ConversationEvaluator`)
followed by `LLMStep` (structured `HarnessVerdict` critique). Config is
loaded from `judge_config.json` as a sibling of the `--input` file. Copy
the appropriate tier config from `configs/` before running (see
**Evaluation Tiers** below).

**D. Review and verify:**

```bash
autopilot -p harness --workspace . --experiment <slug> --epoch <N> propose verify --proposal-id <id> --json
```

Compare experiments:

```bash
autopilot -p harness --workspace . experiment compare <exp_a> <exp_b> --json
```

Review metrics:
- `experiments/<slug>/epoch_N_metrics.json` -- per-epoch metric snapshots
- `.optimization/epoch_N.md` -- optimizer feedback

**E. Branch, merge, and collaborate:**

Merge analysis (read-only classification):

```bash
autopilot -p harness --workspace . store merge-analysis <target_exp> <from_exp>
```

Merge preview (materialize conflicts):

```bash
autopilot -p harness --workspace . store merge-preview <target_exp> <from_exp> --context 'previewing merge from branch'
```

Resolve conflicts:

```bash
autopilot -p harness --workspace . store merge-resolve --token <token> <param_key> --ours --context 'keeping our prompt version'
autopilot -p harness --workspace . store merge-resolve --token <token> <param_key> --theirs --context 'adopting their tool changes'
autopilot -p harness --workspace . store merge-resolve --token <token> <param_key> --content <file> --context 'manual merge resolution'
```

Apply merge:

```bash
autopilot -p harness --workspace . store merge-apply --token <token> --context 'merging tool improvements into main'
```

**F. Stabilize and deploy:**

```bash
autopilot -p harness --workspace . stabilize <experiment_id> --json --context 'locking winning params for deployment'
```

### Inspection commands

```bash
autopilot -p harness --workspace . tree list --json
autopilot -p harness --workspace . query --json
autopilot -p harness --workspace . debug executions list --json
autopilot -p harness --workspace . --experiment <slug> debug cost --json
```

Global `--epoch` flag for epoch-scoped operations:

```bash
autopilot -p harness --workspace . --experiment <slug> --epoch 3 propose verify --proposal-id <id> --json
```

### Trainer script (non-CLI path)

```bash
uv run python run_trainer.py --max-epochs 5 --use-judge
uv run python run_trainer.py --max-epochs 3 --no-judge
uv run python run_trainer.py --max-epochs 5 --env prod --json
```

## Evaluation Tiers

Evaluation is split into explicit tiers with matching scenario JSONL files and
per-tier judge configs. Scenario files live under `harness/scenarios/`; configs
live under `configs/` at the harness project root (`examples/harness/configs/`).

### Tier overview

| Tier | Scenario file | Config file | Size | Purpose |
|------|--------------|-------------|------|---------|
| Smoke | `harness/scenarios/smoke.jsonl` | `configs/judge_smoke.json` | 5--10 | Fast sanity check; stable subset of `val.jsonl` task_ids |
| Full | `harness/scenarios/val.jsonl` | `configs/judge_config.json` | All val | Comprehensive evaluation against full validation set |
| Regression | `harness/scenarios/regression.jsonl` | `configs/judge_regression.json` | 3+ pinned | Must-pass gate; critical behaviors (refund math, cancel-all, exchanges) |
| Safety | `harness/scenarios/safety.jsonl` | `configs/judge_safety.json` | 2+ adversarial | Adversarial scenarios; policy bypass, data exfiltration, pressure tactics |
| Cost | (any tier) | (any config) | -- | Usage reporting via `CostTrackerCallback` / `debug cost` |

### How to run each tier

The `ai judge run` CLI discovers config by looking for `judge_config.json` as a
sibling of the `--input` file (i.e. `Path(input).parent / 'judge_config.json'`).
Canonical tier configs live under `configs/` at the project root. To use a
tier-specific config, **copy** (or symlink) it to `harness/scenarios/judge_config.json`
before running:

**Smoke (fast, after local edits):**

```bash
cp configs/judge_smoke.json harness/scenarios/judge_config.json
autopilot -p harness ai judge run --input harness/scenarios/smoke.jsonl
```

**Full (comprehensive):**

```bash
cp configs/judge_config.json harness/scenarios/judge_config.json
autopilot -p harness ai judge run --input harness/scenarios/val.jsonl
```

**Regression (must-pass gate, before merge):**

```bash
cp configs/judge_regression.json harness/scenarios/judge_config.json
autopilot -p harness ai judge run --input harness/scenarios/regression.jsonl
```

**Safety (adversarial, before deploy):**

```bash
cp configs/judge_safety.json harness/scenarios/judge_config.json
autopilot -p harness ai judge run --input harness/scenarios/safety.jsonl
```

The canonical configs under `configs/` are the source of truth; the copy in
`harness/scenarios/judge_config.json` is ephemeral and should not be committed.
Add it to `.gitignore` if desired.

### Multi-dimensional rubrics

`HarnessVerdict.dimension_scores` maps rubric dimension names to float scores
(0.0--1.0). Standard dimensions are `accuracy`, `tone`, and
`policy_compliance`. The safety tier adds `harm_avoidance`.

`HarnessJudge.build_summary()` aggregates per-item dimension scores into means.
Tier-specific weighting is not yet automated -- the safety judge config uses a
`system_prompt` that instructs the LLM to score `policy_compliance` and
`harm_avoidance` strictly, producing lower scores when the agent misbehaves.
Safety evaluation should overweight `policy_compliance` when interpreting
results.

Each config's `custom.rubric_dimensions` field documents which dimensions that
tier expects. Agents and scripts can read this field to know which dimensions to
report on.

### Agent workflow for tiers

1. **After local edits:** run **smoke** for fast validation.
2. **Before merge:** run **regression** + **full** to ensure no pinned scenarios regressed and overall quality is maintained.
3. **Before deploy:** run **safety** + **full** to validate adversarial robustness alongside comprehensive quality.

## Feature Request Workflow

Use the pipeline script for end-to-end feature landing:

```bash
uv run python scripts/run_feature_request.py \
  --description "Add gift card refund handling" \
  --max-epochs 5 --json
```

This drives: experiment add -> optimize loop -> propose verify.

## Regression Investigation

When a gate fails or metrics regress, investigate in this order:

1. **Experiment directory:** `experiments/<slug>/`
   - `epoch_N_metrics.json` -- per-epoch metric snapshots
   - `result.json` -- final experiment result

2. **Optimizer feedback:** `.optimization/epoch_N.md`
   - Shows what the optimizer changed and why

3. **Store snapshots:** `.autopilot/store/`
   - Content-addressed parameter versions per epoch
   - Use `autopilot -p harness --experiment <slug> store diff --source harness/ --epoch-a <N> --epoch-b <M>` for unified diffs

4. **Execution logs:** `executions.jsonl`
   - Full command history with stdout/stderr capture
   - `autopilot -p harness debug executions list --json`

5. **Forest/tree refs:** `.autopilot/store/refs.json`
   - Branch lineage and epoch pointers

## JSON Envelopes

All `--json` commands emit structured envelopes:

```json
{
  "ok": true,
  "result": { ... },
  "messages": []
}
```

On failure:

```json
{
  "ok": false,
  "error": "description of what went wrong",
  "messages": [{"level": "error", "message": "..."}]
}
```

Parse `ok` first; inspect `error` on failure. Message objects have `level` (info/success/warn/error) and `message` fields.

## JSONL Schema Reference

### `executions.jsonl`

One JSON object per line. Each row is an `ExecutionRecord`:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | ISO 8601 UTC timestamp |
| `command` | string | Resolved subcommand path (e.g. `optimize train`) |
| `args` | list[string] | argv list for reproducibility |
| `duration_ms` | float | Wall-clock duration in milliseconds |
| `exit_code` | int | Process exit code (0 = success) |
| `stdout` | string \| null | Captured stdout text |
| `stderr` | string \| null | Captured stderr text |
| `experiment` | string \| null | Experiment slug from CLI context |
| `project` | string \| null | Project slug from CLI context |
| `extra` | object | Forward-compatible metadata (git SHA, agent ID, etc.) |
| `context` | string \| null | Reason/provenance string from `--context` |

### `evaluation.jsonl`

One JSON object per line. Each row corresponds to one evaluated sample from
the judge pipeline. Contains per-sample `success`, `metrics`, `feedback`,
`metadata`, and `error_message` fields matching `EvalDatum`.

## Cost Attribution

Token and API usage flows from the inference agent through evaluation metrics
to `cost_summary.json` in the experiment directory.

### Data flow

1. **Per-turn `result.usage()`** inside `HarnessAgent.run_conversation()`:
   Pydantic AI `AgentRunResult.usage()` returns `RunUsage` with
   `input_tokens`, `output_tokens`, and `requests`. Running sums accumulate
   across all turns in one conversation.

2. **`ConversationResult`** carries `input_tokens`, `output_tokens`,
   `api_calls` (defaults 0).

3. **`EvalDatum.metadata`** receives these three keys from
   `HarnessModule.forward()` on the success path. Error paths omit token
   keys; downstream aggregation treats missing keys as zero.

4. **Harness metrics** (`TotalInputTokens`, `TotalOutputTokens`,
   `TotalApiCalls` in `harness/metrics.py`) sum values from
   `EvalDatum.metadata` during `update()`. These are registered on
   `HarnessMetrics` alongside the nine quality metrics.

5. **Epoch `Result.metrics`** merges train scalars with `val_*`-prefixed
   validation scalars (when both splits run). Keys include
   `total_input_tokens`, `total_output_tokens`, `total_api_calls` (train)
   and `val_total_input_tokens`, `val_total_output_tokens`,
   `val_total_api_calls` (validation).

6. **`HarnessCostTrackerCallback`** (subclass of `CostTrackerCallback`)
   reads these metric keys in `measure()`, setting `CostEntry.api_calls`
   to train + val API sums and `CostEntry.tokens_used` to total input +
   output tokens across both splits. On loop end, `cost_summary.json` is
   written to the experiment directory.

   - On **standalone `build_trainer()`** paths,
     `HarnessCostTrackerCallback` is automatically registered.
   - On **`optimize loop`** paths, the framework injects its own base
     `CostTrackerCallback`; the harness does **not** pre-register a second
     tracker to avoid duplication. Epoch `Result.metrics` still carries
     token sums for attribution.

7. **CLI:** `autopilot -p harness --experiment <slug> debug cost` reads `cost_summary.json`
   from the experiment directory (requires a run that attached cost
   tracking).

### Cross-experiment cost rollup

There is no first-party command for multi-experiment cost aggregation.
Aggregate `cost_summary.json` files across experiments via external scripts:

```bash
for f in experiments/*/cost_summary.json; do echo "--- $f ---"; cat "$f"; done
```

Or use `jq` for structured rollup:

```bash
jq -s '[.[].tokens_used] | add' experiments/*/cost_summary.json
```

## Key Files

| Path | Purpose |
|------|---------|
| `harness/module.py` | HarnessModule with 3 PathParameters, `use_judge` toggle |
| `harness/judge.py` | HarnessJudge (hybrid PythonStep + LLMStep), HarnessVerdict |
| `harness/agents.py` | PydanticAgent (AgentCollator bridge) |
| `harness/loss.py` | HarnessLoss + HarnessGradient (heuristic path) |
| `harness/metrics.py` | 12-metric MetricCollection (9 quality + 3 cost) |
| `harness/callbacks.py` | MetricsWriter, OptimizerContext, Deploy, HarnessCostTracker |
| `harness/cli.py` | HarnessCLI with HarnessJudge, HarnessOptimizeCommand |
| `harness/environments.py` | Dev/prod presets |
| `harness/prompts/` | System prompt + policies (optimized) |
| `harness/tools/retail_tools.py` | Tool code (optimized) |
| `configs/judge_config.json` | Full/default judge eval config |
| `configs/judge_smoke.json` | Smoke tier judge config (higher parallelism, tighter output) |
| `configs/judge_regression.json` | Regression tier judge config (pinned must-pass gate) |
| `configs/judge_safety.json` | Safety tier judge config (strict policy_compliance prompt) |
| `harness/scenarios/smoke.jsonl` | Smoke tier scenarios (5--10 subset of val.jsonl) |
| `harness/scenarios/regression.jsonl` | Regression tier scenarios (pinned critical behaviors) |
| `harness/scenarios/safety.jsonl` | Safety tier scenarios (adversarial/edge-case) |
| `.autopilot/store/` | Content-addressed parameter store |
| `experiments/<slug>/` | Per-experiment artifacts |
| `.optimization/` | Optimizer epoch feedback |
| `executions.jsonl` | Execution tracking log |

## Operational Recipes

Named recipes for production operations, team collaboration, traceability, and
debugging. Each recipe follows a Scenario / Steps / Key insights structure.

---

### Recipe: Production incident rollback

#### Scenario

A deployed agent regresses in production. Restore a known-good experiment or
epoch, verify with evals, then investigate diffs.

#### Steps

1. Identify last known-good experiment and epoch:
   ```bash
   autopilot -p harness tree show
   autopilot -p harness --experiment <slug> store log --source harness/
   ```

2. Restore working tree files to that state:
   ```bash
   autopilot -p harness checkout <good-experiment-id> --context 'rollback to known-good state after regression'
   ```
   Or, within one experiment, restore a specific epoch:
   ```bash
   autopilot -p harness --experiment <slug> --epoch <epoch> store checkout --source harness/ --context 'restore epoch N params'
   ```

3. Verify rollback with evaluation:
   ```bash
   autopilot -p harness ai judge run --input harness/scenarios/val.jsonl
   autopilot -p harness experiment compare <good-id> <bad-id>
   ```

4. Redeploy restored parameters (deployment is external to AutoPilot).

5. Investigate what changed:
   ```bash
   autopilot -p harness --experiment <slug> store diff --source harness/ --epoch-a <good-epoch> --epoch-b <bad-epoch>
   autopilot -p harness debug gradients
   autopilot -p harness debug executions list
   ```

#### Key insights

- `store checkout` restores real `PathParameter` files on disk, not metadata alone.
- Pair `checkout` / `store checkout` with `ai judge run` to confirm behavior before redeploying.

---

### Recipe: Team collaboration and parallel hypotheses

#### Scenario

Two engineers pursue different hypotheses on separate trees, compare outcomes,
then merge the winning line into main.

#### Steps

1. Create a tree per hypothesis branch:
   ```bash
   autopilot -p harness tree create tool-improvements --context 'branch for tool refactoring'
   autopilot -p harness tree create prompt-improvements --context 'branch for empathy experiments'
   ```

2. Engineer A works on tool improvements:
   ```bash
   autopilot -p harness tree switch tool-improvements --context 'switching to tool branch'
   autopilot -p harness experiment add --hypothesis 'refactor lookup tools for edge cases' --context 'tool edge case experiment'
   autopilot -p harness optimize loop --max-epochs 5 --context 'optimize tool handling'
   ```

3. Engineer B works on prompt improvements:
   ```bash
   autopilot -p harness tree switch prompt-improvements --context 'switching to prompt branch'
   autopilot -p harness experiment add --hypothesis 'improve empathy in system prompt' --context 'empathy prompt experiment'
   autopilot -p harness optimize loop --max-epochs 5 --context 'optimize empathy tone'
   ```

4. Compare experiments:
   ```bash
   autopilot -p harness report compare tool-exp-1 prompt-exp-1
   ```

5. Merge winner into target experiment (positional IDs: target, from):
   ```bash
   autopilot -p harness store merge-analysis main-exp tool-exp-1
   autopilot -p harness store merge-preview main-exp tool-exp-1 --context 'previewing tool improvements merge'
   ```
   For each conflict key:
   ```bash
   autopilot -p harness store merge-resolve --token <token> <key> --ours --context 'keeping main prompt'
   autopilot -p harness store merge-resolve --token <token> <key> --theirs --context 'adopting tool changes'
   autopilot -p harness store merge-resolve --token <token> <key> --content <path> --context 'manual merge of policies'
   ```
   Apply the merge:
   ```bash
   autopilot -p harness store merge-apply --token <token> --context 'merging tool experiment into main'
   ```

#### Key insights

- Trees isolate parallel work; `store merge-*` composes results with explicit per-key resolution when conflicts exist.
- Skip `merge-resolve` when `merge-preview` reports no conflicts; use `merge-apply` with the issued token.

---

### Recipe: Selective parameter rollback

#### Scenario

Metrics improved for tools but regressed on prompts. Keep current tools,
restore prior prompt artifacts via per-key merge resolution.

#### Steps

1. Inspect changes between epochs:
   ```bash
   autopilot -p harness --experiment <slug> store diff --source harness/ --epoch-a 2 --epoch-b 5
   ```

2. Preview merge from the experiment that holds the desired prompt state:
   ```bash
   autopilot -p harness store merge-preview current-exp old-prompt-exp --context 'previewing selective prompt rollback'
   ```

3. Resolve each `PathParameter` key (positional key after `--token`):
   ```bash
   autopilot -p harness store merge-resolve --token <token> system_prompt --theirs --context 'restoring prior prompt'
   autopilot -p harness store merge-resolve --token <token> policies --theirs --context 'restoring prior policies'
   autopilot -p harness store merge-resolve --token <token> tools_code --ours --context 'keeping current tools'
   ```

4. Apply:
   ```bash
   autopilot -p harness store merge-apply --token <token> --context 'selective rollback: prompts reverted, tools kept'
   ```

#### Key insights

- Granularity follows module parameters (`system_prompt`, `policies`, `tools_code`).
- Order: `merge-preview` -> one or more `merge-resolve` -> `merge-apply` with the same token.

---

### Recipe: Traceability and decision audit

#### Scenario

Answer why behavior changed between versions (epochs or experiments) with an
evidence chain.

#### Steps

1. Parameter diffs across epochs:
   ```bash
   autopilot -p harness --experiment <slug> store diff --source harness/ --epoch-a 2 --epoch-b 5
   ```

2. Gradient and collator feedback:
   ```bash
   autopilot -p harness debug gradients
   ```
   Inspect `.optimization/epoch_N.md` under the experiment workspace for
   per-epoch `AgentOptimizer` feedback.

3. Metrics progression:
   ```bash
   autopilot -p harness --experiment <slug> store log --source harness/
   autopilot -p harness report compare exp-v2 exp-v5
   ```

4. Proposal verdicts (global `--epoch` before subcommand):
   ```bash
   autopilot -p harness --experiment <slug> --epoch 5 propose verify --proposal-id <id>
   ```

5. Command history:
   ```bash
   autopilot -p harness debug executions list
   ```

#### Key insights

- Trace path: hypothesis and notes -> gradient attribution -> `store diff` -> `report compare` -> `propose verify`.

---

### Recipe: Checkpoint resume after crash

#### Scenario

Long training or eval job stops mid-way. Resume without rerunning completed
epochs or judge items.

#### Steps

1. **Trainer resume:** pass checkpoint path into the harness trainer entrypoint:
   ```bash
   uv run python run_trainer.py --max-epochs 20 --ckpt-path .autopilot/checkpoints/epoch_14.json
   ```
   Align with flags implemented in `run_trainer.py` for this project.

2. **Judge pipeline resume:**
   ```bash
   autopilot -p harness ai judge resume --checkpoint <checkpoint_file> --input harness/scenarios/val.jsonl
   ```

#### Key insights

- Training checkpoints (`Trainer` / `CheckpointCallback` / store snapshots) and judge JSONL checkpoints are different systems; pick the path that matches the failure surface.

---

### Recipe: Scheduled re-evaluation and drift detection

#### Scenario

CI or cron re-fingerprints scenario files, detects drift, re-runs judge,
compares metrics to baseline.

#### Steps

1. Fingerprint current scenario files (with `cwd` `examples/harness/`; paths
   relative to that root):
   ```bash
   uv run python -c "from pathlib import Path; from autopilot.ai.fingerprint import compute_fingerprint; fp = compute_fingerprint([Path('harness/scenarios/val.jsonl')]); print(fp.to_dict())"
   ```

2. Compare to stored fingerprint using `detect_drift(old_fp, new_fp)` from
   `autopilot.ai.fingerprint` (module-level functions, not methods on
   `DatasetFingerprint`).

3. Run judge on deployed parameters:
   ```bash
   autopilot -p harness ai judge run --input harness/scenarios/val.jsonl
   ```

4. Compare experiments:
   ```bash
   autopilot -p harness report compare deployed-baseline latest-eval
   ```

#### Key insights

- AutoPilot supplies fingerprinting, judge, and `report compare`; scheduling lives in CI/cron.

---

### Recipe: New team member onboarding

#### Scenario

A new engineer needs orientation on branches, best experiments, notes, and
history.

#### Steps

1. View the experiment tree:
   ```bash
   autopilot -p harness tree show
   ```

2. List completed experiments:
   ```bash
   autopilot -p harness query --completed
   ```

3. Find the best experiment by metric:
   ```bash
   autopilot -p harness query --best task_success_rate
   ```

4. Read experiment notes:
   ```bash
   autopilot -p harness experiment notes show <id>
   ```

5. View store history:
   ```bash
   autopilot -p harness --experiment <slug> store log --source harness/
   ```

6. Compare experiments:
   ```bash
   autopilot -p harness report compare <baseline-id> <candidate-id>
   ```

7. Switch to a production tree and inspect:
   ```bash
   autopilot -p harness tree switch prod --context 'onboarding: inspecting prod tree'
   autopilot -p harness tree show
   ```

#### Key insights

- `query --best` discovers peak experiments; `experiment notes show` captures human intent and context.

---

### Recipe: Debugging a specific failing scenario

#### Scenario

One scenario regresses after a given epoch. Narrow the parameter and gradient
changes that correlate.

#### Steps

1. Locate failing samples in per-epoch evaluation artifacts (`evaluation.jsonl`
   or project evaluation logs) by scenario id.

2. Diff parameters between epochs:
   ```bash
   autopilot -p harness --experiment <slug> store diff --source harness/ --epoch-a 2 --epoch-b 3
   ```

3. Read `.optimization/epoch_2.md` (or adjacent epoch) for collator/optimizer-facing guidance.

4. Inspect gradients:
   ```bash
   autopilot -p harness debug gradients
   ```

5. Promote the scenario into the regression tier, then run:
   ```bash
   autopilot -p harness ai judge run --input harness/scenarios/regression.jsonl
   ```

#### Key insights

- Tie together store diffs, written epoch feedback, and a small regression JSONL slice for fast iteration.

---

### Recipe: Performance plateau diagnosis

#### Scenario

`EpochOrchestrator` stopped for plateau. Decide whether data, hypothesis,
optimizer, or branching is next.

#### Steps

1. Read trainer output / experiment logs for orchestrator stop reason (plateau
   vs rollback vs policy).

2. Inspect evaluation JSONL for failure concentration by scenario type.

3. Choose a branch: broaden eval and re-fingerprint, change hypothesis, adjust
   optimizer strategy, or open parallel trees (cross-link: team collaboration
   recipe above).

4. Start a new experiment when ready:
   ```bash
   autopilot -p harness experiment add --hypothesis 'plateau at 85% -- trying tool-level changes' --context 'escaping prompt plateau with structural change'
   autopilot -p harness optimize loop --max-epochs 10 --context 'tool-level optimization after prompt plateau'
   ```

#### Key insights

- Plateau is a signal to change the experiment design, not only micro-edits; use parallel trees when hypotheses compete.

---

## Decision Guide

Strategy and capability-extension guidance for choosing *what* to try next,
complementing the operational recipes above that explain *how* to run commands.

---

## Types of Local Optima

"Local optimum" in this context means *stagnation in the optimization loop*
-- the agent keeps iterating on a parameter surface (prompt wording, tool code,
evaluation data, or metric definition) without meaningful improvement. It is not
a mathematical property of a loss landscape; it is a practical observation that
the current axis of change has been exhausted.

Recognizing which type of plateau you are in determines the escape route. The
table below maps four common stagnation patterns to their symptoms, evidence
sources, and escape strategies:

| Type | Symptom | Evidence | Escape |
|------|---------|----------|--------|
| **Prompt** | Micro-edits to same section, marginal gains | `store diff` shows same region changing; plateau | Rewrite prompt structure (not wording); branch experiment |
| **Tool** | Individual tools work but composition fails | High tool recall but low task success; many repair turns | Change tool contracts, response formats, orchestration |
| **Metric** | Proxy metric high but real quality low | Secondary metrics flat/declining while primary improves | Add/promote metrics; adjust policy gates; change judge rubric |
| **Dataset** | Great on eval, weak in production | Narrow failure slices on perturbations; fingerprint unchanged | Refresh eval data; add adversarial scenarios; holdout validation |

**Detecting the type.** Use `store diff --epoch-a <start> --epoch-b <end>` to
see whether changes concentrate in the same prompt region (prompt optimum) or
scatter across tool code (tool optimum). `report compare` across experiments
reveals whether the primary metric improves while secondaries stall (metric
optimum). `debug gradients` shows whether the collator keeps recommending the
same kind of change epoch over epoch -- a strong signal that the current
parameter surface is saturated.

**Dataset fingerprinting.** When `compute_fingerprint` returns the same digest
across experiments but production quality lags, the evaluation data may not
cover real failure modes. Use `detect_drift` (from `autopilot.ai.fingerprint`)
to confirm the dataset has not changed, then add adversarial or slice-specific
scenarios to break out.

**EpochOrchestrator signals.** When the orchestrator stops with a plateau
reason, read the stop reason from trainer output. A plateau after few epochs
suggests a structural problem (wrong parameter surface); a plateau after many
epochs suggests genuine convergence or a metric ceiling.

---

## External Agent Decision Rubric

An eight-phase rubric for external coding agents choosing what to do next.
Follow the phases in order; each phase produces the input for the next.

```
1. SURVEY: what's been tried?
   autopilot -p harness tree show
   autopilot -p harness query --completed --json
   autopilot -p harness query --best task_success_rate --higher --json

2. COMPARE: what worked and what didn't?
   autopilot -p harness report compare <best-exp> <recent-exp> --json
   # read .optimization/epoch_*.md for gradient history

3. DIAGNOSE: why are we stuck?
   autopilot -p harness --experiment <slug> store diff --source harness/ --epoch-a <plateau-start> --epoch-b <latest>
   # if diffs are small and in same region -> prompt local optimum
   # if diffs are large but metrics flat -> wrong parameter being changed
   autopilot -p harness debug gradients
   # if gradients keep recommending same thing -> structural change needed

4. HYPOTHESIZE: what's the next experiment?
   # formulate a FALSIFIABLE hypothesis
   # "structured output format reduces repair turns" not "improve prompt"
   autopilot -p harness experiment add --hypothesis "<specific, falsifiable>" --context '<reason for this hypothesis>'

5. BRANCH: isolate the experiment
   # if it's a radical change, branch the tree
   autopilot -p harness tree create <hypothesis-branch> --context 'isolating radical change'
   autopilot -p harness tree switch <hypothesis-branch> --context 'switching to hypothesis branch'

6. EXECUTE: run optimization
   autopilot -p harness optimize loop --max-epochs 5 --context '<what we expect to learn>'

7. EVALUATE: did it work?
   autopilot -p harness report compare <baseline> <candidate> --json
   autopilot -p harness --experiment <slug> --epoch <N> propose verify --proposal-id <id>

8. DECIDE: continue, branch, or stop?
   # if improved -> merge into main, continue
   # if plateau -> try different hypothesis type (tools vs prompts vs data)
   # if regressed -> revert, try opposite direction
   # if N branches all plateau near same level -> accept current best, ship
```

**Phase details.** SURVEY uses `tree show` and `query` to build a map of prior
work. COMPARE uses `report compare` and optimizer feedback files to understand
relative strengths. DIAGNOSE combines `store diff` (parameter-level changes)
with `debug gradients` (collator recommendations) to identify the stagnation
type (see **Types of local optima** above). HYPOTHESIZE forces a falsifiable
statement -- vague goals like "improve prompt" waste epochs. BRANCH isolates
radical changes so regression is contained. EXECUTE runs the optimization loop.
EVALUATE uses both metric comparison and proposal verification. DECIDE closes
the loop: merge winners, revert losers, or accept convergence and ship.

---

## Experiment notes conventions

Write structured, grep-friendly notes on every experiment using
`experiment notes write`. Structured notes make it possible for agents and
humans to query experiment history programmatically.

```
autopilot -p harness experiment notes show <id>
autopilot -p harness experiment notes write <id> \
  'hypothesis: structured output reduces repair turns
   edit_focus: prompt_structure
   parent_strategy: prompt_wording (saturated at 72%)
   stopped_because: plateau after 3 epochs
   key_finding: structured output helped accuracy but hurt tone
   next_steps: try tool-level changes for remaining failures' \
  --context 'recording findings after prompt structure experiment'
```

### Required fields

- **`hypothesis`** -- the falsifiable claim being tested.
- **`edit_focus`** -- which parameter surface was changed. Allowed values:
  `prompt_wording` | `prompt_structure` | `tool_contract` | `tool_implementation` | `data` | `evaluator`.
- **`stopped_because`** -- why the experiment ended (plateau, regression,
  hypothesis confirmed/refuted, manual stop).
- **`key_finding`** -- the main takeaway, positive or negative.

### Optional fields (encouraged)

- **`parent_strategy`** -- the approach that preceded this experiment and its
  outcome (e.g. "prompt_wording (saturated at 72%)").
- **`next_steps`** -- what to try if this line continues.

Consistent use of these fields enables `query` filtering and cross-experiment
audits. When an agent reads notes from prior experiments (SURVEY phase of the
decision rubric), structured fields let it extract hypotheses and findings
without free-text parsing.

---

## Adding a New Tool

Adding a new tool to the harness involves both optimizable parameters and
application wiring. Adding a function to
`examples/harness/harness/tools/retail_tools.py` (the versioned `PathParameter`
/ `tools_code` surface) is **not** sufficient -- the new tool must also be
registered in **`TOOL_NAMES`** in `examples/harness/harness/tool_loader.py`,
which is **application code outside** the store's versioned parameters. The
optimizer can edit the Python tool file but cannot complete registration without
that code change.

### Workflow

1. Add eval scenarios covering the new capability to
   `harness/scenarios/train.jsonl` and `harness/scenarios/val.jsonl` (paths
   relative to `examples/harness/` as in repo layout).
2. Update `RetailDB` / seed data if needed (application code change in
   `harness/database.py`).
3. Add the function to `retail_tools.py` (PathParameter -- optimizer/coding
   agent can edit).
4. Add the function name to `TOOL_NAMES` in `tool_loader.py` (application
   code -- human or coding agent).
5. Update `ConversationEvaluator` if new assertion types are needed (application
   code in `harness/evaluator.py`).
6. Re-fingerprint dataset (see drift detection recipe above), create new
   experiment, optimize.

### Boundary: optimizable vs application

Optimizable parameters (`system_prompt`, `policies`, `tools_code`) live inside
the store and are versioned, diffed, merged, and rolled back by the framework.
Application wiring (`TOOL_NAMES` in `tool_loader.py`, `ConversationEvaluator`,
`RetailDB`) sits outside the store. Agents must respect this boundary when
planning work: a prompt-only optimization epoch cannot add new tools, and a tool
addition requires both a code change in `retail_tools.py` *and* a registration
change in `tool_loader.py`.

---

## Prompt architecture decisions

When to use multiple `PathParameter`s vs one combined file for prompt content.

| Split when | Keep combined when |
|------------|-------------------|
| Different lifecycle (instructions change weekly, policies change monthly) | Tight coupling (one references the other) |
| Different owners (prompt engineer vs policy team) | Small total size |
| Need selective rollback (revert policies but keep tone) | Optimizer needs to see full context to make coherent changes |
| Want finer-grained gradient attribution from collator | Fewer store merge keys to manage |

The harness's three-parameter layout (`system_prompt`, `policies`, `tools_code`
on `HarnessModule` in `examples/harness/harness/module.py`) is the reference
balance. `system_prompt` and `policies` have different change cadences (prompt
structure vs policy rules); `tools_code` is a completely different medium
(Python vs markdown). This split lets the optimizer target one surface without
disturbing others, enables selective rollback via `store merge-resolve`, and
produces more specific gradient attribution from the collator.

Extending the layout to four or more parameters requires: adding a new
`PathParameter` field on `HarnessModule`, updating the concatenation / read
path in `_read_instructions` (or analogous method), and registering the new
parameter in the loss's parameter list (`JudgeLoss(parameters=...)`) so the
collator can produce per-parameter gradients. The store automatically picks up
new parameters via `Module.parameters()`.

---

## Structural vs prompt-only changes

A decision tree for when to move beyond prompt optimization into code-level
changes:

```
Failure pattern persists after 3+ prompt-only epochs?
  YES -> Inspect failure taxonomy:
    Same tool errors -> consider tool contract changes (code)
    Same flow issues -> consider conversation strategy (code: agent, simulator)
    Same scope gaps -> consider new tools/capabilities (code + data)
  NO -> Continue prompt optimization
```

### Signals that structural changes are needed

- `EpochOrchestrator` plateau on the monitored metric
- `store diff` shows repeated edits to the same prompt section
- `debug gradients` keeps recommending the same type of change
- `propose verify` returns `inconclusive` on multiple branches

### Signals that prompt optimization is working

- Clear monotonic improvement in early epochs
- Failures concentrate in one bucket that maps to a prompt section
- Gradient attribution is specific and varied across epochs

When structural signals dominate, shift from prompt edits to capability
changes: new tools (see **Adding a new tool** above), modified tool contracts,
conversation flow adjustments, or evaluator refinements. These are code changes
(tier 3 in the agent execution interface), not parameter tuning.

---

## Slice-driven Optimization

Improve a weak failure slice without rerunning the full suite every inner-loop
step. Useful when overall metrics are acceptable but a specific scenario
category underperforms.

### Workflow

1. Create focused scenario JSONL: `harness/scenarios/complex_returns.jsonl`
2. Create matching judge config: `configs/judge_complex_returns.json` (configs
   directory is `examples/harness/configs/`, not under the `harness/` package)
3. Run targeted eval:
   ```bash
   autopilot -p harness ai judge run --input harness/scenarios/complex_returns.jsonl
   ```
4. Create experiment with targeted hypothesis:
   ```bash
   autopilot -p harness experiment add --hypothesis 'improve handling of complex multi-item returns' --context 'targeting weak complex_returns slice'
   ```
5. Optimize with focus (training data biased toward weak scenarios)
6. Verify no regression on full suite:
   ```bash
   autopilot -p harness ai judge run --input harness/scenarios/val.jsonl
   ```
7. Compare:
   ```bash
   autopilot -p harness report compare <baseline> <candidate>
   ```

### Guard rails

Always run `regression.jsonl` (must-pass) alongside any slice-focused
optimization to catch regressions:

```bash
cp configs/judge_regression.json harness/scenarios/judge_config.json
autopilot -p harness ai judge run --input harness/scenarios/regression.jsonl
```

Regression scenarios are pinned behaviors that must never break. If a
slice-focused optimization regresses a pinned scenario, revert and adjust the
hypothesis to account for the regression constraint.
