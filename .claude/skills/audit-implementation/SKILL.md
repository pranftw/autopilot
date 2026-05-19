---
name: audit-implementation
description: Post-implementation audit of a master plan. Systematic dogfooding with uv run, /tmp copies, parallel stress testing, multi-round discovery, issue cataloging, and fix master plan generation. Use after implementing a master plan, when dogfooding, stress-testing, or ruthlessly testing implemented features.
---

## Philosophy

- **This is about IMPLEMENTED code, not plan files.** create-master-plan Phase 1 also runs `uv run python -c` -- but that is discovery on the existing codebase to inform plan writing. This skill tests code that was RECENTLY IMPLEMENTED from a master plan. Different temporal context, same tools.
- **Find EVERYTHING that's broken**, not just obvious failures.
- "Don't be lazy, be enthusiastic and ruthless and explore all possibilities."
- "Ruthlessly test out all features and whether everything works."
- Discovery first, fix plan second -- full exploration and testing, THEN the master plan.
- Test from all angles: unit, integration, usability, concurrency, edge cases, real-world examples.
- **Testing must verify logic and behavior**, not just that code runs without crashing. Check that computed values are correct, state transitions are valid, side effects modify the right state, error paths raise specific exceptions. Surface-level "it doesn't crash" tests are insufficient for autonomous implementation.
- **Design tests with edge cases in mind from the start.** Before testing a function, enumerate its edge cases (empty inputs, boundary values, None, concurrent access, malformed data). Test each one explicitly.
- **The agentic loop:** implement -> test -> verify -> fix -> repeat. Testing thoroughness directly determines whether this loop converges. Weak tests mean infinite fix cycles. Strong tests mean quick convergence.
- Iterate with fresh lenses -- each round uses a different angle. Don't stop when you hit N issues; stop when new rounds stop finding new CATEGORIES of issues.
- **Use subagents generously.** Every discovery round, every testing scope, every deep dive should be delegated to subagents. This keeps the context window clean, enables parallel investigation, and ensures thorough coverage. Launch many subagents per round (8-11 per round is normal). Skipping subagents when they are warranted is a process failure.
- **No backward compatibility.** The implemented code and any fix plans must enforce a clean break unless the user explicitly requested otherwise. No aliases, no deprecation shims, no fallback imports, no bridge code, no `getattr` with fallback, no legacy support. Flag any backward-compat patterns found during testing as issues in the catalog.

## When to use

After a master plan has been implemented. When you want to systematically verify the implementation works, find all issues, and produce a comprehensive fix plan. When the user asks to dogfood, stress-test, or ruthlessly test an implemented feature.

## Phase 0: Read the implemented plan

Before testing anything, launch a subagent to read the master plan directory (`.cursor/plans/<feature>/`):
- Read `00-master-plan.md` + all sub-plans (`01-NN.md`)
- Map claimed scope: what subsystems were touched, what classes were added/modified/deleted, what APIs changed
- Extract the verification protocol and checkpoint definitions from the master plan
- Produce a testing guide: which subsystems to prioritize, which APIs to exercise, which edge cases the plan claimed to handle

This gives Phase 1 subagents a focused testing map instead of exploring blindly. Without this step, discovery rounds may miss subsystems the plan touched or waste time on untouched code.

## Phase 1: Systematic discovery

### Round 0 -- Automated quality gate

Before any AI-driven testing, run the deterministic quality gate (see CLAUDE.md ## Verification):

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check src/
uv run ast-grep scan --config sgconfig.yml src/ tests/
uv run pytest -x -v
```

Any failures are instant catalog entries (Phase 2) with severity "code quality" -- no AI interpretation required. If the gate passes, all CLAUDE.md prohibited patterns are confirmed clean.

### Round 1 -- Layer-scoped testing

Parallel subagents by subsystem. Each reads AND tests its scope using `uv run python -c "..."`.

Example scopes:
- Trainer.fit: every code path, dry_run, checkpoints, callbacks
- Node/Tree: topological sort, serialization round-trip, query builder
- FileStore: merge, snapshot, restore, concurrent access
- Config/FilePath: path resolution, cascading, parameterization
- Module state_dict: save/load, nested modules
- DataLoader: edge cases (empty, single item, len vs iter mismatch)
- Epoch loop: line-by-line stepping, advance_epoch, hooks
- Callbacks: dispatch, ordering, exception handling
- Metrics: accumulation, reset, collection, metadata
- Environment/IsolatedEnvironment: setup, teardown, worktree creation, isolation

Each subagent produces: PASS/FAIL per test, specific error messages, `uv run python -c` reproduction commands.

Each subagent must test at multiple levels:
- **Smoke test:** basic import and instantiation
- **Logic test:** verify computed values are correct (not just non-None)
- **Behavior test:** verify state changes, side effects, ordering
- **Edge case test:** empty, None, boundary, malformed inputs
- **Error path test:** verify specific exception types and messages

### Round 2 -- Cross-cutting testing

Parallel subagents by concern:
- Import cycle detection: `uv run python -c "from autopilot.x import Y"` for all public modules
- Integration paths: end-to-end flows crossing subsystem boundaries
- Concurrency: parallel experiment execution, store isolation, file locking
- **Prohibited patterns:** Confirmed clean by Round 0 automated gate. Focus remaining effort on AI-only semantic review (see checklist below).

### AI-Only Semantic Review

These require human/AI judgment and CANNOT be automated by tooling. Check each one explicitly during Round 2:

- No defensive programming / needless fallbacks / bloat (judgment: is this guard necessary?)
- No try/except for things that won't fail in normal flow (judgment: will this fail?)
- No `.get(x, default)` defensively everywhere (judgment: is the key required by contract?)
- No comments explaining obvious code (judgment: is it obvious?)
- Let errors propagate naturally (judgment: is this catch adding value?)
- Validate at system boundaries only (judgment: is this a boundary?)
- No fake fallback objects on precondition failure (judgment: is this a fallback?)
- No inline file-content strings for templates (judgment: is this a template?)
- No module-level constants caching function calls (judgment: is it caching?)
- No bridge/adapter code preserving old APIs (judgment: is this backward compat?)
- No config-file-driven workflows (judgment: is this a workflow config?)
- Store uses only snapshot()/restore() (API usage pattern)
- No private extension/customization methods (not all `_` methods are hooks)
- graph.py zero autopilot imports (`rg 'from autopilot' src/autopilot/core/graph.py`)
- No `__init__.py` files (`find src/ -name '__init__.py'`)
- Test quality: logic verified not just "doesn't crash"; enumerate edge cases before testing
- Test performance: no recursive `pytest` inside tests, no `subprocess.run` for import validation (use `exec()` / direct import), no `shutil.copytree` of examples to run via subprocess
- DRY: no reimplemented utilities (DictMixin, MetricsComparator, tracking.io)
- Architecture: right layer, right responsibility

### Round 3 -- Dogfooding with real examples

- Copy examples (e.g. `examples/textmatch`) to `/tmp`
- May need to fix the autopilot path in `pyproject.toml` for the copy
- Create MULTIPLE copies in `/tmp` for parallel testing
- Run through the full workflow as a real user/agent would
- Exercise every CLI command, every Python API
- Test features like: tree branching, parallel experiments, store, stabilize, environment, trainer fit

**Acceptance bar:** The primary example (e.g. `examples/textmatch`) must complete a full run with exit code 0. This is the minimum bar for declaring dogfooding successful. If the primary example fails, implementation is not complete.

Why `/tmp`:
- Isolated from main repo -- tests the "install and use" experience
- Catches path assumptions, missing files, broken imports
- Multiple copies enable parallel stress testing and concurrency verification
- Reveals real usability issues vs lab-condition testing

### Round 4+ -- Targeted deep dives

Based on findings from earlier rounds, launch focused subagents on suspicious areas.

**PoC verification for novel patterns:** If the implementation introduced novel mechanics (e.g. FilePath descriptors, VFS symlinks, content-addressed storage), build throwaway PoC scripts:
- `cd /tmp && uv init test_xyz`
- Write a self-contained test script with assertions that print PASS/FAIL
- `uv run python test_xyz.py`
- A failed PoC is a confirmed bug for the catalog

### Coverage justification

Before moving to Phase 2, answer concretely: "how are you sure you have explored all possibilities and all edge cases?" In practice, full discovery takes 3+ rounds and 20+ subagent runs. A premature catalog after only one round will be rejected.

Don't limit scope to the implemented master plan's subsystem. Implementation may have broken things outside its direct scope. "Launch subagents to uncover issues with the full codebase."

## Phase 2: Issue cataloging

After each discovery round, synthesize findings. Per issue:
- **Issue ID** (sequential, 1-N)
- **Severity** (crash, wrong behavior, missing feature, code quality, test gap, test performance)
- **Subsystem** (core, AI, CLI, data, tracking, etc.)
- **Description** (specific: "Tree.remove raises NameError on line 115" not "Tree has issues")
- **Reproduction** (`uv run python -c "..."` command)
- **Root cause** (if identified)
- **Proposed fix** (code-level, not just "fix it")

**Catalog format** (markdown table or list per issue):

```
### BUG-001: WidgetFactory.create raises TypeError on empty config (illustrative -- not a real bug)
- **Severity:** crash
- **Subsystem:** core/widgets
- **Description:** `WidgetFactory.create({})` raises TypeError instead of returning default widget
- **Reproduction:** `uv run python -c "from myapp.core.widgets import WidgetFactory; WidgetFactory().create({})"`
- **Root cause:** Missing empty-dict guard before unpacking config keys
- **Proposed fix:** Add `if not config: return DefaultWidget()` guard at line 42 of `core/widgets.py`
```

Key: iterate with fresh lenses. In practice, issue catalogs grow from ~13 (premature) to ~53 to ~64 across 4 rounds as new angles surface new categories of issues.

## Phase 3: Fix master plan generation

Use the **create-master-plan** skill to produce a fix master plan from the issue catalog:

- One master plan (`00-master-plan.md`) + sub-plans (`01-NN.md`)
- Group by logical dependency, not discovery order
- Design principles for fix plans:
  - **Fix bugs before adding tests** -- a test for broken code is a test that expects failure
  - **Bottom-up order** -- IO/serialization first, then core, then AI, then CLI, then integration
  - **Each sub-plan includes both fixes AND tests** for what it touches
- Include banned patterns, verification protocol, dependency graph
- Include transcript reference for traceability

Then follow the **verify-plan** skill to prove the fix master plan is implementation-ready.

## Gotchas

- Don't rush from discovery to plan. Full discovery (3+ rounds, 20+ subagents) comes first.
- Don't limit scope to just the implemented master plan's subsystem. Check the full codebase.
- `uv run python -c` must test BEHAVIOR, not just imports. Call methods, create objects, exercise edge cases.
- `/tmp` copies must be fully isolated. Don't accidentally test against the workspace.
- The fix master plan is itself a master plan -- the full create-master-plan + verify-plan process applies.
