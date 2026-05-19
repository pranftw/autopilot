---
name: create-plan
description: Feature planning guide for AutoPilot covering architecture boundaries, config layering, naming, testing, and subplan structure. Use when planning a new feature, designing a new module or command, or writing an implementation plan.
---

## Architecture: base engine vs project overlay

**Base engine** (`src/autopilot/`):
- Reusable library code: protocols (`Module`, `Policy`, `Metric`), CLI infrastructure, tracking, state machine.
- No project-specific URLs, credentials, or product defaults baked in as the only way to run; those belong in the consuming project’s wiring.
- Shared defaults for protocol shapes (message types, path segments) live as module-level constants where they are protocol, not tenant config.

**Project overlay** (`autopilot/projects/<name>/` or equivalent):
- Concrete `Module`, `Trainer` wiring, project `CLI` subclass, and constructor-time kwargs on adapters.
- Python objects instantiated in the project CLI and trainer are the source of truth for what runs; there is no separate workflow config file that overrides that wiring.

## Config layering

```
CLI args  >  constructor kwargs  >  defaults in project CLI subclass
```

- **Constructor kwargs**: module instance configuration passed when objects are built (timeouts, endpoints, paths, metric objects).
- **CLI args**: override runtime values exposed on the project CLI (flags registered on `Command` / `CLI`).
- **Project CLI subclass**: owns argparse defaults and composition; keep shared-library constructors free of one-off product constants.

Wire protocol constants (SSE message types, API path segments) are module-level constants in the engine where they describe the protocol, not tenant-specific config.

## Naming conventions

- **ML vocabulary**: experiment, datum, metrics, result, split, epoch, checkpoint, hyperparams, evaluate, infer. Not 'run', 'job', 'batch', 'probe'.
- **Acronyms capitalized in class names**: `TGInferAdapter` not `TgInferAdapter`. Files stay snake_case.
- **Follow existing patterns exactly**: `forest.json`, `results.jsonl`, slugs for directories.

## Data model conventions

- `str | None = None` for optional strings. Never `str = ''`.
- `int | None = None` or `float | None = None` for optional numerics where 0 isn't meaningful.
- `list[...] = field(default_factory=list)` and `dict[...] = field(default_factory=dict)` for collections.
- `metadata: dict[str, Any] = field(default_factory=dict)` -- always a dict, never optional.
- All dataclasses get `to_dict()` / `from_dict()`.

## Agent-first design

All tooling is primarily used by agents. Every design decision from the agent's perspective:

- Every command supports `--json` producing `{'ok': bool, 'result': {...}, 'messages': [...]}` envelope
- JSON results include everything an agent needs for the next step -- metrics, slugs, status, paths
- No interactive prompts or confirmations
- Session/evaluation slugs returned in results so agents can store and reuse them
- Error messages must be actionable and machine-parseable

## DRY -- reuse existing implementations

One canonical implementation per concern. Reuse:
- `atomic_write_json`, `append_jsonl`, `read_jsonl`, `read_json`, `utc_now_iso`, `read_json_dict` from `tracking.io`
- `Logger` / `JSONLogger` from `core.logger` for metrics, events, and hyperparams logging
- `Datum` from `core.types`, `Result` from `core.models`
- `Output` class with `result()`, `info()`, `table()` from `cli.output`
- `make_subparser` from `cli.resolvers` (used internally by `Command.register()` in `cli/command.py`)

Don't create parallel tracking, storage, or output systems.

## Storage patterns

Experiment state persists through Forest/Tree (`forest.json`) and
`Experiment.state_dict()`. Logger records metrics and events. Training
checkpoints use `CheckpointIO` / `JSONCheckpointIO`. Evaluation pipelines
use their own append-only JSONL storage:

```
autopilot/<feature>/
  evaluations/<slug>/
    results.jsonl       # one result per line (append-only)
  experiments/
    forest.json         # Forest persistence (experiment tree)
```

## Subagent usage

Use subagents generously for exploration, verification, and coherence checking. Delegate reading large codebases, running tests, verifying paths, and checking consistency to subagents rather than doing everything in the main context. This keeps the context window clean and allows thorough investigation. When in doubt, launch a subagent.

## Plan structure

**For complex individual plans, use create-meta-plan first** to iterate on discovery, POC verification, and design, then use this skill to write the final plan file. For simpler features where the design is obvious, you may use this skill directly.

### Plan file format (mandatory)

Every plan file MUST use this exact structure. No deviations. This ensures consistency across all master plan systems and enables autonomous implementation by agents that have never seen the original discussion.

```markdown
---
name: <short descriptive title>
overview: '<one-line summary of what this plan does>'
phase: '<phase letter or name> (NN-NN)'
sub_plan: 'NN / total'
depends_on: [<list of plan filenames this depends on>]
---

# Sub-plan NN: <title> (Phase X)

## 1. Context

**Phase:** X (description), sub-plan **N of total**.

**Purpose:** <what this plan accomplishes and why it matters>

**Affected files:**
- `src/autopilot/path/to/file.py` (MODIFY)
- `src/autopilot/path/to/new_file.py` (NEW)
- `tests/path/to/test_file.py` (NEW)

## 2. Subplans

### 2.1 <first implementation step>

<description of what to implement and why>

<reference implementation -- target signatures, contracts, code samples>

\```python
class Foo:
  """Target class with full contract."""

  def bar(self, x: int) -> str:
    """Does bar. Returns formatted string."""
    ...
\```

<any additional implementation notes, constraints, gotchas>

**Tests for 2.1:**
- `test_foo_bar_normal` -- assert `Foo().bar(1) == '1'`
- `test_foo_bar_edge` -- assert `Foo().bar(0) == '0'`

**Gate:**
\```bash
uv run ruff check src/autopilot/path/to/file.py
uv run ruff format --check src/autopilot/path/to/file.py
uv run ty check src/
uv run ast-grep scan --config sgconfig.yml src/autopilot/path/to/file.py
uv run pytest tests/path/to/test_file.py -x -v
uv run python -c "from autopilot.path.to.file import Foo; print('ok')"
\```

### 2.2 <next implementation step>

<description, reference code, tests, gate>

...

## 3. Key Decisions

| Decision | Rationale |
|----------|-----------|
| <what was decided> | <why> |
| ... | ... |

## 4. Tests

New file: `tests/path/to/test_file.py`

### 4.1 <test group name>

1. **`test_function_name`** -- <what it verifies>. Assert `<specific assertion>`.
2. **`test_another`** -- <what it verifies>. Assert `<specific assertion>`.

### 4.2 <another test group>
...

## 5. Verification (full plan gate)

\```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check src/
uv run ast-grep scan --config sgconfig.yml src/ tests/
uv run pytest -x -v
uv run python -c "from autopilot.x import Y; print('ok')"
\```

ast-grep scan enforces all CLAUDE.md prohibited patterns. See CLAUDE.md ## Verification for remaining `rg` checks.

**Expected:** <what success looks like for each command>

## 6. Docs

- **`src/autopilot/path/file.py`** -- update class docstring to reflect <change>
- **`CLAUDE.md`** -- add <specific bullet> to <specific section>
- Or: "None (internal only)"

## Dependencies (existing code)

- **`autopilot.core.types.Datum`** -- used in isinstance checks
- **`autopilot.core.module.module.Module`** -- base class for new component
```

**Note:** When part of a master plan, this format applies to each sub-plan file (not the `00-master-plan.md` which has its own structure per the create-master-plan skill).

### Frontmatter field definitions

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | short title (matches the `# Sub-plan` heading) |
| `overview` | yes | one-line description of scope |
| `phase` | yes | which phase this belongs to (e.g. `'A (01-04)'`) |
| `sub_plan` | yes | position in sequence (e.g. `'03 / 13'`) |
| `depends_on` | yes | list of plan filenames (empty `[]` if none) |

### Section rules

1. **Sections are always numbered 1-6** in this order: Context, Subplans, Key Decisions, Tests, Verification, Docs. The Dependencies section is unnumbered and placed last.

2. **Section 2 (Subplans)** uses `2.1`, `2.2`, `2.3` numbering. Each subplan is a logical unit executed and verified independently. Each subplan MUST include:
   - description of what to implement and why
   - reference implementation: target signatures, contracts, and code samples showing the desired end state (not file-level diffs -- show the target code/design the implementor should produce)
   - tests specific to this subplan step (listed inline)
   - **gate:** runnable verification commands (ruff check, ruff format, ty check, ast-grep scan, pytest on relevant files, import check). Never proceed to the next subplan until the current gate passes.

   For reference implementations, show enough code to be unambiguous -- full method signatures with types, class contracts, key logic. The implementor writes the actual code guided by these references. This is NOT a diff format; it is "here is what the code should look like when you are done."

3. **Section 3 (Key Decisions)** is a markdown table with `Decision | Rationale` columns. Every non-obvious choice made during planning goes here. If the plan has no non-obvious decisions, write `| Format only (no behavioral decisions) | mechanical change |`.

4. **Section 4 (Tests)** uses `4.1`, `4.2` subsection numbering. This is the consolidated test inventory for the entire plan (individual subplan steps also list their own tests inline for gating). Each test case is a numbered list item with:
   - bold `test_function_name`
   - what it verifies (one phrase)
   - specific assertion (not "test that it works" but `assert result.metrics == {'accuracy': 0.95}`)

5. **Section 5 (Verification)** is the full-plan gate run after ALL subplans pass. Contains runnable bash commands covering the entire scope. After each command or block, state the expected outcome. The canonical quality gate (see CLAUDE.md ## Verification) replaces individual banned-pattern greps. Include any plan-specific `rg` checks for patterns not covered by tooling.

6. **Section 6 (Docs)** names exact files and what to update. Use "None (internal only)" only when the plan truly changes no public surface.

7. **Dependencies** lists existing modules/classes this plan relies on. This helps the implementor know what imports and APIs are available without reading unrelated code.

### Per-subplan gating (critical)

The implementing agent MUST treat each subplan (2.1, 2.2, ...) as an atomic unit with its own gate. The workflow is:

1. Implement subplan 2.1
2. Write its tests
3. Run its gate (ruff check + format + ty + ast-grep + pytest + import verification)
4. Gate passes -> proceed to 2.2
5. Gate fails -> fix before proceeding

This prevents cascading failures. If subplan 2.3 depends on 2.1 being correct, the gate after 2.1 catches issues immediately rather than discovering them late. The full-plan verification (section 5) is the final pass after all subplan gates have passed.

### Plans must be self-contained

Implementation happens in a separate session. The plan IS the handoff document. Include:
- Full context: why, current state, desired state
- All design decisions with rationale
- Key constraints stated directly (not "as discussed")
- Reference file paths for existing patterns
- Complete implementation details per subplan

### Split into subplans

Each subplan is a logical unit executed and verified independently:
1. Group by dependency (helper module -> CLI -> overlay)
2. Each subplan ends with: write tests, run quality gate (ruff check + format + ty + ast-grep + pytest), verify imports
3. Never proceed to next subplan until current one passes

### Comprehensive testing per subplan

Testing is given the HIGHEST priority. Since implementation is fully autonomous, tests are the primary mechanism for verifying correctness. Surface-level tests that only check "it doesn't crash" are insufficient. Tests must verify logic, behavior, integration, and edge cases.

**Design with edge cases in mind.** Before writing implementation details, enumerate the edge cases for each function/class. Design the solution to handle them, then write tests that verify each one. Don't add edge case handling as an afterthought.

**Unit tests:**
- Every function: normal cases, edge cases, error conditions
- Dataclass round-trips: `to_dict()` -> `from_dict()` -> assert equal
- Pure functions with realistic inputs matching actual data shapes
- Configurable behavior tested with varied inputs
- **Logic verification:** test that computed values are correct, not just that the function returns without error
- **Boundary conditions:** empty inputs, single element, maximum size, zero values, None where applicable
- **Error paths:** verify specific exception types and messages, not just "raises something"

**Behavioral tests:**
- State transitions: verify objects move through expected states correctly
- Side effects: verify that methods modify the right state and leave other state untouched
- Ordering: verify that operations produce correct results regardless of call order where applicable
- Idempotency: verify operations that should be idempotent actually are

**Integration tests:**
- End-to-end flows: module -> helpers -> I/O
- CLI parser tests: verify flag combinations parse correctly
- Handler tests: mock module, verify params and `ctx.output.result()` structure
- **Cross-boundary flows:** verify data passes correctly between subsystems (e.g. Trainer -> Store -> Tree)
- **Full lifecycle:** create -> use -> modify -> serialize -> deserialize -> verify state preserved

**Mocking:**
- HTTP: `unittest.mock.patch('requests.post')` with realistic responses
- File I/O: `tmp_path` fixture, not filesystem mocking
- Time: mock `time.sleep` for ramp-up, `time.monotonic` for latency

**Test data:**
- Realistic dataset items: `{'id': '...', 'turns': [{'role': 'user', 'content': '...'}]}`
- Realistic API response formats (actual SSE, actual JSON structures)
- No trivial "hello world" data

**Test performance:**
- Tests must run in-process. Never shell out to `uv run pytest`, `uv run ruff`, or `subprocess.run(['uv', 'run', 'python', '-c', ...])` for validation that can be done with direct imports, `exec()`, or library calls.
- Never re-run `pytest` from inside a test (recursive invocation).
- Never `shutil.copytree` example directories to run them via subprocess. Test the logic in-process instead.
- Full suite target < 15s, no individual test > 1s.

**Test DRY:** Before writing test stubs, check existing shared doubles:
- `tests/doubles.py` -- NoopEvalModule, DirectNumericLoss, NoOpOptimizer, make_run_config
- `tests/data/conftest.py` -- SizedDataset, ds5/ds9/ds10/empty_ds fixtures
- `tests/core/conftest.py` -- make_experiment, completed_exp/pending_exp/running_exp/failed_exp/cancelled_exp

Only create local stubs when behavior is genuinely unique. Duplicate test doubles are treated as DRY violations.

**Test organization:** one file per subplan: `test_<feature>_helpers.py`, `test_<feature>_adapter.py`, `test_<feature>_cli.py`

### Verify everything through code

If you have any doubt about whether an approach is correct, whether an API works the way you think, whether a class has the attributes you expect, or whether your logic is sound -- verify it with `uv run python -c "..."` before writing it into the plan. Do not guess. Do not assume. Run it.

This applies throughout plan writing. Any claim about existing code behavior must be verified, not assumed.

**Novel patterns:** If the plan introduces novel patterns (new serialization approaches, OS-level mechanics, composable APIs, import resolution tricks), build a lightweight proof-of-concept and run it before finalizing:

- **Fresh/novel patterns**: create a throwaway project in `/tmp` via `uv init`, write a self-contained test script with assertions, run with `uv run python test_xyz.py`
- **Patterns touching existing codebase**: verify directly in the repo with `uv run python -c "..."`
- If the PoC fails, fix the design in the plan -- don't leave unverified assumptions
- Document findings (ordering gotchas, edge cases discovered, required workarounds) back into the plan

### Verify before finalizing

```bash
uv run python -c "from autopilot.x.y import Z"            # confirm imports exist
uv run ruff check src/ tests/                              # lint clean
uv run ruff format --check src/ tests/                     # format clean
uv run ty check src/                                       # type check clean
uv run ast-grep scan --config sgconfig.yml src/ tests/     # prohibited patterns clean
uv run pytest                                              # existing tests pass
```

Check referenced paths exist, functions have expected signatures.

**Sequential subplan simulation:** Walk subplans in order. After each, verify:
- The codebase would be in a valid state (imports resolve, no undefined references)
- If subplan N depends on subplan M, M's outputs match N's inputs
- No subplan leaves a broken intermediate state

**Implementor-POV check:** Read the plan as a separate agent seeing it for the first time:
- Self-contained? Implementable without the discussion context?
- All file paths concrete and verified to exist (or clearly marked "new file")?
- All code samples syntactically valid (not pseudocode unless labeled)?
- All method signatures complete (parameters, types, return types)?
- All tests specific enough to write without guessing?

This is a lightweight write-time check; for exhaustive verification use the verify-plan skill.

## Policy directives

- **No backward compatibility.** All renames, removals, and API changes are clean breaks. No aliases, no deprecation shims, no fallback imports. Every consumer (library, tests, examples, docs, skills) is updated in the same plan.
- **Thorough and comprehensive testing.** Implementation is fully autonomous -- the agentic loop is: implement, test, verify, fix, repeat. Testing must be thorough enough to drive this loop without human intervention. Every subplan must include tests covering: normal paths, edge cases, error conditions, round-trips, subclass-override scenarios, and integration flows. Tests must verify logic and behavior, not just that code runs without crashing. Consider all edge cases during design and test for them explicitly. Tests are written alongside the code, not as an afterthought. All existing tests must be updated for any rename or API change.

## Code quality rules

Tool-enforced (ast-grep + ruff catch these automatically at implementation time):
- `str = ''` defaults, `.get('key', '')`, `getattr` on args, deferred imports, etc.
- Full list: see CLAUDE.md ## Verification and `rules/`

AI-judgment required (review during implementation):
- No defensive programming, needless fallbacks, or bloat
- No try/except for things that won't fail in normal flow
- No `.get(x, default)` defensively everywhere
- No comments explaining obvious code
- Let errors propagate naturally -- catch only when adding real value
- Validate at system boundaries only (CLI input, external API responses)
- No hardcoded values or magic numbers

## Migration alongside new features

When adding a feature, if existing code violates these principles, migrate in the same plan -- don't defer. If you're about to follow a pattern and existing code does it wrong, fix both together.

## Plan coherence check

After every round of edits, grep for:
- Stale references (old names, removed features, dead parameters)
- Examples matching what constructors and CLI flags actually accept
- Test descriptions matching actual function signatures
- CLI flag names aligning with module params and `CLIContext` fields
- Gates referencing metrics that will actually be produced

### Ambiguity scan

After every round of edits, hunt for these 10 patterns. The implementor should NEVER need to guess:

1. "Consider" / "Option A or B" / "either...or" without a picked winner
2. Design "if" without resolution
3. Vague verbs: might, could, should consider, worth exploring
4. Missing specifics: "update callers" without listing which callers
5. "Document" / "add docstring" without the exact text to write
6. Proposed changes with unresolved "or"
7. Vague tests: "edge cases" without specifying which
8. Missing full method signatures for new methods
9. "Follow existing patterns" without citing which file
10. Open behavioral decisions: what happens on error? what is the default?

If you chose between options, record which and why. This is a lightweight write-time check; for exhaustive auditing use the verify-plan skill.

### Cross-plan consistency (when part of a master plan)

When this plan is one of many sub-plans:
- API assumptions match sibling plans (e.g. if this plan assumes `Experiment.store` exists, another plan must add it)
- No issue/bug ID collisions with other plans (IDs use format `BUG-NNN`, each owned by exactly one sub-plan)
- Dependency claims match the master plan's dependency graph
- Naming consistent across plans (not `experiment_data` in one and `experiment_state` in another)

### DRY at implementation level

Would implementing this plan create duplicate logic?
- Helper methods on the wrong class when a better host exists?
- Reimplemented serialization when `DictMixin` exists?
- Metric comparison logic when `MetricsComparator` exists?
- Think about the resulting codebase, not just the plan text.

## When part of a master plan

When this plan is one sub-plan of a master plan, additional requirements apply:

- Must work with only: this plan file + master plan (`00-master-plan.md`) + codebase. No other sub-plan files required.
- Cross-references to other sub-plans are informational ("this builds on plan 02's Store redesign"), never blocking ("see plan 02 for the method signature" without inlining it).
- Key signatures, class hierarchies, and API contracts that this plan depends on from other plans must be inlined.
- Transcript reference is available via the master plan if additional context is needed.
- If at a checkpoint boundary (~every 3-4 plans), the verification section must include a codebase-wide check (full quality gate per CLAUDE.md ## Verification, all imports, examples), not just files this plan touched.
- **Per-plan verification:** Every plan's verification section must include: quality gate (ruff check, ruff format, ty check, ast-grep scan, pytest), import verification for touched modules. If ANY check fails, the implementing agent must first check whether a later sub-plan is planned to address it (read remaining plan files). If yes, note as "expected -- deferred to sub-plan NN." If no, fix it in place.
- **Self-healing awareness:** The implementing agent has full read/write access to source code and plan files. If a fix reveals a gap in this plan, update the plan file to document what was actually done. Genuine regressions are fixed in place; work planned for later sub-plans is not duplicated.
- Testing-first: every subplan has specific tests with concrete expectations, not vague "add tests for edge cases."

## AI module planning guidance

When planning features that involve LLM operations:

- **Built-in vs extensible**: Consider whether new components should ship as built-in defaults (like `SlotPlanner`/`VarDef`) or require project-level implementation. If most projects would use it directly, make it built-in.
- **Step-based workflows**: Prefer `LLMStep`/`PythonStep` sequences over tool-call-driven agent loops. Code controls the flow; LLM steps produce structured data only.
- **Checkpointable protocol**: Use `state_dict()`/`load_state_dict()` for anything that needs state persistence across resume boundaries (taxonomy state, RNG state, etc.).
- **Extension points via protocols**: Define `Protocol` classes for structural typing alongside base classes with shared defaults.
- **Base vs overlay split**: Universal fields in base models (`DataItem`: id, turns, split). Everything project-specific goes in `custom: T` generic field.

## Documentation updates are per-subplan, not a separate pass

Every subplan that changes a public API must update the affected class/module docstrings in the same subplan. All code-knowledge documentation (contracts, extension points, gotchas, invariants) lives in source docstrings, not in external skill files or concept docs. Project-level meta-docs (CLAUDE.md, README.md, PHILOSOPHY.md) still exist for agent guidance, project structure, and design philosophy -- update those when structural changes affect them.

### What to update (check every item for each subplan)

For **every** subplan, scan this list and update anything affected:

1. **Source docstrings** -- the class or module docstring in the file you changed IS the documentation. If you change a constructor signature, add a method, or change behavior, update the docstring in the same subplan.
2. **CLAUDE.md** (symlinked to AGENTS.md) -- extension model bullets, configuration invariants, key files list, prohibited anti-patterns. If you add a class, change a convention, or add a key file, update CLAUDE.md.
3. **README.md** -- code examples, package layout, component mapping table. If an example would break with your change, fix it now.
4. **PHILOSOPHY.md** -- if the feature introduces a new design principle or changes how an existing principle applies.
5. **`examples/`** -- if this subplan changes public API, templates, or CLI commands, update all affected examples (imports, config, scripts). Broken examples are broken documentation.

### How to enforce this in the plan

Each subplan's task list must include a **"Docs"** line item that names the specific files to update. Example:

```
## Subplan 3: Add FooCallback

1. Implement FooCallback in core/callbacks/stage.py
2. Wire into optimize loop defaults
3. Tests: test_foo_callback.py
4. Docs: FooCallback docstring (contract + hooks), CLAUDE.md (stage callbacks bullet)
```

If a subplan changes nothing public, the Docs line says "None (internal only)". But it must be present -- the absence of a Docs line is a plan defect.

### Verification at subplan close

After each subplan passes tests and lint, verify that the class docstring reflects the new behavior. If a docstring still describes the old behavior, the subplan is not done.

### Stale docs are bugs

Stale docstrings are worse than none. They cause agents to follow outdated patterns and generate code that doesn't work. Treat a stale docstring the same way you'd treat a failing test -- fix it before moving on.
