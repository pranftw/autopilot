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
- **Follow existing patterns exactly**: `manifest.json`, `events.jsonl`, slugs for directories.

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
- `atomic_write_json`, `append_jsonl`, `read_jsonl`, `read_json` from `tracking.io`
- `Event` + `create_event` / `append_event` from `tracking.events` (delegates to `tracking.io`)
- `Datum` from `core.types`, `Result` and `Manifest` from `core.models`
- `Output` class with `result()`, `info()`, `table()` from `cli.output`
- `make_subparser` from `cli.resolvers` (used internally by `Command.register()` in `cli/command.py`)

Don't create parallel tracking, storage, or output systems.

## Storage patterns

Follow existing experiment storage:
```
autopilot/<feature>/
  evaluations/<slug>/
    manifest.json       # atomic write
    results.jsonl       # one result per line
    events.jsonl        # append-only
  sessions/<slug>/
    manifest.json
    events.jsonl
```

## Plan structure

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
2. Each subplan ends with: write tests, run ruff, run pytest, verify imports
3. Never proceed to next subplan until current one passes

### Comprehensive testing per subplan

**Unit tests:**
- Every function: normal cases, edge cases, error conditions
- Dataclass round-trips: `to_dict()` -> `from_dict()` -> assert equal
- Pure functions with realistic inputs matching actual data shapes
- Configurable behavior tested with varied inputs

**Integration tests:**
- End-to-end flows: module -> helpers -> I/O
- CLI parser tests: verify flag combinations parse correctly
- Handler tests: mock module, verify params and `ctx.output.result()` structure

**Mocking:**
- HTTP: `unittest.mock.patch('requests.post')` with realistic responses
- File I/O: `tmp_path` fixture, not filesystem mocking
- Time: mock `time.sleep` for ramp-up, `time.monotonic` for latency

**Test data:**
- Realistic dataset items: `{'id': '...', 'turns': [{'role': 'user', 'content': '...'}]}`
- Realistic API response formats (actual SSE, actual JSON structures)
- No trivial "hello world" data

**Test organization:** one file per subplan: `test_<feature>_helpers.py`, `test_<feature>_adapter.py`, `test_<feature>_cli.py`

### Verify before finalizing

```bash
uv run python -c "from autopilot.x.y import Z"   # confirm imports exist
uv run pytest                                    # existing tests pass
uv run ruff check src/                           # lint clean
```

Check referenced paths exist, functions have expected signatures.

## Policy directives

- **No backward compatibility.** All renames, removals, and API changes are clean breaks. No aliases, no deprecation shims, no fallback imports. Every consumer (library, tests, examples, docs, skills) is updated in the same plan.
- **Thorough and comprehensive testing.** Every subplan must include tests covering: normal paths, edge cases, error conditions, round-trips, subclass-override scenarios, and integration flows. Tests are written alongside the code, not as an afterthought. All existing tests must be updated for any rename or API change.

## Code quality rules

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
