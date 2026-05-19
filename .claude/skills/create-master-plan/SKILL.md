# Create Master Plan

## What this skill produces

This skill generates a **complete master plan system** -- a directory of plan files that together form the full spec for a feature or redesign. The final deliverable is:

```
.cursor/plans/<feature>/
  00-master-plan.md    # top-level spec: context, design, dependency graph, protocols
  01-name.md           # sub-plan 1
  02-name.md           # sub-plan 2
  ...
  NN-name.md           # sub-plan N
```

**Input:** A meta plan document (from the **create-meta-plan** skill) or inline design discussion for simpler features. For complex redesigns, use create-meta-plan first to iterate on discovery, POC verification, and design before generating plan files. For simpler multi-plan features where the design is obvious, you may use this skill directly.

**Your job is to generate plans, not implement them.** This skill produces only plan files. Do not write source code, do not modify the codebase, do not run implementations. The only code you run is `uv run python -c "..."` to verify assumptions during plan creation. Implementation happens later, by separate agents executing each sub-plan sequentially.

**Ask the user for the target directory** if not specified. The default convention is `.cursor/plans/<feature>/` where `<feature>` is a short slug describing the scope (e.g. `experiment-testing`, `cli-redesign`). Confirm the directory with the user before writing any files.

## Philosophy

- **Nothing left to the implementor.** Every design decision must be pinned. Ambiguity in a plan becomes inconsistency when separate agents execute each plan.
- **Self-contained plans.** Each sub-plan will be executed by a separate agent with no access to other sub-plans or the discussion history.
- **Testing as highest priority.** Implementation is fully autonomous -- the agentic loop is: implement, test, verify, fix, repeat. Testing and verification must be thorough enough to drive this loop without human intervention. Plans must specify tests that cover logic, behavior, integration, and edge cases -- not just surface-level smoke tests. Every feature must be designed with edge cases in mind from the start, not patched after the fact.
- **Point of no return.** Once implementation starts, ~20 plans execute sequentially. Plans must be verified (via the verify-plan skill) before this is safe.
- **Use subagents generously.** Delegate plan generation, auditing, and verification to subagents. This keeps the context window clean and allows thorough parallel work. When in doubt, launch more subagents rather than fewer.
- **No backward compatibility.** All plans must enforce a clean break unless the user explicitly requests otherwise. No aliases, no deprecation shims, no fallback imports, no bridge code, no `getattr` with fallback, no legacy support patterns. This is non-negotiable.

## When to use

When redesigning a major subsystem from scratch. When multiple interdependent plans are needed. When the scope is too large for a single create-plan skill invocation.

## Process

### Phase 1: Write the Master Plan

The master plan deliverable is the entire plan system described in "What this skill produces" above -- NOT just `00-master-plan.md`. Every reference to "the master plan" in this skill means the complete directory of plan files. The `00-master-plan.md` file is the top-level spec; the sub-plans (`01-NN.md`) are the implementation-ready specs for each logical unit.

Create the top-level master plan file (`00-master-plan.md`) with these sections:

1. **Context**: what's wrong, what we're building, why it's a clean break
2. **Design Principles**: every key decision from the discussion, with justification
3. **Class Hierarchy**: core classes and builtin classes with file locations and purpose
4. **Detailed Class Design**: attributes with justification, method signatures, code samples
5. **Kill List**: what gets deleted/moved with reason
6. **Sub-Plans**: one per logical unit, dependency-ordered. Each tracked issue/bug gets a unique ID (e.g. `BUG-001`, `BUG-002`) owned by exactly one sub-plan. IDs must not collide across plans. Each sub-plan contains:
   - Context (what this plan does)
   - Subplans (grouped implementation steps)
   - Key decisions
   - Tests -- this section must be comprehensive and is the highest priority:
     - Unit tests: every function with normal, edge, and error cases
     - Logic tests: verify computed values are correct, not just non-None
     - Behavior tests: verify state transitions, side effects, ordering
     - Edge case tests: empty inputs, boundary values, None, malformed data (enumerate explicitly)
     - Error path tests: verify specific exception types and messages
     - Round-trip tests: `to_dict()` -> `from_dict()` -> assert equal for all dataclasses
     - Integration tests: cross-boundary flows between subsystems
     - Lifecycle tests: create -> use -> modify -> serialize -> deserialize -> verify
     - Subclass/override tests: verify customization points work
   - Examples migration: if this plan changes public API, templates, or CLI commands, include explicit `examples/` update steps (imports, config, scripts)
   - Verification commands
7. **Dependency Graph**: ASCII diagram showing plan ordering. Plans MUST be numbered so that numeric order IS execution order (01, 02, 03, ...). All plans are executed strictly sequentially -- no parallel execution. This prevents nasty bugs from out-of-order execution and ensures clean linear progression. If dependency analysis reveals that a later-numbered plan must run before an earlier one, renumber before finalizing. Never claim plans can run in parallel. Additionally, define **named implementation phases** grouping plans into atomic bundles with verification gates at each boundary (e.g. "Phase A (plans 01-03): foundation -- checkpoint after", "Phase B (plans 04-06): core modules -- checkpoint after"). This gives checkpoint audits concrete scope and helps agents understand the logical progression.
8. **Implementation Notes**: naming conventions, DRY rules, testing requirements
9. **Verification Protocol**: two tiers -- per sub-plan (ruff check, ruff format, ty check, ast-grep scan, pytest, imports, docstrings match plan) and final (full quality gate per CLAUDE.md ## Verification, real examples, `/tmp` full-stack test, CLAUDE.md imports, grep stale names from kill list)
10. **Banned Patterns**: CODE patterns from CLAUDE.md are enforced deterministically by ast-grep + ruff at implementation time (see CLAUDE.md ## Verification). PLAN patterns listed below still require AI/`rg` checking on plan markdown files. Each with WHY. Grows during verification. Required plan-level bans include:
    - "see POC for details" without inlining -- POC is ephemeral; critical patterns must be inlined in the master plan
    - Sub-plan files referencing `/tmp` paths, POC source files, or agent transcript logs -- sub-plans must be self-contained
    - "test X works" without specific assertions -- tests must verify specific values/types/states
    - Deferring tests to a later plan -- each plan must include its own comprehensive tests
    - Cross-referencing sibling sub-plans for required context -- each sub-plan must be self-contained
    - Backward compatibility shims -- clean break, no aliases, no fallbacks
    - POC-only API patterns in sub-plan code samples -- use production APIs only
11. **Final Verification Plan**: dedicated plan that audits the entire redesign
12. **Transcript Reference**: full path to agent transcript JSONL (main + subagent transcripts if applicable). "Launch a subagent to search/read sections if any decision needs clarification." Update on context compaction. Note all transcript segments (discovery rounds, POC phases, audit results). **Sub-plan prohibition:** sub-plan files must NOT reference transcript paths, `/tmp` POC paths, or agent logs. All critical patterns validated during planning must be inlined in the master plan's "Critical POC Code Samples" section (item 18). Sub-plans copy from there. Transcripts are for master-plan-level authoring context only.
13. **Implementation Execution Protocol**: the master plan MUST include a detailed execution protocol section covering all five subsections below (A-E). This is critical because once implementation starts, all plans execute sequentially with full autonomy -- there is no turning back.

    **A. Per-plan verification:** After each plan is implemented, the implementing agent MUST run:
    - `uv run ruff check src/ tests/`
    - `uv run ruff format --check src/ tests/`
    - `uv run ty check src/`
    - `uv run ast-grep scan --config sgconfig.yml src/ tests/`
    - `uv run pytest -x -v`
    - Import verification: `uv run python -c "from autopilot.x import Y"` for all modules touched
    - Docstring check: verify docstrings reflect new behavior, not old
    - Reference verify-plan Pass 6b-B for AI-judged patterns; tool-enforced patterns are caught by ast-grep scan above
    - If ANY check fails: first check whether a later sub-plan is explicitly planned to address it (read the remaining plan files). If yes, note it as "expected -- deferred to sub-plan NN" and move on. If no, fix the issue immediately -- the implementing agent owns the fix.
    - Never proceed to the next plan until all non-deferred checks pass (see self-healing protocol in section C for deferral rules).

    **B. Checkpoint audits (~every 3-4 plans):** Define specific checkpoint gates (Checkpoint A after plans 01-04, Checkpoint B after plans 05-08, etc.) covering every plan. Each checkpoint runs a comprehensive codebase-wide verification:
    - Full quality gate (ruff check, ruff format, ty check, ast-grep scan, pytest) -- not just files touched by recent plans
    - Import verification for all public modules up to this point
    - Remaining `rg` checks per CLAUDE.md ## Verification (graph.py isolation, no `__init__.py` files)
    - Specific behavioral verification commands (e.g. "verify epochs are 0-based", "verify Store uses snapshot/restore only")
    - Example runs if applicable
    - CLAUDE.md / README consistency check
    - These are mandatory gates -- do NOT proceed to the next batch until the checkpoint passes.

    **C. Self-healing protocol:** If a checkpoint or per-plan verification finds ANY issue:
    1. Diagnose the root cause (which plan introduced it, what went wrong)
    2. Check whether a subsequent plan already addresses this issue. Read remaining sub-plan files to see if the issue is explicitly planned for a later plan. Examples: epoch values may be 1-based until a later plan flips them; private methods may remain until a later plan renames them; `str = ''` defaults may persist until a later plan cleans them up.
    3. If the issue IS planned for a later sub-plan: note it as "expected -- deferred to sub-plan NN" and move on. Do NOT fix it or duplicate the effort.
    4. If the issue is NOT planned for any later sub-plan (genuine regression or gap): fix the issue directly in the source code.
    5. If the fix reveals a gap in the sub-plan that caused the issue, update that sub-plan file to document what was actually done (so the plan reflects reality).
    6. If the fix affects the master plan (e.g. a new bug discovered, a dependency change), update the master plan.
    7. Re-run the full checkpoint verification to confirm the fix (excluding known-deferred items).
    8. Only proceed to the next batch once all checks pass (with deferred items explicitly noted).
    - The checkpoint agent has full read/write access to both source code and plan files. Genuine regressions must be fixed in place; planned future work must not be duplicated.

    **D. Self-containment:** Each plan must be implementable by a separate agent with only this plan + master plan + codebase. No cross-references that require reading other sub-plan files.

    **E. Final plan special case:** The last plan in the sequence is audit-and-fix, not audit-only. Since there are no subsequent plans to defer to, every issue found must be resolved. The agent must diagnose, fix source code, update relevant plan files, and re-run verification until all checks pass.
14. **Future Work**: noted but not implemented
15. **Additive-First Deletion Policy**: new code lands and is proven before old code is deleted. Green tests at EVERY intermediate step. Coordinated deletion in a dedicated late plan.
16. **create-master-plan skill note**: so this process can be reused
17. **Pre-Plan POC Verification** (if POCs were run during planning): document results as a structured appendix. Include: test phases (what each cluster validates), per-test bullet descriptions, bugs discovered during POC, all-pass confirmation with counts. This makes the POC audit trail part of the spec, not ephemeral `/tmp` artifacts that disappear.
18. **Critical POC Code Samples**: inline the validated implementation patterns from POC that sub-plans must follow. Annotate bugs inline. Include a "POC vs production API differences" mapping if the POC used simplified APIs (e.g. POC `Parameter(name=...)` vs production `Parameter` with no name field). Sub-plan files copy what they need from here; they must NOT reference `/tmp` paths or POC source files directly. POC code must be written in the target repo's style (indentation, quotes, naming, design patterns, prohibited patterns) so that samples are directly liftable into implementation without style translation.
19. **Bug Registry**: table mapping each BUG-NNN to description and owning sub-plan. If POCs found design bugs, document them as a chronicle (what failed, root cause, fix applied to the design) so implementers understand why the design looks the way it does.
20. **Resolved Ambiguities**: Q&A format for design decisions that were debated or could cause implementer divergence. Each entry: question, answer, rationale. Pin non-obvious choices so separate agents don't re-open settled debates.
21. **Risk Mitigations**: numbered risks with concrete mitigations. Focus on the 3-5 riskiest plans and what guards are in place (POC validation, checkpoint gates, specific tests, fallback approaches).
22. **Reference Framework Alignment** (if applicable): document what aligns with reference frameworks (PyTorch, Lightning, torchmetrics, git), what intentionally diverges, and why. Validated via subagent exploration of reference codebases.
23. **File Locations**: standalone map of every new/modified file with purpose annotation (NEW/MODIFY). Separate from Class Hierarchy (which focuses on relationships between classes, not file layout).

### Phase 2: Coherence Verification (Handoff to verify-plan)

After writing the master plan and sub-plan files, follow the **verify-plan** skill to prove the plan is implementation-ready through multi-pass auditing and execution simulation.

The verify-plan skill provides structured audit passes (structural coherence, ambiguity elimination, three-source verification, DRY, simulation, readiness gates) and a fix-plan generation loop. The heavy verification machinery lives there, not here.

**Scope expansion:** Verification or later discovery rounds may reveal work that requires adding or splitting sub-plans. This is expected and normal. New plans must be numbered to maintain strict sequential order (renumber existing plans if needed) and the full verify-plan process re-run on the expanded set.

### Phase 3: Sub-Plan Files

Create individual sub-plan files in the target directory (e.g., `.cursor/plans/<feature>/`). Each uses the create-plan skill format. Symlink the master plan to the same directory as `00-master-plan.md`.

**Important**: This skill ONLY generates plan files. Never implement code, never modify source files, never make codebase changes. The only code execution allowed is `uv run python -c "..."` to verify assumptions. Implementation is a completely separate phase initiated by the user -- do not start it unless explicitly asked.

## Plan file location

Master plan: `.cursor/plans/<feature>/00-master-plan.md`
Sub-plans: `.cursor/plans/<feature>/01-name.md` through `NN-name.md`

## Key principles (non-negotiable)

- Every attribute must be justified (why it exists, why it's on THIS class)
- Every design decision from the discussion must be captured in the plan
- Comprehensive tests per plan — not an afterthought
- The implementing agent runs per-plan verification (quality gate per CLAUDE.md ## Verification: ruff check, ruff format, ty check, ast-grep scan, pytest, imports) directly after each plan. A separate verification subagent is launched at checkpoint boundaries (~every 3-4 plans) for codebase-wide audits.
- Final plan is always a full verification against the master plan
- No git operations — the plan creator only creates plan files, never implements
- No implementation during planning — plans and implementation are distinct phases. Do not implement unless the user explicitly asks you to.
