---
name: verify-plan
description: Pre-implementation verification of master plans and standalone plans. Multi-pass auditing (coherence, ambiguity, DRY, simulation, readiness), fix plan generation, and readiness gates. Use when verifying a plan is implementation-ready, running audits, or checking plan coherence.
---

## Philosophy

- **Plans are specs consumed by autonomous agents.** Any ambiguity becomes an inconsistency when separate agents implement each plan.
- **Audits are iterative.** Fix after each pass, re-audit to confirm. A single pass is never enough.
- **Simulation catches what reading misses.** Plans coherent in isolation can fail when executed sequentially -- API deletions, rename propagation, migration ordering.
- **Binary readiness.** A plan is READY FOR IMPLEMENTATION or NOT READY. No "probably fine."
- **Fix plans, not code.** When audits find issues, fix the PLAN FILES. Code comes later.
- **Testing thoroughness determines autonomous success.** Implementation follows the agentic loop: implement -> test -> verify -> fix -> repeat. If plans specify weak tests, this loop never converges. Audit Pass 7 (Testing thoroughness) must verify that plans specify tests covering logic, behavior, integration, and edge cases -- not just surface-level smoke tests.
- **No backward compatibility.** Plans must enforce a clean break unless the user explicitly requested otherwise. No aliases, no deprecation shims, no fallback imports, no bridge code, no `getattr` with fallback, no legacy support. Audit passes must actively grep for these patterns and flag them as violations.
- **This skill audits plan files, not code.** It spot-checks plans against existing source for accuracy, but does NOT implement or test new code. That separates it from audit-implementation.
- **Use subagents generously.** Every audit pass, every simulation step, every coherence check should be delegated to subagents. This keeps the context window clean, allows thorough investigation without truncation, and ensures each audit gets a fresh, focused lens. Launch subagents for each pass -- do not try to do multiple passes in a single context. When in doubt, use more subagents rather than fewer.

## When to use

After a master plan and sub-plans have been written (via create-master-plan). When you need to prove a plan is implementation-ready. When the user asks for audits, coherence checks, simulation, or readiness verification.

Also works for standalone plans (not part of a master plan) with reduced scope.

## Multi-pass audit system

Twelve structured passes (1-6, 6b, 7-11), each with a specific rubric. After each pass: fix issues found, then re-audit to confirm fixes landed before moving to the next pass.

### Pass 1 -- Structural coherence audit

One subagent reads ALL plan files with this 11-point rubric:

1. **Issue/bug ID consistency** -- IDs use format `BUG-NNN`, every issue owned by exactly one sub-plan, no orphans, no duplicate IDs across plans
2. **Dependency graph correctness** -- no cycles, dependencies real, strict sequential ordering (no parallel execution claims), prerequisite chains verified, numeric order matches execution order
3. **File path accuracy** -- referenced files exist or clearly marked "new file to create"
4. **Fix descriptions vs actual code** -- spot-check that plan descriptions match actual source (line numbers, method names, behaviors)
5. **Test coverage completeness** -- tests cover all claimed fixes, no obvious missing scenarios
6. **Cross-plan consistency** -- if plan A fixes something plan B depends on, dependency noted? Contradictory changes?
7. **Missing coverage** -- anything in issue list NOT covered by any sub-plan?
8. **Naming and organization** -- filenames match content, numbering sequential and logical
9. **Rename propagation** -- every renamed symbol uses the new name consistently across ALL plans. Grep kill-list names against all plan text; any stale name from the kill list still appearing in plan instructions or code samples is a violation.
10. **Verification sections** -- every sub-plan ends with correct verification commands (see CLAUDE.md ## Verification for the canonical command set)
11. **Docs sections** -- every sub-plan has a Docs line, docstrings aligned with code changes

Tone: "Be ruthless -- if something is vague, contradictory, or missing, call it out."

### Pass 2 -- Ambiguity audit (de-vaguification)

Subagent(s) read all plans hunting for 10 patterns:

1. "Consider" / "Option A or B" / "either...or" without a decision
2. Design "if" without resolution
3. Vague verbs: might, could, should consider, worth exploring
4. Missing specifics: "update callers" without listing them
5. "Document" / "add docstring" without exact text
6. Proposed changes with unresolved "or"
7. Vague tests: "edge cases" without specifying which
8. Missing full method signatures for new methods
9. "Follow existing patterns" without citing which file
10. Open behavioral decisions: what happens on error? default?

Quote the EXACT ambiguous text. Propose the resolution. Goal is BINARY: the final sweep must return zero ambiguities. In practice this takes 3 sweeps (initial, final, absolute-final with IGNORE rules for false positives).

**IGNORE rules** (known false positives to skip):
- `or` inside Python code snippets (e.g. `prefix or ''`, `value or default`)
- `should` in test descriptions (e.g. "test should raise ValueError")
- `may` in docstrings describing optional behavior (e.g. "caller may override this hook")
- `consider` in comments explaining WHY a decision was made (e.g. "we considered X but chose Y")
- Conditional language in Future Work sections (explicitly deferred, not ambiguous)
- `either` in error messages describing user choices (e.g. "pass either a path or a stream")

### Pass 3 -- Three-source verification

When plans evolved from earlier work, compare three sources:
1. Old pending todos / open questions from prior sessions
2. Current plan files as written
3. Transcript design decisions from the discussion

Per plan file, produce: what it says, what each source requires, GAP (in source but not plan), CONFLICT (sources disagree). This prevents requirements from being dropped as designs evolve.

Skip for greenfield plans with no prior work.

### Pass 4 -- Unfinalized-decisions audit

Hunt for: TBD, may, might, optional without resolution, missing signatures/types, missing error handling/edge cases, cross-plan contradictions, deferred "evaluate/rethink" items, implementer judgment calls.

Produce a numbered decision inventory (1, 2, ... N). Each entry: file, quoted text, decision needed, suggested resolution. Fold into the master plan or a fix plan.

### Pass 5 -- DRY audit

Two dimensions:
- **Plan prose:** same decisions/constraints duplicated across plan files? Identify the canonical owner.
- **Code-level:** would implementing plans as written create NEW duplicate logic? (Helper on wrong class, reimplemented serialization when DictMixin exists, etc.)

Audit both the plan TEXT and what the plan would PRODUCE in code.

### Pass 6 -- Skill compliance + alignment audit

- Plans match create-plan, create-master-plan, and create-meta-plan skill requirements (if a meta plan was used as input, verify that its decisions are captured in the plan files)
- Philosophy alignment with reference frameworks (PyTorch, Lightning, torchmetrics, git) if applicable
- Every sub-plan has verification commands and a Docs line

### Pass 6b -- Banned patterns + code quality audit

Each pattern below must be explicitly checked -- not spot-checked, not "generally scanned."

#### Pass 6b-A: Tool-Enforced Patterns (reference only)

These patterns are checked by `ast-grep scan` and `ruff check` at implementation time. During plan verification, the auditor must confirm that:
- The plan's verification section includes the full quality gate (see CLAUDE.md ## Verification)
- Proposed code samples in the plan don't visibly contain these patterns

For tool-enforced patterns, "explicit check" means confirming the plan's gate INCLUDES ast-grep/ruff; the tools themselves do the exhaustive checking at implementation time.

Patterns now tool-enforced (21 ast-grep rules + ruff):
- `getattr(args, 'x', default)` -- `no-getattr-on-args`
- `str = ''` defaults -- `no-empty-string-default`
- `.get('key', '')` -- `no-get-empty-string`
- `except Exception: return {}` -- `no-except-return-empty-dict`
- `# noqa` -- `no-noqa-comments`
- `trainer._store` -- `no-trainer-private-store`
- `getattr(experiment, 'store'/'rollback')` -- `no-getattr-base-attrs`
- `isinstance` on concrete leaves -- `no-isinstance-concrete-leaf`
- Registry/decorator patterns -- `no-registry-pattern`
- Phase/mode in forward() -- `no-phase-string-in-forward`
- `if TYPE_CHECKING:` -- `no-type-checking-block`
- Conditional/fallback imports -- `no-conditional-import`
- Deferred/inner imports -- `no-deferred-import`
- Dynamic imports -- `no-dynamic-import`
- `param.grad =` (non-None) -- `no-param-grad-assign`
- `SequenceModule`/`ParallelModule` -- `no-banned-class-names`
- `input()` calls -- `no-input-call`
- `pytest.raises(Exception)` -- `no-broad-pytest-raises`
- `@staticmethod` -- `no-staticmethod`
- Multi-for comprehensions -- `no-multi-for-comprehension`
- `# type:` comments -- `no-type-comment`
- `os.environ`/`os.getenv`/`load_dotenv` -- ruff TID251 banned-api
- `except Exception: pass` -- ruff S110
- Relative imports -- ruff TID
- Import ordering -- ruff I001/isort
- 2-space indentation -- ruff format
- Single quotes -- ruff format + Q rules
- Line length 100 -- ruff E501 + format

#### Pass 6b-B: AI-Judged Patterns (must still be explicitly checked)

These require human/AI interpretation and CANNOT be automated:

**Clean-break enforcement:**
- Grep plan text for: backward, compat, bridge, legacy, deprecat, shim, fallback, alias
- No bridge/adapter code preserving old APIs (intent-based)

**Semantic code quality:**
- No fake fallback objects on precondition failure
- No inline file-content strings for templates
- No module-level constants caching function calls
- No private extension methods that should be public hooks
- No config-file-driven workflows
- Store only uses snapshot()/restore()
- Naming follows project conventions (no magic strings)
- `__init__.py` absence (`find src/ -name '__init__.py'`)
- graph.py isolation (`rg 'from autopilot' src/autopilot/core/graph.py`)

Per pattern: CLEAN or VIOLATION (with sub-plan ID and exact text).

### Pass 7 -- Testing thoroughness audit

Since implementation is fully autonomous, testing is the primary verification mechanism. Surface-level tests that only check "it doesn't crash" are insufficient. Per class/module that the plans touch, assess: COMPREHENSIVE or GAPS.

- Normal paths, edge cases, error conditions all tested?
- **Logic and behavior verified**, not just "runs without error"?
- Boundary conditions covered (empty, single, max, zero, None)?
- Error paths test specific exception types and messages?
- Round-trip tests (`to_dict` -> `from_dict`) for all dataclasses?
- Integration tests for cross-boundary flows (e.g. Trainer -> Store -> Tree)?
- Full lifecycle tests (create -> use -> modify -> serialize -> deserialize)?
- State transition tests (objects move through expected states)?
- Subclass/override scenarios tested?
- Are test descriptions specific enough to write without guessing?
- Are edge cases enumerated and tested explicitly, not left to "add edge case tests"?
- **Test performance:** do any planned tests invoke `subprocess.run` for validation achievable in-process (e.g. `uv run python -c` for import checks, `uv run pytest` recursively, `shutil.copytree` + subprocess for example scripts)? Flag as GAPS.

### Pass 8 -- Extensibility / design quality audit

Per class, score: GOOD / CONCERN / BLOCKER.
- Extension points: can users subclass and override hooks?
- Protocol compliance: does the class follow the framework's protocols?
- Banned pattern compliance: no prohibited patterns in proposed code?
- Professional quality: naming, organization, documentation?
- Flexibility: not over-rigid, not over-generic?

### Pass 9 -- Existing-codebase DRY audit

Distinct from Pass 5. This audits the CURRENT codebase for DRY violations that the plans should address. If the plans are meant to clean up the codebase, verify they catch all existing violations.

### Pass 10 -- Thematic spot-check

Fast boolean-themed sanity pass after bulk edits. Pick 3-5 critical themes and check PASS/FAIL across all plans. Examples:
- "paths.py is only referenced as delete/replace, never as keep"
- "Config used consistently, never old ProjectConfig"
- "Branching precondition present in all tree-related plans"

Useful as a quick verification after a round of StrReplace edits, before running heavier passes again.

### Pass 11 -- Transcript decision verification

**When to run:** If the plan files reference transcript files (in a "Transcript Reference" section, meta plan, or anywhere else), this pass is mandatory. If no transcripts are referenced, skip.

**Purpose:** Verify that every design decision, constraint, and requirement discussed in the transcripts is faithfully captured in the plan files. Transcripts are the raw record of what was agreed; plans are the frozen spec. This pass ensures nothing was lost in translation.

**Procedure:**

1. **Locate transcripts.** Read the master plan's Transcript Reference section (or the meta plan's). Collect all referenced transcript JSONL paths (main + subagent transcripts).
2. **Mine decisions.** Launch subagent(s) to search each transcript for design decisions. Target patterns:
   - Explicit user directives ("do X", "don't do Y", "make sure Z")
   - Agreed-upon constraints ("we decided", "let's go with", "confirmed")
   - Rejected alternatives ("not X because", "we ruled out")
   - Behavioral specifications ("when X happens, Y should", "the default is")
   - Naming decisions ("call it X", "rename to Y")
   - Scope inclusions/exclusions ("include X", "defer Y", "out of scope")
   - Testing requirements ("test that", "must verify", "edge case: X")
3. **Build a decision ledger.** Each entry: transcript quote (abbreviated), decision summary, which plan file should own it.
4. **Cross-reference against plans.** For each decision in the ledger, verify it appears in the appropriate plan file. Produce:
   - **CAPTURED** -- decision is in the plan (quote the plan text)
   - **MISSING** -- decision is in transcript but not in any plan file (this is a gap)
   - **CONTRADICTED** -- plan says something different from what was decided (this is a conflict)
5. **Report.** Per decision: CAPTURED / MISSING / CONTRADICTED with evidence from both transcript and plan. Any MISSING or CONTRADICTED item is a finding that must be fixed.

**Subagent strategy:** For large transcripts, split by transcript segment (discovery rounds, POC phases, design discussion, audit results) and launch parallel subagents. Each subagent mines its segment and returns a partial decision ledger. Merge before cross-referencing.

**Relationship to other passes:**
- Pass 3 (Three-source verification) compares old todos, plans, and transcript decisions at a high level. Pass 11 is the exhaustive, line-by-line transcript mining.
- Step 5 (Meta-audit in execution simulation) is a lighter-weight version focused on catching obvious gaps. Pass 11 is the systematic, complete verification.

### After all passes -- Final coherence check

Launch a subagent for a final consolidated coherence check across ALL plan files. This catches issues introduced by fixes from later passes breaking earlier ones. Specifically:
- Re-verify the master plan's dependency graph against all sub-plan dependencies
- Re-verify Pass 6b-A tool gate is referenced in each sub-plan's verification section; re-verify Pass 6b-B AI-judged patterns against all sub-plan content
- Re-verify naming consistency across all plans (no plan using old names that another plan renamed)
- Cross-reference master plan class hierarchy against sub-plan detailed designs
- Verify every decision from the discussion is captured (not just those that survived edits)
- Confirm no stale references to old/removed concepts remain

This is the coherence gate: if it fails, go back and fix before declaring readiness.

## Report templates

**A-E coherence template:** Optional structured output for coherence reports:
- A. Decision coverage: every design decision from discussion captured
- B. Skill compliance: plans match create-plan + create-master-plan skills
- C. Internal consistency: no contradictions within/across plans
- D. Completeness: all requirements addressed
- E. Overall: PASS / FAIL with specific issues

**Spec-diff A-K template:** When verifying that an update spec was applied correctly, use per-bullet checks (A through K or however many) against the actual plan files. Each check: PASS / FAIL with quoted evidence.

## Execution simulation

Plans coherent in isolation can fail when executed sequentially. Simulation catches ordering and migration bugs.

**Step 1 -- Verify execution order:** All plans execute strictly sequentially in numeric order (01, 02, 03, ...). No parallel execution. Verify that plan numbering matches the intended execution order and that no plan depends on a higher-numbered plan.

**Step 2 -- Sequential simulation:** Walk plans in strict numeric order. For each plan, assume all prior plans have been fully implemented. Evaluate from the IMPLEMENTOR perspective:
1. Full context? Implementable without asking questions?
2. File paths correct, referenced files exist?
3. API names/signatures accurate against cumulative state?
4. Docs/examples/CLAUDE.md updated to track cumulative state?
5. Verification steps present and correct?

Per plan: READY / NEEDS FIX (with specifics). Report cross-plan issues and cumulative state after each plan.

**Step 3 -- Sequential boundary checking:** Walk execution order plan by plan. At each boundary:
- **GREEN:** tests pass, imports resolve, plan N+1 assumptions match plan N outputs
- **YELLOW:** minor issues, easily fixable, not blocking
- **RED:** major breakage, plan revision required

Any RED boundary means the plan is NOT implementation-ready.

**Step 4 -- Readiness audit:** 7 categories:
1. Sequential execution integrity
2. Self-containment (per-plan; BLOCKING if cross-refs require other sub-plan files)
3. Implementation Execution Protocol completeness -- verify the master plan defines ALL of:
   - Per-plan verification steps (ruff check, ruff format, ty check, ast-grep scan, pytest, imports, docstring check)
   - Per-plan failure handling ("check if later plan owns it, defer or fix")
   - Checkpoint audits (~every 3-4 plans) with specific gates covering every plan
   - Each checkpoint has concrete verification commands (not just "run pytest")
   - Self-healing protocol (diagnose, check later plans, defer-or-fix, update plan files, re-run)
   - Final plan marked as audit-and-fix with "no subsequent plans to defer to"
   - Self-containment (each plan + master plan + codebase is sufficient)
   - If ANY of these are missing or vague: BLOCKING
4. Source code accuracy (spot-check plan claims vs actual source)
5. Completeness (all issues addressed, all sections present, kill list complete)
6. Prohibited patterns (grep plans for banned patterns; grep sub-plans for `/tmp` paths, transcript references, POC-only APIs)
7. Risk assessment (top 3 riskiest plans + mitigation strategies)
8. POC documentation (if POCs were run): Pre-Plan POC Verification section present with test counts, Critical POC Code Samples inlined, Bug Registry mapping IDs to sub-plans, no "see POC" without inlining
9. Resolved Ambiguities present for non-obvious design decisions (prevents implementer divergence)

Verdict: **READY FOR IMPLEMENTATION** or **NOT READY -- N blocking issues.**

**Step 5 -- Meta-audit (transcript mining):** Subagent mines the transcript for user requirements or decisions NOT captured in plans. For each finding: quote the transcript, identify the target plan, state whether captured or not, suggest the addition. Closes the loop between the discovery session and the frozen spec.

## Fix plan generation loop

When audits or simulation find issues:
1. Produce a consolidated fix plan with (plan file | issue | concrete fix) tables
2. Apply fixes to plan files
3. Re-run the specific audit pass that found the issue
4. If new issues emerge, iterate
5. After all fixes, run full simulation to confirm no RED boundaries remain

Fix plans may use numbered decisions (A1-AN) organized as: Part A (decisions), Part B (per-file fixes), Part C (testing gaps), Part D (autonomy/testing protocol).

## Fix proposal workflow

**Direct apply (default):** Small fixes (typos, missing Docs lines, stale references) applied directly via StrReplace, then re-audited.

**Propose-then-apply (on user request or for structural changes):** For larger changes:
1. Produce a fix plan document listing all proposed changes
2. Present to user for review
3. User approves, modifies, or rejects individual fixes
4. Apply approved fixes
5. Re-audit

## Meta fix plans

When audits produce a substantial set of fixes, the fix artifact becomes a tracked plan file (e.g. "Fix experiment plans v2"). Treat these as first-class specs:
- Spec-diff style checks apply to "changes per file" sections
- Numbered decisions (A1-AN) in the fix plan are auditable
- After applying fixes, verify the fix plan is fully resolved (all decisions applied, all PASS)

## Renumbering protocol

If execution order differs from numeric order and renumbering is desired:
1. Produce old-to-new mapping table
2. Rename all plan files
3. Update ALL cross-references in ALL plan files + master plan
4. Dedicated verification subagent: read all files, check for stale old numbers, verify titles match filenames, verify dependency claims use new numbers, verify semantic references are still correct

## Standalone plan verification

For individual plans (not part of a master plan), use reduced scope:
- Skip three-source verification (no prior work to reconcile)
- Skip execution simulation (single plan has no sequential boundaries)
- Skip renumbering (single file)
- Keep: ambiguity audit, DRY, implementor-POV, structural checks within subplans, testing thoroughness
- Verification sections should reference CLAUDE.md ## Verification for the canonical code quality gate
- Key question: "Could a separate agent implement this without asking questions?"

## Gotchas

- A single audit pass is NEVER enough. Each pass surfaces issues that subsequent passes must re-check.
- Ambiguity false positives: `or` inside code/docstrings, `prefix or ''` in Python expressions. Add IGNORE rules for these.
- Three-source verification only applies when plans evolved from earlier work. Skip for greenfield.
- Simulation uses strict numeric order (01, 02, 03, ...). If numbering doesn't match execution order, the plan must be renumbered first.
- Self-containment is strict: "see sub-plan X for the signature" without inlining is BLOCKING.
- DRY at code level requires understanding what the plan WOULD produce, not just what it says.
