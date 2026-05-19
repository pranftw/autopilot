---
name: create-meta-plan
description: Meta-planning skill for design exploration, POC verification, and iterative design refinement. Produces a meta plan document that feeds into create-master-plan (for multi-plan systems) or create-plan (for individual plans). Use when you want to think through a design before generating plan files, when POC validation is needed, or when the user asks to plan out a feature without generating plan files yet.
---

## What this skill produces

This skill produces a **meta plan** -- a single evolving document that captures design exploration, POC results, code samples, resolved ambiguities, and everything needed to eventually generate plan files. The meta plan is the "thinking" artifact; plan files are the "doing" artifacts.

**A meta plan is NOT a plan file.** It does not get implemented directly. Instead, it feeds into:
- **create-master-plan** -- to generate a master plan system (`00-master-plan.md` + `01-NN.md`) for multi-plan redesigns
- **create-plan** -- to generate an individual plan file for focused features

**Two target modes:**
- **Meta plan for a master plan system:** design a full subsystem redesign with multiple interdependent plans. The meta plan captures class hierarchies, detailed design, POC validation, and all decisions. It is then used as input to create-master-plan.
- **Meta plan for an individual plan:** design a focused feature or module with enough depth that the plan file will be comprehensive. The meta plan captures the design iteration, then feeds into create-plan.

**Your job is to design and verify, not generate plan files.** This skill explores, discusses, runs POCs, and iterates. It does NOT create the final plan directory or individual plan files. That happens later via create-master-plan or create-plan, only when the user is ready.

**Ask the user for the target directory** for the meta plan if not specified. The meta plan itself is typically a single `.plan.md` file (e.g. `Autograd Redesign.plan.md`).

## Philosophy

- **Explore before planning.** Discovery (subagents, code execution, dogfooding) must happen BEFORE any plan is written. The first goal is full exploration, THEN the meta plan.
- **Epistemic humility.** Never assume coverage is complete. Challenge yourself: "how are you sure you have explored all possibilities and all edge cases?" If you cannot justify coverage, keep exploring.
- **Iteration over speed.** Minimum 3 discovery rounds with fresh lenses. Rushing to generate the plan is an anti-pattern: "dont be in a hurry to generate the plan."
- **Nothing left to the implementor.** Every design decision must be pinned in the meta plan. Ambiguity in a meta plan becomes inconsistency when plan files are generated from it.
- **Testing as highest priority.** Implementation is fully autonomous -- the agentic loop is: implement, test, verify, fix, repeat. Testing and verification must be thorough enough to drive this loop without human intervention. The meta plan must specify tests that cover logic, behavior, integration, and edge cases -- not just surface-level smoke tests. Every feature must be designed with edge cases in mind from the start.
- **Use subagents generously.** Delegate exploration, auditing, verification, and testing to subagents. This keeps the context window clean and allows thorough parallel investigation. Skipping subagents when they are warranted is a process failure. When in doubt, launch more subagents rather than fewer.
- **No backward compatibility.** All designs must enforce a clean break unless the user explicitly requests otherwise. No aliases, no deprecation shims, no fallback imports, no bridge code, no `getattr` with fallback, no legacy support patterns. This is non-negotiable.
- **POCs are outside the meta plan's task list.** POCs run separately (in `/tmp`, via `uv run python -c`, etc.) and their results drive updates to the meta plan. The meta plan records POC outcomes, not POC execution steps.

## When to use

- When the user wants to **think through a design** before committing to plan files
- When **POC validation** is needed before the design can be locked
- When the user says things like "let's plan this out", "let's design this", "I want to think through this first"
- When the scope is large enough that jumping straight to plan files would skip important design exploration
- When novel patterns need validation before the design can be frozen

**When to skip:** For simple/obvious features where the design is clear, go directly to create-master-plan or create-plan. Not every feature needs a meta plan.

**When updating an existing meta plan or plan set:** Read existing plan files (`.cursor/plans/<feature>/`) first as the primary source of truth. Use transcripts only when plans are incomplete, for traceability, or after compaction. Written plans in the repo are canonical; transcripts are secondary recovery material.

## Process

### Phase 1: Deep Exploration (Multi-Round Discovery)

Run at least 3 rounds of parallel subagent discovery, each with a fresh lens.

**Round 1 -- Layer-scoped discovery:** Parallel subagents by subsystem layer. Each reads every file in its scope, catalogs issues, behaviors, and interfaces.

Example scopes: data pipeline, policy, module/parameter/gradient, trainer/callbacks/loops, AI modules, tracking/IO/serialization, CLI infrastructure, remaining CLI commands.

**Round 2 -- Cross-cutting discovery:** Parallel subagents by cross-cutting concern.

Example scopes: import cycles, orchestrator/graph integration, concurrency/parallelism, end-to-end dogfood with real examples, quality gate compliance (see CLAUDE.md ## Verification).

**Round 3+ -- Execute-to-verify:** Shift from reading source to EXECUTING code. Subagents run `uv run python -c "..."`, `uv run pytest`, and exercise real code paths. This catches issues that reading alone theorizes but cannot confirm.

This is discovery on the EXISTING codebase to inform design. It is NOT audit-implementation (which tests newly implemented code after a plan was executed).

**Additional rounds as needed.** If scope covers the full codebase, expect 4+ rounds and 20-30+ subagent runs.

**Post-draft discovery extension:** If the meta plan has been drafted but the user challenges completeness, execute-to-verify rounds can continue. Findings get patched back into the meta plan.

**Batching:** Synthesize findings after each round before proceeding. Rounds build on each other -- Round 1 findings inform Round 2 scope.

**Coverage justification:** Before proceeding to Phase 2, explicitly justify: what subsystems were covered, what angles were used, why discovery is complete. If challenged, go back and run more rounds.

### Phase 2: Interactive Discussion

Discuss the design with the user BEFORE writing the meta plan. Cover:
- What's wrong with the current system (concrete diagnosis, not vague complaints)
- Core classes and their responsibilities (with attribute justification)
- Code samples showing how everything composes (PyTorch-style, Lightning-style)
- What gets killed/moved/created

Key principles to apply:
- **Database normalization**: no dual state, single source of truth, derive don't store
- **Real Python objects**: not string keys or FK wrappers. In-memory = objects, serialization handles ID conversion
- **Extensibility via subclassing**: no metadata dict catch-all fields
- **Agent decides, framework tracks**: framework provides raw data, agent interprets significance
- **OOP enums**: proper Python enums, lowercase members, no magic strings
- **Centralized paths**: one module for all path computation
- **Clean break**: no backward compatibility. No aliases, no deprecation shims, no fallback imports, no bridge code, no legacy support. Every rename, removal, and API change is a clean break with all consumers updated in the same plan. Only deviate if the user explicitly requests backward compatibility.

**Scope negotiation:** Confirm whether the meta plan targets a full master plan system (15-20 sub-plans) or a single focused plan. The user may want broader scope than you assume.

**Reference codebase exploration:** If the project models after reference frameworks (PyTorch, Lightning, torchmetrics, git), launch subagents to explore those codebases for high-level design, philosophy, and UX alignment. Not line-by-line reads, but enough to verify alignment claims.

**Anti-hurry signals:** If the user says "don't hurry", challenges coverage, or asks "how are you sure?", that is a signal to go back to Phase 1 with more rounds, not to proceed with disclaimers.

Iterate until the user confirms the direction. Do NOT proceed to the meta plan until design is agreed.

### Phase 3: Write the Meta Plan

Create or update the meta plan document. This is a single file that evolves iteratively. Include these sections (adapt based on scope -- not all sections needed for simpler plans):

1. **Context**: what's wrong, what we're building, why it's a clean break
2. **Design Principles**: every key decision from the discussion, with justification
3. **Class Hierarchy**: core classes and builtin classes with file locations and purpose
4. **Detailed Class Design**: attributes with justification, method signatures, code samples
5. **File Locations**: standalone map of every new/modified file with purpose annotation (NEW/MODIFY)
6. **Kill List**: what gets deleted/moved with reason
7. **Sub-Plans Overview** (for master plan targets): one per logical unit, dependency-ordered, with scope summary
8. **Pre-Plan POC Verification**: document POC results as a structured appendix. Include: test phases (what each cluster validates), per-test bullet descriptions, bugs discovered during POC, all-pass confirmation with counts.
9. **Critical POC Code Samples**: inline the validated implementation patterns from POC. Annotate bugs inline. Include a "POC vs production API differences" mapping if the POC used simplified APIs.
10. **Bug Registry**: table mapping each BUG-NNN to description and (for master plan targets) owning sub-plan. Document bugs found during POC as a chronicle (what failed, root cause, fix applied to the design).
11. **Resolved Ambiguities**: Q&A format for design decisions that were debated. Each entry: question, answer, rationale. Pin non-obvious choices.
12. **Risk Mitigations**: numbered risks with concrete mitigations. Focus on the 3-5 riskiest areas.
13. **Reference Framework Alignment** (if applicable): what aligns with reference frameworks, what intentionally diverges, and why.
14. **Transcript Reference**: full path to agent transcript JSONL (main + subagent transcripts). "Launch a subagent to search/read sections if any decision needs clarification." Note all transcript segments.

### Phase 4: Proof-of-Concept Verification

Before finalizing, verify ALL assumptions through `uv run python -c "..."`. If you have any doubt about whether an approach is correct, whether an API works the way you think, whether a class has the attributes you expect, or whether your logic is sound -- verify it by running code. Do not guess. Do not assume. Run it.

This applies throughout meta plan creation, not just at the end. Any time you write something in the meta plan that depends on how the existing code behaves, verify it first.

**Verify existing behavior:** Before claiming "line 115 has a NameError" or "this method returns a dict", confirm with `uv run python -c "..."`. Plans with unverified claims about code behavior are plans with bugs.

**Novel patterns:** If the design introduces novel patterns (new serialization approaches, OS-level mechanics, composable APIs, import resolution tricks), build a lightweight proof-of-concept:
- **Fresh/novel patterns**: create a throwaway project in `/tmp` via `uv init`, write a self-contained test script with assertions, run with `uv run python test_xyz.py`
- **Patterns touching existing codebase**: verify directly in the repo dir with `uv run python -c "..."`
- Each PoC should be self-contained with assertions that print PASS/FAIL
- If a PoC fails, fix the design in the meta plan before proceeding
- Document findings (ordering gotchas, edge cases discovered, required workarounds) back into the meta plan

**POC must mirror the target repo's soul.** Even though POCs are throwaway software, they must be written in the same style, conventions, and design philosophy as the repo they model. This means:
- Follow the repo's style rules (indentation, quotes, import order, line length)
- Use the same naming conventions (variable names, class names, method names)
- Follow the same design patterns (subclassing, protocols, composition style)
- Use the same data model conventions (`str | None = None`, `field(default_factory=...)`, etc.)
- Respect the same prohibited patterns (no `getattr` fallbacks, no defensive `.get('key', '')`, etc. -- enforced by ast-grep scan; see `rules/`)
- Use realistic variable names and structures that match what production code would look like

The goal is zero style divergence between POC and production. POC code samples should be directly liftable into plan files and then into implementation without needing style translation, renaming, or restructuring. A POC that validates the right logic but uses wrong conventions is a POC that will cause drift when implementors copy from it.

**POC results feed the meta plan.** After running POCs, update:
- Pre-Plan POC Verification section (test phases, counts, all-pass confirmation)
- Critical POC Code Samples section (inline validated patterns)
- Bug Registry (any design bugs found and fixed)
- Resolved Ambiguities (any decisions made during POC)

**Verify before finalizing every section.** Don't defer verification to the end. If Phase 1 discovery claims a behavior, verify it. If Phase 2 discussion assumes an API shape, verify it. If Phase 3 writing references a file path or method signature, verify it.

### Phase 5: Handoff

When the meta plan is complete and the user is satisfied:

**For master plan systems:** Hand off to the **create-master-plan** skill to generate the plan directory (`00-master-plan.md` + `01-NN.md`). The meta plan is the source; create-master-plan structures it into the plan file format.

**For individual plans:** Hand off to the **create-plan** skill to write the plan file. The meta plan provides the design; create-plan structures it into the plan file format.

Do NOT generate plan files during meta-planning. The meta plan and plan files are distinct phases. Only proceed to plan generation when the user explicitly asks.

**Compaction resilience:** The meta plan must carry enough inlined decisions and pointers to survive context compaction. When compaction occurs mid-session: update the transcript reference, verify key decisions are inlined in the meta plan text (not only in the compacted transcript), launch a subagent to mine the transcript and inline critical decisions if needed. The meta plan should be reconstructable by an agent that has never seen the original conversation.
