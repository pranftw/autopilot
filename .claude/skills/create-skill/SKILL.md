---
name: create-skill
description: Comprehensive guide for authoring new Agent Skills in this repo following the agentskills.io specification. Use when creating a new skill, updating an existing skill definition, or reviewing skill quality.
---

## What is a skill?

A skill is a directory containing a `SKILL.md` file that gives an AI agent specialized knowledge and step-by-step instructions for a specific domain. Skills live in `.claude/skills/` (also accessible via `.agents/skills/` symlink).

## Directory structure

```
.claude/skills/<skill-name>/
  SKILL.md              # Required: YAML frontmatter + markdown instructions
  scripts/              # Optional: executable code the agent can run
  references/           # Optional: detailed docs loaded on demand
  assets/               # Optional: templates, schemas, static resources
```

## SKILL.md format

The file has two parts: YAML frontmatter between `---` delimiters, then a markdown body.

### Frontmatter (required fields)

```yaml
---
name: skill-name
description: What the skill does and when to use it. Include trigger keywords.
---
```

**`name`** (required):
- Must match the parent directory name exactly
- 1-64 characters
- Lowercase letters, numbers, and hyphens only
- No leading/trailing hyphens, no consecutive hyphens (`--`)
- Examples: `module-model`, `cli-conventions`, `create-skill`

**`description`** (required):
- 1-1024 characters
- Must describe both WHAT the skill does AND WHEN to use it
- Include specific keywords that help agents match user prompts to this skill
- The description is loaded at startup for all skills (~50-100 tokens each), so it must be concise but precise

Good: `'Extract PDF text, fill forms, merge files. Use when handling PDFs or when the user mentions document extraction.'`
Bad: `'Helps with PDFs.'`

### Frontmatter (optional fields)

```yaml
---
name: skill-name
description: What it does and when to use it.
license: Apache-2.0
compatibility: Requires Python 3.12+ and uv
metadata:
  author: team-name
  version: '1.0'
allowed-tools: Bash(git:*) Read
---
```

- **`license`**: license name or reference to bundled LICENSE file
- **`compatibility`**: max 500 chars, environment requirements
- **`metadata`**: arbitrary key-value pairs (string keys and values)
- **`allowed-tools`**: space-delimited pre-approved tools (experimental)

### Body content

The markdown body after frontmatter contains the actual instructions. This is loaded only when the skill activates (Tier 2 of progressive disclosure).

## Progressive disclosure model

Skills use three tiers to manage context efficiently:

1. **Tier 1 -- Catalog** (~100 tokens): `name` + `description` loaded at session start for ALL skills
2. **Tier 2 -- Instructions** (<5000 tokens): full `SKILL.md` body loaded when the skill activates
3. **Tier 3 -- Resources** (as needed): files in `scripts/`, `references/`, `assets/` loaded on demand

Keep `SKILL.md` under 500 lines. Move detailed reference material to separate files.

## Writing effective body content

### Include what the agent wouldn't know

Focus on project-specific conventions, domain procedures, non-obvious edge cases, and which tools/APIs to use. Skip things the agent already knows (what a PDF is, how HTTP works).

### Recommended sections

- **Protocol/interface definitions** -- method signatures, required contracts
- **Step-by-step procedures** -- numbered steps for common tasks
- **Key files** -- paths to the most important source files
- **Gotchas** -- concrete corrections to mistakes the agent will make without guidance

### Gotchas sections are high-value

```markdown
## Gotchas

- The `users` table uses soft deletes. Always add `WHERE deleted_at IS NULL`.
- Backend role identifiers use underscores, not hyphens: `staging_backend`, not `staging-backend`.
- Modules must never access environment variables directly.
```

### Provide defaults, not menus

Pick one recommended approach. Mention alternatives briefly only when needed:

```markdown
<!-- Good: clear default -->
Use pdfplumber for text extraction. For scanned PDFs, fall back to pdf2image with pytesseract.

<!-- Bad: too many equal options -->
You can use pypdf, pdfplumber, PyMuPDF, or pdf2image...
```

### Favor procedures over declarations

Teach the agent HOW to approach a class of problems, not WHAT to produce for one specific case.

## Project-specific conventions for this repo

When writing skills for AutoPilot, follow these rules:

- **2-space indentation** in all code examples
- **Single quotes** for all Python strings
- **No env vars** in any `src/autopilot/` code -- all config through function args
- **Absolute imports only**: `from autopilot.core.models import Manifest`
- **No dynamic imports**: no `importlib`, no runtime discovery
- **No registries**: components are wired as explicit objects (constructors, `Trainer`, `CLI`), not string-key lookups
- Code examples should match the Google Python Style Guide baseline with the project overrides above

## When to update existing skills

Skills must stay in sync with the codebase. When a feature is added, changed, or removed:

1. **New feature** -- create a new skill if it's a distinct domain, or update the relevant existing skill's SKILL.md
2. **Changed protocol/interface** -- update the skill that documents it (e.g. new Module hook -> update `module-model`)
3. **New CLI command** -- update `cli-conventions` skill and/or create a domain-specific skill
4. **Changed config schema** -- update `project-bootstrap` and any skills that reference the config patterns
5. **Always update README.md** -- the project README commands table and feature list must reflect current state

Stale skills are worse than no skills -- they cause agents to follow outdated patterns.

## Step-by-step: creating a new skill

1. **Choose a name**: lowercase, hyphens, matches directory name. Keep it descriptive but short.

2. **Create the directory**:
   ```
   mkdir -p .claude/skills/<skill-name>
   ```

3. **Write the `SKILL.md`**:
   - Start with frontmatter: `name` + `description`
   - Write the body: protocols, procedures, key files, gotchas
   - Keep under 500 lines / 5000 tokens

4. **Validate**:
   - `name` matches directory name
   - `description` is under 1024 chars and includes trigger keywords
   - Body has actionable instructions, not just descriptions
   - No line in the body is exactly `---` (confuses frontmatter parser)

5. **Test activation**: start a new session and verify the skill appears in the catalog and activates on relevant prompts.

## When to use optional directories

**`scripts/`**: when the agent repeatedly reinvents the same logic across runs. Write a tested script once and reference it from SKILL.md.

**`references/`**: when detailed documentation exceeds the 5000-token Tier 2 budget. Split into focused files and tell the agent WHEN to load each: "Read `references/api-errors.md` if the API returns a non-200 status."

**`assets/`**: for templates, schemas, or lookup tables the agent needs verbatim.

## Validation checklist

- [ ] Directory name matches `name` field in frontmatter
- [ ] `name` is lowercase, hyphens only, 1-64 chars, no leading/trailing/consecutive hyphens
- [ ] `description` is 1-1024 chars, describes what AND when
- [ ] Body is under 500 lines
- [ ] No bare `---` lines in body (use `****` for horizontal rules)
- [ ] Key files referenced in body actually exist in the codebase
- [ ] Instructions are actionable (procedures, not just descriptions)
- [ ] Code examples follow repo conventions (2-space indent, single quotes)
- [ ] Gotchas section covers non-obvious pitfalls

## Template

```markdown
---
name: my-skill-name
description: What this skill does. Use when <specific trigger conditions>.
---

## Overview

Brief description of the domain and key concepts.

## Key patterns

Step-by-step procedures for common tasks.

## Key files

- `src/autopilot/path/to/file.py` -- what it contains

## Gotchas

- Non-obvious fact 1
- Non-obvious fact 2
```
