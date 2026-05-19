---
name: code-reorganization
description: Deterministic, programmatic techniques for restructuring Python codebases -- splitting monoliths into packages, rewriting imports via AST, updating patch targets, and validating with quality gates. Use when reorganizing files, splitting large modules, moving code between packages, refactoring import paths, or restructuring directory layout.
---

## Overview

This skill covers safe, repeatable code reorganization for Python projects that use namespace packages (no `__init__.py`). Every structural change must be programmatic and validated -- no manual file-by-file editing of imports.

The core principle: **move files, rewrite imports, validate**. Each step is deterministic and reversible.

## Mandatory workflow

Every reorganization follows this sequence. Never skip steps.

1. **Copy to `/tmp`** -- work on a disposable copy first
2. **Establish green baseline** -- full quality gate must pass before any changes
3. **One structural change at a time** -- move one file/module, rewrite all references, validate
4. **Quality gate after every step** -- lint, format, type check, tests
5. **Document findings** -- record every issue and its fix for the real implementation

## Technique 1: splitting a monolith into a package

When `foo.py` (>500 lines) needs to become `foo/{a.py, b.py}`:

```python
import ast
from pathlib import Path

source = Path('src/pkg/foo.py')
content = source.read_text()
tree = ast.parse(content)

# 1. identify class/function boundaries
for node in ast.iter_child_nodes(tree):
  if isinstance(node, ast.ClassDef):
    # CRITICAL: include decorators
    start = node.lineno - 1
    if node.decorator_list:
      start = node.decorator_list[0].lineno - 1
    print(f'L{start}-{node.end_lineno} {node.name}')
```

Then split by copying line ranges into new files with their required imports.

**After splitting:** delete the original file, create the package directory, write the new files.

## Technique 2: rewriting imports with Python AST

Use `ast` to parse each `.py` file and rewrite `from X import Y` statements. Two modes:

### Mode A: whole-module move (foo.py -> foo/foo.py)

Every `from pkg.foo import X` becomes `from pkg.foo.foo import X`:

```python
import ast
from pathlib import Path

def rewrite_file(path: Path, old_module: str, new_module: str) -> bool:
  content = path.read_text()
  try:
    tree = ast.parse(content)
  except SyntaxError:
    return False

  lines = content.splitlines(keepends=True)
  replacements = []

  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module == old_module:
      lineno = node.lineno - 1
      end_lineno = node.end_lineno
      old_text = ''.join(lines[lineno:end_lineno])
      new_text = old_text.replace(old_module, new_module, 1)
      replacements.append((lineno, end_lineno, new_text))

  if not replacements:
    return False

  for start, end, new_text in reversed(replacements):
    lines[start:end] = [new_text]

  path.write_text(''.join(lines))
  return True
```

### Mode B: symbol-level split (types go to types.py, class stays in base.py)

Build an explicit symbol-to-module map and rewrite each import:

```python
def rewrite_split_import(
  path: Path,
  old_module: str,
  symbol_to_module: dict[str, str],
) -> bool:
  content = path.read_text()
  try:
    tree = ast.parse(content)
  except SyntaxError:
    return False

  lines = content.splitlines(keepends=True)
  replacements = []

  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module == old_module:
      groups: dict[str, list[str]] = {}
      for alias in node.names:
        target = symbol_to_module.get(alias.name, old_module)
        name = f'{alias.name} as {alias.asname}' if alias.asname else alias.name
        groups.setdefault(target, []).append(name)

      if len(groups) == 1 and old_module in groups:
        continue

      lineno = node.lineno - 1
      end_lineno = node.end_lineno
      indent = ''.join(c for c in lines[lineno] if c in ' \t')

      new_lines = []
      for mod, symbols in sorted(groups.items()):
        new_lines.append(f'{indent}from {mod} import {", ".join(symbols)}\n')

      replacements.append((lineno, end_lineno, ''.join(new_lines)))

  if not replacements:
    return False

  for start, end, new_text in reversed(replacements):
    lines[start:end] = [new_text]

  path.write_text(''.join(lines))
  return True
```

**Always use Mode B** when symbols from one file split across multiple destinations. Mode A is only safe when the entire module moves to a single new path.

## Technique 3: rewriting patch() string targets

Test files use `mock.patch('pkg.module.ClassName')` which breaks when the module moves. Scan for string references:

```python
def rewrite_patch_strings(path: Path, old: str, new: str) -> bool:
  content = path.read_text()
  if old not in content:
    return False
  new_content = content.replace(f"'{old}", f"'{new}")
  new_content = new_content.replace(f'"{old}', f'"{new}')
  if new_content == content:
    return False
  path.write_text(new_content)
  return True
```

## Technique 4: scanning for all affected files

Always scan `src/`, `tests/`, AND `examples/`. Template files may also need updating:

```python
from pathlib import Path

def collect_python_files(*roots: Path) -> list[Path]:
  files = []
  for root in roots:
    if root.is_dir():
      files.extend(sorted(root.rglob('*.py')))
  return files

# also scan non-Python files for import references
def scan_docs_for_paths(old_module: str, *doc_files: Path) -> list[Path]:
  affected = []
  dotted = old_module  # e.g. 'autopilot.core.store'
  for f in doc_files:
    if f.exists() and dotted in f.read_text():
      affected.append(f)
  return affected
```

Always check: `README.md`, `CLAUDE.md`, `AGENTS.md`, `PHILOSOPHY.md`, `pyproject.toml`, template files under `src/autopilot/templates/`, and skill files under `.claude/skills/`.

## Post-move checklist

Run after every structural change:

1. **Clean stale bytecache**: `find src/ -path '*/__pycache__/<moved_module>.cpython-*.pyc' -delete` (Python caches file-module resolution in `.pyc`; when a file becomes a directory, the stale `.pyc` shadows the new package)
2. **Auto-fix imports**: `uv run ruff check --fix src/ tests/ examples/`
3. **Format**: `uv run ruff format src/ tests/ examples/`
4. **Type check**: `uv run ty check`
5. **Tests**: `uv run pytest -x -q`
6. **Test count**: `uv run pytest --co -q` (must match baseline)

## Gotchas

### Decorator loss during AST extraction

`ast.ClassDef.lineno` points to the `class` keyword, not preceding decorators. When extracting a class, always check `node.decorator_list`:

```python
start = node.lineno - 1
if node.decorator_list:
  start = node.decorator_list[0].lineno - 1
```

Missing a `@dataclass` decorator causes silent runtime `TypeError` on construction.

### Namespace packages have `__file__ = None` and `__doc__ = None`

When `foo.py` becomes `foo/bar.py`, any test doing `import pkg.foo as mod; mod.__file__` breaks. Update to `from pkg.foo import bar as mod`.

Same for `__doc__` -- namespace packages have no module docstring. Point docstring tests at the specific file module.

### Private names crossing module boundaries (PLC2701)

When `_helper()` moves from `foo.py` to `foo/helpers.py`, importing it in `foo/main.py` triggers ruff's PLC2701 (private name import). Rename to public: `_helper -> helper`.

Audit all `_`-prefixed symbols being extracted. If they will be imported from another module in the same package, drop the underscore.

### pyproject.toml per-file lint ignores

`pyproject.toml` may have per-file rule suppressions like `"src/pkg/big.py" = ["PLR0904"]`. When the file moves, the path becomes stale and the suppression stops working. Update paths in lockstep.

### Self-referencing calls after rename

When renaming `_foo` to `foo` in a file, calls to `_foo()` within the *same* file also need updating. Use file-wide `str.replace('_foo(', 'foo(')` rather than targeted import-only rewrites.

### README contract tests

`tests/unit/test_readme_contracts.py` executes import statements found in `README.md`. Any stale import path in README causes test failure. This is a feature -- it catches missed documentation updates. Always update README in the same step as the code move.

### Template files cause SyntaxError

Files under `templates/` may contain `{name}` placeholders that are not valid Python. Wrap `ast.parse()` in `try/except SyntaxError` to skip them gracefully.

### Import rewriter must not over-match

Never use blanket module rewriting when only specific symbols moved. If `infer_direction` moved from `cli.commands.experiment` to `core.metric_utils`, do NOT rewrite all imports from `cli.commands.experiment` -- only rewrite imports that specifically name `infer_direction`.

### Bytecache shadows new package directories

When `foo.py` becomes `foo/`, Python's `__pycache__/foo.cpython-*.pyc` still resolves `foo` as a file-module. This causes `ModuleNotFoundError: 'pkg.foo' is not a package` even after the directory exists. Fix: delete the stale `.pyc` files for the moved module immediately after the file-to-directory rename:

```bash
find src/ -path '*/__pycache__/foo.cpython-*.pyc' -delete
```

No `uv sync --reinstall-package` or `uv pip install -e .` is needed -- editable installs already point at the source tree. The issue is purely stale bytecache.

## /tmp dry-run protocol

Before touching the real codebase:

```bash
# 1. fresh copy
rm -rf /tmp/autopilot-reorg && cp -a /path/to/repo /tmp/autopilot-reorg

# 2. clean environment
cd /tmp/autopilot-reorg && rm -rf .venv && uv sync

# 3. green baseline
uv run pytest --co -q  # record test count
uv run pytest -x -q    # must pass

# 4. apply changes (one at a time)
# ... move files, rewrite imports ...

# 5. validate after each change (no reinstall needed for editable installs)
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
uv run pytest -x -q
uv run pytest --co -q  # count must match baseline

# 6. full quality gate at the end
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check
uv run ast-grep scan --config sgconfig.yml src/ tests/
uv run pytest -x -v
```

Only after the `/tmp` dry-run is fully green should changes be applied to the real codebase.
