"""Tests that documentation stays accurate relative to the codebase.

Validates template import paths, API references, root documentation parity,
and CLI command matrix completeness.
"""

from pathlib import Path
import ast
import inspect
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = REPO_ROOT / 'src' / 'autopilot' / 'templates' / 'project' / 'CLAUDE.md'
ROOT_CLAUDE_MD = REPO_ROOT / 'CLAUDE.md'
ROOT_AGENTS_MD = REPO_ROOT / 'AGENTS.md'


def _extract_fenced_python_blocks(text: str) -> list[str]:
  """Extract content of fenced code blocks tagged as python."""
  pattern = re.compile(r'```python\n(.*?)```', re.DOTALL)
  return pattern.findall(text)


def _extract_import_lines(code: str) -> list[str]:
  """Extract lines starting with 'from' or 'import' from code block."""
  lines = []
  for line in code.splitlines():
    stripped = line.strip()
    if stripped.startswith(('from ', 'import ')):
      lines.append(stripped)
  return lines


class TestTemplateFidelity:
  """Tests that template code blocks contain valid imports and API references."""

  def test_template_imports_valid(self):
    """Extract python import lines from template; exec in isolated namespace."""
    content = TEMPLATE_PATH.read_text()
    blocks = _extract_fenced_python_blocks(content)
    assert len(blocks) > 0, 'no python blocks found in template'

    all_imports: list[str] = []
    for block in blocks:
      all_imports.extend(_extract_import_lines(block))

    assert len(all_imports) > 0, 'no import lines found in template python blocks'

    failures = []
    for imp in all_imports:
      if 'from {name}' in imp or 'from textmatch' in imp:
        continue
      ns: dict = {}
      try:
        exec(imp, ns)
      except (ImportError, ModuleNotFoundError) as exc:
        failures.append(f'{imp!r} -> {exc}')

    assert not failures, f'{len(failures)} import(s) failed:\n' + '\n'.join(failures)

  def test_template_api_references_exist(self):
    """Verify documented Module.attr patterns resolve via getattr or __init__."""
    from autopilot.core.comparison import ComparatorMetric
    from autopilot.core.metric_utils import infer_direction
    from autopilot.core.query import QueryBuilder
    from autopilot.policy.gates import (
      BudgetGate,
      CustomGate,
      MaxGate,
      MinGate,
      MonotonicGate,
      RangeGate,
    )

    qb_methods = [
      'completed',
      'pending',
      'failed',
      'running',
      'cancelled',
      'order_by_metric',
      'all',
    ]
    failures = [
      f'QueryBuilder.{m} not found' for m in qb_methods if getattr(QueryBuilder, m, None) is None
    ]

    sig = inspect.signature(ComparatorMetric.__init__)
    if 'metric_name' not in sig.parameters:
      failures.append('ComparatorMetric.__init__ missing param metric_name')

    assert infer_direction is not None, 'infer_direction not importable'

    gate_classes = [MinGate, MaxGate, RangeGate, CustomGate, MonotonicGate, BudgetGate]
    failures.extend(f'gate class {cls} is None' for cls in gate_classes if cls is None)

    assert not failures, f'{len(failures)} API reference(s) broken:\n' + '\n'.join(failures)

  def test_doc_template_missing_sections_detected(self):
    """Template must contain all required section headings."""
    content = TEMPLATE_PATH.read_text()
    required_headings = [
      'Common Imports',
      'Metric Key Naming',
      'Module Attribute Assignment Trap',
      'EvalDatum vs Datum',
      'Autograd Graph Preservation',
      'Git-to-Store Glossary',
      'Store Bootstrap Sequence',
      'Worktree Isolation Semantics',
      'Python API',
      'Key Commands Reference',
    ]

    missing = [h for h in required_headings if h not in content]

    assert not missing, f'Template missing section(s): {missing}'

  def test_doc_fenced_import_syntax_error(self):
    """All python fenced blocks must be syntactically valid."""
    content = TEMPLATE_PATH.read_text()
    blocks = _extract_fenced_python_blocks(content)

    failures = []
    for i, block in enumerate(blocks):
      try:
        ast.parse(block)
      except SyntaxError as exc:
        failures.append(f'block {i}: {exc.msg} (line {exc.lineno})')

    assert not failures, (
      f'{len(failures)} syntax error(s) in template python blocks:\n' + '\n'.join(failures)
    )

  def test_doc_template_stale_api_reference(self):
    """Template must not reference removed/changed API paths in code blocks."""
    content = TEMPLATE_PATH.read_text()
    blocks = _extract_fenced_python_blocks(content)
    code_content = '\n'.join(blocks)

    stale_patterns = [
      'from autopilot.policy.gate import',
      'from autopilot.policy.policy import QualityFirstPolicy',
      'from autopilot.ai.store import FileStore',
      'from autopilot.ai.config import',
      'status_filter',
      'spec_version_filter',
    ]

    found = [p for p in stale_patterns if p in code_content]

    assert not found, f'Template code blocks contain stale API reference(s): {found}'


class TestRootDocumentationParity:
  """Tests that root CLAUDE.md and AGENTS.md stay synchronized."""

  def test_claude_md_equals_agents_md(self):
    """CLAUDE.md and AGENTS.md must be byte-for-byte identical."""
    agents_path = ROOT_AGENTS_MD

    if agents_path.is_symlink():
      target = agents_path.resolve()
      assert target == ROOT_CLAUDE_MD.resolve(), (
        f'AGENTS.md symlink points to {target}, expected {ROOT_CLAUDE_MD.resolve()}'
      )
    else:
      claude_bytes = ROOT_CLAUDE_MD.read_bytes()
      agents_bytes = agents_path.read_bytes()
      assert claude_bytes == agents_bytes, 'CLAUDE.md and AGENTS.md differ; they must be identical'

  def test_new_commands_in_matrix(self):
    """Root CLAUDE.md CLI matrix includes required command slugs."""
    content = ROOT_CLAUDE_MD.read_text()

    required_commands = [
      'workspace journal',
      'experiment lineage',
      'experiment timeline',
      'undo-guide',
      'store reflog list',
    ]

    missing = [cmd for cmd in required_commands if cmd not in content]

    assert not missing, f'CLI matrix missing command(s): {missing}'
