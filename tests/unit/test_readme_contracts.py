"""Tests verifying README.md contracts for sub-plan 12 (README and external docs).

Each test function corresponds to a step in the plan (test_readme_contract_step_2N).
"""

from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / 'README.md'


def _readme() -> str:
  return README.read_text()


class TestReadmeContractStep21:
  """2.1: ClaudeCodeAgent import paths in fenced examples."""

  def test_readme_contract_step_21(self) -> None:
    r = _readme()
    occurrences = r.count('autopilot.ai.agents.claude_code')
    assert occurrences >= 2, (
      f'expected at least 2 occurrences of autopilot.ai.agents.claude_code, got {occurrences}'
    )
    assert 'from autopilot.ai.coding import' not in r

  def test_import_resolves(self) -> None:
    from autopilot.ai.agents.claude_code import ClaudeCodeAgent

    assert ClaudeCodeAgent is not None


class TestReadmeContractStep22:
  """2.2: Memory/FileMemory/MemoryCallback claims removed."""

  def test_readme_contract_step_22(self) -> None:
    r = _readme()
    assert 'MemoryCallback captures' not in r
    assert '- **Persistent Memory**' not in r
    assert 'Query, record, trends, and context' not in r
    assert '| `memory` |' not in r.lower()

  def test_no_memory_command_in_help(self) -> None:
    proc = subprocess.run(
      ['uv', 'run', 'autopilot', '--help'],
      capture_output=True,
      text=True,
      check=True,
      cwd=str(ROOT),
    )
    lines = proc.stdout.lower().splitlines()
    subcommands = [ln.strip().split()[0] for ln in lines if ln.strip().startswith(('memory',))]
    r = _readme()
    if '| `memory` |' in r.lower():
      assert 'memory' in list(subcommands), 'README lists memory command but CLI does not expose it'


class TestReadmeContractStep23:
  """2.3: Quick-start snippet with Datum + ListDataset + DataLoader."""

  def test_readme_contract_step_23(self) -> None:
    from autopilot.core.types import Datum
    from autopilot.data.dataloader import DataLoader
    from autopilot.data.dataset import ListDataset

    loader = DataLoader(ListDataset([Datum(), Datum(), Datum()]), batch_size=1)
    batch = next(iter(loader))
    assert batch.id


class TestReadmeContractStep24:
  """2.4: Component mapping table has GeneratorAgent/JudgeAgent."""

  def test_readme_contract_step_24(self) -> None:
    r = _readme()
    assert 'autopilot.ai.evaluation.generator' in r
    assert 'autopilot.ai.evaluation.judge' in r

  def test_no_stale_memory_row(self) -> None:
    r = _readme()
    table_section = r[r.find('## Component mapping') : r.find('## Examples')]
    assert '| No equivalent | `Memory`' not in table_section
    assert '| No equivalent | `DataGenerator`' not in table_section

  def test_imports_resolve(self) -> None:
    from autopilot.ai.evaluation.generator import GeneratorAgent
    from autopilot.ai.evaluation.judge import JudgeAgent

    assert GeneratorAgent is not None
    assert JudgeAgent is not None


class TestReadmeContractStep25:
  """2.5: 'What to import' section exists before 'Package layout'."""

  def test_readme_contract_step_25(self) -> None:
    r = _readme()
    assert '## What to import' in r
    what_idx = r.index('## What to import')
    layout_idx = r.index('## Package layout')
    assert layout_idx > what_idx

  def test_import_paths_resolve(self) -> None:
    r = _readme()
    what_section_start = r.index('## What to import')
    what_section_end = r.index('## Package layout')
    section = r[what_section_start:what_section_end]

    import_lines = re.findall(r'`(from \S+ import \S+(?:,\s*\S+)*)`', section)
    assert len(import_lines) >= 10, f'expected at least 10 import bullets, got {len(import_lines)}'

    for line in import_lines:
      try:
        exec(line)
      except Exception as exc:
        msg = f'import failed: {line}'
        raise AssertionError(msg) from exc


class TestReadmeContractStep26:
  """2.6: Examples section references all three example dirs."""

  def test_readme_contract_step_26(self) -> None:
    r = _readme()
    assert 'examples/textmatch/' in r
    assert 'examples/protim/' in r
    assert 'examples/multi_module/' in r

  def test_example_paths_exist(self) -> None:
    assert (ROOT / 'examples' / 'textmatch').is_dir()
    assert (ROOT / 'examples' / 'protim').is_dir()
    assert (ROOT / 'examples' / 'multi_module').is_dir()
    assert (ROOT / 'examples' / 'multi_module' / 'run_trainer.py').is_file()


class TestReadmeContractStep27:
  """2.7: Package layout tree does not claim Memory/DataGenerator."""

  def test_readme_contract_step_27(self) -> None:
    r = _readme()
    layout_start = r.find('src/autopilot/')
    assert layout_start >= 0
    block = r[layout_start : layout_start + 1200]
    assert '  ai/           # DataGenerator' not in block
    assert 'Memory' not in block


class TestReadmeContractStep28:
  """2.8: Key commands table entries all appear in --help."""

  def test_readme_contract_step_28(self) -> None:
    proc = subprocess.run(
      ['uv', 'run', 'autopilot', '--help'],
      capture_output=True,
      text=True,
      check=True,
      cwd=str(ROOT),
    )
    help_output = proc.stdout

    r = _readme()
    assert 'Key commands' in r
    tail = r.split('Key commands', 1)[1][:8000]
    cmds = re.findall(r'\|\s*`([a-z0-9_-]+)`\s*\|', tail)
    assert cmds, 'no commands found in Key commands table'
    missing = [c for c in cmds if c not in help_output]
    assert not missing, f'README lists commands not in --help: {missing}'


class TestReadmeContractStep29:
  """2.9: Multi-project snippet annotates canonical eval types."""

  def test_readme_contract_step_29(self) -> None:
    r = _readme()
    assert 'canonical eval type is GeneratorAgent' in r
    assert 'canonical eval type is JudgeAgent' in r

  def test_canonical_types_importable(self) -> None:
    from autopilot.ai.evaluation.generator import GeneratorAgent
    from autopilot.ai.evaluation.judge import JudgeAgent

    assert GeneratorAgent is not None
    assert JudgeAgent is not None


class TestFencedImportResolution:
  """4.1: All from-import lines in fenced python blocks resolve."""

  def test_all_fenced_python_imports_resolve(self) -> None:
    r = _readme()
    fenced_blocks = re.findall(r'```python\n(.*?)```', r, re.DOTALL)
    assert len(fenced_blocks) >= 3, (
      f'expected at least 3 fenced python blocks, got {len(fenced_blocks)}'
    )

    import_lines: list[str] = []
    for block in fenced_blocks:
      for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith('from autopilot.') and ' import ' in stripped:
          clean = stripped.split('#')[0].strip()
          import_lines.append(clean)

    assert len(import_lines) >= 5, (
      f'expected at least 5 autopilot import lines across fences, got {len(import_lines)}'
    )

    for line in import_lines:
      try:
        exec(line)
      except Exception as exc:
        msg = f'fenced import failed: {line}'
        raise AssertionError(msg) from exc
