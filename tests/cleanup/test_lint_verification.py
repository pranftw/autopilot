"""Sub-plan 08: final verification tests for zero-error lint compliance.

Confirms all lint tools exit 0 and banned structural patterns are absent.
"""

from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ruff_check_zero() -> None:
  """Assert ruff check finds 0 errors across src/ and tests/."""
  result = subprocess.run(
    ['uv', 'run', 'ruff', 'check', 'src/', 'tests/'],
    capture_output=True,
    text=True,
    check=False,
    cwd=str(REPO_ROOT),
  )
  assert result.returncode == 0, f'ruff check failed:\n{result.stdout[-1000:]}'


def test_ruff_format_clean() -> None:
  """Assert ruff format finds no files needing reformatting."""
  result = subprocess.run(
    ['uv', 'run', 'ruff', 'format', '--check', 'src/', 'tests/'],
    capture_output=True,
    text=True,
    check=False,
    cwd=str(REPO_ROOT),
  )
  assert result.returncode == 0, f'ruff format failed:\n{result.stderr[-1000:]}'


def test_ty_check_zero() -> None:
  """Assert ty check finds 0 diagnostics."""
  result = subprocess.run(
    ['uv', 'run', 'ty', 'check'],
    capture_output=True,
    text=True,
    check=False,
    cwd=str(REPO_ROOT),
  )
  assert result.returncode == 0, f'ty check failed:\n{result.stdout[-1000:]}'


def test_astgrep_zero() -> None:
  """Assert ast-grep scan finds 0 rule violations."""
  result = subprocess.run(
    ['uv', 'run', 'ast-grep', 'scan', '--config', 'sgconfig.yml', 'src/', 'tests/'],
    capture_output=True,
    text=True,
    check=False,
    cwd=str(REPO_ROOT),
  )
  assert result.returncode == 0, f'ast-grep failed:\n{result.stdout[-1000:]}'


def test_graph_isolation() -> None:
  """Assert graph.py has no autopilot imports (leaf module isolation)."""
  graph_path = REPO_ROOT / 'src' / 'autopilot' / 'core' / 'graph.py'
  content = graph_path.read_text()
  violating_lines = [
    line for line in content.splitlines() if line.strip().startswith('from autopilot')
  ]
  assert violating_lines == [], f'graph.py must not import from autopilot, found: {violating_lines}'


def test_no_init_files() -> None:
  """Assert no __init__.py files exist under src/."""
  src_dir = REPO_ROOT / 'src'
  init_files = list(src_dir.rglob('__init__.py'))
  assert init_files == [], f'unexpected __init__.py files: {init_files}'
