"""Integration tests for example script fixes (sub-plan 04).

Verifies:
  - All example pyproject.toml files have the relocation comment (P2#26)
  - textmatch README documents workspace scaffold for -p textmatch (P2#27)
  - multi_module dead experiment branch is removed (P3#35)
  - textmatch accuracy extraction handles zero correctly (P3#36)
"""

from pathlib import Path
import ast

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / 'examples'
RELOCATION_COMMENT = '# This path reference requires running from within the autopilot repo.'


def _example_pyproject_files() -> list[Path]:
  """Collect top-level example pyproject.toml files, excluding .venv and caches."""
  results = []
  for path in sorted(EXAMPLES_DIR.rglob('pyproject.toml')):
    parts = path.relative_to(EXAMPLES_DIR).parts
    if any(p.startswith('.') or p == '__pycache__' for p in parts[:-1]):
      continue
    results.append(path)
  return results


def test_example_pyproject_has_relocation_comment():
  """Every example pyproject.toml must carry the relocation warning comment."""
  pyproject_files = _example_pyproject_files()
  assert len(pyproject_files) >= 1, 'no example pyproject.toml files found'

  missing = []
  for path in pyproject_files:
    content = path.read_text(encoding='utf-8')
    if RELOCATION_COMMENT.strip() not in content:
      missing.append(str(path.relative_to(REPO_ROOT)))

  assert not missing, f'pyproject.toml files missing relocation comment: {missing}'


def test_textmatch_cli_wiring_documented():
  """textmatch README must document workspace scaffold requirements."""
  readme = EXAMPLES_DIR / 'textmatch' / 'README.md'
  assert readme.exists(), 'examples/textmatch/README.md does not exist'

  content = readme.read_text(encoding='utf-8')
  assert 'textmatch' in content
  assert 'workspace' in content.lower()
  assert '-p' in content or ('project' in content.lower() and 'textmatch' in content)


def test_multi_module_no_dead_code():
  """multi_module run_trainer.py must not contain the dead experiment branch."""
  script = EXAMPLES_DIR / 'multi_module' / 'run_trainer.py'
  assert script.exists(), 'examples/multi_module/run_trainer.py does not exist'

  source = script.read_text(encoding='utf-8')
  tree = ast.parse(source, filename=str(script))

  for node in ast.walk(tree):
    if isinstance(node, ast.If):
      test = ast.unparse(node.test)
      assert 'trainer.experiment is not None' not in test, (
        'dead branch "if trainer.experiment is not None" still present '
        'in examples/multi_module/run_trainer.py'
      )


def test_accuracy_extraction_handles_zero():
  """Accuracy extraction must preserve 0.0 (not treat it as falsy)."""
  metrics_with_zero = {'accuracy': 0.0, 'train_accuracy': 0.75}
  metrics_with_none = {'train_accuracy': 0.75}
  metrics_with_value = {'accuracy': 0.85, 'train_accuracy': 0.75}

  def extract_accuracy(train: dict) -> float:
    """Mirrors the logic in examples/textmatch/run_trainer.py."""
    acc = train.get('accuracy')
    return acc if acc is not None else train.get('train_accuracy', 0.0)

  assert extract_accuracy(metrics_with_zero) == 0.0
  assert extract_accuracy(metrics_with_none) == 0.75
  assert extract_accuracy(metrics_with_value) == 0.85

  script = EXAMPLES_DIR / 'textmatch' / 'run_trainer.py'
  source = script.read_text(encoding='utf-8')
  assert "train.get('accuracy') or" not in source, (
    'textmatch run_trainer.py still uses truthiness-based `or` for accuracy extraction'
  )
