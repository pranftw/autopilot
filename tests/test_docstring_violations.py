"""Per-package verification that all D*/DOC* ruff violations are resolved.

Tests run ``ruff check --select D,DOC`` against each source package and
assert zero errors, ensuring Google-style docstring coverage is maintained.
"""

from pathlib import Path
import subprocess

SRC = Path(__file__).resolve().parent.parent / 'src' / 'autopilot'


def _ruff_docstring_errors(target: str) -> tuple[int, str]:
  """Run ruff docstring checks on *target* and return (exit_code, output).

  Args:
    target: Path to check.

  Returns:
    Tuple of (exit_code, combined stdout+stderr).
  """
  result = subprocess.run(
    ['uv', 'run', 'ruff', 'check', target, '--select', 'D,DOC'],
    capture_output=True,
    text=True,
    check=False,
  )
  return result.returncode, result.stdout + result.stderr


class TestCoreNoDocstringViolations:
  """Assert ruff check src/autopilot/core/ --select D,DOC exits 0."""

  def test_core_no_docstring_violations(self) -> None:
    code, output = _ruff_docstring_errors(str(SRC / 'core'))
    assert code == 0, f'core/ has D/DOC violations:\n{output}'


class TestAINoDocstringViolations:
  """Assert ruff check src/autopilot/ai/ --select D,DOC exits 0."""

  def test_ai_no_docstring_violations(self) -> None:
    code, output = _ruff_docstring_errors(str(SRC / 'ai'))
    assert code == 0, f'ai/ has D/DOC violations:\n{output}'


class TestCLINoDocstringViolations:
  """Assert ruff check src/autopilot/cli/ --select D,DOC exits 0."""

  def test_cli_no_docstring_violations(self) -> None:
    code, output = _ruff_docstring_errors(str(SRC / 'cli'))
    assert code == 0, f'cli/ has D/DOC violations:\n{output}'


class TestDataNoDocstringViolations:
  """Assert ruff check src/autopilot/data/ --select D,DOC exits 0."""

  def test_data_no_docstring_violations(self) -> None:
    code, output = _ruff_docstring_errors(str(SRC / 'data'))
    assert code == 0, f'data/ has D/DOC violations:\n{output}'


class TestTrackingNoDocstringViolations:
  """Assert ruff check src/autopilot/tracking/ --select D,DOC exits 0."""

  def test_tracking_no_docstring_violations(self) -> None:
    code, output = _ruff_docstring_errors(str(SRC / 'tracking'))
    assert code == 0, f'tracking/ has D/DOC violations:\n{output}'


class TestPolicyNoDocstringViolations:
  """Assert ruff check src/autopilot/policy/ --select D,DOC exits 0."""

  def test_policy_no_docstring_violations(self) -> None:
    code, output = _ruff_docstring_errors(str(SRC / 'policy'))
    assert code == 0, f'policy/ has D/DOC violations:\n{output}'


class TestAllDocstringRulesClean:
  """Assert ruff check src/ --select D,DOC exits 0 with 0 errors."""

  def test_all_docstring_rules_clean(self) -> None:
    code, output = _ruff_docstring_errors(str(SRC.parent))
    assert code == 0, f'src/ has D/DOC violations:\n{output}'
