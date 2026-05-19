"""AP-1: Enforce read-only CLI tests use run_cli_no_context.

Meta-test scanning tests/cli/ to verify that files not on the allowlist
do not import or call ``run_cli`` (the mutating helper that injects
``--context 'test'``).  Using ``run_cli`` for context-exempt commands
masks enforcement bugs where a read-only command accidentally requires
``--context``.

See plan 22, subplan 2.1.
"""

from pathlib import Path

CLI_TESTS_DIR = Path(__file__).parent
ALLOWLIST_FILE = CLI_TESTS_DIR / 'run_cli_allowlist.txt'


def _load_allowlist() -> frozenset[str]:
  """Load the allowlist of files permitted to use run_cli."""
  lines = ALLOWLIST_FILE.read_text(encoding='utf-8').splitlines()
  return frozenset(
    line.strip() for line in lines if line.strip() and not line.strip().startswith('#')
  )


def _file_uses_run_cli_for_exempt_only(content: str) -> bool:
  """Return True if file uses run_cli( without run_cli_no_context."""
  has_run_cli = 'run_cli(' in content
  has_no_context = 'run_cli_no_context(' in content
  if not has_run_cli:
    return False
  if has_no_context:
    return False
  for line in content.splitlines():
    if not line.startswith('from tests.cli.conftest import'):
      continue
    if 'run_cli' in line and 'run_cli_no_context' not in line and line.strip() != 'run_cli_text':
      return True
  return has_run_cli


def test_read_only_cli_uses_run_cli_no_context() -> None:
  """Files not on the allowlist must not use run_cli (the mutating helper).

  Scans all ``test_*.py`` files under ``tests/cli/`` and verifies that
  only allowlisted files import or call ``run_cli(``.  Non-allowlisted
  files should use ``run_cli_no_context`` for read-only / exempt commands.
  """
  allowlist = _load_allowlist()
  violations: list[str] = []

  for test_file in sorted(CLI_TESTS_DIR.glob('test_*.py')):
    if test_file.name in allowlist or test_file.name == Path(__file__).name:
      continue
    content = test_file.read_text(encoding='utf-8')
    if _file_uses_run_cli_for_exempt_only(content):
      violations.append(test_file.name)

  assert violations == [], (
    f'these test files use run_cli() but are not on the allowlist '
    f'(tests/cli/run_cli_allowlist.txt). Read-only / exempt command tests '
    f'should use run_cli_no_context() instead: {violations}'
  )
