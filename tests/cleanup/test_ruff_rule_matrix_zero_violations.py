"""Tests for sub-plan 05: code quality, naming and style lint compliance."""

import subprocess


def _ruff_count(select: str, paths: str = 'src/ tests/') -> int:
  """Run ruff with given select and return violation count.

  Args:
    select: Comma-separated rule codes to check.
    paths: Space-separated paths to check.

  Returns:
    Number of violations found.
  """
  result = subprocess.run(
    ['uv', 'run', 'ruff', 'check', *paths.split(), '--select', select, '--output-format=json'],
    capture_output=True,
    text=True,
    check=False,
  )
  if result.returncode == 0:
    return 0
  import json

  return len(json.loads(result.stdout))


class TestRuffNamingRulesClean:
  """Section 2.1: A002, N806, N801."""

  def test_naming_clean(self) -> None:
    assert _ruff_count('A002,N806,N801') == 0


class TestRuffAsyncPrivateBoolLambdaPrint:
  """Section 2.2: RUF029, SLF001, FBT003, PLW0108, T201."""

  def test_async_clean(self) -> None:
    assert _ruff_count('RUF029') == 0

  def test_private_access_clean(self) -> None:
    assert _ruff_count('SLF001') == 0

  def test_boolean_trap_clean(self) -> None:
    assert _ruff_count('FBT003') == 0

  def test_lambda_clean(self) -> None:
    assert _ruff_count('PLW0108') == 0

  def test_print_clean(self) -> None:
    assert _ruff_count('T201', paths='src/') == 0


class TestRuffPerfSubprocessMembership:
  """Section 2.3: PERF401, PERF102, PLW1510, PLR6201."""

  def test_perf_clean(self) -> None:
    assert _ruff_count('PERF401,PERF102') == 0

  def test_subprocess_clean(self) -> None:
    assert _ruff_count('PLW1510') == 0

  def test_membership_clean(self) -> None:
    assert _ruff_count('PLR6201') == 0


class TestRuffTestSecurityTryArg:
  """Section 2.4: PT011, PT006, PT018, ARG, BLE001, S*, TRY*."""

  def test_pt_clean(self) -> None:
    assert _ruff_count('PT011,PT006,PT018') == 0

  def test_arg_clean(self) -> None:
    assert _ruff_count('ARG004,ARG001') == 0

  def test_ble_clean(self) -> None:
    assert _ruff_count('BLE001') == 0

  def test_try_clean(self) -> None:
    assert _ruff_count('TRY300,TRY004,TRY301') == 0


class TestRuffMisc:
  """Section 2.5: miscellaneous remaining rules."""

  def test_misc_clean(self) -> None:
    assert (
      _ruff_count(
        'FURB113,FURB118,FURB171,ISC004,RET503,PIE810,LOG014,'
        'PLR0124,PLW1514,PLW2901,PLW3201,RUF052,C416'
      )
      == 0
    )


class TestRuffAggregate:
  """Aggregate verification across all SP-05 rules."""

  def test_all_sp05_rules_clean(self) -> None:
    all_rules = (
      'A002,N806,N801,'
      'RUF029,SLF001,FBT003,PLW0108,T201,'
      'PERF401,PERF102,PLW1510,PLR6201,'
      'PT011,PT006,PT018,ARG004,ARG001,BLE001,TRY300,TRY004,TRY301,'
      'S311,S404,S603,S105,'
      'FURB113,FURB118,FURB171,ISC004,RET503,PIE810,LOG014,'
      'PLR0124,PLW1514,PLW2901,PLW3201,RUF052,C416'
    )
    assert _ruff_count(all_rules) == 0
