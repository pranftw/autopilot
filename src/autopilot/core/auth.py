"""Generic auth validation utilities.

Preflight check functions for common auth providers (AWS, GitHub CLI).
Projects compose these into their own preflight checks as needed.
"""

from pathlib import Path
from typing import Any
import subprocess


def check_aws_auth_mount(aws_dir: Path | None = None) -> list[str]:
  """Verify host AWS auth directory exists with credentials or config."""
  aws_dir = aws_dir or Path.home() / '.aws'
  errors: list[str] = []
  if not aws_dir.is_dir():
    errors.append(f'AWS auth directory not found: {aws_dir}')
    return errors
  has_creds = (aws_dir / 'credentials').is_file()
  has_config = (aws_dir / 'config').is_file()
  if not has_creds and not has_config:
    errors.append(f'no credentials or config found in {aws_dir}')
  return errors


def check_gh_auth_mount(gh_dir: Path | None = None) -> list[str]:
  """Verify host GitHub CLI auth directory exists with hosts.yml."""
  gh_dir = gh_dir or Path.home() / '.config' / 'gh'
  errors: list[str] = []
  if not gh_dir.is_dir():
    errors.append(f'GitHub CLI auth directory not found: {gh_dir}')
    return errors
  if not (gh_dir / 'hosts.yml').is_file():
    errors.append(f'hosts.yml not found in {gh_dir}')
  return errors


def check_gh_auth_status() -> list[str]:
  """Run gh auth status and check it succeeds."""
  errors: list[str] = []
  try:
    result = subprocess.run(
      ['gh', 'auth', 'status'],
      capture_output=True,
      text=True,
      timeout=15,
    )
    if result.returncode != 0:
      stderr = result.stderr.strip() or 'unknown error'
      errors.append(f'gh auth status failed: {stderr}')
  except FileNotFoundError:
    errors.append('gh CLI not found on PATH')
  except subprocess.TimeoutExpired:
    errors.append('gh auth status timed out')
  return errors


def check_aws_sts_identity(region: str = 'ap-south-1') -> list[str]:
  """Run aws sts get-caller-identity and check it succeeds."""
  errors: list[str] = []
  try:
    result = subprocess.run(
      ['aws', 'sts', 'get-caller-identity', '--region', region],
      capture_output=True,
      text=True,
      timeout=15,
    )
    if result.returncode != 0:
      stderr = result.stderr.strip() or 'unknown error'
      errors.append(f'aws sts get-caller-identity failed: {stderr}')
  except FileNotFoundError:
    errors.append('aws CLI not found on PATH')
  except subprocess.TimeoutExpired:
    errors.append('aws sts get-caller-identity timed out')
  return errors


def check_binary_on_path(binary: str) -> list[str]:
  """Verify a binary is available on PATH."""
  errors: list[str] = []
  try:
    subprocess.run(
      [binary, '--version'],
      capture_output=True,
      timeout=10,
    )
  except FileNotFoundError:
    errors.append(f'{binary} not found on PATH')
  except subprocess.TimeoutExpired:
    errors.append(f'{binary} --version timed out')
  return errors


def check_file_exists(path: str | Path, label: str = '') -> list[str]:
  """Verify a file exists at the given path."""
  p = Path(path)
  if not p.is_file():
    name = label or str(p)
    return [f'file not found: {name}']
  return []


def check_dir_exists(path: str | Path, label: str = '') -> list[str]:
  """Verify a directory exists at the given path."""
  p = Path(path)
  if not p.is_dir():
    name = label or str(p)
    return [f'directory not found: {name}']
  return []


def run_preflight_checks(
  checks: dict[str, list[str]],
) -> dict[str, Any]:
  """Aggregate preflight check results.

  Takes a dict mapping check names to their error lists.
  Returns structured results with per-check status and overall pass/fail.
  """
  all_errors: list[str] = []
  for check_name, errs in checks.items():
    for err in errs:
      all_errors.append(f'[{check_name}] {err}')
  return {
    'checks': {k: {'ok': len(v) == 0, 'errors': v} for k, v in checks.items()},
    'all_errors': all_errors,
    'passed': len(all_errors) == 0,
  }
