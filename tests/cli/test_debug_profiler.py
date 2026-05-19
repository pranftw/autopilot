"""CLI tests for debug profiler command."""

from pathlib import Path
from tests.cli.conftest import run_cli_no_context
import json
import pytest


def _seed_profiler_summary(workspace: Path, experiment_id: str) -> None:
  """Write a profiler_summary.json for the given experiment."""
  exp_dir = workspace / '.autopilot' / 'experiments' / experiment_id
  exp_dir.mkdir(parents=True, exist_ok=True)
  summary = {
    'training_step': {'count': 10, 'total_ms': 500.0, 'mean_ms': 50.0},
    'backward': {'count': 10, 'total_ms': 200.0, 'mean_ms': 20.0},
  }
  (exp_dir / 'profiler_summary.json').write_text(json.dumps(summary))


class TestDebugProfilerCliJson:
  """debug profiler --json: envelope ok, result schema includes parsed summary."""

  def test_debug_profiler_cli_json(self, cli_workspace: Path) -> None:
    exp_id = 'prof-exp'
    _seed_profiler_summary(cli_workspace, exp_id)

    result = run_cli_no_context(
      cli_workspace,
      ['--experiment', exp_id, 'debug', 'profiler'],
    )
    assert result['ok'] is True
    data = result['result']
    assert 'training_step' in data
    assert data['training_step']['count'] == 10
    assert data['backward']['total_ms'] == 200.0


class TestDebugProfilerExitCode:
  """No profiler_summary.json: CLI exits non-zero with actionable message."""

  def test_debug_profiler_exit_code(self, cli_workspace: Path) -> None:
    exp_id = 'missing-prof'
    exp_dir = cli_workspace / '.autopilot' / 'experiments' / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(SystemExit):
      run_cli_no_context(
        cli_workspace,
        ['--experiment', exp_id, 'debug', 'profiler'],
      )
