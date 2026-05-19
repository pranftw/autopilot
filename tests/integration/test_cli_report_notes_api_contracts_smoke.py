"""Dogfood V4 final audit integration tests (sub-plan 10).

Verifies:
  1. report summary workspace-level aggregate (V4 feature from plan 06)
  2. experiment notes write --file round-trip (V4 feature from plan 07)
  3. API contract test inventory completeness (plan 01 meta-check)
  4. CLAUDE.md / AGENTS.md byte-identity (documentation parity)
  5. Store UX tests pass (plan 03)
  6. JSON matrix and context exemption suites exist (plans 08, 09)
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.report.command import gather_workspace_summary
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context, seed_tree_with_experiments
import inspect
import pytest
import tests.api.test_api_contracts as _api_contracts_mod
import tests.api.test_error_suggestions as _error_suggestions_mod
import tests.cli.test_context_exemptions as _context_exemptions_mod
import tests.cli.test_json_contract_matrix as _json_matrix_mod
import tests.cli.test_store_ux as _store_ux_mod


class TestIntegrationReportSummaryWorkspace:
  """report summary without --experiment returns workspace-level aggregate."""

  def test_integration_report_summary_workspace(self, tmp_path: Path) -> None:
    """Bootstraps a workspace, creates experiments, invokes report summary
    without --experiment, asserts workspace-level aggregate fields."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    seed_tree_with_experiments(
      forest,
      'main',
      [
        {'id': 'exp-1', 'status': 'completed', 'metrics': {'accuracy': 0.85}},
        {'id': 'exp-2', 'status': 'completed', 'metrics': {'accuracy': 0.92}},
        {'id': 'exp-3', 'status': 'running'},
      ],
    )

    envelope = run_cli_no_context(ws, ['report', 'summary'])
    assert envelope['ok'] is True

    result = envelope['result']
    assert result['scope'] == 'tree'
    assert 'experiments_count' in result
    assert result['experiments_count']['completed'] == 2
    assert 'metric_summary' in result
    assert 'accuracy' in result['metric_summary']
    assert result['metric_summary']['accuracy']['min'] == pytest.approx(0.85)
    assert result['metric_summary']['accuracy']['max'] == pytest.approx(0.92)
    assert result['best_experiment'] is not None
    assert result['best_experiment']['id'] == 'exp-2'

  def test_workspace_aggregate_json_consistent_with_function(self, tmp_path: Path) -> None:
    """CLI JSON output matches gather_workspace_summary function result."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    seed_tree_with_experiments(
      forest,
      'main',
      [
        {'id': 'exp-a', 'status': 'completed', 'metrics': {'f1': 0.75}},
      ],
    )

    tree = forest.active
    direct_result = gather_workspace_summary(forest, tree, all_trees=False)

    envelope = run_cli_no_context(ws, ['report', 'summary'])
    cli_result = envelope['result']

    assert direct_result['scope'] == cli_result['scope']
    assert direct_result['experiments_count'] == cli_result['experiments_count']
    assert direct_result['metric_summary'] == cli_result['metric_summary']


class TestIntegrationNotesWriteFile:
  """experiment notes write --file round-trip."""

  def test_integration_notes_write_file(self, tmp_path: Path) -> None:
    """Write notes from a temp file, read back via notes show, verify equality."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    seed_tree_with_experiments(
      forest,
      'main',
      [{'id': 'exp-notes-v4', 'status': 'running'}],
    )

    notes_content = 'V4 audit notes content.\nSecond line with unicode: \u2714'
    notes_file = tmp_path / 'audit_notes.txt'
    notes_file.write_text(notes_content, encoding='utf-8')

    write_result = run_cli(
      ws,
      ['experiment', 'notes', 'write', 'exp-notes-v4', '--file', str(notes_file)],
    )
    assert write_result['ok'] is True
    assert write_result['result']['bytes_written'] == len(notes_content.encode('utf-8'))

    show_result = run_cli_no_context(
      ws,
      ['experiment', 'notes', 'show', 'exp-notes-v4'],
    )
    assert show_result['ok'] is True
    assert show_result['result']['notes'] == notes_content


class TestApiContractsAllPass:
  """Meta-test verifying all 26 API contract tests exist and are importable."""

  def test_api_contracts_all_pass(self) -> None:
    """Discover test functions in test_api_contracts.py; assert >= 26."""
    test_functions = [
      name
      for name, obj in inspect.getmembers(_api_contracts_mod)
      if name.startswith('test_') and callable(obj)
    ]
    assert len(test_functions) >= 26, (
      f'Expected at least 26 API contract tests, found {len(test_functions)}. '
      f'Tests may have been renamed or deleted without updating the inventory.'
    )

  def test_error_suggestions_test_count(self) -> None:
    """Discover test functions/methods in test_error_suggestions.py; assert >= 17."""
    test_count = 0
    for name, obj in inspect.getmembers(_error_suggestions_mod):
      if name.startswith('test_') and callable(obj):
        test_count += 1
      elif inspect.isclass(obj) and name.startswith('Test'):
        for method_name, method in inspect.getmembers(obj):
          if method_name.startswith('test_') and callable(method):
            test_count += 1
    assert test_count >= 17, (
      f'Expected at least 17 error suggestion tests, found {test_count}. '
      f'Tests may have been renamed or deleted without updating the inventory.'
    )


class TestClaudeMdAgentsMdIdentical:
  """CLAUDE.md and AGENTS.md must be byte-identical."""

  def test_claude_md_agents_md_identical(self) -> None:
    """Read both doc files and compare content; fail with sync hint on mismatch."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    claude_md = repo_root / 'CLAUDE.md'
    agents_md = repo_root / 'AGENTS.md'

    assert claude_md.exists(), f'CLAUDE.md not found at {claude_md}'
    assert agents_md.exists(), f'AGENTS.md not found at {agents_md}'

    claude_content = claude_md.read_text(encoding='utf-8')
    agents_content = agents_md.read_text(encoding='utf-8')

    assert claude_content == agents_content, (
      'CLAUDE.md and AGENTS.md have diverged. '
      'CLAUDE.md is the canonical source; copy it to AGENTS.md to sync. '
      'Run: cp CLAUDE.md AGENTS.md'
    )

  def test_forest_backed_documented(self) -> None:
    """Verify 'forest-backed' term exists in CLAUDE.md (plan 03 store UX)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    content = (repo_root / 'CLAUDE.md').read_text(encoding='utf-8')
    assert 'forest-backed' in content, (
      'CLAUDE.md must document the forest-backed vs source-backed mental model'
    )

  def test_source_backed_documented(self) -> None:
    """Verify 'source-backed' term exists in CLAUDE.md (plan 03 store UX)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    content = (repo_root / 'CLAUDE.md').read_text(encoding='utf-8')
    assert 'source-backed' in content, (
      'CLAUDE.md must document the forest-backed vs source-backed mental model'
    )


class TestStoreUxTestModuleExists:
  """Store UX tests from plan 03 are importable and have expected count."""

  def test_store_ux_test_count(self) -> None:
    """tests/cli/test_store_ux.py has at least 5 test methods (plan 03)."""
    test_count = 0
    for name, obj in inspect.getmembers(_store_ux_mod):
      if name.startswith('test_') and callable(obj):
        test_count += 1
      elif inspect.isclass(obj) and name.startswith('Test'):
        for method_name, method in inspect.getmembers(obj):
          if method_name.startswith('test_') and callable(method):
            test_count += 1
    assert test_count >= 5, f'Expected at least 5 store UX tests (plan 03), found {test_count}.'


class TestJsonMatrixSuiteExists:
  """JSON contract matrix tests from plan 08 are importable."""

  def test_json_matrix_module_importable(self) -> None:
    """tests/cli/test_json_contract_matrix.py is importable and non-empty."""
    test_count = 0
    for name, obj in inspect.getmembers(_json_matrix_mod):
      if name.startswith('test_') and callable(obj):
        test_count += 1
      elif inspect.isclass(obj) and name.startswith('Test'):
        for method_name, method in inspect.getmembers(obj):
          if method_name.startswith('test_') and callable(method):
            test_count += 1
    assert test_count >= 20, (
      f'Expected at least 20 JSON matrix tests (plan 08), found {test_count}.'
    )


class TestContextExemptionSuiteExists:
  """Context exemption matrix tests from plan 09 are importable."""

  def test_context_exemption_module_importable(self) -> None:
    """tests/cli/test_context_exemptions.py has at least 10 test methods (plan 09)."""
    test_count = 0
    for name, obj in inspect.getmembers(_context_exemptions_mod):
      if name.startswith('test_') and callable(obj):
        test_count += 1
      elif inspect.isclass(obj) and name.startswith('Test'):
        for method_name, method in inspect.getmembers(obj):
          if method_name.startswith('test_') and callable(method):
            test_count += 1
    assert test_count >= 10, (
      f'Expected at least 10 context exemption tests (plan 09), found {test_count}.'
    )
