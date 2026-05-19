"""CLI tests for experiment metadata subcommands and query --metadata-contains.

Includes cross-tree metadata resolution tests (plan 02).
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
from unittest.mock import patch
import pytest


def _setup_experiment(cli_workspace: Path, cli_forest) -> str:
  """Create a tree with one experiment and return its id."""
  tree = cli_forest.create_tree('main')
  exp = Experiment(experiment_id='exp-meta-01', hypothesis='metadata test')
  exp.start()
  exp.complete(metrics={'accuracy': 0.85})
  tree.add(Node(experiment=exp))
  cli_forest.switch('main')
  cli_forest.save()
  return 'exp-meta-01'


class TestMetadataSetGet:
  """CLI metadata set then get returns value."""

  def test_metadata_set_get(self, cli_workspace: Path, cli_forest) -> None:
    """Set then get returns the value in JSON mode."""
    exp_id = _setup_experiment(cli_workspace, cli_forest)

    set_result = run_cli(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'set',
        exp_id,
        'env',
        'production',
      ],
    )
    assert set_result['ok'] is True
    assert set_result['result']['key'] == 'env'
    assert set_result['result']['value'] == 'production'

    get_result = run_cli_no_context(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'get',
        exp_id,
        'env',
      ],
    )
    assert get_result['ok'] is True
    assert get_result['result']['value'] == 'production'

  def test_metadata_show(self, cli_workspace: Path, cli_forest) -> None:
    """Show lists all pairs in JSON."""
    exp_id = _setup_experiment(cli_workspace, cli_forest)

    run_cli(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'set',
        exp_id,
        'tag',
        'v1',
      ],
    )
    run_cli(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'set',
        exp_id,
        'env',
        'staging',
      ],
    )

    result = run_cli_no_context(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'show',
        exp_id,
      ],
    )
    assert result['ok'] is True
    metadata = result['result']['metadata']
    assert metadata['tag'] == 'v1'
    assert metadata['env'] == 'staging'

  def test_metadata_overwrite(self, cli_workspace: Path, cli_forest) -> None:
    """Second set replaces prior value."""
    exp_id = _setup_experiment(cli_workspace, cli_forest)

    run_cli(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'set',
        exp_id,
        'key',
        'first',
      ],
    )
    run_cli(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'set',
        exp_id,
        'key',
        'second',
      ],
    )

    result = run_cli_no_context(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'get',
        exp_id,
        'key',
      ],
    )
    assert result['result']['value'] == 'second'


class TestMetadataJson:
  """JSON envelope for each metadata subcommand."""

  def test_metadata_json_set(self, cli_workspace: Path, cli_forest) -> None:
    """Set returns proper JSON envelope."""
    exp_id = _setup_experiment(cli_workspace, cli_forest)
    result = run_cli(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'set',
        exp_id,
        'k',
        'v',
      ],
    )
    assert 'ok' in result
    assert 'result' in result
    assert result['result']['experiment_id'] == exp_id

  def test_metadata_json_get(self, cli_workspace: Path, cli_forest) -> None:
    """Get returns proper JSON envelope."""
    exp_id = _setup_experiment(cli_workspace, cli_forest)
    result = run_cli_no_context(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'get',
        exp_id,
        'missing',
      ],
    )
    assert result['ok'] is True
    assert result['result']['value'] is None

  def test_metadata_json_show(self, cli_workspace: Path, cli_forest) -> None:
    """Show returns proper JSON envelope."""
    exp_id = _setup_experiment(cli_workspace, cli_forest)
    result = run_cli_no_context(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'show',
        exp_id,
      ],
    )
    assert result['ok'] is True
    assert 'metadata' in result['result']


class TestMetadataContextEnforcement:
  """Context enforcement for metadata subcommands."""

  def test_metadata_requires_context(self) -> None:
    """metadata set requires --context per CLI rules."""
    from autopilot.cli.command import CLI

    cli = CLI()
    assert cli.requires_context('experiment metadata set') is True

  def test_metadata_show_exempt(self, cli_workspace: Path, cli_forest) -> None:
    """metadata show does not require context (read-only)."""
    from autopilot.cli.command import CLI

    cli = CLI()
    assert cli.requires_context('experiment metadata show') is False

    exp_id = _setup_experiment(cli_workspace, cli_forest)
    result = run_cli_no_context(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'show',
        exp_id,
      ],
    )
    assert result['ok'] is True

  def test_metadata_get_exempt(self, cli_workspace: Path, cli_forest) -> None:
    """metadata get does not require context (read-only)."""
    from autopilot.cli.command import CLI

    cli = CLI()
    assert cli.requires_context('experiment metadata get') is False

    exp_id = _setup_experiment(cli_workspace, cli_forest)
    result = run_cli_no_context(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'get',
        exp_id,
        'key',
      ],
    )
    assert result['ok'] is True


class TestMetadataEmpty:
  """Empty metadata behaviors."""

  def test_metadata_empty_show(self, cli_workspace: Path, cli_forest) -> None:
    """metadata show with no file returns empty JSON dict, exit 0."""
    exp_id = _setup_experiment(cli_workspace, cli_forest)
    result = run_cli_no_context(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'show',
        exp_id,
      ],
    )
    assert result['ok'] is True
    assert result['result']['metadata'] == {}


class TestMetadataJsonSchema:
  """Stable keys in --json mode."""

  def test_metadata_json_schema_set(self, cli_workspace: Path, cli_forest) -> None:
    """Set JSON includes experiment_id, key, value."""
    exp_id = _setup_experiment(cli_workspace, cli_forest)
    result = run_cli(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'set',
        exp_id,
        'k',
        'v',
      ],
    )
    payload = result['result']
    assert 'experiment_id' in payload
    assert 'key' in payload
    assert 'value' in payload

  def test_metadata_json_schema_show(self, cli_workspace: Path, cli_forest) -> None:
    """Show JSON includes experiment_id, metadata."""
    exp_id = _setup_experiment(cli_workspace, cli_forest)
    result = run_cli_no_context(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'show',
        exp_id,
      ],
    )
    payload = result['result']
    assert 'experiment_id' in payload
    assert 'metadata' in payload


class TestMetadataCliExitCodes:
  """Success vs failure exit codes."""

  def test_metadata_cli_exit_codes_success(self, cli_workspace: Path, cli_forest) -> None:
    """Successful operations return ok=True."""
    exp_id = _setup_experiment(cli_workspace, cli_forest)
    result = run_cli(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'set',
        exp_id,
        'k',
        'v',
      ],
    )
    assert result['ok'] is True

  def test_metadata_set_requires_context_via_requires_context(self) -> None:
    """CLI.requires_context confirms metadata set is mutating."""
    from autopilot.cli.command import CLI

    cli = CLI()
    assert cli.requires_context('experiment metadata set') is True


class TestMetadataContainsQuery:
  """query --metadata-contains tests."""

  def test_metadata_contains_no_match(self, cli_workspace: Path, cli_forest) -> None:
    """Query with filter yields empty list, exit 0."""
    _setup_experiment(cli_workspace, cli_forest)
    result = run_cli_no_context(
      cli_workspace,
      [
        'query',
        '--metadata-contains',
        'env:prod',
      ],
    )
    assert result['ok'] is True
    assert result['result']['experiments'] == []

  def test_metadata_contains_with_match(self, cli_workspace: Path, cli_forest) -> None:
    """Query returns matching experiment after metadata set."""
    exp_id = _setup_experiment(cli_workspace, cli_forest)
    run_cli(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'set',
        exp_id,
        'env',
        'prod',
      ],
    )
    result = run_cli_no_context(
      cli_workspace,
      [
        'query',
        '--metadata-contains',
        'env:prod',
      ],
    )
    assert result['ok'] is True
    ids = [r['id'] for r in result['result']['experiments']]
    assert exp_id in ids

  def test_metadata_contains_colon_in_value(self, cli_workspace: Path, cli_forest) -> None:
    """URL-style value with colons parses and matches."""
    exp_id = _setup_experiment(cli_workspace, cli_forest)
    run_cli(
      cli_workspace,
      [
        'experiment',
        'metadata',
        'set',
        exp_id,
        'url',
        'https://example.com',
      ],
    )
    result = run_cli_no_context(
      cli_workspace,
      [
        'query',
        '--metadata-contains',
        'url:https://example.com',
      ],
    )
    assert result['ok'] is True
    ids = [r['id'] for r in result['result']['experiments']]
    assert exp_id in ids


def _setup_cross_tree_workspace(ws: Path) -> str:
  """Create a workspace with experiment on non-active tree, return experiment id."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  forest.create_tree('alpha')
  tree_beta = forest.create_tree('beta')
  exp = Experiment(experiment_id='exp-cross', hypothesis='cross-tree')
  exp.start()
  exp.complete(metrics={'f1': 0.9})
  tree_beta.add(Node(experiment=exp))
  forest.switch('alpha')
  forest.save()
  return 'exp-cross'


@pytest.fixture
def cross_tree_workspace(tmp_path: Path) -> Path:
  """Workspace with experiment on non-active tree."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  _setup_cross_tree_workspace(ws)
  return ws


class TestMetadataCrossTree:
  """Cross-tree metadata resolution tests (plan 02, section 4.2)."""

  @pytest.fixture(autouse=True)
  def _patch_checkout(self):
    """Patch FileStore.checkout for tests that don't create snapshots."""
    with patch('autopilot.ai.store.file_store.FileStore.checkout'):
      yield

  def test_metadata_get_cross_tree(self, cross_tree_workspace: Path) -> None:
    """metadata get succeeds when id exists off active tree."""
    run_cli(
      cross_tree_workspace,
      ['experiment', 'metadata', 'set', 'exp-cross', 'env', 'staging'],
    )
    result = run_cli_no_context(
      cross_tree_workspace,
      ['experiment', 'metadata', 'get', 'exp-cross', 'env'],
    )
    assert result['ok'] is True
    assert result['result']['value'] == 'staging'
    assert result['result']['experiment_id'] == 'exp-cross'

  def test_metadata_show_cross_tree(self, cross_tree_workspace: Path) -> None:
    """metadata show succeeds off active tree."""
    run_cli(
      cross_tree_workspace,
      ['experiment', 'metadata', 'set', 'exp-cross', 'tag', 'v2'],
    )
    result = run_cli_no_context(
      cross_tree_workspace,
      ['experiment', 'metadata', 'show', 'exp-cross'],
    )
    assert result['ok'] is True
    assert result['result']['metadata']['tag'] == 'v2'

  def test_metadata_set_cross_tree(self, cross_tree_workspace: Path) -> None:
    """metadata set succeeds for experiment on non-active tree."""
    result = run_cli(
      cross_tree_workspace,
      ['experiment', 'metadata', 'set', 'exp-cross', 'model', 'gpt-4'],
    )
    assert result['ok'] is True
    assert result['result']['key'] == 'model'
    assert result['result']['value'] == 'gpt-4'
