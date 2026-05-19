"""Tests for tree CLI commands: list, create, show, switch."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.forest import Forest, validate_tree_name
from autopilot.core.store.base import Store
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_text, seed_tree_with_experiments
from unittest.mock import MagicMock
import contextlib
import io
import json
import pytest


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def seeded_ws(ws: Path) -> Path:
  """Workspace with a tree containing experiments."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {
        'id': 'baseline',
        'hypothesis': 'default prompts',
        'status': 'completed',
        'metrics': {'accuracy': 0.72},
      },
      {
        'id': 'cot',
        'hypothesis': 'chain of thought',
        'status': 'completed',
        'metrics': {'accuracy': 0.78},
        'parent': 'baseline',
        'baseline': 'baseline',
      },
    ],
  )
  return ws


class TestTreeList:
  def test_empty_forest(self, ws: Path) -> None:
    result = run_cli(ws, ['tree', 'list'])
    assert result['ok'] is True
    assert result['result']['trees'] == []

  def test_multiple_trees(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('tree-a', description='first tree')
    forest.create_tree('tree-b', description='second tree')
    forest.switch('tree-a')
    forest.save()

    result = run_cli(ws, ['tree', 'list'])
    trees = result['result']['trees']
    assert len(trees) == 2
    names = {t['name'] for t in trees}
    assert names == {'tree-a', 'tree-b'}
    active_tree = next(t for t in trees if t['active'])
    assert active_tree['name'] == 'tree-a'

  def test_json_valid(self, ws: Path) -> None:
    result = run_cli(ws, ['tree', 'list'])
    assert 'ok' in result
    assert 'result' in result

  def test_text_output(self, ws: Path) -> None:
    text = run_cli_text(ws, ['tree', 'list'])
    assert '(none)' in text or 'name' in text.lower()


class TestTreeCreate:
  def test_creates_tree(self, ws: Path) -> None:
    result = run_cli(ws, ['tree', 'create', 'my-tree'])
    assert result['ok'] is True
    assert result['result']['ok'] is True
    assert result['result']['tree'] == 'my-tree'

    result2 = run_cli(ws, ['tree', 'list'])
    trees = result2['result']['trees']
    assert len(trees) == 1
    assert trees[0]['name'] == 'my-tree'

  def test_create_with_description(self, ws: Path) -> None:
    result = run_cli(ws, ['tree', 'create', 'desc-tree', '--description', 'test desc'])
    assert result['result']['ok'] is True

    list_result = run_cli(ws, ['tree', 'list'])
    trees = list_result['result']['trees']
    assert trees[0]['description'] == 'test desc'

  def test_duplicate_name_error(self, ws: Path) -> None:
    run_cli(ws, ['tree', 'create', 'dup-tree'])
    with pytest.raises(SystemExit):
      run_cli(ws, ['tree', 'create', 'dup-tree'])

  def test_create_sets_active(self, ws: Path) -> None:
    run_cli(ws, ['tree', 'create', 'active-tree'])
    result = run_cli(ws, ['tree', 'list'])
    trees = result['result']['trees']
    assert trees[0]['active'] is True

  def test_json_output(self, ws: Path) -> None:
    result = run_cli(ws, ['tree', 'create', 'json-tree'])
    assert 'ok' in result
    assert result['result']['ok'] is True

  def test_tree_create_cli_rejects_empty_name(self, ws: Path) -> None:
    """tree create '' exits non-zero; JSON ok is False; error mentions empty/whitespace."""
    from tests.cli.conftest import build_context, build_parser

    parser = build_parser()
    full_argv = ['tree', 'create', '', '--workspace', str(ws), '--json', '--context', 'test']
    parsed = parser.parse_args(full_argv)
    ctx = build_context(parsed)

    buf = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stdout(buf):
      parsed.handler(ctx, parsed)

    output = buf.getvalue()
    envelope = json.loads(output)
    assert envelope['ok'] is False
    error_lower = envelope['error'].lower()
    assert 'empty' in error_lower or 'whitespace' in error_lower

  def test_tree_create_cli_rejects_whitespace_only_name(self, ws: Path) -> None:
    """tree create '  ' exits non-zero; no trees created."""
    with pytest.raises(SystemExit):
      run_cli(ws, ['tree', 'create', '  '])
    result = run_cli(ws, ['tree', 'list'])
    assert result['result']['trees'] == []

  def test_tree_create_cli_rejects_leading_trailing_whitespace(self, ws: Path) -> None:
    """tree create ' mytree' exits non-zero (leading whitespace invalid)."""
    with pytest.raises(SystemExit):
      run_cli(ws, ['tree', 'create', ' mytree'])

  def test_tree_create_cli_valid_name_succeeds(self, ws: Path) -> None:
    """Valid name with allowed chars creates tree successfully."""
    result = run_cli(ws, ['tree', 'create', 'valid-tree_1.0'])
    assert result['result']['ok'] is True
    assert result['result']['tree'] == 'valid-tree_1.0'


class TestTreeShow:
  def test_renders_populated_tree(self, seeded_ws: Path) -> None:
    text = run_cli_text(seeded_ws, ['tree', 'show'])
    assert 'baseline' in text
    assert 'cot' in text

  def test_renders_by_name(self, seeded_ws: Path) -> None:
    text = run_cli_text(seeded_ws, ['tree', 'show', 'main'])
    assert 'main' in text

  def test_json_returns_dict(self, seeded_ws: Path) -> None:
    result = run_cli(seeded_ws, ['tree', 'show'])
    tree_data = result['result']
    assert tree_data['name'] == 'main'
    assert 'nodes' in tree_data

  def test_nonexistent_name_error(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    FileStore(config)
    with pytest.raises(SystemExit):
      run_cli(ws, ['tree', 'show', 'nonexistent'])

  def test_no_active_tree_error(self, ws: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli(ws, ['tree', 'show'])


class TestTreeSwitch:
  def test_switches_active(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('a')
    forest.create_tree('b')
    forest.save()

    result = run_cli(ws, ['tree', 'switch', 'b', '--no-checkout'])
    assert result['result']['ok'] is True
    assert result['result']['active'] == 'b'

    list_result = run_cli(ws, ['tree', 'list'])
    active = [t for t in list_result['result']['trees'] if t['active']]
    assert active[0]['name'] == 'b'

  def test_nonexistent_error(self, ws: Path) -> None:
    with pytest.raises(Exception, match='not found'):
      run_cli(ws, ['tree', 'switch', 'ghost', '--no-checkout'])

  def test_json_output(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('sw')
    forest.save()

    result = run_cli(ws, ['tree', 'switch', 'sw', '--no-checkout'])
    assert 'ok' in result
    assert result['result']['ok'] is True


class TestTreeNameValidation:
  """Tests for validate_tree_name and Forest.create_tree name validation."""

  def _make_forest(self) -> Forest:
    store = MagicMock(spec=Store)
    return Forest(store=store)

  def test_tree_create_rejects_empty_name(self) -> None:
    forest = self._make_forest()
    with pytest.raises(ValueError, match='name'):
      forest.create_tree('')

  def test_tree_create_rejects_whitespace_name(self) -> None:
    forest = self._make_forest()
    with pytest.raises(ValueError, match='name'):
      forest.create_tree('  ')

  def test_tree_create_rejects_slash(self) -> None:
    forest = self._make_forest()
    with pytest.raises(ValueError, match='invalid characters'):
      forest.create_tree('a/b')

  def test_tree_create_rejects_space(self) -> None:
    forest = self._make_forest()
    with pytest.raises(ValueError, match='invalid characters'):
      forest.create_tree('a b')

  def test_tree_create_accepts_valid_chars(self) -> None:
    forest = self._make_forest()
    tree = forest.create_tree('my-tree_v1.0')
    assert tree.name == 'my-tree_v1.0'

  def test_tree_create_leading_dot_rejected(self) -> None:
    forest = self._make_forest()
    with pytest.raises(ValueError, match='must not start or end'):
      forest.create_tree('.hidden')

  def test_tree_create_double_dot_rejected(self) -> None:
    forest = self._make_forest()
    with pytest.raises(ValueError, match='must not start or end'):
      forest.create_tree('..')

  def test_tree_create_embedded_double_dot_rejected(self) -> None:
    forest = self._make_forest()
    with pytest.raises(ValueError, match='must not contain'):
      forest.create_tree('a..b')

  def test_tree_create_exactly_128_chars_accepted(self) -> None:
    forest = self._make_forest()
    name = 'a' * 128
    tree = forest.create_tree(name)
    assert tree.name == name

  def test_tree_create_leading_trailing_hyphen_accepted(self) -> None:
    forest = self._make_forest()
    tree = forest.create_tree('-name-')
    assert tree.name == '-name-'

  def test_tree_create_rejects_too_long(self) -> None:
    forest = self._make_forest()
    name = 'a' * 129
    with pytest.raises(ValueError, match='exceeds'):
      forest.create_tree(name)

  def test_tree_create_tab_in_name_rejected(self) -> None:
    forest = self._make_forest()
    with pytest.raises(ValueError, match='invalid characters'):
      forest.create_tree('na\tme')

  def test_tree_create_newline_rejected(self) -> None:
    forest = self._make_forest()
    with pytest.raises(ValueError, match='invalid characters'):
      forest.create_tree('na\nme')

  def test_validate_tree_name_standalone(self) -> None:
    validate_tree_name('ok')
    validate_tree_name('a' * 128)
    with pytest.raises(ValueError, match='name'):
      validate_tree_name('')
