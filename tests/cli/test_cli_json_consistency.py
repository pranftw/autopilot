"""JSON consistency tests for all experiment-related CLI commands.

For every experiment-related command, verify:
- --json produces valid JSON
- JSON has {ok, result, messages} envelope
- Error cases also produce JSON envelope

Commands tested: tree list/create/show/switch, experiment add/status/compare,
query, checkout, stabilize, store snapshot/status/diff/log/worktree list/
worktree create.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import build_context
from autopilot.cli.main import AutoPilotCLI, build_parser
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from io import StringIO
from pathlib import Path
from tests.cli.conftest import run_cli, seed_tree_with_experiments
from unittest.mock import patch
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
def ws_with_experiments(ws: Path) -> Path:
  """Workspace with a tree, experiments, and store snapshots."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {
        'id': 'exp-a',
        'hypothesis': 'baseline approach',
        'status': 'completed',
        'metrics': {'accuracy': 0.72, 'latency': 120.0},
      },
      {
        'id': 'exp-b',
        'hypothesis': 'improved approach',
        'status': 'completed',
        'metrics': {'accuracy': 0.85, 'latency': 100.0},
        'parent': 'exp-a',
        'baseline': 'exp-a',
      },
    ],
  )
  return ws


def _assert_json_envelope(result: dict) -> None:
  """Assert the standard JSON envelope structure."""
  assert 'ok' in result, f'missing "ok" key in {result}'
  assert 'result' in result, f'missing "result" key in {result}'
  assert 'messages' in result, f'missing "messages" key in {result}'
  assert isinstance(result['ok'], bool)
  assert isinstance(result['messages'], list)


def _run_store_json(
  workspace: Path,
  store_action: str,
  source: Path,
  *,
  experiment: str = 'exp-store',
  store_dir: Path | None = None,
  extra_args: list[str] | None = None,
) -> dict:
  """Run a store subcommand with --json and capture output."""
  parser = build_parser()
  argv = [
    'store',
    store_action,
    '--context',
    'test',
    '--workspace',
    str(workspace),
    '--experiment',
    experiment,
    '--source',
    str(source),
    '--json',
  ]
  if store_dir:
    argv += ['--store', str(store_dir)]
  if extra_args:
    argv.extend(extra_args)
  args = parser.parse_args(argv)
  ctx = build_context(args)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    args.handler(ctx, args)
  output = buf.getvalue().strip()
  return json.loads(output)


# -- Tree commands --


class TestTreeListJsonEnvelope:
  def test_empty_forest(self, ws: Path) -> None:
    result = run_cli(ws, ['tree', 'list'])
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert isinstance(result['result']['trees'], list)

  def test_populated_forest(self, ws_with_experiments: Path) -> None:
    result = run_cli(ws_with_experiments, ['tree', 'list'])
    _assert_json_envelope(result)
    assert result['ok'] is True
    trees = result['result']['trees']
    assert len(trees) >= 1
    tree_names = [t['name'] for t in trees]
    assert 'main' in tree_names
    main_tree = next(t for t in trees if t['name'] == 'main')
    assert 'experiment_count' in main_tree or 'experiments' in main_tree or 'active' in main_tree


class TestTreeCreateJsonEnvelope:
  def test_creates_tree(self, ws: Path) -> None:
    result = run_cli(ws, ['tree', 'create', 'test-tree'])
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert result['result']['tree'] == 'test-tree'

  def test_duplicate_error(self, ws: Path) -> None:
    run_cli(ws, ['tree', 'create', 'dup'])
    with pytest.raises(SystemExit):
      run_cli(ws, ['tree', 'create', 'dup'])


class TestTreeShowJsonEnvelope:
  def test_shows_tree(self, ws_with_experiments: Path) -> None:
    result = run_cli(ws_with_experiments, ['tree', 'show'])
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert result['result']['name'] == 'main'
    assert 'nodes' in result['result']

  def test_by_name(self, ws_with_experiments: Path) -> None:
    result = run_cli(ws_with_experiments, ['tree', 'show', 'main'])
    _assert_json_envelope(result)
    assert result['result']['name'] == 'main'


class TestTreeSwitchJsonEnvelope:
  def test_switches(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('alpha')
    forest.create_tree('beta')
    forest.save()

    result = run_cli(ws, ['tree', 'switch', 'beta', '--no-checkout'])
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert result['result']['active'] == 'beta'


# -- Experiment commands --


class TestExperimentAddJsonEnvelope:
  def test_add_root(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('main')
    forest.switch('main')
    forest.save()

    result = run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'json test',
        '--id',
        'j-001',
      ],
    )
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert result['result']['experiment_id'] == 'j-001'
    assert result['result']['hypothesis'] == 'json test'

  def test_add_with_parent(self, ws_with_experiments: Path) -> None:
    result = run_cli(
      ws_with_experiments,
      [
        'experiment',
        'add',
        '--hypothesis',
        'child',
        '--id',
        'child-j',
        '--parent',
        'exp-a',
      ],
    )
    _assert_json_envelope(result)
    assert result['result']['parent'] == 'exp-a'


class TestExperimentStatusJsonEnvelope:
  def test_by_id(self, ws_with_experiments: Path) -> None:
    result = run_cli(ws_with_experiments, ['experiment', 'status', 'exp-a'])
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert result['result']['id'] == 'exp-a'
    assert result['result']['status'] == 'completed'
    assert 'metrics' in result['result']
    assert result['result']['metrics']['accuracy'] == 0.72
    assert result['result']['hypothesis'] == 'baseline approach'

  def test_by_head(self, ws_with_experiments: Path) -> None:
    with patch('autopilot.ai.store.file_store.FileStore.checkout'):
      run_cli(ws_with_experiments, ['checkout', 'exp-b'])
    result = run_cli(ws_with_experiments, ['experiment', 'status'])
    _assert_json_envelope(result)
    assert result['result']['id'] == 'exp-b'


class TestExperimentCompareJsonEnvelope:
  def test_compare_two(self, ws_with_experiments: Path) -> None:
    result = run_cli(
      ws_with_experiments,
      [
        'experiment',
        'compare',
        'exp-a',
        'exp-b',
      ],
    )
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert result['result']['a'] == 'exp-a'
    assert result['result']['b'] == 'exp-b'
    assert 'deltas' in result['result']
    deltas_by_metric = {d['metric']: d for d in result['result']['deltas']}
    assert 'accuracy' in deltas_by_metric


# -- Query --


class TestQueryJsonEnvelope:
  def test_completed_query(self, ws_with_experiments: Path) -> None:
    result = run_cli(ws_with_experiments, ['query', '--completed'])
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert 'experiments' in result['result']
    assert 'count' in result['result']
    assert isinstance(result['result']['experiments'], list)

  def test_metric_filter(self, ws_with_experiments: Path) -> None:
    result = run_cli(ws_with_experiments, ['query', '--metric-gt', 'accuracy:0.8'])
    _assert_json_envelope(result)
    assert result['result']['count'] == 1

  def test_best_query(self, ws_with_experiments: Path) -> None:
    result = run_cli(ws_with_experiments, ['query', '--best', 'accuracy'])
    _assert_json_envelope(result)
    assert result['result']['best']['id'] == 'exp-b'

  def test_empty_results(self, ws_with_experiments: Path) -> None:
    result = run_cli(ws_with_experiments, ['query', '--metric-gt', 'accuracy:0.99'])
    _assert_json_envelope(result)
    assert result['result']['count'] == 0
    assert result['result']['experiments'] == []


# -- Checkout --


class TestCheckoutJsonEnvelope:
  def test_checkout(self, ws_with_experiments: Path) -> None:
    with patch('autopilot.ai.store.file_store.FileStore.checkout'):
      result = run_cli(ws_with_experiments, ['checkout', 'exp-a'])
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert result['result']['experiment_id'] == 'exp-a'
    assert result['result']['status'] == 'completed'
    assert 'hypothesis' in result['result']


# -- Stabilize --


class TestStabilizeJsonEnvelope:
  def test_stabilize_completed(self, tmp_path: Path) -> None:
    """Stabilize a completed experiment with snapshots produces JSON envelope."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)

    prompts_dir = ws / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.txt').write_text('hello')

    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('stab-exp', 0)

    exp = Experiment(experiment_id='stab-exp', hypothesis='test stabilize')
    exp.start()
    exp.complete(metrics={'accuracy': 0.9})

    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')
    tree.add(Node(experiment=exp))
    forest.save()

    result = run_cli(ws, ['stabilize', 'stab-exp'])
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert 'copied' in result['result']


# -- Store commands --


class TestStoreSnapshotJsonEnvelope:
  def test_snapshot(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    src = ws / 'src'
    src.mkdir()
    (src / 'main.py').write_text('v0')
    store_dir = ws / 'store'

    result = _run_store_json(
      ws,
      'create',
      src,
      experiment='snap-exp',
      store_dir=store_dir,
    )
    _assert_json_envelope(result)
    assert result['ok'] is True

    (src / 'main.py').write_text('v1')
    result2 = _run_store_json(
      ws,
      'snapshot',
      src,
      experiment='snap-exp',
      store_dir=store_dir,
    )
    _assert_json_envelope(result2)
    assert result2['ok'] is True
    assert 'epoch' in result2['result']


class TestStoreStatusJsonEnvelope:
  def test_status(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    src = ws / 'src'
    src.mkdir()
    (src / 'main.py').write_text('v0')
    store_dir = ws / 'store'

    _run_store_json(ws, 'create', src, experiment='st-exp', store_dir=store_dir)
    result = _run_store_json(
      ws,
      'status',
      src,
      experiment='st-exp',
      store_dir=store_dir,
    )
    _assert_json_envelope(result)
    assert result['ok'] is True


class TestStoreDiffJsonEnvelope:
  def test_diff(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    src = ws / 'src'
    src.mkdir()
    (src / 'main.py').write_text('v0')
    store_dir = ws / 'store'

    _run_store_json(ws, 'create', src, experiment='diff-exp', store_dir=store_dir)
    (src / 'main.py').write_text('v1')
    _run_store_json(ws, 'snapshot', src, experiment='diff-exp', store_dir=store_dir)

    result = _run_store_json(
      ws,
      'diff',
      src,
      experiment='diff-exp',
      store_dir=store_dir,
      extra_args=['--epoch-a', '0', '--epoch-b', '1'],
    )
    _assert_json_envelope(result)
    assert result['ok'] is True


class TestStoreLogJsonEnvelope:
  def test_log(self, tmp_path: Path) -> None:
    ws = tmp_path / 'ws'
    ws.mkdir()
    src = ws / 'src'
    src.mkdir()
    (src / 'main.py').write_text('v0')
    store_dir = ws / 'store'

    _run_store_json(ws, 'create', src, experiment='log-exp', store_dir=store_dir)
    (src / 'main.py').write_text('v1')
    _run_store_json(ws, 'snapshot', src, experiment='log-exp', store_dir=store_dir)

    result = _run_store_json(
      ws,
      'log',
      src,
      experiment='log-exp',
      store_dir=store_dir,
    )
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert 'count' in result['result']


class TestStoreWorktreeListJsonEnvelope:
  def test_empty(self, ws: Path) -> None:
    result = run_cli(ws, ['store', 'worktree', 'list'])
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert result['result']['worktrees'] == []

  def test_with_worktrees(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    src = ws / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})
    store.snapshot('wt-exp', 0)
    store.create_worktree('wt-exp')

    result = run_cli(ws, ['store', 'worktree', 'list'])
    _assert_json_envelope(result)
    assert 'wt-exp' in result['result']['worktrees']


class TestStoreWorktreeCreateJsonEnvelope:
  def test_create(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    src = ws / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})
    store.snapshot('wt-create', 0)

    result = run_cli(ws, ['store', 'worktree', 'create', 'wt-create'])
    _assert_json_envelope(result)
    assert result['ok'] is True
    assert result['result']['experiment_id'] == 'wt-create'
    assert 'path' in result['result']


# -- Error cases with --json via full CLI dispatch --


def _run_full_cli_json(argv: list[str]) -> tuple[int, str]:
  """Run AutoPilotCLI with captured stdout, returning (exit_code, stdout)."""
  cli = AutoPilotCLI()
  out = StringIO()
  exit_code = 0
  with patch('sys.stdout', out), patch('sys.stderr', StringIO()):
    try:
      cli(argv=argv)
    except SystemExit as e:
      exit_code = int(e.code) if e.code is not None else 0
  return exit_code, out.getvalue()


class TestErrorJsonEnvelopes:
  """Error cases via full CLI dispatch produce JSON envelope with {ok, error, messages}."""

  def test_nonexistent_experiment_status(self, ws_with_experiments: Path) -> None:
    code, stdout = _run_full_cli_json(
      [
        'experiment',
        'status',
        'ghost-999',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'error' in envelope
    assert 'not found' in envelope['error']
    assert 'messages' in envelope

  def test_nonexistent_checkout(self, ws_with_experiments: Path) -> None:
    code, stdout = _run_full_cli_json(
      [
        'checkout',
        'ghost-999',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'not found' in envelope['error']
    assert 'messages' in envelope

  def test_nonexistent_tree_switch(self, ws_with_experiments: Path) -> None:
    code, stdout = _run_full_cli_json(
      [
        'tree',
        'switch',
        'ghost-tree',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'not found' in envelope['error']
    assert 'messages' in envelope

  def test_no_active_tree_query(self, ws: Path) -> None:
    code, stdout = _run_full_cli_json(
      [
        'query',
        '--completed',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'no active tree' in envelope['error']
    assert 'messages' in envelope

  def test_no_active_tree_experiment_add(self, ws: Path) -> None:
    code, stdout = _run_full_cli_json(
      [
        'experiment',
        'add',
        '--hypothesis',
        'test',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'no active tree' in envelope['error']
    assert 'messages' in envelope

  def test_compare_nonexistent(self, ws_with_experiments: Path) -> None:
    code, stdout = _run_full_cli_json(
      [
        'experiment',
        'compare',
        'exp-a',
        'ghost',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
      ]
    )
    assert code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'not found' in envelope['error']
    assert 'messages' in envelope

  def test_nonexistent_tree_show(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    FileStore(config)
    code, stdout = _run_full_cli_json(
      [
        'tree',
        'show',
        'nonexistent',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert 'not found' in envelope['error']
    assert 'messages' in envelope
