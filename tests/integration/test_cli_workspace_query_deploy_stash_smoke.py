"""Cross-subplan integration smoke tests for dogfood V3 (Plan 13).

Verifies end-to-end flows spanning multiple subplans:
  1. Deploy + compare with direction-aware verdict (plans 01, 02)
  2. Stash + snapshot with context forwarding (plans 05)
  3. Query --best with metric constraints + --all-trees (plans 04, 07)
  4. CLAUDE.md / AGENTS.md byte-identity check
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
from tests.doubles import make_completed_experiment


def test_cli_deploy_and_compare_respect_metric_direction(tmp_path: Path) -> None:
  """Full workflow: create workspace, add experiments with metrics,
  deploy one, compare with direction-aware verdict, verify JSON shape."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_base = make_completed_experiment(
    'exp-base',
    'baseline model',
    {'accuracy': 0.80, 'loss': 1.0, 'latency_ms': 200.0},
  )
  node_base = Node(experiment=exp_base)
  tree.add(node_base)

  exp_candidate = make_completed_experiment(
    'exp-candidate',
    'improved model',
    {'accuracy': 0.85, 'loss': 0.7, 'latency_ms': 180.0},
  )
  node_cand = Node(experiment=exp_candidate, parent=node_base)
  tree.add(node_cand)

  forest.deploy(node_base, 'production')
  forest.save()

  result = run_cli(ws, ['experiment', 'deploy', 'exp-candidate', '--as', 'staging'])
  assert result['ok'] is True
  assert result['result']['deployed_as'] == 'staging'

  result = run_cli_no_context(
    ws,
    ['experiment', 'compare', 'exp-base', 'exp-candidate', '--lower-metric', 'loss'],
  )
  assert result['ok'] is True
  deltas = result['result']['deltas']
  assert len(deltas) >= 3

  for delta in deltas:
    assert 'higher_is_better' in delta

  loss_delta = next(d for d in deltas if d['metric'] == 'loss')
  assert loss_delta['higher_is_better'] is False
  assert loss_delta['delta'] < 0

  accuracy_delta = next(d for d in deltas if d['metric'] == 'accuracy')
  assert accuracy_delta['higher_is_better'] is True
  assert accuracy_delta['delta'] > 0

  assert result['result']['verdict'] == 'improved'


def test_store_stash_and_snapshot_forward_context(tmp_path: Path) -> None:
  """Stash + snapshot with context forwarding."""
  from autopilot.tracking.io import read_jsonl

  ws = tmp_path / 'ws'
  ws.mkdir()
  source_dir = ws / 'src'
  source_dir.mkdir()
  (source_dir / 'main.py').write_text('print("hello")', encoding='utf-8')

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)

  param = PathParameter(source=str(source_dir), pattern='*.py')
  store = FileStore(config)
  store.register_parameters({'source': param})

  experiment_id = 'exp-stash'
  store.snapshot(experiment_id, 0, context='epoch 0 snapshot')

  forest = FileForest(store)
  tree = forest.create_tree('main')
  exp = Experiment(experiment_id=experiment_id, hypothesis='test stash')
  exp.start()
  exp.complete(metrics={'score': 0.5})
  tree.add(Node(experiment=exp))
  forest.switch('main')
  forest.save()

  reflog_path = config.store_path / 'reflog.jsonl'
  entries = read_jsonl(reflog_path, strict=False)
  snapshot_entries = [e for e in entries if e.get('operation') == 'snapshot']
  assert len(snapshot_entries) >= 1
  assert snapshot_entries[0].get('context') == 'epoch 0 snapshot'

  result = run_cli(ws, ['--experiment', experiment_id, 'store', 'stash'])
  assert result['ok'] is True

  stash_entries = [
    e for e in read_jsonl(reflog_path, strict=False) if e.get('operation') == 'stash'
  ]
  assert len(stash_entries) >= 1


def test_query_constraints_best(tmp_path: Path) -> None:
  """--best with metric constraints + --all-trees."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  forest.switch('alpha')
  exp_a1 = make_completed_experiment(
    'exp-a1',
    'alpha cheap',
    {'accuracy': 0.85, 'cost_usd': 0.3},
  )
  tree_a.add(Node(experiment=exp_a1))
  exp_a2 = make_completed_experiment(
    'exp-a2',
    'alpha good',
    {'accuracy': 0.92, 'cost_usd': 2.0},
  )
  tree_a.add(Node(experiment=exp_a2))

  tree_b = forest.create_tree('beta')
  forest.switch('beta')
  exp_b1 = make_completed_experiment(
    'exp-b1',
    'beta best',
    {'accuracy': 0.95, 'cost_usd': 3.0},
  )
  tree_b.add(Node(experiment=exp_b1))

  forest.switch('alpha')
  forest.save()

  result = run_cli_no_context(
    ws,
    ['query', '--best', 'accuracy', '--metric-gt', 'cost_usd:0.5', '--all-trees'],
  )
  assert result['ok'] is True
  best = result['result']['best']
  assert best['id'] == 'exp-b1'
  assert best['metrics']['accuracy'] == 0.95
  assert best['metrics']['cost_usd'] > 0.5
  assert 'tree' in best


def test_claude_md_matches_agents_md() -> None:
  """Verify CLAUDE.md and AGENTS.md are byte-identical."""
  repo_root = Path(__file__).resolve().parents[2]
  claude_md = repo_root / 'CLAUDE.md'
  agents_md = repo_root / 'AGENTS.md'

  assert claude_md.exists(), 'CLAUDE.md not found at repo root'
  assert agents_md.exists(), 'AGENTS.md not found at repo root'

  claude_content = claude_md.read_bytes()
  agents_content = agents_md.read_bytes()

  assert claude_content == agents_content, (
    'CLAUDE.md and AGENTS.md are not byte-identical. '
    f'CLAUDE.md: {len(claude_content)} bytes, AGENTS.md: {len(agents_content)} bytes'
  )
