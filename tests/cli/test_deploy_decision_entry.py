"""Deploy/undeploy CLI handlers emit typed DecisionEntry metadata.

Verifies that deploy, replace, and undeploy paths use
DecisionEntry.deployment() so context entries are machine-filterable
by _type == 'deployment'.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.decision import DecisionEntry
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli


def _setup_forest_with_experiment(
  tmp_path: Path,
  experiment_id: str,
) -> tuple[Path, FileForest]:
  """Create workspace with one tree and one completed experiment."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  exp = Experiment(experiment_id=experiment_id, hypothesis='test')
  exp.start()
  exp.complete(metrics={'acc': 0.9})
  tree.add(Node(experiment=exp))
  forest.switch('main')
  forest.save()
  return ws, forest


def test_deploy_emits_decision_entry_metadata(tmp_path: Path):
  """Deploy writes typed DecisionEntry.deployment() metadata to context log."""
  ws, _forest = _setup_forest_with_experiment(tmp_path, 'exp-001')
  run_cli(ws, ['experiment', 'deploy', 'exp-001', '--as', 'production'])

  forest_reloaded = FileForest(FileStore(AutoPilotConfig(workspace=ws)))
  result = forest_reloaded.find_experiment('exp-001')
  assert result is not None
  node, _ = result
  deploy_entries = [
    e
    for e in node.experiment.context_log
    if e.metadata.get('_type') == DecisionEntry.DEPLOYMENT_TYPE
  ]
  assert len(deploy_entries) >= 1
  entry = deploy_entries[0]
  assert entry.metadata['label'] == 'production'
  assert entry.metadata['experiment_id'] == 'exp-001'
  assert entry.source == 'deployment'


def _setup_two_experiments(tmp_path: Path) -> Path:
  """Create workspace with two completed experiments for replace testing."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  forest = FileForest(FileStore(config))
  tree = forest.create_tree('main')

  for eid, hyp, acc in [('exp-old', 'old', 0.8), ('exp-new', 'new', 0.9)]:
    exp = Experiment(experiment_id=eid, hypothesis=hyp)
    exp.start()
    exp.complete(metrics={'acc': acc})
    tree.add(Node(experiment=exp))

  forest.switch('main')
  forest.save()
  return ws


def test_replace_emits_decision_entry_both_experiments(tmp_path: Path):
  """Replace path emits typed metadata on both old and new experiments."""
  ws = _setup_two_experiments(tmp_path)
  run_cli(ws, ['experiment', 'deploy', 'exp-old', '--as', 'prod'])
  run_cli(ws, ['experiment', 'deploy', 'exp-new', '--as', 'prod', '--replace'])

  forest_reloaded = FileForest(FileStore(AutoPilotConfig(workspace=ws)))

  new_result = forest_reloaded.find_experiment('exp-new')
  assert new_result is not None
  new_node, _ = new_result
  new_deploys = [
    e
    for e in new_node.experiment.context_log
    if e.metadata.get('_type') == DecisionEntry.DEPLOYMENT_TYPE
  ]
  assert len(new_deploys) >= 1

  old_result = forest_reloaded.find_experiment('exp-old')
  assert old_result is not None
  old_node, _ = old_result
  old_deploys = [
    e
    for e in old_node.experiment.context_log
    if e.metadata.get('_type') == DecisionEntry.DEPLOYMENT_TYPE
  ]
  assert len(old_deploys) >= 1
  transfer_entry = [e for e in old_deploys if 'previous_id' in e.metadata]
  assert len(transfer_entry) >= 1


def test_undeploy_emits_decision_entry(tmp_path: Path):
  """Undeploy writes typed DecisionEntry.deployment() metadata."""
  ws, _forest = _setup_forest_with_experiment(tmp_path, 'exp-u')
  run_cli(ws, ['experiment', 'deploy', 'exp-u', '--as', 'staging'])
  run_cli(ws, ['experiment', 'undeploy', 'staging'])

  forest_reloaded = FileForest(FileStore(AutoPilotConfig(workspace=ws)))
  result = forest_reloaded.find_experiment('exp-u')
  assert result is not None
  node, _ = result
  undeploy_entries = [
    e
    for e in node.experiment.context_log
    if e.metadata.get('_type') == DecisionEntry.DEPLOYMENT_TYPE and 'undeployed' in e.reason
  ]
  assert len(undeploy_entries) >= 1
  assert undeploy_entries[0].metadata['label'] == 'staging'


def test_decision_entry_metadata_machine_filterable(tmp_path: Path):
  """Context entries can be filtered by _type == DEPLOYMENT_TYPE."""
  ws, _forest = _setup_forest_with_experiment(tmp_path, 'exp-f')
  run_cli(ws, ['experiment', 'deploy', 'exp-f', '--as', 'canary'])
  run_cli(ws, ['experiment', 'undeploy', 'canary'])

  forest_reloaded = FileForest(FileStore(AutoPilotConfig(workspace=ws)))
  result = forest_reloaded.find_experiment('exp-f')
  assert result is not None
  node, _ = result
  typed = [
    e
    for e in node.experiment.context_log
    if e.metadata.get('_type') == DecisionEntry.DEPLOYMENT_TYPE
  ]
  assert len(typed) == 2
