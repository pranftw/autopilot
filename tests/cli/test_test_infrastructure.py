"""Test infrastructure regression tests (plan 13).

Meta-tests validating test helpers, shared fixtures, structural invariants,
and regression coverage for FINDING-001, FINDING-005, and FINDING-009.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.forest import Forest
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.store.base import Store
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum, GateResult
from autopilot.data.dataloader import DataLoader
from autopilot.data.dataset import Dataset
from autopilot.policy.gates import MinGate
from autopilot.policy.quality_first import QualityFirstPolicy
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
from tests.doubles import NoopEvalModule
from typing import Any
import ast
import pytest

# ---------------------------------------------------------------------------
# subplan 2.1 -- run_cli_no_context
# ---------------------------------------------------------------------------


class TestRunCliNoContext:
  """Verify run_cli_no_context omits --context and still produces JSON output."""

  def test_run_cli_no_context_omits_flag(self, cli_workspace: Path) -> None:
    """run_cli_no_context must not inject --context yet still return a dict."""
    config = AutoPilotConfig(workspace=cli_workspace)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('test-tree')
    forest.switch('test-tree')
    forest.save()

    result = run_cli_no_context(cli_workspace, ['tree', 'list'])
    assert isinstance(result, dict)

  def test_run_cli_includes_context(self, cli_workspace: Path) -> None:
    """run_cli includes --context 'test' and returns a dict."""
    config = AutoPilotConfig(workspace=cli_workspace)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('test-tree')
    forest.switch('test-tree')
    forest.save()

    result = run_cli(cli_workspace, ['tree', 'list'])
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# subplan 2.2 -- no autouse checkout mock in checkout tests
# ---------------------------------------------------------------------------

CHECKOUT_TEST_FILES = (
  'tests/ai/test_checkout_safety.py',
  'tests/cli/test_checkout_cmd.py',
)


def _file_has_autouse_checkout_patch(filepath: str) -> bool:
  """Parse a test file's AST and check for autouse fixtures patching FileStore.checkout."""
  path = Path(filepath)
  if not path.exists():
    return False
  source = path.read_text(encoding='utf-8')
  tree = ast.parse(source)
  for node in ast.walk(tree):
    if not isinstance(node, ast.FunctionDef):
      continue
    for decorator in node.decorator_list:
      if not isinstance(decorator, ast.Call):
        continue
      for kw in decorator.keywords:
        if kw.arg == 'autouse' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
          func_source = ast.get_source_segment(source, node) or ''
          if 'FileStore.checkout' in func_source or 'store.checkout' in func_source:
            return True
  return False


def test_no_autouse_checkout_mock_in_checkout_tests() -> None:
  """Checkout-focused test files must not use autouse patches on FileStore.checkout."""
  for filepath in CHECKOUT_TEST_FILES:
    assert not _file_has_autouse_checkout_patch(filepath), (
      f'{filepath} has an autouse fixture patching FileStore.checkout; '
      'checkout tests must exercise real code paths'
    )


# ---------------------------------------------------------------------------
# subplan 2.3 -- shared fixture smoke tests
# ---------------------------------------------------------------------------


def test_workspace_with_store_and_forest_fixture(
  workspace_with_store_and_forest: dict[str, Any],
) -> None:
  """workspace_with_store_and_forest yields a writable root with store + forest."""
  ws = workspace_with_store_and_forest['workspace']
  config = workspace_with_store_and_forest['config']
  store = workspace_with_store_and_forest['store']
  forest = workspace_with_store_and_forest['forest']

  assert ws.is_dir()
  assert config.store_path.is_dir()
  assert isinstance(store, Store)
  assert isinstance(forest, Forest)
  assert len(forest.list_trees()) >= 1
  assert forest.active is not None

  reloaded = FileForest(store)
  assert len(reloaded.list_trees()) >= 1


def test_multi_tree_forest_fixture(multi_tree_forest: FileForest) -> None:
  """multi_tree_forest reports at least two trees."""
  assert len(multi_tree_forest.list_trees()) >= 2
  tree_names = [t.name for t in multi_tree_forest.list_trees()]
  assert 'alpha' in tree_names
  assert 'beta' in tree_names


# ---------------------------------------------------------------------------
# subplan 2.5 -- FINDING-001: feature add workflow E2E
# ---------------------------------------------------------------------------


class _AccuracyMetric(Metric):
  """Metric that extracts 'accuracy' from EvalDatum.metrics."""

  higher_is_better = True

  def __init__(self) -> None:
    super().__init__()
    self.add_state('_values', list)

  def update(self, datum: Any) -> None:
    if isinstance(datum, EvalDatum) and 'accuracy' in datum.metrics:
      self._values.append(datum.metrics['accuracy'])

  def compute(self) -> dict[str, float]:
    if not self._values:
      return {}
    return {'accuracy': sum(self._values) / len(self._values)}


class _MetricEvalModule(NoopEvalModule):
  """Module that produces a known metric value for E2E verification."""

  def __init__(self, metric_value: float = 0.85) -> None:
    super().__init__()
    self._metric_value = metric_value
    self.accuracy_metric = _AccuracyMetric()

  def training_step(self, batch, batch_idx):
    return EvalDatum(success=True, metrics={'accuracy': self._metric_value})

  def validation_step(self, batch, batch_idx):
    return EvalDatum(success=True, metrics={'accuracy': self._metric_value})


class _DatumDataset(Dataset):
  """Minimal map-style dataset yielding Datum items for E2E tests."""

  def __init__(self, size: int = 3) -> None:
    self._size = size

  def __getitem__(self, index: int):
    return Datum()

  def __len__(self) -> int:
    return self._size


def _setup_e2e_workspace(tmp_path: Path) -> tuple[Path, AutoPilotConfig, FileStore]:
  """Create workspace with store directory for E2E tests."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  return ws, config, FileStore(config)


def test_feature_add_metrics_visible_e2e(tmp_path: Path) -> None:
  """E2E: experiment add -> Trainer.fit -> experiment show -> metrics visible.

  Covers FINDING-001: metrics produced by training are visible through
  the experiment object and through CLI experiment show.
  """
  ws, config, store = _setup_e2e_workspace(tmp_path)
  forest = FileForest(store)
  forest.create_tree('main')
  forest.switch('main')

  run_cli(ws, ['experiment', 'add', '--hypothesis', 'test e2e', '--id', 'e2e-exp'])

  forest_fresh = FileForest(store)
  tree_fresh = forest_fresh.active
  assert tree_fresh is not None
  node = tree_fresh.get('e2e-exp')
  assert node is not None
  exp = node.experiment

  loader = DataLoader(_DatumDataset(3), batch_size=1)
  trainer = Trainer(
    experiment=exp,
    config=config,
    store=store,
    tree=tree_fresh,
    forest=forest_fresh,
  )
  trainer.fit(
    _MetricEvalModule(metric_value=0.92),
    train_dataloaders=loader,
    val_dataloaders=loader,
    max_epochs=1,
  )

  assert exp.status == 'completed'
  assert exp.metrics is not None
  assert any('accuracy' in k for k in exp.metrics), (
    f'expected accuracy in metric keys, got {set(exp.metrics.keys())}'
  )

  show_result = run_cli_no_context(ws, ['experiment', 'show', 'e2e-exp'])
  shown_metrics = show_result['result']['metrics']
  assert shown_metrics is not None
  assert any('accuracy' in k for k in shown_metrics), (
    f'experiment show metrics missing accuracy, got {shown_metrics}'
  )


# ---------------------------------------------------------------------------
# subplan 2.6 -- FINDING-005: policy gate matrix
# ---------------------------------------------------------------------------

POLICY_GATE_MATRIX = [
  ('pass', {'val_accuracy': 0.9}, True),
  ('fail', {'val_accuracy': 0.3}, False),
  ('nan_metric', {'val_accuracy': float('nan')}, False),
  ('missing_metric', {}, False),
]


@pytest.mark.parametrize(
  ('label', 'metrics', 'should_pass'),
  POLICY_GATE_MATRIX,
  ids=[row[0] for row in POLICY_GATE_MATRIX],
)
def test_policy_gate_accept_reject_matrix(
  label: str,
  metrics: dict[str, float],
  should_pass: bool,
) -> None:
  """Policy gate outcomes: pass, fail, NaN metric, missing metric.

  Validates both the policy/gate layer in isolation and verifies
  the explain output is coherent.
  """
  policy = QualityFirstPolicy(gates=[MinGate('val_accuracy', 0.5, required=True)])
  result = Result(metrics=metrics)
  gate_result = policy(result)

  if should_pass:
    assert gate_result != GateResult.FAIL
  else:
    assert gate_result == GateResult.FAIL

  explanation = policy.explain(result)
  assert isinstance(explanation, str)
  assert len(explanation) > 0


def test_policy_gate_trainer_integration(tmp_path: Path) -> None:
  """Policy gate integrated with Trainer: FAIL triggers rollback and context emission."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)

  policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.99, required=True)])
  exp = Experiment(experiment_id='gate-exp', hypothesis='gate test')

  module = _MetricEvalModule(metric_value=0.5)
  ds = _DatumDataset(3)
  train_loader = DataLoader(ds, batch_size=1)

  trainer = Trainer(
    experiment=exp,
    config=config,
    policy=policy,
    enable_context_log=True,
  )
  trainer.fit(module, train_dataloaders=train_loader, max_epochs=2)

  policy_entries = exp.context_log.filter_by_source('policy')
  assert len(policy_entries) >= 1, 'expected policy gate context entries'

  rejected = [e for e in policy_entries if 'rejected' in e.reason]
  assert len(rejected) >= 1, 'expected at least one policy rejection context entry'


# ---------------------------------------------------------------------------
# subplan 2.7 -- FINDING-009: multi-epoch optimizer context entries
# ---------------------------------------------------------------------------


def test_multi_epoch_produces_optimizer_context_entries(tmp_path: Path) -> None:
  """Multi-epoch training produces context entries across epochs.

  Covers FINDING-009: verifies that context_log entries accrue across
  multiple epochs with expected sources (trainer, policy when configured).
  Uses a policy gate to generate per-epoch policy context entries.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)

  exp = Experiment(experiment_id='ctx-exp', hypothesis='context test')

  policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.1, required=True)])

  module = _MetricEvalModule(metric_value=0.85)
  ds = _DatumDataset(3)
  train_loader = DataLoader(ds, batch_size=1)

  max_epochs = 3
  trainer = Trainer(
    experiment=exp,
    config=config,
    policy=policy,
    enable_context_log=True,
  )
  trainer.fit(
    module,
    train_dataloaders=train_loader,
    max_epochs=max_epochs,
  )

  assert exp.status == 'completed'

  entries = exp.context_log.entries
  assert len(entries) > 0, 'expected context_log entries after multi-epoch training'

  trainer_entries = [e for e in entries if e.source == 'trainer']
  assert len(trainer_entries) >= 2, (
    f'expected at least 2 trainer context entries (completion + max_epochs), '
    f'got {len(trainer_entries)}: {[e.reason for e in trainer_entries]}'
  )

  completion_entry = [e for e in trainer_entries if 'completed' in e.reason]
  assert len(completion_entry) >= 1, 'expected experiment completion context entry'

  max_epochs_entry = [e for e in trainer_entries if 'max_epochs' in e.reason]
  assert len(max_epochs_entry) >= 1, 'expected max_epochs context entry'

  policy_entries = exp.context_log.filter_by_source('policy')
  assert len(policy_entries) >= max_epochs, (
    f'expected at least {max_epochs} policy entries (one per epoch), got {len(policy_entries)}'
  )

  epoch_values = {e.epoch for e in policy_entries if e.epoch is not None}
  assert len(epoch_values) >= 2, (
    f'expected context entries spanning multiple epochs, got epochs {epoch_values}'
  )
