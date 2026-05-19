"""Tests for build_trainer and next_slug."""

from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.loops.orchestrator import EpochOrchestrator
from autopilot.core.trainer.trainer import Trainer
from autopilot.policy.gates import MinGate
from autopilot.policy.quality_first import QualityFirstPolicy
from harness.callbacks import (
  DeployCallback,
  HarnessCostTrackerCallback,
  MetricsWriterCallback,
  OptimizerContextCallback,
)
from harness.data import HarnessDataModule
from harness.environments import EnvironmentConfig
from harness.module import HarnessModule
from harness.trainer import build_trainer, next_slug
import json
import pytest

STUB_TOOLS_CODE = '''\
def calculate(ctx, expression):
  """Stub."""
  return str(expression)
'''


@pytest.fixture
def harness_project(tmp_path, scenarios_dir):
  """Build a complete project tree under tmp_path for build_trainer."""
  root = tmp_path / 'project'
  root.mkdir()

  pkg = root / 'harness'
  prompts = pkg / 'prompts'
  prompts.mkdir(parents=True)
  tools = pkg / 'tools'
  tools.mkdir(parents=True)
  db_dir = pkg / 'db'
  db_dir.mkdir(parents=True)

  (prompts / 'system_prompt.md').write_text('test system prompt', encoding='utf-8')
  (prompts / 'policies.md').write_text('test policies', encoding='utf-8')
  (tools / 'retail_tools.py').write_text(STUB_TOOLS_CODE, encoding='utf-8')
  (db_dir / 'retail.json').write_text(
    json.dumps({'products': {}, 'users': {}, 'orders': {}}),
    encoding='utf-8',
  )

  scenarios = pkg / 'scenarios'
  scenarios.mkdir()
  for name in ('train.jsonl', 'val.jsonl', 'test.jsonl'):
    src = scenarios_dir / name
    (scenarios / name).write_text(src.read_text(encoding='utf-8'), encoding='utf-8')

  return root


def test_build_trainer_returns_tuple(harness_project):
  """build_trainer returns a (Trainer, HarnessModule, HarnessDataModule) triple."""
  result = build_trainer(harness_project)
  assert isinstance(result, tuple)
  assert len(result) == 3
  trainer, module, datamodule = result
  assert isinstance(trainer, Trainer)
  assert isinstance(module, HarnessModule)
  assert isinstance(datamodule, HarnessDataModule)


def test_build_trainer_store_registered(harness_project):
  """Store has registered parameter keys matching module.named_parameters()."""
  trainer, module, _ = build_trainer(harness_project)
  store = trainer.store
  module_keys = {name for name, _ in module.named_parameters()}
  registered_keys = set(store._param_names.keys())
  assert registered_keys == module_keys


def test_build_trainer_default_gates(harness_project):
  """Default gates include MinGate for task_success_rate and tool_recall with correct thresholds."""
  trainer, _, _ = build_trainer(harness_project)
  policy = trainer.policy
  assert isinstance(policy, QualityFirstPolicy)
  gates = policy._gates
  assert len(gates) == 2
  gate_map = {g.metric: g for g in gates}
  assert set(gate_map) == {'task_success_rate', 'tool_recall'}
  for gate in gates:
    assert isinstance(gate, MinGate)
  assert gate_map['task_success_rate'].threshold == 0.3
  assert gate_map['tool_recall'].threshold == 0.4


def test_build_trainer_custom_gates(harness_project):
  """Custom gates override the defaults."""
  custom_gate = MinGate('error_rate', threshold=0.1, required=True)
  trainer, _, _ = build_trainer(harness_project, gates=[custom_gate])
  policy = trainer.policy
  assert isinstance(policy, QualityFirstPolicy)
  assert len(policy._gates) == 1
  assert policy._gates[0] is custom_gate


def test_build_trainer_callbacks(harness_project):
  """Callback list always includes StoreCheckpointCallback."""
  from autopilot.core.callbacks.callback import Callback

  class DummyCallback(Callback):
    """Test callback."""

  dummy = DummyCallback()
  trainer, _, _ = build_trainer(harness_project, callbacks=[dummy])
  cb_types = [type(cb) for cb in trainer.callbacks]
  assert StoreCheckpointCallback in cb_types
  assert DummyCallback in cb_types


def test_next_slug_no_refs(tmp_path):
  """next_slug returns harness-1 when refs.json does not exist."""
  store_path = tmp_path / 'store'
  store_path.mkdir()
  assert next_slug(store_path) == 'harness-1'


def test_next_slug_with_existing(tmp_path):
  """next_slug increments based on existing harness- branches."""
  store_path = tmp_path / 'store'
  store_path.mkdir()
  refs = {'branches': {'harness-1': {}, 'harness-2': {}, 'other-1': {}}}
  (store_path / 'refs.json').write_text(json.dumps(refs), encoding='utf-8')
  assert next_slug(store_path) == 'harness-3'


def test_next_slug_sparse_branches(tmp_path):
  """Sparse branches use max suffix, not count."""
  store = tmp_path / 'store'
  store.mkdir()
  (store / 'refs.json').write_text(
    json.dumps({'branches': {'harness-1': 'a', 'harness-10': 'b'}}),
    encoding='utf-8',
  )
  assert next_slug(store) == 'harness-11'


def test_next_slug_ignores_non_numeric(tmp_path):
  """Non-numeric harness-* branches are ignored."""
  store = tmp_path / 'store'
  store.mkdir()
  (store / 'refs.json').write_text(
    json.dumps({'branches': {'harness-beta': 'a', 'harness-3': 'b'}}),
    encoding='utf-8',
  )
  assert next_slug(store) == 'harness-4'


def test_build_trainer_has_orchestrator(harness_project):
  """Trainer uses EpochOrchestrator as its loop."""
  trainer, _, _ = build_trainer(harness_project)
  assert isinstance(trainer.loop, EpochOrchestrator)


def test_orchestrator_config_values(harness_project):
  """Orchestrator config matches harness plateau detection settings."""
  trainer, _, _ = build_trainer(harness_project)
  orch = trainer.loop
  assert isinstance(orch, EpochOrchestrator)
  cfg = orch._config
  assert cfg.monitor == 'task_success_rate'
  assert cfg.auto_rollback is True
  assert cfg.plateau_window == 3
  assert cfg.plateau_threshold == 0.01


def test_dataset_fingerprint_stamped(harness_project):
  """Experiment carries a non-empty dataset fingerprint dict."""
  trainer, _, _ = build_trainer(harness_project)
  meta = trainer.experiment.dataset_meta
  assert isinstance(meta, dict)
  assert isinstance(meta['bundle_hash'], str)
  assert len(meta['bundle_hash']) > 0
  assert isinstance(meta['paths'], list)
  assert isinstance(meta['hashes'], list)
  assert len(meta['paths']) == len(meta['hashes'])
  assert len(meta['paths']) > 0
  assert 'timestamp' in meta


def test_build_trainer_use_judge_param(harness_project):
  """use_judge keyword threads into module."""
  _, module_false, _ = build_trainer(harness_project, use_judge=False)
  assert module_false.use_judge is False

  _, module_true, _ = build_trainer(harness_project, use_judge=True)
  assert module_true.use_judge is True

  _, module_default, _ = build_trainer(harness_project)
  assert module_default.use_judge is True


def test_build_trainer_has_essential_callbacks(harness_project):
  """build_trainer installs essential callbacks but not DeployCallback."""
  trainer, _, _ = build_trainer(harness_project)
  cb_types = [type(cb) for cb in trainer.callbacks]
  assert MetricsWriterCallback in cb_types
  assert OptimizerContextCallback in cb_types
  assert DeployCallback not in cb_types


def test_build_trainer_callback_ordering(harness_project):
  """Essential callbacks in order; HarnessCostTracker after StoreCheckpoint."""
  from autopilot.core.callbacks.callback import Callback

  class UserCallback(Callback):
    """User-provided callback."""

  user_cb = UserCallback()
  trainer, _, _ = build_trainer(harness_project, callbacks=[user_cb])
  cb_types = [type(cb) for cb in trainer.callbacks]

  user_idx = cb_types.index(UserCallback)
  metrics_idx = cb_types.index(MetricsWriterCallback)
  optimizer_idx = cb_types.index(OptimizerContextCallback)
  store_idx = cb_types.index(StoreCheckpointCallback)
  cost_idx = cb_types.index(HarnessCostTrackerCallback)

  assert user_idx < metrics_idx
  assert user_idx < optimizer_idx
  assert metrics_idx < store_idx
  assert optimizer_idx < store_idx
  assert cost_idx > store_idx


def test_build_trainer_env_parameter(harness_project):
  """env parameter overrides default environment config."""
  custom_env = EnvironmentConfig(
    model='test-model',
    max_epochs=2,
    max_turns=5,
    use_judge=False,
  )
  trainer, module, _ = build_trainer(harness_project, env=custom_env)
  assert module.use_judge is False
  assert module._max_turns == 5


def test_build_trainer_use_judge_overrides_env(harness_project):
  """Explicit use_judge takes precedence over env.use_judge."""
  custom_env = EnvironmentConfig(
    model='test-model',
    max_epochs=2,
    max_turns=5,
    use_judge=False,
  )
  _, module, _ = build_trainer(harness_project, env=custom_env, use_judge=True)
  assert module.use_judge is True


def test_build_trainer_max_turns_from_env(harness_project):
  """Module receives max_turns from resolved environment config."""
  custom_env = EnvironmentConfig(
    model='test-model',
    max_epochs=2,
    max_turns=7,
    use_judge=False,
  )
  _, module, _ = build_trainer(harness_project, env=custom_env)
  assert module._max_turns == 7
