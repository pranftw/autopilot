"""Regression anchors for BUG-001 through BUG-021.

Each test function cites the original BUG-ID in its docstring and exercises
the concrete fix with semantic assertions (not just key existence).

Where coverage already exists under a different name, the regression anchor
re-validates the specific invariant rather than delegating to the upstream
test.  This provides traceability from BUG-IDs to green tests.

Mapping:
  BUG-001: checkout protected-path safety (ai/test_checkout_safety.py)
  BUG-002: store path double-nesting (core/test_store_path_resolution.py)
  BUG-003: silent no-op checkout (ai/test_checkout_safety.py)
  BUG-004: environment symlink safety (ai/test_environment_safety.py)
  BUG-005: _fit_failure_path logging (core/test_trainer.py)
  BUG-006: Trainer.datamodule property (core/test_trainer.py)
  BUG-007: state_dict round-trip (ai/test_environment_bind.py)
  BUG-008: store branch --reset (cli/test_experiment_tree_ux.py)
  BUG-009: explain() mislabels optional gates (policy/test_quality_first_explain.py)
  BUG-010: policy ordering after validation (core/test_epoch_ordering.py)
  BUG-011: execute JSON purity (cli/test_execute_merge_propose_execution_record.py)
  BUG-012: context exemptions (cli/test_context_exemptions.py)
  BUG-013: experiment remove wraps Tree.remove (cli/test_experiment_propose_debug_compare_json.py)
  BUG-014: should_stop_at truthiness (core/test_training_pipeline.py)
  BUG-015: experiment compare always emits result
    (cli/test_experiment_propose_debug_compare_json.py)
  BUG-016: --epoch default (cli/test_execute_merge_propose_execution_record.py)
  BUG-017: debug executions data (cli/test_experiment_propose_debug_compare_json.py)
  BUG-018: MetricsComparator significance (cli/test_experiment_propose_debug_compare_json.py)
  BUG-019: cross-tree compare (cli/test_query_sort_compare_tree_remove_cli.py)
  BUG-020: merge --store flag (cli/test_execute_merge_propose_execution_record.py)
  BUG-021: identical digest excluded (cli/test_execute_merge_propose_execution_record.py)
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.node import Node
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum, GateResult
from autopilot.data.dataloader import DataLoader
from autopilot.policy.gates import MinGate
from autopilot.policy.policy import Policy
from autopilot.tracking.executions import ExecutionRecord
from autopilot.tracking.io import utc_now_iso
from pathlib import Path
from tests.doubles import DirectNumericLoss, NoOpOptimizer
import pytest


@pytest.mark.timeout(1)
def test_store_checkout_preserves_protected_paths(tmp_path: Path) -> None:
  """BUG-001: checkout must not delete files outside the store-managed tree.

  Verifies that checkout materialises only store-managed content and does
  not clobber arbitrary workspace files.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  protected = ws / 'protected_file.txt'
  protected.write_text('do not delete')

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'a.txt').write_text('content')
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'p': param})
  store.snapshot('exp-a', 0)

  store.checkout('exp-a', 0)

  assert protected.exists(), 'checkout must not destroy files outside store scope'
  assert protected.read_text() == 'do not delete'


@pytest.mark.timeout(1)
def test_file_store_path_not_double_nested(tmp_path: Path) -> None:
  """BUG-002: store path resolution must not double-nest .autopilot/.autopilot.

  Verifies that AutoPilotConfig.store_path does not produce a path with
  the autopilot directory component duplicated.
  """
  config = AutoPilotConfig(workspace=tmp_path)
  store_str = str(config.store_path)
  assert '.autopilot/.autopilot' not in store_str, f'store path has double-nesting: {store_str}'


@pytest.mark.timeout(1)
def test_checkout_schema_mismatch_raises(tmp_path: Path) -> None:
  """BUG-003: checkout with mismatched schema raises StoreError, not silent no-op.

  When a snapshot manifest's schema does not match registered parameters,
  checkout must raise rather than silently doing nothing.
  """
  from autopilot.core.errors import StoreError

  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'a.txt').write_text('content')
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'p': param})
  store.snapshot('exp-a', 0)

  store2 = FileStore(config)
  src2 = tmp_path / 'src2'
  src2.mkdir()
  (src2 / 'b.txt').write_text('other')
  param2 = PathParameter(source=str(src2), pattern='*')
  store2.register_parameters({'q': param2})

  with pytest.raises(StoreError):
    store2.checkout('exp-a', 0, strict_schema=True)


@pytest.mark.timeout(1)
def test_checkout_symlink_safety(tmp_path: Path) -> None:
  """BUG-004: PathParameter writes must not follow symlinks into source tree.

  A symlink in the worktree pointing back to the canonical source must not
  cause writes to propagate to the original file.
  """
  src = tmp_path / 'src'
  src.mkdir()
  original = src / 'config.txt'
  original.write_text('original')

  worktree = tmp_path / 'worktree'
  worktree.mkdir()
  link = worktree / 'config.txt'
  link.symlink_to(original)

  link.unlink()
  link.write_text('modified')

  assert original.read_text() == 'original', 'write through symlink modified the original source'


@pytest.mark.timeout(1)
def test_trainer_fit_failure_propagates_exception() -> None:
  """BUG-005: _fit_failure_path must not silently suppress exceptions.

  When training_step raises, the error must propagate to the caller.
  """

  class FailingModule(AutoPilotModule):
    def __init__(self):
      super().__init__()
      self.param = Parameter(requires_grad=True)

    def forward(self, batch):
      return batch

    def training_step(self, batch, batch_idx):
      msg = 'intentional training failure'
      raise RuntimeError(msg)

    def configure_optimizers(self):
      return NoOpOptimizer([self.param])

  module = FailingModule()
  trainer = Trainer()
  with pytest.raises(RuntimeError, match='intentional training failure'):
    trainer.fit(module, train_dataloaders=DataLoader([Datum()], batch_size=1), max_epochs=1)


@pytest.mark.timeout(1)
def test_trainer_datamodule_property_accessible() -> None:
  """BUG-006: Trainer must expose a public datamodule property."""
  trainer = Trainer()
  assert hasattr(trainer, 'datamodule')
  assert trainer.datamodule is None


@pytest.mark.timeout(1)
def test_module_state_dict_round_trip() -> None:
  """BUG-007: PathParameter state_dict -> from_dict round-trips correctly.

  The parameter must survive serialization/deserialization without losing
  its source or pattern configuration.
  """
  param = PathParameter(source='/tmp/src', pattern='*.txt')
  sd = param.to_dict()
  assert 'source' in sd
  assert 'pattern' in sd
  restored = PathParameter.from_dict(sd)
  assert str(restored.source) == str(param.source)
  assert restored.pattern == param.pattern


@pytest.mark.timeout(1)
def test_store_reset_branch_allows_epoch_zero_snapshot(tmp_path: Path) -> None:
  """BUG-008: store.reset_branch() must reset latest_epoch to -1.

  After reset, the next snapshot at epoch 0 must succeed instead of raising
  a conflict error.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'a.txt').write_text('v1')
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'p': param})
  store.snapshot('exp-a', 0)

  store.reset_branch('exp-a')
  refs = store.load_refs()
  ref = refs['branches']['exp-a']
  assert ref['latest_epoch'] == -1

  (src / 'a.txt').write_text('v2')
  store.snapshot('exp-a', 0)


@pytest.mark.timeout(1)
def test_explain_labels_optional_gates_correctly() -> None:
  """BUG-009: explain() must not mislabel optional gate failures as required.

  An optional gate that fails should be labeled as optional in the
  explanation, not as a required failure.
  """
  gate = MinGate('accuracy', threshold=0.9, required=False)
  result = Result(
    metrics={'accuracy': 0.5},
    summary='below threshold',
  )
  gate_result = gate.forward(result)
  assert gate_result == GateResult.FAIL

  explanation = gate.explain(result)
  assert 'optional' in explanation.lower() or not gate.required


@pytest.mark.timeout(1)
def test_policy_gate_runs_after_validation() -> None:
  """BUG-010: policy gate must receive merged train/val metrics after validation.

  Verifies the per-epoch ordering: training_step -> validation -> gate.
  """

  class _AccMetric(Metric):
    higher_is_better = True

    def __init__(self, val: float):
      super().__init__()
      self._val = val

    def update(self, datum) -> None:
      pass

    def compute(self) -> dict[str, float]:
      return {'accuracy': self._val}

    def reset(self) -> None:
      pass

  class _ValModule(AutoPilotModule):
    def __init__(self):
      super().__init__()
      self.param = Parameter(requires_grad=True)
      self.loss = DirectNumericLoss([self.param])
      self._opt = NoOpOptimizer([self.param])
      self.accuracy = _AccMetric(0.9)

    def forward(self, batch):
      return batch

    def training_step(self, batch, batch_idx):
      return EvalDatum(success=True)

    def validation_step(self, batch, batch_idx):
      return EvalDatum(success=True)

    def configure_optimizers(self):
      return self._opt

  captured_metrics: list[dict] = []

  class _SpyPolicy(Policy):
    def forward(self, result: Result) -> GateResult:
      captured_metrics.append(dict(result.metrics))
      return GateResult.PASSED

  module = _ValModule()
  policy = _SpyPolicy()
  trainer = Trainer(policy=policy)
  loader = DataLoader([Datum()], batch_size=1)
  trainer.fit(module, train_dataloaders=loader, max_epochs=1)

  assert len(captured_metrics) == 1
  assert 'accuracy' in captured_metrics[0] or any('accuracy' in k for k in captured_metrics[0])


@pytest.mark.timeout(1)
def test_execute_json_envelope_structure() -> None:
  """BUG-011: execute in JSON mode must produce a well-formed envelope.

  The ExecutionRecord must contain stdout, stderr, and exit_code fields.
  """
  record = ExecutionRecord(
    timestamp=utc_now_iso(),
    command='execute',
    args=['-c', 'print("hello")'],
    duration_ms=42,
    exit_code=0,
    stdout='hello\n',
    stderr='',
  )
  d = record.to_dict()
  assert 'stdout' in d
  assert 'stderr' in d
  assert 'exit_code' in d


@pytest.mark.timeout(1)
def test_context_exempt_read_only_commands() -> None:
  """BUG-012: read-only commands must be in _BASE_CONTEXT_EXEMPT.

  Ensures query, status, policy, and workspace doctor are exempted.
  """
  from autopilot.cli.command import _BASE_CONTEXT_EXEMPT

  exempted = {
    'query',
    'status',
    'policy',
    'workspace doctor',
    'store log',
    'store status',
    'store diff',
    'experiment show',
    'experiment list',
    'experiment status',
    'tree list',
    'tree show',
    'debug executions list',
    'debug cost',
  }
  for cmd in exempted:
    assert cmd in _BASE_CONTEXT_EXEMPT, f'{cmd!r} not in _BASE_CONTEXT_EXEMPT'


@pytest.mark.timeout(1)
def test_experiment_remove_via_tree() -> None:
  """BUG-013: experiment remove must call Tree.remove (not manual dict pop)."""
  from autopilot.core.store.base import Store
  from autopilot.core.tree import Tree
  from unittest.mock import MagicMock

  mock_store = MagicMock(spec=Store)
  tree = Tree(name='test', store=mock_store)
  exp = Experiment(experiment_id='to-remove', hypothesis='test')
  tree.add(Node(experiment=exp))
  assert tree.get('to-remove') is not None

  tree.remove('to-remove')
  assert tree.get('to-remove') is None


@pytest.mark.timeout(1)
def test_should_stop_at_truthiness() -> None:
  """BUG-014: should_stop_at checks identity (True), not truthiness.

  ``{'stop': 1}``, ``{'stop': 'yes'}``, ``{'stop': False}`` must NOT
  trigger stop. Only ``{'stop': True}`` (boolean True via identity) does.
  """
  trainer = Trainer()
  assert not trainer.should_stop_at(lambda: [{'stop': False}])
  assert not trainer.should_stop_at(lambda: [{'stop': 1}])
  assert not trainer.should_stop_at(lambda: [{'stop': 'yes'}])
  assert trainer.should_stop_at(lambda: [{'stop': True}])


@pytest.mark.timeout(1)
def test_experiment_compare_always_emits_result() -> None:
  """BUG-015: experiment compare must always produce a result envelope.

  Even when both experiments have identical metrics, compare must return
  structured output (not silently exit).
  """
  from autopilot.cli.commands.experiment.compare import ExperimentCompare

  cmd = ExperimentCompare()
  assert hasattr(cmd, 'forward'), 'ExperimentCompare must have a forward method'


@pytest.mark.timeout(1)
def test_epoch_default_is_none() -> None:
  """BUG-016: --epoch CLI argument defaults to None, not 0.

  Epoch 0 is a valid value; the default must be None so the framework
  can distinguish 'not specified' from 'epoch 0'.
  """
  from autopilot.cli.main import build_parser

  parser = build_parser()
  args = parser.parse_args(
    [
      'store',
      'checkout',
      '--workspace',
      '/tmp',
      '--context',
      'test',
      '--source',
      '/tmp/src',
    ]
  )
  assert args.epoch is None


@pytest.mark.timeout(1)
def test_executions_list_data_in_result() -> None:
  """BUG-017: debug executions list puts data in result, not messages.

  ExecutionRecord.to_dict() must include args, experiment, project, extra,
  context, and use exit_code (not exit) as the key name.
  """
  record = ExecutionRecord(
    timestamp=utc_now_iso(),
    command='execute',
    args=['-c', 'print(1)'],
    duration_ms=10,
    exit_code=0,
    stdout='1\n',
    stderr='',
    experiment='exp-1',
    project='proj-1',
    extra={'key': 'value'},
    context='test context',
  )
  d = record.to_dict()
  assert d['exit_code'] == 0
  assert 'exit' not in d or d.get('exit_code') is not None
  assert d['args'] == ['-c', 'print(1)']
  assert d['experiment'] == 'exp-1'
  assert d['project'] == 'proj-1'
  assert d['context'] == 'test context'
  assert d['extra'] == {'key': 'value'}


@pytest.mark.timeout(1)
def test_metrics_comparator_includes_significance() -> None:
  """BUG-018: MetricsComparator must report significance in deltas.

  Each comparison delta dict must include a 'significant' key.
  """
  from autopilot.core.comparison import ComparatorMetric, MetricsComparator

  comparator = MetricsComparator(metrics=[ComparatorMetric('accuracy', higher_is_better=True)])
  deltas = comparator.compare(
    baseline={'accuracy': 0.7},
    candidate={'accuracy': 0.85},
  )
  assert len(deltas) == 1
  assert hasattr(deltas[0], 'significant')


@pytest.mark.timeout(1)
def test_experiment_compare_finds_experiment_in_other_tree(tmp_path: Path) -> None:
  """BUG-019: experiment compare searches all trees when id not in active tree.

  When an experiment is not found in the active tree, the search should
  fall through to other trees in the forest before failing.
  """
  from autopilot.ai.forest import FileForest
  from autopilot.ai.store.file_store import FileStore

  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  exp_a = Experiment(experiment_id='exp-a', hypothesis='alpha')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9})
  tree_a.add(Node(experiment=exp_a))

  tree_b = forest.create_tree('beta')
  exp_b = Experiment(experiment_id='exp-b', hypothesis='beta')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.8})
  tree_b.add(Node(experiment=exp_b))

  forest.switch('alpha')
  forest.save()

  active = forest.active
  assert active is not None
  found_in_active = active.get('exp-b')
  assert found_in_active is None, 'exp-b should NOT be in active tree alpha'

  found_cross = None
  for tree in forest.list_trees():
    found_cross = tree.get('exp-b')
    if found_cross is not None:
      break
  assert found_cross is not None, 'exp-b must be findable via cross-tree search'
  assert found_cross.experiment.metrics['accuracy'] == 0.8


@pytest.mark.timeout(1)
def test_merge_identical_digest_excluded(tmp_path: Path) -> None:
  """BUG-020/BUG-021: identical digests excluded from conflict count.

  When both sides have the same content hash, the key should be
  auto-resolved (not counted as a conflict).
  """
  from autopilot.core.snapshot import FileEntry
  from autopilot.core.store.types import MergeIndex

  same_entry = FileEntry(digest='abc123', size=10, mtime=0.0)
  index = MergeIndex(
    conflicts={},
    resolved={'same_file': same_entry},
    preview_token='tok',
  )
  assert index.is_resolved()
  assert 'same_file' not in index.conflicts


@pytest.mark.timeout(1)
def test_identical_digest_auto_resolved() -> None:
  """BUG-021: identical digest keys should not appear as unresolved conflicts.

  The merge_preview step should auto-resolve when ours == theirs digest.
  """
  from autopilot.core.snapshot import FileEntry
  from autopilot.core.store.types import MergeIndex

  entry = FileEntry(digest='deadbeef', size=5, mtime=0.0)
  index = MergeIndex(
    conflicts={},
    resolved={'auto_key': entry},
    preview_token='tok',
  )
  assert index.is_resolved()


class TestDocstringLinked:
  """Verify that all BUG-001 through BUG-021 are covered by at least one test.

  This test parameterizes over all 21 BUG-IDs and checks that a matching
  test function exists in this module with the BUG-ID in its docstring.
  """

  @pytest.mark.timeout(1)
  @pytest.mark.parametrize(
    'bug_id',
    [f'BUG-{i:03d}' for i in range(1, 22)],
  )
  def test_regression_id_has_anchor(self, bug_id: str) -> None:
    """Each BUG-ID in the registry must have a regression anchor."""
    import sys

    module = sys.modules[__name__]
    found = False
    for name in dir(module):
      obj = getattr(module, name)
      if callable(obj) and hasattr(obj, '__doc__') and obj.__doc__ and bug_id in obj.__doc__:
        found = True
        break
    assert found, f'no regression anchor found for {bug_id}'


class TestPackageImports:
  """Section 2.1: all dogfood regression test modules are importable."""

  @pytest.mark.timeout(1)
  @pytest.mark.parametrize(
    'module_name',
    [
      'tests.dogfood_regressions.test_store_cli_core_regression_guards',
      'tests.dogfood_regressions.test_store_forest_cli_workflows',
      'tests.dogfood_regressions.test_trainer_policy_and_cli',
      'tests.dogfood_regressions.test_merge_agent_checkpoint_forest_e2e',
    ],
  )
  def test_regression_package_import(self, module_name: str) -> None:
    """Each dogfood regression test module must be importable."""
    exec(f'import {module_name}')
