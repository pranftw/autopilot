"""End-to-end dry-run integration test for the full optimization workflow.

Exercises the walkthrough from the plan:
  workspace init -> workspace doctor -> dataset list
  -> experiment setup -> optimize loop --dry-run
Then verifies Forest-backed experiments, events, command logs, results,
dataset snapshots, runtime override wiring, and records/ boundary.

Tests set up experiments via Forest nodes and events on disk. The experiment
CLI tests live in test_experiment_cmd.py.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.core.callbacks.callback import Callback
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.logger import append_event, create_event, load_events
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.node import Node
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum
from autopilot.policy.gates import MaxGate, MinGate
from autopilot.policy.quality_first import QualityFirstMetric, QualityFirstPolicy
from pathlib import Path
from typing import Any
import importlib.util
import json
import pytest
import tomllib


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
  """Set up a temporary workspace with full autopilot overlay structure."""
  ws = tmp_path / 'my-project'
  ws.mkdir()
  ap_dir = ws / '.autopilot'
  ap_dir.mkdir()

  (ap_dir / 'experiments').mkdir()
  (ap_dir / 'records').mkdir()
  (ap_dir / 'records' / 'promotions').mkdir()
  (ap_dir / 'records' / 'notes').mkdir()
  (ap_dir / 'datasets').mkdir()
  (ap_dir / 'datasets' / 'train').mkdir()

  (ap_dir / 'datasets' / 'val').mkdir()
  (ap_dir / 'datasets' / 'test').mkdir()
  (ap_dir / 'workflows').mkdir()
  (ap_dir / 'helpers').mkdir()

  _write_workflow_toml(ap_dir)
  _write_dataset_files(ap_dir)

  return ws


def _write_workflow_toml(autopilot: Path) -> None:
  (autopilot / 'workflows' / 'reasoning_v3_ci_staging.toml').write_text(
    """[workflow]
name = 'reasoning_v3_ci_staging'
target = 'reasoning_v3_ci'
environment = 'staging'

[targets.dev]
url = 'https://dev.example.com'
auth_token = 'test-token-abc'

[targets.prod]
url = 'https://prod.example.com'
auth_token = ''

[policy]
name = 'quality_first'
human_review_on_warn = true

[datasets]
registry = 'datasets/registry.toml'

[datasets.splits]
train = 'reasoning_v3_ci_train.jsonl'
val = 'reasoning_v3_ci_val.jsonl'
test = 'reasoning_v3_ci_test.jsonl'

[scoring.validate]
[scoring.validate.gates]
accuracy = { min = 0.7, required = true }
loss = { max = 0.5, required = true }

[scoring.test]
[scoring.test.gates]
accuracy = { min = 0.7, required = true }
loss = { max = 0.5, required = true }
""",
    encoding='utf-8',
  )


def _write_dataset_files(autopilot: Path) -> None:
  for split_dir, filename in [
    ('train', 'reasoning_v3_ci_train.jsonl'),
    ('val', 'reasoning_v3_ci_val.jsonl'),
    ('test', 'reasoning_v3_ci_test.jsonl'),
  ]:
    path = autopilot / 'datasets' / split_dir / filename
    path.write_text(
      '{"prompt": "test 1"}\n{"prompt": "test 2"}\n{"prompt": "test 3"}\n',
      encoding='utf-8',
    )


class _DryRunModule(AutoPilotModule):
  """Module for dry-run tests. All optimize commands return success."""

  def __init__(self, config: dict) -> None:
    super().__init__()
    gates = _build_gates(config)
    policy_cfg = config.get('policy', {})
    self.policy = QualityFirstPolicy(
      gates=gates,
      human_review_on_warn=policy_cfg.get('human_review_on_warn', True),
    )
    self.metric = QualityFirstMetric(gates=gates)

  def forward(self, ctx: dict[str, Any], params: dict[str, Any]) -> EvalDatum:
    return EvalDatum(success=True, metadata={'dry_run': True, 'command': params.get('command')})

  def training_step(self, batch, batch_idx):
    return self.forward({}, {})

  def configure_optimizers(self):
    return None


def _build_gates(workflow: dict) -> list:
  """Build Gate objects from workflow scoring config."""
  scoring = workflow.get('scoring', {})
  gates_cfg = (scoring.get('validate', {}) or {}).get('gates') or {}
  gate_objects = []
  for metric, spec in gates_cfg.items():
    required = spec.get('required', True)
    if 'min' in spec:
      gate_objects.append(MinGate(metric, spec['min'], required=required))
    if 'max' in spec:
      gate_objects.append(MaxGate(metric, spec['max'], required=required))
  return gate_objects


def build_trainer(
  workflow: dict,
  dry_run: bool = False,
  callbacks: list[Callback] | None = None,
) -> Trainer:
  """Build a Trainer from workflow config (mirrors downstream project overlay)."""
  return Trainer(callbacks=callbacks or [], dry_run=dry_run)


def _run_cli(workspace: Path, argv: list[str]) -> None:
  """Parse and run a CLI command from an explicit argument list."""
  parser = build_parser()
  full_argv = [*list(argv), '--workspace', str(workspace), '--json']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)

  try:
    profile = parsed.profile if hasattr(parsed, 'profile') else 'reasoning_v3_ci_staging'
    cfg = AutoPilotConfig(workspace=workspace, project=ctx.project)
    wf_path = cfg.root / 'workflows' / f'{profile}.toml'
    with Path(wf_path).open('rb') as wf_file:
      workflow = tomllib.load(wf_file)
    ctx.module = _DryRunModule(workflow)
    ctx.trainer = build_trainer(workflow, dry_run=ctx.dry_run)
  except (FileNotFoundError, KeyError):
    pass

  handler = parsed.handler
  assert handler is not None, f'no handler for {argv}'
  handler(ctx, parsed)


def _create_legacy_experiment(workspace: Path, slug: str, **kwargs: Any) -> Path:
  """Create an experiment directory with Forest node and events."""
  cfg = AutoPilotConfig(workspace=workspace)
  exp_dir = cfg.experiment_path(slug=slug)
  exp_dir.mkdir(parents=True, exist_ok=True)

  cfg.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(cfg)
  forest = FileForest(store)
  if forest.get_tree('main') is None:
    forest.create_tree('main')
    forest.switch('main')
  tree = forest.get_tree('main')
  assert tree is not None
  existing = tree.get(slug)
  if existing is None:
    exp = Experiment(
      experiment_id=slug,
      hypothesis=kwargs.get('hypothesis') or kwargs.get('idea'),
    )
    tree.add(Node(experiment=exp))
  forest.save()

  append_event(exp_dir, create_event('created', message='test setup'))
  return exp_dir


def _prepare_experiment(workspace: Path, slug: str) -> Path:
  """Create an experiment for optimize tests."""
  exp_dir = _create_legacy_experiment(workspace, slug)
  return exp_dir


class TestWorkspaceInit:
  def test_workspace_doctor_passes(self, workspace: Path) -> None:
    _run_cli(workspace, ['workspace', 'doctor'])
    ap_dir = workspace / '.autopilot'
    assert ap_dir.exists()
    assert (ap_dir / 'experiments').exists()
    assert (ap_dir / 'records').exists()
    assert (ap_dir / 'datasets').exists()
    assert (ap_dir / 'workflows').exists()


class TestDatasetCommands:
  def test_dataset_list(self, workspace: Path) -> None:
    _run_cli(workspace, ['dataset', 'list'])


class TestLegacyExperimentCreate:
  """Tests for experiment creation via Forest API."""

  def test_create_experiment(self, workspace: Path) -> None:
    exp_dir = _create_legacy_experiment(
      workspace,
      'tool-contract-fix',
      idea='align reasoning and think_web tool contracts',
    )

    assert exp_dir.exists()
    assert (exp_dir / 'events.jsonl').is_file()

    cfg = AutoPilotConfig(workspace=workspace)
    store = FileStore(cfg)
    forest = FileForest(store)
    nodes = forest.query().filter(id='tool-contract-fix').all()
    assert len(nodes) == 1
    assert nodes[0].experiment.id == 'tool-contract-fix'


class TestOptimizeLoopDryRun:
  def test_dry_run_reports_plan(self, workspace: Path) -> None:
    _create_legacy_experiment(workspace, 'dry-run-exp')
    _run_cli(
      workspace,
      [
        'optimize',
        'loop',
        '--experiment',
        'dry-run-exp',
        '--dry-run',
      ],
    )

    cfg = AutoPilotConfig(workspace=workspace)
    store = FileStore(cfg)
    forest = FileForest(store)
    nodes = forest.query().filter(id='dry-run-exp').all()
    assert len(nodes) >= 1

  def test_preflight_dry_run(self, workspace: Path) -> None:
    _create_legacy_experiment(workspace, 'preflight-exp')
    _run_cli(
      workspace,
      [
        'optimize',
        'preflight',
        '--experiment',
        'preflight-exp',
        '--dry-run',
      ],
    )

  def test_deploy_dry_run(self, workspace: Path) -> None:
    exp_dir = _prepare_experiment(workspace, 'deploy-exp')

    _run_cli(
      workspace,
      [
        'optimize',
        'deploy',
        '--experiment',
        'deploy-exp',
        '--dry-run',
      ],
    )

    events = load_events(exp_dir)
    assert len(events) >= 1

  def test_train_dry_run(self, workspace: Path) -> None:
    exp_dir = _prepare_experiment(workspace, 'train-exp')

    _run_cli(
      workspace,
      [
        'optimize',
        'train',
        '--experiment',
        'train-exp',
        '--split',
        'train',
        '--epoch',
        '1',
        '--limit',
        '5',
        '--dry-run',
      ],
    )

    events = load_events(exp_dir)
    assert len(events) >= 1

  def test_validate_dry_run(self, workspace: Path) -> None:
    _prepare_experiment(workspace, 'val-exp')

    _run_cli(
      workspace,
      [
        'optimize',
        'validate',
        '--experiment',
        'val-exp',
        '--dry-run',
      ],
    )

  def test_test_dry_run(self, workspace: Path) -> None:
    _prepare_experiment(workspace, 'test-exp')

    _run_cli(
      workspace,
      [
        'optimize',
        'test',
        '--experiment',
        'test-exp',
        '--dry-run',
      ],
    )


class TestExperimentStateIntegrity:
  def test_experiment_state_dict_has_required_fields(self, workspace: Path) -> None:
    _create_legacy_experiment(workspace, 'manifest-exp', idea='test fields')

    cfg = AutoPilotConfig(workspace=workspace)
    store = FileStore(cfg)
    forest = FileForest(store)
    nodes = forest.query().filter(id='manifest-exp').all()
    assert len(nodes) == 1
    state = nodes[0].experiment.state_dict()

    required_keys = ['id', 'status', 'epoch', 'hypothesis']
    for key in required_keys:
      assert key in state, f'experiment state missing field: {key}'

  def test_events_log_is_append_only(self, workspace: Path) -> None:
    exp_dir = _prepare_experiment(workspace, 'events-exp')

    _run_cli(
      workspace,
      [
        'optimize',
        'deploy',
        '--experiment',
        'events-exp',
        '--dry-run',
      ],
    )

    events = load_events(exp_dir)
    assert len(events) >= 1

    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps), 'events are not in chronological order'

  def test_commands_json_records_optimize_subcommand(self, workspace: Path) -> None:
    exp_dir = _prepare_experiment(workspace, 'cmd-exp')

    _run_cli(
      workspace,
      [
        'optimize',
        'deploy',
        '--experiment',
        'cmd-exp',
        '--dry-run',
      ],
    )

    cmd_path = exp_dir / 'commands.json'
    assert cmd_path.is_file()
    data = json.loads(cmd_path.read_text(encoding='utf-8'))
    assert len(data) >= 1
    assert 'deploy' in data[-1]['args']


class TestRecordsBoundary:
  def test_experiments_not_in_records(self, workspace: Path) -> None:
    _create_legacy_experiment(workspace, 'boundary-exp')
    cfg = AutoPilotConfig(workspace=workspace)
    records = cfg.records_path
    experiments = cfg.experiments_path
    assert (experiments / 'boundary-exp').exists()
    assert not (records / 'boundary-exp').exists()

  def test_experiment_index_exists(self, workspace: Path) -> None:
    records = AutoPilotConfig(workspace=workspace).records_path
    (records / 'experiment_index.jsonl').write_text('', encoding='utf-8')
    assert (records / 'experiment_index.jsonl').is_file()


class TestRegressionRemovedFeatures:
  def test_manifest_module_deleted(self) -> None:
    assert importlib.util.find_spec('autopilot.tracking.manifest') is None

  def test_no_state_module(self) -> None:
    assert importlib.util.find_spec('autopilot.core.state') is None

  def test_no_state_transition_error(self) -> None:
    import autopilot.core.errors as errors_mod

    assert not hasattr(errors_mod, 'StateTransitionError')

  def test_services_module_removed(self) -> None:
    assert importlib.util.find_spec('autopilot.core.services') is None

  def test_checkpoints_module_removed(self) -> None:
    assert importlib.util.find_spec('autopilot.core.checkpoints') is None

  def test_no_trainer_run(self) -> None:
    from autopilot.core.trainer.trainer import Trainer

    t = Trainer()
    assert not hasattr(t, 'run')

  def test_callback_no_removed_hooks(self) -> None:
    from autopilot.core.callbacks.callback import Callback

    cb = Callback()
    assert not hasattr(cb, 'on_status_transition')
    assert not hasattr(cb, 'on_experiment_created')
    assert not hasattr(cb, 'on_result_computed')
    assert not hasattr(cb, 'on_policy_evaluated')
