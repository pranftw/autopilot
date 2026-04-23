"""End-to-end dry-run integration test for the full optimization workflow.

Exercises the walkthrough from the plan:
  workspace init -> workspace doctor -> dataset list
  -> experiment create -> optimize loop --dry-run
Then verifies manifests, events, command logs, results, dataset snapshots,
runtime override wiring, and records/ boundary.
"""

from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.core.callbacks.callback import Callback
from autopilot.core.hyperparams import load_hyperparams, update_hyperparams
from autopilot.core.module import Module
from autopilot.core.trainer import Trainer
from autopilot.core.types import Datum
from autopilot.policy.gates import MaxGate, MinGate
from autopilot.policy.quality_first import QualityFirstMetric, QualityFirstPolicy
from autopilot.tracking.events import load_events
from autopilot.tracking.manifest import load_manifest
from pathlib import Path
from typing import Any
import autopilot.core.paths as paths
import json
import pytest
import tomllib


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
  """Set up a temporary workspace with full autopilot overlay structure."""
  ws = tmp_path / 'my-project'
  ws.mkdir()
  autopilot = ws / 'autopilot'
  autopilot.mkdir()

  (autopilot / 'experiments').mkdir()
  (autopilot / 'records').mkdir()
  (autopilot / 'records' / 'promotions').mkdir()
  (autopilot / 'records' / 'notes').mkdir()
  (autopilot / 'datasets').mkdir()
  (autopilot / 'datasets' / 'train').mkdir()

  (autopilot / 'datasets' / 'val').mkdir()
  (autopilot / 'datasets' / 'test').mkdir()
  (autopilot / 'workflows').mkdir()
  (autopilot / 'helpers').mkdir()

  _write_workflow_toml(autopilot)
  _write_dataset_files(autopilot)

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


class _DryRunModule(Module):
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

  def forward(self, ctx: dict[str, Any], params: dict[str, Any]) -> Datum:
    return Datum(success=True, metadata={'dry_run': True, 'command': params.get('command')})


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
  full_argv = list(argv) + ['--workspace', str(workspace), '--json']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)

  try:
    profile = parsed.profile if hasattr(parsed, 'profile') else 'reasoning_v3_ci_staging'
    wf_path = paths.root(workspace, ctx.project) / 'workflows' / f'{profile}.toml'
    with open(wf_path, 'rb') as wf_file:
      workflow = tomllib.load(wf_file)
    ctx.module = _DryRunModule(workflow)
    ctx.trainer = build_trainer(workflow, dry_run=ctx.dry_run)
  except (FileNotFoundError, KeyError):
    pass

  handler = parsed.handler
  assert handler is not None, f'no handler for {argv}'
  handler(ctx, parsed)


def _prepare_experiment(workspace: Path, slug: str) -> Path:
  """Create an experiment for optimize tests."""
  _run_cli(
    workspace,
    [
      'experiment',
      'create',
      '--slug',
      slug,
    ],
  )
  exp_dir = workspace / 'autopilot' / 'experiments' / slug
  update_hyperparams(exp_dir, {'deploy_id': 'test-uid-abc123'})
  return exp_dir


class TestWorkspaceInit:
  def test_workspace_doctor_passes(self, workspace: Path) -> None:
    _run_cli(workspace, ['workspace', 'doctor'])
    autopilot = workspace / 'autopilot'
    assert autopilot.exists()
    assert (autopilot / 'experiments').exists()
    assert (autopilot / 'records').exists()
    assert (autopilot / 'datasets').exists()
    assert (autopilot / 'workflows').exists()


class TestDatasetCommands:
  def test_dataset_list(self, workspace: Path) -> None:
    _run_cli(workspace, ['dataset', 'list'])


class TestExperimentCreate:
  def test_create_experiment(self, workspace: Path) -> None:
    _run_cli(
      workspace,
      [
        'experiment',
        'create',
        '--slug',
        'tool-contract-fix',
        '--idea',
        'align reasoning and think_web tool contracts',
      ],
    )

    exp_dir = workspace / 'autopilot' / 'experiments' / 'tool-contract-fix'
    assert exp_dir.exists()
    assert (exp_dir / 'manifest.json').is_file()
    assert (exp_dir / 'events.jsonl').is_file()

    manifest = load_manifest(exp_dir)
    assert manifest.slug == 'tool-contract-fix'

  def test_experiment_list_after_create(self, workspace: Path) -> None:
    _run_cli(
      workspace,
      [
        'experiment',
        'create',
        '--slug',
        'test-exp',
      ],
    )
    _run_cli(workspace, ['experiment', 'list'])

  def test_experiment_show(self, workspace: Path) -> None:
    _run_cli(
      workspace,
      [
        'experiment',
        'create',
        '--slug',
        'show-exp',
      ],
    )
    _run_cli(workspace, ['experiment', 'show', '--experiment', 'show-exp'])

  def test_experiment_status(self, workspace: Path) -> None:
    _run_cli(
      workspace,
      [
        'experiment',
        'create',
        '--slug',
        'status-exp',
      ],
    )
    _run_cli(workspace, ['experiment', 'status', '--experiment', 'status-exp'])


class TestOptimizeLoopDryRun:
  def test_dry_run_reports_plan(self, workspace: Path) -> None:
    _run_cli(
      workspace,
      [
        'experiment',
        'create',
        '--slug',
        'dry-run-exp',
      ],
    )
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

    exp_dir = workspace / 'autopilot' / 'experiments' / 'dry-run-exp'
    checkpoint = load_manifest(exp_dir, strict=False)
    assert checkpoint is not None

  def test_preflight_dry_run(self, workspace: Path) -> None:
    _run_cli(
      workspace,
      [
        'experiment',
        'create',
        '--slug',
        'preflight-exp',
      ],
    )
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


class TestSetHyperparams:
  def test_set_hparams(self, workspace: Path) -> None:
    _run_cli(
      workspace,
      [
        'experiment',
        'create',
        '--slug',
        'hparams-exp',
      ],
    )
    _run_cli(
      workspace,
      [
        'optimize',
        'set-hparams',
        '--experiment',
        'hparams-exp',
        '--values',
        '{"deploy_id": "test-uid-123", "concurrency": 10}',
      ],
    )

    exp_dir = workspace / 'autopilot' / 'experiments' / 'hparams-exp'
    hparams = load_hyperparams(exp_dir)
    assert hparams.values['deploy_id'] == 'test-uid-123'
    assert hparams.values['concurrency'] == 10
    assert hparams.version >= 1


class TestManifestIntegrity:
  def test_manifest_has_required_fields(self, workspace: Path) -> None:
    _run_cli(
      workspace,
      [
        'experiment',
        'create',
        '--slug',
        'manifest-exp',
        '--idea',
        'test manifest fields',
      ],
    )

    exp_dir = workspace / 'autopilot' / 'experiments' / 'manifest-exp'
    data = json.loads((exp_dir / 'manifest.json').read_text(encoding='utf-8'))

    required_keys = [
      'slug',
      'title',
      'current_epoch',
      'idea',
      'hypothesis',
      'hyperparams',
      'decision',
      'decision_reason',
      'metadata',
    ]
    for key in required_keys:
      assert key in data, f'manifest missing required field: {key}'

    removed_keys = [
      'status',
      'profile',
      'target',
      'environment',
      'constraints',
      'baseline',
      'candidate',
      'dataset_snapshot',
    ]
    for key in removed_keys:
      assert key not in data, f'manifest should not have field: {key}'

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
    _run_cli(
      workspace,
      [
        'experiment',
        'create',
        '--slug',
        'boundary-exp',
      ],
    )
    records = workspace / 'autopilot' / 'records'
    experiments = workspace / 'autopilot' / 'experiments'
    assert (experiments / 'boundary-exp').exists()
    assert not (records / 'boundary-exp').exists()

  def test_experiment_index_exists(self, workspace: Path) -> None:
    records = workspace / 'autopilot' / 'records'
    (records / 'experiment_index.jsonl').write_text('', encoding='utf-8')
    assert (records / 'experiment_index.jsonl').is_file()


class TestRegressionRemovedFeatures:
  def test_manifest_no_status_field(self) -> None:
    from autopilot.core.models import Manifest

    manifest = Manifest(slug='test')
    assert 'status' not in manifest.to_dict()

  def test_no_state_module(self) -> None:
    with pytest.raises(ModuleNotFoundError):
      import autopilot.core.state  # noqa: F401

  def test_no_state_transition_error(self) -> None:
    import autopilot.core.errors as errors_mod

    assert not hasattr(errors_mod, 'StateTransitionError')

  def test_services_module_removed(self) -> None:
    import importlib.util

    assert importlib.util.find_spec('autopilot.core.services') is None

  def test_checkpoints_module_removed(self) -> None:
    import importlib.util

    assert importlib.util.find_spec('autopilot.core.checkpoints') is None

  def test_no_update_manifest_status(self) -> None:
    from autopilot.tracking import manifest

    assert not hasattr(manifest, 'update_manifest_status')

  def test_no_trainer_run(self) -> None:
    from autopilot.core.trainer import Trainer

    t = Trainer()
    assert not hasattr(t, 'run')

  def test_callback_no_removed_hooks(self) -> None:
    from autopilot.core.callbacks.callback import Callback

    cb = Callback()
    assert not hasattr(cb, 'on_status_transition')
    assert not hasattr(cb, 'on_experiment_created')
    assert not hasattr(cb, 'on_result_computed')
    assert not hasattr(cb, 'on_policy_evaluated')
