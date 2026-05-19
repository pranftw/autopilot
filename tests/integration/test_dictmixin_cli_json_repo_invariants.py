"""Dogfood V5 final verification audit integration tests (sub-plan 11).

Verifies:
  1. DictMixin round-trip for all V5 serializable types
  2. New V5 CLI commands produce valid --json output
  3. Multi-feature integration: deploy, metadata, query, trend
  4. No __init__.py files under src/
  5. graph.py import isolation rule
  6. CLAUDE.md / AGENTS.md byte-identity
"""

from autopilot.ai.deployment import (
  DeploymentEvent,
  deployment_log_for_workspace,
  emit_deployment_event,
)
from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.constraint import ConstraintResult
from autopilot.core.decision import DecisionEntry
from autopilot.core.diagnostic import DiagnosticEntry
from autopilot.core.recommend import Recommendation
from autopilot.core.trend import TrendResult
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context, seed_tree_with_experiments
import pytest


class TestDictMixinPrimitivesRoundtrip:
  """DictMixin to_dict/from_dict round-trip for all V5 serializable types."""

  def test_deployment_event_roundtrip(self) -> None:
    """DeploymentEvent survives to_dict -> from_dict."""
    event = DeploymentEvent(
      label='production',
      experiment_id='exp-001',
      action='deploy',
      previous_experiment_id='exp-000',
      timestamp='2026-01-15T10:00:00Z',
      context='initial deploy',
    )
    d = event.to_dict()
    restored = DeploymentEvent.from_dict(d)
    assert restored.label == event.label
    assert restored.experiment_id == event.experiment_id
    assert restored.action == event.action
    assert restored.previous_experiment_id == event.previous_experiment_id
    assert restored.timestamp == event.timestamp
    assert restored.context == event.context

  def test_constraint_result_roundtrip(self) -> None:
    """ConstraintResult survives to_dict -> from_dict."""
    cr = ConstraintResult(
      name='accuracy_gate',
      passed=True,
      metric='accuracy',
      value=0.95,
      threshold='>= 0.9',
      message=None,
    )
    d = cr.to_dict()
    restored = ConstraintResult.from_dict(d)
    assert restored.name == cr.name
    assert restored.passed == cr.passed
    assert restored.metric == cr.metric
    assert restored.value == pytest.approx(cr.value)
    assert restored.threshold == cr.threshold
    assert restored.message == cr.message

  def test_recommendation_roundtrip(self) -> None:
    """Recommendation survives to_dict -> from_dict."""
    rec = Recommendation(
      action='deploy',
      experiment_id='exp-best',
      confidence='high',
      reasoning=['best accuracy', 'passed all gates'],
      alternatives=['exp-runner-up'],
      evidence={'accuracy_delta': 0.05},
    )
    d = rec.to_dict()
    restored = Recommendation.from_dict(d)
    assert restored.action == rec.action
    assert restored.experiment_id == rec.experiment_id
    assert restored.confidence == rec.confidence
    assert restored.reasoning == rec.reasoning
    assert restored.alternatives == rec.alternatives
    assert restored.evidence == rec.evidence

  def test_diagnostic_entry_roundtrip(self) -> None:
    """DiagnosticEntry survives to_dict -> from_dict."""
    entry = DiagnosticEntry(
      code='orphan_blob',
      severity='warning',
      path='/store/objects/abc123',
      message='unreferenced blob found',
      repairable=True,
      repair_action='delete',
    )
    d = entry.to_dict()
    restored = DiagnosticEntry.from_dict(d)
    assert restored.code == entry.code
    assert restored.severity == entry.severity
    assert restored.path == entry.path
    assert restored.message == entry.message
    assert restored.repairable == entry.repairable
    assert restored.repair_action == entry.repair_action

  def test_trend_result_roundtrip(self) -> None:
    """TrendResult survives to_dict -> from_dict."""
    tr = TrendResult(
      metric='accuracy',
      values=[0.7, 0.8, 0.85, 0.9],
      experiment_ids=['e1', 'e2', 'e3', 'e4'],
      direction='improving',
      best_value=0.9,
      best_experiment_id='e4',
      latest_value=0.9,
      improvement_rate=0.0667,
    )
    d = tr.to_dict()
    restored = TrendResult.from_dict(d)
    assert restored.metric == tr.metric
    assert restored.values == pytest.approx(tr.values)
    assert restored.experiment_ids == tr.experiment_ids
    assert restored.direction == tr.direction
    assert restored.best_value == pytest.approx(tr.best_value)
    assert restored.best_experiment_id == tr.best_experiment_id
    assert restored.latest_value == pytest.approx(tr.latest_value)
    assert restored.improvement_rate == pytest.approx(tr.improvement_rate)

  def test_all_primitives_roundtrip(self) -> None:
    """Composite test: all 5 V5 DictMixin types round-trip in one test."""
    types_and_instances = [
      DeploymentEvent(
        label='staging',
        experiment_id='e-1',
        action='replace',
        previous_experiment_id='e-0',
        timestamp='2026-05-17T00:00:00Z',
        context='upgrade',
      ),
      ConstraintResult(
        name='budget',
        passed=False,
        metric='cost_usd',
        value=55.0,
        threshold='<= 50.0',
        message='budget exceeded',
      ),
      Recommendation(
        action='investigate',
        experiment_id=None,
        confidence='low',
        reasoning=['insufficient data'],
      ),
      DiagnosticEntry(
        code='stale_lock',
        severity='info',
        path='/tmp/store.lock',
        message='lock file from pid 99999',
        repairable=True,
        repair_action='delete',
      ),
      TrendResult(
        metric='loss',
        values=[1.0, 0.8, 0.6],
        experiment_ids=['a', 'b', 'c'],
        direction='improving',
        best_value=0.6,
        best_experiment_id='c',
        latest_value=0.6,
        improvement_rate=-0.2,
      ),
    ]
    for instance in types_and_instances:
      d = instance.to_dict()
      cls = type(instance)
      restored = cls.from_dict(d)
      assert restored.to_dict() == d, (
        f'{cls.__name__} round-trip failed: original={d}, restored={restored.to_dict()}'
      )


class TestNewCliCommandsJson:
  """New V5 CLI commands produce valid --json envelopes."""

  def _make_workspace(self, tmp_path: Path) -> Path:
    """Bootstrap a workspace with a tree and experiments."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    config.experiments_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [
        {'id': 'exp-a', 'status': 'completed', 'metrics': {'accuracy': 0.85}},
        {'id': 'exp-b', 'status': 'completed', 'metrics': {'accuracy': 0.92}},
        {'id': 'exp-c', 'status': 'completed', 'metrics': {'accuracy': 0.78}},
      ],
    )
    return ws

  def test_experiment_deploy_log_json(self, tmp_path: Path) -> None:
    """experiment deploy-log --json returns valid envelope with events list."""
    ws = self._make_workspace(tmp_path)
    autopilot_dir = ws / '.autopilot'
    autopilot_dir.mkdir(exist_ok=True)
    log = deployment_log_for_workspace(ws)
    emit_deployment_event(log, 'prod', 'exp-a', 'deploy', context='test deploy')

    envelope = run_cli_no_context(ws, ['experiment', 'deploy-log'])
    assert envelope['ok'] is True
    assert 'events' in envelope['result']
    assert len(envelope['result']['events']) >= 1
    assert envelope['result']['events'][0]['label'] == 'prod'

  def test_recommend_json(self, tmp_path: Path) -> None:
    """recommend --json returns valid envelope with recommendation fields."""
    ws = self._make_workspace(tmp_path)
    envelope = run_cli_no_context(ws, ['recommend', '--metric', 'accuracy'])
    assert envelope['ok'] is True
    result = envelope['result']
    assert 'action' in result
    assert 'confidence' in result
    assert 'experiment_id' in result

  def test_workspace_doctor_repair_dry_run_json(self, tmp_path: Path) -> None:
    """workspace doctor --repair --dry-run --json returns valid envelope."""
    ws = self._make_workspace(tmp_path)
    autopilot_dir = ws / '.autopilot'
    autopilot_dir.mkdir(exist_ok=True)
    envelope = run_cli_no_context(
      ws,
      ['workspace', 'doctor', '--repair', '--dry-run'],
    )
    assert envelope['ok'] is True
    result = envelope['result']
    assert 'dry_run' in result
    assert result['dry_run'] is True

  def test_report_trend_json(self, tmp_path: Path) -> None:
    """report trend --json returns valid TrendResult-shaped envelope."""
    ws = self._make_workspace(tmp_path)
    envelope = run_cli_no_context(ws, ['report', 'trend', 'accuracy'])
    assert envelope['ok'] is True
    result = envelope['result']
    assert 'metric' in result
    assert 'direction' in result
    assert 'values' in result

  def test_experiment_metadata_show_json(self, tmp_path: Path) -> None:
    """experiment metadata show --json returns envelope with metadata dict."""
    ws = self._make_workspace(tmp_path)
    envelope = run_cli_no_context(
      ws,
      ['experiment', 'metadata', 'show', 'exp-a'],
    )
    assert envelope['ok'] is True
    result = envelope['result']
    assert 'experiment_id' in result
    assert 'metadata' in result
    assert isinstance(result['metadata'], dict)

  def test_experiment_metadata_get_json(self, tmp_path: Path) -> None:
    """experiment metadata get --json returns envelope with key/value."""
    ws = self._make_workspace(tmp_path)
    run_cli(ws, ['experiment', 'metadata', 'set', 'exp-a', 'env', 'staging'])
    envelope = run_cli_no_context(
      ws,
      ['experiment', 'metadata', 'get', 'exp-a', 'env'],
    )
    assert envelope['ok'] is True
    result = envelope['result']
    assert result['key'] == 'env'
    assert result['value'] == 'staging'

  def test_query_metric_between_json(self, tmp_path: Path) -> None:
    """query --metric-between --json returns filtered experiments."""
    ws = self._make_workspace(tmp_path)
    envelope = run_cli_no_context(
      ws,
      ['query', '--metric-between', 'accuracy:0.80:0.93'],
    )
    assert envelope['ok'] is True
    result = envelope['result']
    experiments = result['experiments'] if isinstance(result, dict) else result
    ids = {r['id'] for r in experiments}
    assert 'exp-a' in ids
    assert 'exp-b' in ids
    assert 'exp-c' not in ids

  def test_query_metadata_contains_json(self, tmp_path: Path) -> None:
    """query --metadata-contains --json returns filtered experiments."""
    ws = self._make_workspace(tmp_path)
    run_cli(ws, ['experiment', 'metadata', 'set', 'exp-a', 'env', 'prod'])
    envelope = run_cli_no_context(
      ws,
      ['query', '--metadata-contains', 'env:prod'],
    )
    assert envelope['ok'] is True
    result = envelope['result']
    experiments = result['experiments'] if isinstance(result, dict) else result
    ids = {r['id'] for r in experiments}
    assert 'exp-a' in ids

  def test_query_metric_gt_and_lt_json(self, tmp_path: Path) -> None:
    """query --metric-gt and --metric-lt --json still work with V5 dedup."""
    ws = self._make_workspace(tmp_path)
    envelope = run_cli_no_context(
      ws,
      ['query', '--metric-gt', 'accuracy:0.80', '--metric-lt', 'accuracy:0.93'],
    )
    assert envelope['ok'] is True
    result = envelope['result']
    experiments = result['experiments'] if isinstance(result, dict) else result
    assert len(experiments) > 0
    for r in experiments:
      assert r['metrics']['accuracy'] > 0.80
      assert r['metrics']['accuracy'] < 0.93


def test_new_cli_commands_json(tmp_path: Path) -> None:
  """Composite: all new V5 CLI commands produce valid JSON envelopes.

  Exercises: experiment deploy-log, recommend, workspace doctor --repair
  --dry-run, report trend, experiment metadata show/get, query
  --metric-between, query --metadata-contains, query --metric-gt/--metric-lt.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  config.experiments_path.mkdir(parents=True, exist_ok=True)
  (ws / '.autopilot').mkdir(exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {'id': 'exp-a', 'status': 'completed', 'metrics': {'accuracy': 0.85}},
      {'id': 'exp-b', 'status': 'completed', 'metrics': {'accuracy': 0.92}},
      {'id': 'exp-c', 'status': 'completed', 'metrics': {'accuracy': 0.78}},
    ],
  )

  log = deployment_log_for_workspace(ws)
  emit_deployment_event(log, 'prod', 'exp-a', 'deploy', context='test')

  env = run_cli_no_context(ws, ['experiment', 'deploy-log'])
  assert env['ok'] is True
  assert 'events' in env['result']
  assert isinstance(env['result']['events'], list)

  env = run_cli_no_context(ws, ['recommend', '--metric', 'accuracy'])
  assert env['ok'] is True
  assert 'action' in env['result']
  assert isinstance(env['result']['action'], str)
  assert env['result']['action'] in {'deploy', 'continue', 'rollback', 'branch', 'investigate'}

  env = run_cli_no_context(ws, ['workspace', 'doctor', '--repair', '--dry-run'])
  assert env['ok'] is True
  assert 'dry_run' in env['result']
  assert env['result']['dry_run'] is True

  env = run_cli_no_context(ws, ['report', 'trend', 'accuracy'])
  assert env['ok'] is True
  assert 'direction' in env['result']
  assert isinstance(env['result']['direction'], str)

  env = run_cli_no_context(ws, ['experiment', 'metadata', 'show', 'exp-a'])
  assert env['ok'] is True
  assert 'metadata' in env['result']
  assert isinstance(env['result']['metadata'], dict)

  run_cli(ws, ['experiment', 'metadata', 'set', 'exp-a', 'env', 'staging'])
  env = run_cli_no_context(ws, ['experiment', 'metadata', 'get', 'exp-a', 'env'])
  assert env['ok'] is True
  assert env['result']['value'] == 'staging'

  env = run_cli_no_context(ws, ['query', '--metric-between', 'accuracy:0.80:0.93'])
  assert env['ok'] is True
  exps = env['result']['experiments']
  ids = {r['id'] for r in exps}
  assert 'exp-a' in ids
  assert 'exp-b' in ids

  env = run_cli_no_context(ws, ['query', '--metadata-contains', 'env:staging'])
  assert env['ok'] is True
  assert any(r['id'] == 'exp-a' for r in env['result']['experiments'])

  env = run_cli_no_context(
    ws,
    ['query', '--metric-gt', 'accuracy:0.80', '--metric-lt', 'accuracy:0.93'],
  )
  assert env['ok'] is True
  for r in env['result']['experiments']:
    assert 0.80 < r['metrics']['accuracy'] < 0.93


class TestMultiFeatureIntegration:
  """End-to-end integration: deploy, metadata, query, and trend interop."""

  def test_multi_feature_integration(self, tmp_path: Path) -> None:
    """Deploy an experiment, set metadata, query with filters, run trend."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    config.experiments_path.mkdir(parents=True, exist_ok=True)
    (ws / '.autopilot').mkdir(exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    seed_tree_with_experiments(
      forest,
      'main',
      [
        {'id': 'exp-1', 'status': 'completed', 'metrics': {'f1': 0.70}},
        {'id': 'exp-2', 'status': 'completed', 'metrics': {'f1': 0.80}},
        {'id': 'exp-3', 'status': 'completed', 'metrics': {'f1': 0.90}},
      ],
    )

    deploy_result = run_cli(
      ws,
      ['experiment', 'deploy', 'exp-3', '--as', 'production'],
    )
    assert deploy_result['ok'] is True

    run_cli(ws, ['experiment', 'metadata', 'set', 'exp-3', 'team', 'ml-core'])

    log_result = run_cli_no_context(ws, ['experiment', 'deploy-log'])
    assert log_result['ok'] is True
    events = log_result['result']['events']
    assert any(e['experiment_id'] == 'exp-3' for e in events)

    meta_result = run_cli_no_context(
      ws,
      ['experiment', 'metadata', 'get', 'exp-3', 'team'],
    )
    assert meta_result['result']['value'] == 'ml-core'

    query_result = run_cli_no_context(
      ws,
      ['query', '--metadata-contains', 'team:ml-core'],
    )
    assert query_result['ok'] is True
    qr = query_result['result']
    experiments = qr['experiments'] if isinstance(qr, dict) else qr
    ids = {r['id'] for r in experiments}
    assert 'exp-3' in ids

    trend_result = run_cli_no_context(ws, ['report', 'trend', 'f1'])
    assert trend_result['ok'] is True
    assert trend_result['result']['direction'] == 'improving'
    assert trend_result['result']['best_experiment_id'] == 'exp-3'


class TestNoInitPyUnderSrc:
  """src/ tree contains no __init__.py files."""

  def test_no_init_files(self) -> None:
    """Walk src/ and assert zero __init__.py files exist."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    src_dir = repo_root / 'src'
    init_files = list(src_dir.rglob('__init__.py'))
    assert init_files == [], (
      f'Found __init__.py files in src/: {[str(f) for f in init_files]}. '
      'This repo uses no __init__.py -- all imports from terminal modules.'
    )


class TestCoreGraphImportIsolation:
  """graph.py must not import from autopilot (leaf module isolation)."""

  def test_graph_isolation(self) -> None:
    """Read graph.py text and assert no 'from autopilot' substring."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    graph_path = repo_root / 'src' / 'autopilot' / 'core' / 'graph.py'
    assert graph_path.exists(), f'graph.py not found at {graph_path}'
    content = graph_path.read_text(encoding='utf-8')
    assert 'from autopilot' not in content, (
      'graph.py must not import from autopilot (leaf module isolation rule). '
      'Found "from autopilot" substring in graph.py.'
    )


class TestClaudeMdAgentsMdIdenticalV5:
  """CLAUDE.md and AGENTS.md must be byte-identical."""

  def test_claude_md_agents_md_identical(self) -> None:
    """Read both doc files and compare content; fail with sync hint on mismatch."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    claude_md = repo_root / 'CLAUDE.md'
    agents_md = repo_root / 'AGENTS.md'

    assert claude_md.exists(), f'CLAUDE.md not found at {claude_md}'
    assert agents_md.exists(), f'AGENTS.md not found at {agents_md}'

    claude_content = claude_md.read_bytes()
    agents_content = agents_md.read_bytes()

    assert claude_content == agents_content, (
      'CLAUDE.md and AGENTS.md have diverged. '
      'CLAUDE.md is the canonical source; copy it to AGENTS.md to sync. '
      'Run: cp CLAUDE.md AGENTS.md'
    )


class TestDecisionEntryFactories:
  """DecisionEntry classmethods produce correctly typed dicts."""

  def test_deployment_has_type_key(self) -> None:
    """deployment() returns dict with _type='deployment'."""
    result = DecisionEntry.deployment('prod', 'exp-1')
    assert isinstance(result, dict)
    assert result['_type'] == 'deployment'
    assert result['label'] == 'prod'
    assert result['experiment_id'] == 'exp-1'

  def test_rollback_has_type_key(self) -> None:
    """rollback() returns dict with _type='rollback'."""
    result = DecisionEntry.rollback(0, 'bad metrics')
    assert isinstance(result, dict)
    assert result['_type'] == 'rollback'
    assert result['target_epoch'] == 0

  def test_comparison_has_type_key(self) -> None:
    """comparison() returns dict with _type='comparison'."""
    result = DecisionEntry.comparison('base', 'cand', 'improved')
    assert isinstance(result, dict)
    assert result['_type'] == 'comparison'

  def test_policy_gate_has_type_key(self) -> None:
    """policy_gate() returns dict with _type='policy_gate'."""
    result = DecisionEntry.policy_gate('MinGate', passed=True)
    assert isinstance(result, dict)
    assert result['_type'] == 'policy_gate'
    assert result['passed'] is True
