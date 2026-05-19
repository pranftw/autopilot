"""Integration smoke tests for the harness pipeline.

End-to-end verification that all components compose correctly without
external API calls. Every test uses mocks/stubs to keep execution
deterministic and offline.
"""

from autopilot.ai.store.file_store import FileStore
from autopilot.core.gradient import Gradient
from autopilot.core.types import EvalDatum
from autopilot.policy.policy import Policy
from harness.agent import ConversationResult
from harness.callbacks import MetricsWriterCallback
from harness.evaluator import EvaluationResult
from harness.loss import HarnessGradient, HarnessLoss
from harness.metrics import HarnessMetrics
from harness.module import HarnessModule
from harness.trainer import build_trainer
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import pytest


def _make_scenario(task_id: str) -> dict:
  """Build a minimal scenario for testing."""
  return {
    'task_id': task_id,
    'initial_message': f'Help with {task_id}',
    'user_instructions': {
      'reason_for_call': f'need help with {task_id}',
      'known_info': {'sku': f'sku-{task_id}'},
      'task_instructions': f'do {task_id}',
    },
    'evaluation_criteria': {
      'expected_actions': [{'tool': 'find_user_id_by_name_zip', 'args': {'name': 'Alice'}}],
      'communicate_info': ['your order has been processed'],
      'nl_assertions': ['agent was polite'],
    },
  }


def _make_conv_result(success: bool = True) -> ConversationResult:
  """Build a mock ConversationResult."""
  tool_calls = [{'name': 'find_user_id_by_name_zip', 'arguments': {'name': 'Alice'}}]
  trajectory = [
    {
      'role': 'assistant',
      'content': 'Your order has been processed. The agent was polite and helpful!',
      'turn': 0,
    }
  ]
  if not success:
    tool_calls = []
    trajectory = [{'role': 'assistant', 'content': 'I cannot help.', 'turn': 0}]
  return ConversationResult(
    trajectory=trajectory,
    tool_calls=tool_calls,
    turns=1,
    error=None,
    input_tokens=100,
    output_tokens=50,
    api_calls=1,
  )


@pytest.fixture
def harness_root(tmp_path: Path) -> Path:
  """Create a minimal harness directory structure for tests."""
  harness = tmp_path / 'harness'
  harness.mkdir()
  prompts = harness / 'prompts'
  prompts.mkdir()
  (prompts / 'system_prompt.md').write_text('You are a helpful agent.', encoding='utf-8')
  (prompts / 'policies.md').write_text('Be polite. Follow rules.', encoding='utf-8')
  tools = harness / 'tools'
  tools.mkdir()
  (tools / 'retail_tools.py').write_text(
    'def find_user_id_by_name_zip(ctx, name, zip_code):\n  return "user_1"\n',
    encoding='utf-8',
  )
  db_dir = harness / 'db'
  db_dir.mkdir()
  db_data = {
    'users': [{'user_id': 'u1', 'name': 'Alice Smith', 'email': 'alice@test.com'}],
    'orders': [{'order_id': 'o1', 'user_id': 'u1', 'status': 'pending', 'items': []}],
    'products': [{'product_id': 'p1', 'name': 'Widget', 'variants': []}],
  }
  (db_dir / 'retail.json').write_text(json.dumps(db_data), encoding='utf-8')
  scenarios = harness / 'scenarios'
  scenarios.mkdir()
  records = [_make_scenario('t0'), _make_scenario('t1')]
  for split in ('train.jsonl', 'val.jsonl', 'test.jsonl'):
    (scenarios / split).write_text(
      '\n'.join(json.dumps(r) for r in records) + '\n', encoding='utf-8'
    )
  return harness


class TestModuleForwardMock:
  """Forward two scenarios with mocked agent/db; assert eval datums."""

  def test_forward_returns_eval_datum(self, harness_root: Path):
    """Module forward returns EvalDatum with expected shape under mocks."""
    module = HarnessModule(str(harness_root))
    mock_result = _make_conv_result(success=True)
    with patch.object(module._agent, 'run_conversation', return_value=mock_result):
      scenario = _make_scenario('t0')
      datum = EvalDatum(metadata=scenario)
      result = module.forward(datum)
    assert isinstance(result, EvalDatum)
    assert result.success is True
    assert 'eval_result' in result.metadata
    assert 'trajectory' in result.metadata

  def test_forward_failure_scenario(self, harness_root: Path):
    """Module forward marks failures when eval dimensions fail."""
    module = HarnessModule(str(harness_root))
    mock_result = _make_conv_result(success=False)
    with patch.object(module._agent, 'run_conversation', return_value=mock_result):
      scenario = _make_scenario('t1')
      datum = EvalDatum(metadata=scenario)
      result = module.forward(datum)
    assert isinstance(result, EvalDatum)
    assert result.success is False

  def test_forward_exception_handling(self, harness_root: Path):
    """Module forward catches exceptions and returns error EvalDatum."""
    module = HarnessModule(str(harness_root))
    with patch.object(module._agent, 'run_conversation', side_effect=RuntimeError('api down')):
      scenario = _make_scenario('t0')
      datum = EvalDatum(metadata=scenario)
      result = module.forward(datum)
    assert isinstance(result, EvalDatum)
    assert result.success is False
    assert result.error_message == 'RuntimeError: api down'


class TestLossGradientFlow:
  """forward -> loss.forward -> gradient; assert gradient structure."""

  def test_loss_produces_gradient_from_failed_datum(self):
    """HarnessLoss produces HarnessGradient from failed eval datums."""
    loss = HarnessLoss()
    eval_result = EvaluationResult(
      task_success=False,
      tool_recall=0.5,
      tool_precision=1.0,
      tool_argument_accuracy=0.5,
      communication_recall=0.8,
      policy_compliance=0.7,
      turns=3,
      errored=False,
    )
    datum = EvalDatum(
      success=False,
      metadata={'eval_result': eval_result.to_dict(), 'scenario': {'task_id': 't0'}},
    )
    loss.forward(datum)
    gradient = loss.compute_seed_gradient()
    assert isinstance(gradient, HarnessGradient)
    assert isinstance(gradient, Gradient)
    assert len(gradient.tool_failures) == 1
    assert len(gradient.communication_gaps) == 1
    assert len(gradient.policy_violations) == 1

  def test_gradient_render_produces_markdown(self):
    """Gradient render produces non-empty markdown with recommendations."""
    gradient = HarnessGradient(
      tool_failures=[{'task_id': 't0', 'description': 'tool_recall=0.50'}],
      communication_gaps=[{'task_id': 't0', 'description': 'comm_recall=0.80'}],
    )
    rendered = gradient.render()
    assert '## Tool Call Failures' in rendered
    assert '## Communication Gaps' in rendered
    assert '## Recommendations' in rendered

  def test_gradient_accumulates_correctly(self):
    """Two HarnessGradients accumulate by concatenating buckets."""
    g1 = HarnessGradient(tool_failures=[{'task_id': 't0', 'description': 'miss'}])
    g2 = HarnessGradient(policy_violations=[{'task_id': 't1', 'description': 'violation'}])
    combined = g1.accumulate(g2)
    assert len(combined.tool_failures) == 1
    assert len(combined.policy_violations) == 1


class TestMetricsUpdateCompute:
  """update + compute returns nine metrics with stable keys."""

  def test_twelve_metrics_returned(self):
    """HarnessMetrics compute returns exactly twelve metric keys."""
    metrics = HarnessMetrics()
    eval_result = EvaluationResult(
      task_success=True,
      tool_recall=1.0,
      tool_precision=1.0,
      tool_argument_accuracy=1.0,
      communication_recall=1.0,
      policy_compliance=1.0,
      turns=2,
      errored=False,
    )
    datum = EvalDatum(success=True, metadata={'eval_result': eval_result.to_dict()})
    metrics.update(datum)
    result = metrics.compute()
    expected_keys = {
      'task_success_rate',
      'tool_recall',
      'tool_precision',
      'tool_argument_accuracy',
      'communication_recall',
      'policy_compliance',
      'avg_turns',
      'error_rate',
      'tau_reward',
      'total_input_tokens',
      'total_output_tokens',
      'total_api_calls',
    }
    assert set(result.keys()) == expected_keys

  def test_metrics_values_correct_for_success(self):
    """All metrics reflect perfect evaluation when datum is success."""
    metrics = HarnessMetrics()
    eval_result = EvaluationResult(
      task_success=True,
      tool_recall=1.0,
      tool_precision=1.0,
      tool_argument_accuracy=1.0,
      communication_recall=1.0,
      policy_compliance=1.0,
      turns=3,
      errored=False,
    )
    datum = EvalDatum(success=True, metadata={'eval_result': eval_result.to_dict()})
    metrics.update(datum)
    result = metrics.compute()
    assert result['task_success_rate'] == 1.0
    assert result['tool_recall'] == 1.0
    assert result['error_rate'] == 0.0
    assert result['tau_reward'] == 1.0
    assert result['avg_turns'] == 3.0

  def test_metrics_mixed_datums(self):
    """Metrics correctly aggregate mixed success/failure datums."""
    metrics = HarnessMetrics()
    success_eval = EvaluationResult(
      task_success=True,
      tool_recall=1.0,
      tool_precision=1.0,
      tool_argument_accuracy=1.0,
      communication_recall=1.0,
      policy_compliance=1.0,
      turns=2,
      errored=False,
    )
    fail_eval = EvaluationResult(
      task_success=False,
      tool_recall=0.0,
      tool_precision=0.0,
      tool_argument_accuracy=0.0,
      communication_recall=0.0,
      policy_compliance=0.0,
      turns=5,
      errored=True,
    )
    metrics.update(EvalDatum(success=True, metadata={'eval_result': success_eval.to_dict()}))
    metrics.update(EvalDatum(success=False, metadata={'eval_result': fail_eval.to_dict()}))
    result = metrics.compute()
    assert result['task_success_rate'] == 0.5
    assert result['error_rate'] == 0.5
    assert result['avg_turns'] == 3.5


class TestBuildTrainerWiring:
  """build_trainer returns Trainer with store and policy wired."""

  def test_trainer_has_store(self, tmp_path: Path):
    """build_trainer returns Trainer with a FileStore attached."""
    root = tmp_path / 'project'
    root.mkdir()
    self._setup_project(root)
    trainer, module, dm = build_trainer(root)
    assert trainer.store is not None
    assert isinstance(trainer.store, FileStore)

  def test_trainer_has_policy(self, tmp_path: Path):
    """build_trainer returns Trainer with a Policy attached."""
    root = tmp_path / 'project'
    root.mkdir()
    self._setup_project(root)
    trainer, module, dm = build_trainer(root)
    assert isinstance(trainer.policy, Policy)

  def test_trainer_has_experiment(self, tmp_path: Path):
    """build_trainer returns Trainer with an experiment set."""
    root = tmp_path / 'project'
    root.mkdir()
    self._setup_project(root)
    trainer, module, dm = build_trainer(root)
    assert trainer.experiment is not None

  def test_module_has_parameters(self, tmp_path: Path):
    """Module returned by build_trainer has three PathParameters."""
    root = tmp_path / 'project'
    root.mkdir()
    self._setup_project(root)
    _, module, _ = build_trainer(root)
    params = list(module.named_parameters())
    assert len(params) == 3
    names = [name for name, _ in params]
    assert 'system_prompt' in names
    assert 'policies' in names
    assert 'tools_code' in names

  def test_datamodule_returned(self, tmp_path: Path):
    """build_trainer returns a HarnessDataModule as the third element."""
    from harness.data import HarnessDataModule

    root = tmp_path / 'project'
    root.mkdir()
    self._setup_project(root)
    _, _, dm = build_trainer(root)
    assert isinstance(dm, HarnessDataModule)

  def _setup_project(self, root: Path) -> None:
    """Create minimal project structure for build_trainer."""
    harness = root / 'harness'
    harness.mkdir()
    prompts = harness / 'prompts'
    prompts.mkdir()
    (prompts / 'system_prompt.md').write_text('System prompt.', encoding='utf-8')
    (prompts / 'policies.md').write_text('Policy rules.', encoding='utf-8')
    tools = harness / 'tools'
    tools.mkdir()
    (tools / 'retail_tools.py').write_text(
      'def find_user_id_by_name_zip(ctx, name, zip_code):\n  return "u1"\n',
      encoding='utf-8',
    )
    db_dir = harness / 'db'
    db_dir.mkdir()
    db_data = {
      'users': [{'user_id': 'u1', 'name': 'Test', 'email': 'test@test.com'}],
      'orders': [],
      'products': [],
    }
    (db_dir / 'retail.json').write_text(json.dumps(db_data), encoding='utf-8')
    scenarios = harness / 'scenarios'
    scenarios.mkdir()
    record = _make_scenario('t0')
    for split in ('train.jsonl', 'val.jsonl', 'test.jsonl'):
      (scenarios / split).write_text(json.dumps(record) + '\n', encoding='utf-8')


class TestCallbacksFire:
  """MetricsWriterCallback writes expected files when hook runs."""

  def test_metrics_writer_creates_file(self, tmp_path: Path):
    """MetricsWriterCallback writes epoch_N_metrics.json on epoch end."""
    cb = MetricsWriterCallback()
    exp_dir = tmp_path / 'experiments' / 'test-exp'
    exp_dir.mkdir(parents=True)
    trainer = MagicMock()
    trainer.experiment = MagicMock()
    trainer.experiment.id = 'test-exp'
    trainer.config = MagicMock()
    trainer.config.experiment_path.return_value = exp_dir
    result = MagicMock()
    result.metrics = {'task_success_rate': 0.8, 'tool_recall': 0.9}
    module = MagicMock()
    cb.on_epoch_end(trainer, module, epoch=0, result=result)
    metrics_file = exp_dir / 'epoch_0_metrics.json'
    assert metrics_file.exists()
    written = json.loads(metrics_file.read_text(encoding='utf-8'))
    assert written['task_success_rate'] == 0.8
    assert written['tool_recall'] == 0.9

  def test_metrics_writer_no_result_no_file(self, tmp_path: Path):
    """MetricsWriterCallback does nothing when result is None."""
    cb = MetricsWriterCallback()
    trainer = MagicMock()
    module = MagicMock()
    cb.on_epoch_end(trainer, module, epoch=0, result=None)
    assert not list(tmp_path.iterdir())

  def test_metrics_writer_no_experiment_no_file(self, tmp_path: Path):
    """MetricsWriterCallback does nothing when experiment is None."""
    cb = MetricsWriterCallback()
    trainer = MagicMock()
    trainer.experiment = None
    result = MagicMock()
    result.metrics = {'x': 1.0}
    module = MagicMock()
    cb.on_epoch_end(trainer, module, epoch=0, result=result)


class TestFullPipelineMock:
  """module -> loss -> gradient -> metrics without API calls."""

  def test_end_to_end_pipeline(self, harness_root: Path):
    """Full pipeline: forward -> loss -> gradient -> metrics compute."""
    module = HarnessModule(str(harness_root))
    scenarios = [_make_scenario('t0'), _make_scenario('t1')]

    success_conv = _make_conv_result(success=True)
    fail_conv = _make_conv_result(success=False)
    conv_results = iter([success_conv, fail_conv])

    with patch.object(module._agent, 'run_conversation', side_effect=conv_results):
      eval_datums = []
      for scenario in scenarios:
        datum = EvalDatum(metadata=scenario)
        result = module.forward(datum)
        eval_datums.append(result)

    assert eval_datums[0].success is True
    assert eval_datums[1].success is False

    loss = HarnessLoss()
    for datum in eval_datums:
      loss.forward(datum)
    gradient = loss.compute_seed_gradient()
    assert isinstance(gradient, HarnessGradient)
    assert gradient.metadata['total_failures'] == 1

    metrics = HarnessMetrics()
    for datum in eval_datums:
      metrics.update(datum)
    computed = metrics.compute()
    assert set(computed.keys()) == {
      'task_success_rate',
      'tool_recall',
      'tool_precision',
      'tool_argument_accuracy',
      'communication_recall',
      'policy_compliance',
      'avg_turns',
      'error_rate',
      'tau_reward',
      'total_input_tokens',
      'total_output_tokens',
      'total_api_calls',
    }
    assert computed['task_success_rate'] == 0.5
