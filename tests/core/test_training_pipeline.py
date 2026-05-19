"""Tests for Plan 08: Training Pipeline Fixes.

Covers BUG-048 through BUG-059 and BUG-076 -- validation metrics persistence,
should_stop_at fix, epoch orchestrator min_epoch, Trainer-Environment integration,
Forest/Trainer experiment unification, JudgeLoss.judge, AgentOptimizer agentic
feedback, ClaudeCodeAgent cwd, Module.load_state_dict PathParameter restore,
OrchestratorConfig.monitor default, propose verify, and gradient todo_items.
"""

from autopilot.ai.agents.agent import Agent, AgentResult
from autopilot.ai.agents.claude_code import ClaudeCodeAgent
from autopilot.ai.gradient import TextGradient
from autopilot.ai.loss import JudgeLoss
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.ai.parameter import PathParameter
from autopilot.core.callbacks.callback import Callback
from autopilot.core.config import Config
from autopilot.core.environment import Environment
from autopilot.core.errors import ConfigError
from autopilot.core.experiment import Experiment
from autopilot.core.gradient import Gradient, NumericGradient
from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.loops.loop import LoopConfig
from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.node import Node
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.tree import Tree
from autopilot.core.types import EvalDatum
from autopilot.data.dataloader import DataLoader
from dataclasses import dataclass
from tests.doubles import NoOpOptimizer
from typing import Any
from unittest.mock import MagicMock
import logging
import math
import pytest

# -- helpers ------------------------------------------------------------------


class _StubModule(AutoPilotModule):
  def forward(self, batch):
    return batch

  def training_step(self, batch, batch_idx):
    return self(batch)

  def validation_step(self, batch, batch_idx):
    return self(batch)

  def configure_optimizers(self):
    return NoOpOptimizer([])


class _StubMetric(Metric):
  higher_is_better: bool | None = True

  def __init__(self):
    super().__init__()
    self._vals: list[float] = []

  def update(self, datum):
    pass

  def compute(self):
    return {}


def _mock_agent(output: str = 'done') -> MagicMock:
  agent = MagicMock(spec=Agent)
  agent.run.return_value = AgentResult(output=output)
  agent.limiter = None
  agent._cwd = None
  return agent


def _make_loop_config(**overrides: Any) -> LoopConfig:
  defaults: dict[str, Any] = {
    'max_epochs': 3,
    'min_epoch': 0,
    'dry_run': False,
    'ctx': {},
    'train_loader': DataLoader([EvalDatum(success=True)], batch_size=1),
    'val_loader': None,
    'loss': None,
    'optimizer': None,
    'metrics': {},
    'accumulate_grad_batches': 1,
    'experiment': None,
    'metric_metadata': {},
  }
  defaults.update(overrides)
  return LoopConfig(**defaults)


# -- 4.1 Epoch loop / validation payload (BUG-059) ---------------------------


class TestValMetricsEmptyDict:
  """BUG-059: val_metrics={} must be preserved (not dropped by truthiness)."""

  def test_epoch_result_includes_empty_val_metrics_dict(self):
    loop = EpochLoop()
    trainer = Trainer(dry_run=False)
    mod = _StubModule()
    trainer.fit(mod, max_epochs=0)
    trainer._module = mod
    config = _make_loop_config(
      max_epochs=1,
      val_loader=DataLoader([EvalDatum(success=True)], batch_size=1),
    )
    result = loop.run(trainer, config)
    assert len(result['epochs']) == 1
    epoch = result['epochs'][0]
    assert 'val_metrics' in epoch

  def test_epoch_result_omits_val_metrics_when_validation_skipped(self):
    loop = EpochLoop()
    trainer = Trainer(dry_run=False)
    mod = _StubModule()
    trainer._module = mod
    config = _make_loop_config(max_epochs=1, val_loader=None)
    result = loop.run(trainer, config)
    epoch = result['epochs'][0]
    assert 'val_metrics' not in epoch

  def test_epoch_non_empty_val_metrics_unchanged(self):
    """Non-empty val_metrics should always be included."""
    loop = EpochLoop()

    class _ValMetricModule(_StubModule):
      pass

    class _ReturnsVal(Metric):
      higher_is_better: bool | None = True

      def update(self, datum):
        pass

      def compute(self):
        return {'accuracy': 0.95}

    mod = _ValMetricModule()
    mod.val_metric = _ReturnsVal()
    trainer = Trainer(dry_run=False)
    trainer._module = mod
    config = _make_loop_config(
      max_epochs=1,
      val_loader=DataLoader([EvalDatum(success=True)], batch_size=1),
      metrics={'val_metric': mod.val_metric},
    )
    result = loop.run(trainer, config)
    epoch = result['epochs'][0]
    assert 'val_metrics' in epoch
    assert epoch['val_metrics']['accuracy'] == 0.95


# -- 4.2 Orchestrator epoch range (BUG-058) ----------------------------------


class TestOrchestratorMinEpoch:
  """BUG-058: EpochOrchestrator must respect min_epoch."""

  def test_orchestrator_min_epoch_three_skips_zero_two(self):
    orch = EpochOrchestrator()
    trainer = Trainer(dry_run=False)
    mod = _StubModule()
    trainer._module = mod
    config = _make_loop_config(min_epoch=3, max_epochs=5)
    result = orch.run(trainer, config)
    executed_epochs = [e['epoch'] for e in result['epochs']]
    assert executed_epochs == [3, 4]

  def test_orchestrator_epoch_count_matches_range_width(self):
    orch = EpochOrchestrator()
    trainer = Trainer(dry_run=False)
    mod = _StubModule()
    trainer._module = mod
    config = _make_loop_config(min_epoch=2, max_epochs=6)
    result = orch.run(trainer, config)
    assert result['total_epochs'] == 4


# -- 4.3 Early stopping aggregation (BUG-057) --------------------------------


class TestShouldStopAt:
  """BUG-057/BUG-001: should_stop_at must check value truth, not key existence."""

  def test_should_stop_at_true(self):
    trainer = Trainer()

    def hook(**kwargs):
      return [{'stop': True}]

    assert trainer.should_stop_at(hook) is True

  def test_should_stop_at_false_value(self):
    trainer = Trainer()

    def hook(**kwargs):
      return [{'stop': False}]

    assert trainer.should_stop_at(hook) is False

  def test_should_stop_at_zero_value(self):
    trainer = Trainer()

    def hook(**kwargs):
      return [{'stop': 0}]

    assert trainer.should_stop_at(hook) is False

  def test_should_stop_at_none_value(self):
    trainer = Trainer()

    def hook(**kwargs):
      return [{'stop': None}]

    assert trainer.should_stop_at(hook) is False

  def test_should_stop_at_no_key(self):
    trainer = Trainer()

    def hook(**kwargs):
      return [{}]

    assert trainer.should_stop_at(hook) is False

  def test_should_stop_at_mixed(self):
    trainer = Trainer()

    def hook(**kwargs):
      return [{}, {'stop': True}]

    assert trainer.should_stop_at(hook) is True

  def test_should_stop_at_non_dict(self):
    trainer = Trainer()

    def hook(**kwargs):
      return ['string', None, 42]

    assert trainer.should_stop_at(hook) is False

  def test_should_stop_at_not_list(self):
    trainer = Trainer()

    def hook(**kwargs):
      return 'not a list'

    assert trainer.should_stop_at(hook) is False

  def test_should_stop_at_empty_list(self):
    trainer = Trainer()

    def hook(**kwargs):
      return []

    assert trainer.should_stop_at(hook) is False

  def test_should_stop_at_detects_stop_in_second_callback(self):
    trainer = Trainer()

    def hook(**kwargs):
      return [{}, {'stop': True}]

    assert trainer.should_stop_at(hook) is True

  def test_should_stop_at_handles_non_dict_entries_gracefully(self):
    trainer = Trainer()

    def hook(**kwargs):
      return [None, {'stop': True}]

    assert trainer.should_stop_at(hook) is True

  def test_should_stop_at_stop_false_does_not_stop(self):
    """BUG-014: {'stop': False} must not trigger stop."""
    trainer = Trainer()

    def hook(**kwargs):
      return [{'stop': False}]

    assert trainer.should_stop_at(hook) is False

  def test_should_stop_at_stop_truthy_non_bool(self):
    """BUG-014: {'stop': 'yes'} must not trigger stop -- identity check, not truthiness."""
    trainer = Trainer()

    def hook(**kwargs):
      return [{'stop': 'yes'}]

    assert trainer.should_stop_at(hook) is False

  def test_should_stop_at_stop_one(self):
    """BUG-014: {'stop': 1} must not trigger stop -- int 1 is not True."""
    trainer = Trainer()

    def hook(**kwargs):
      return [{'stop': 1}]

    assert trainer.should_stop_at(hook) is False


# -- 4.4 Experiment complete metrics (BUG-048) --------------------------------


class TestExperimentCompleteMetrics:
  """BUG-048: experiment.complete() should use val metrics when present."""

  def test_complete_prefixed_merge_when_both_train_and_val(self):
    exp = Experiment('test-exp')
    exp.start()
    trainer = Trainer(experiment=exp)
    trainer._experiment = exp
    loop_result = {
      'epochs': [
        {
          'epoch': 0,
          'metrics': {'loss': 0.5},
          'val_metrics': {'accuracy': 0.9},
        }
      ]
    }
    trainer._complete_experiment_success(loop_result)
    assert 'train_loss' in exp.metrics
    assert 'val_accuracy' in exp.metrics
    assert exp.metrics['train_loss'] == 0.5
    assert exp.metrics['val_accuracy'] == 0.9

  def test_complete_val_only_when_train_empty(self):
    exp = Experiment('test-exp')
    exp.start()
    trainer = Trainer(experiment=exp)
    trainer._experiment = exp
    loop_result = {'epochs': [{'epoch': 0, 'metrics': {}, 'val_metrics': {'accuracy': 0.8}}]}
    trainer._complete_experiment_success(loop_result)
    assert exp.metrics == {'accuracy': 0.8}

  def test_complete_train_only_when_val_none(self):
    exp = Experiment('test-exp')
    exp.start()
    trainer = Trainer(experiment=exp)
    trainer._experiment = exp
    loop_result = {'epochs': [{'epoch': 0, 'metrics': {'loss': 0.3}}]}
    trainer._complete_experiment_success(loop_result)
    assert exp.metrics == {'loss': 0.3}


# -- 4.5 Environment activation (BUG-055) ------------------------------------


class TestTrainerEnvironment:
  """BUG-055: Trainer.fit() should activate Config.environment."""

  def test_fit_raises_config_error_when_environment_set_but_experiment_none(self):
    mock_env = MagicMock(spec=Environment)
    config = MagicMock(spec=Config)
    config.environment = mock_env
    trainer = Trainer(config=config)
    mod = _StubModule()
    with pytest.raises(ConfigError, match='requires an experiment'):
      trainer.fit(mod, max_epochs=1)

  def test_fit_local_environment_nullcontext_no_bind(self):
    """LocalEnvironment should not bind PathParameters (cwd unchanged)."""
    exp = Experiment('test-env')
    mod = _StubModule()
    trainer = Trainer(experiment=exp)
    result = trainer.fit(mod, max_epochs=1)
    assert result['total_epochs'] == 1

  def test_fit_unbinds_path_parameters_after_loop_even_on_exception(self, tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'rules.txt').write_text('rules content')

    class _PathModule(AutoPilotModule):
      def __init__(self, src: str):
        super().__init__()
        self.prompts = PathParameter(source=src, pattern='**/*')

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def validation_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return NoOpOptimizer([])

    mod = _PathModule(str(source))
    mod.prompts.bind('/fake/bound')
    assert mod.prompts.working_root == '/fake/bound'
    mod.prompts.unbind()
    assert mod.prompts.working_root == str(source)


# -- 4.6 Experiment identity (BUG-049) ---------------------------------------


class TestExperimentIdentity:
  """BUG-049: Trainer and Tree should share the same Experiment object."""

  def test_trainer_experiment_is_tree_node_experiment(self):
    exp = Experiment('exp-1')
    mock_store = MagicMock()
    tree = Tree(name='test-tree', store=mock_store)
    tree.add(Node(experiment=exp))
    trainer = Trainer(experiment=exp, tree=tree)
    assert trainer.experiment is exp
    node = tree.get('exp-1')
    assert node is not None
    assert trainer.experiment is node.experiment


# -- 4.7 JudgeLoss (BUG-051) -------------------------------------------------


class TestJudgeLossJudge:
  """BUG-051: JudgeLoss should store and use self.judge."""

  def test_judge_loss_assigns_self_judge(self):
    mock_judge = MagicMock()
    mock_collator = MagicMock()
    loss = JudgeLoss(judge=mock_judge, collator=mock_collator)
    assert loss.judge is mock_judge

  def test_judge_loss_forward_uses_instance_judge(self):
    mock_judge = MagicMock()
    mock_collator = MagicMock()
    loss = JudgeLoss(judge=mock_judge, collator=mock_collator)
    assert hasattr(loss, 'judge')
    assert loss.judge is mock_judge


# -- 4.8 AgentOptimizer context + agentic feedback (BUG-052, BUG-076) --------


class TestAgentOptimizerAgentic:
  """BUG-052/076: AgentOptimizer file-based agentic feedback."""

  def test_write_epoch_feedback_creates_file(self, tmp_path):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = TextGradient(text='improve accuracy', attribution='fix rules')
    opt = AgentOptimizer(
      agent,
      [p],
      context={'epoch': 0, 'metrics': {'accuracy': 0.5}},
      feedback_dir=str(tmp_path / '.optimization'),
    )
    path = opt.write_epoch_feedback(0)
    assert path.exists()
    content = path.read_text(encoding='utf-8')
    assert '# Epoch 0' in content
    assert 'accuracy: 0.5' in content
    assert 'fix rules' in content

  def test_write_epoch_feedback_works_with_text_gradient(self, tmp_path):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = TextGradient(
      text='improve latency',
      attribution='optimize queries',
      severity=0.8,
    )
    opt = AgentOptimizer(agent, [p], feedback_dir=str(tmp_path / '.opt'))
    path = opt.write_epoch_feedback(1)
    content = path.read_text(encoding='utf-8')
    assert 'Text: improve latency' in content
    assert 'optimize queries' in content
    assert 'Severity: 0.80' in content

  def test_write_epoch_feedback_works_with_custom_gradient_subclass(self, tmp_path):
    @dataclass
    class _CustomGradient(Gradient):
      message: str = ''

      def render(self):
        return f'custom: {self.message}'

      def accumulate(self, other):
        return _CustomGradient(message=f'{self.message}; {other.message}')

    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = _CustomGradient(message='reduce latency')
    opt = AgentOptimizer(agent, [p], feedback_dir=str(tmp_path / '.opt'))
    path = opt.write_epoch_feedback(0)
    content = path.read_text(encoding='utf-8')
    assert 'custom: reduce latency' in content

  def test_update_todo_creates_todo_from_attributions(self):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = TextGradient(attribution='Fix error handling in parser')
    opt = AgentOptimizer(agent, [p], context={'epoch': 0})
    opt.update_todo()
    assert any('Fix error handling in parser' in item.text for item in opt._todo_items)

  def test_update_todo_marks_addressed_items(self):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = TextGradient(attribution='Fix bug A')
    opt = AgentOptimizer(agent, [p], context={'epoch': 0, 'metrics': {'accuracy': 0.5}})
    opt.update_todo()
    assert not opt._todo_items[0].addressed
    opt._prev_metrics = {'accuracy': 0.5}
    opt._context = {'epoch': 1, 'metrics': {'accuracy': 0.8}}
    p.grad = TextGradient(attribution='Fix bug B')
    opt.update_todo()
    assert opt._todo_items[0].addressed

  def test_update_todo_merges_new_and_existing(self):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = TextGradient(attribution='Fix A')
    opt = AgentOptimizer(agent, [p], context={'epoch': 0})
    opt.update_todo()
    p.grad = TextGradient(attribution='Fix B')
    opt._context = {'epoch': 1}
    opt.update_todo()
    texts = [item.text for item in opt._todo_items]
    assert 'Fix A' in texts
    assert 'Fix B' in texts

  def test_build_task_brief_is_concise(self, tmp_path):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = TextGradient(attribution='improve')
    opt = AgentOptimizer(
      agent,
      [p],
      context={'epoch': 2, 'metrics': {'acc': 0.7}},
      feedback_dir=str(tmp_path / '.optimization'),
    )
    brief = opt.build_task_brief()
    assert len(brief.split('\n')) <= 15

  def test_build_task_brief_includes_file_pointers_and_inline_todo(self, tmp_path):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = TextGradient(attribution='Fix parsing')
    opt = AgentOptimizer(
      agent,
      [p],
      context={'epoch': 0},
      feedback_dir=str(tmp_path / '.optimization'),
    )
    opt.update_todo()
    brief = opt.build_task_brief()
    assert 'epoch_*.md' in brief
    assert '## Todo' in brief
    assert 'Fix parsing' in brief

  def test_agentic_false_uses_build_prompt(self):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = TextGradient(attribution='fix')
    opt = AgentOptimizer(agent, [p], agentic=False)
    opt.step()
    prompt = agent.run.call_args[0][0]
    assert '--- Parameter' in prompt

  def test_step_writes_epoch_file_before_agent_run(self, tmp_path):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = TextGradient(attribution='improve')
    opt = AgentOptimizer(
      agent,
      [p],
      context={'epoch': 0},
      feedback_dir=str(tmp_path / '.optimization'),
    )
    opt.step()
    assert (tmp_path / '.optimization' / 'epoch_0.md').exists()

  def test_feedback_dir_raises_config_error_no_dir_no_cwd(self):
    """BUG-003: _feedback_dir raises ConfigError when no override and no agent _cwd."""
    agent = _mock_agent()
    agent._cwd = None
    p = Parameter(requires_grad=True)
    opt = AgentOptimizer(agent, [p])
    with pytest.raises(ConfigError, match='feedback_dir'):
      _ = opt._feedback_dir

  def test_feedback_dir_configurable(self):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    opt = AgentOptimizer(agent, [p], feedback_dir='/custom/path')
    assert opt._feedback_dir == '/custom/path'

  def test_write_epoch_feedback_no_gradients_writes_metrics_only(self, tmp_path):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = None
    opt = AgentOptimizer(
      agent,
      [p],
      context={'epoch': 0, 'metrics': {'accuracy': 0.6}},
      feedback_dir=str(tmp_path / '.opt'),
    )
    path = opt.write_epoch_feedback(0)
    content = path.read_text(encoding='utf-8')
    assert '# Epoch 0' in content
    assert 'accuracy: 0.6' in content
    assert '## Parameter' not in content

  def test_write_epoch_feedback_overwrites_same_epoch(self, tmp_path):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = TextGradient(attribution='v1')
    opt = AgentOptimizer(agent, [p], context={'epoch': 0}, feedback_dir=str(tmp_path / '.opt'))
    opt.write_epoch_feedback(0)
    p.grad = TextGradient(attribution='v2')
    path = opt.write_epoch_feedback(0)
    content = path.read_text(encoding='utf-8')
    assert 'v2' in content

  def test_agentic_state_not_in_checkpoint(self):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    opt = AgentOptimizer(agent, [p], context={'epoch': 1})
    opt._todo_items.append(MagicMock(text='test', epoch=0, addressed=False))
    opt._prev_metrics = {'acc': 0.5}
    state = opt.state_dict()
    assert 'todo_items' not in state
    assert 'prev_metrics' not in state

  def test_multiple_epochs_accumulate_files_and_todo(self, tmp_path):
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    opt = AgentOptimizer(agent, [p], feedback_dir=str(tmp_path / '.opt'))
    for epoch in range(3):
      p.grad = TextGradient(attribution=f'Fix issue {epoch}')
      opt._context = {'epoch': epoch}
      opt.write_epoch_feedback(epoch)
      opt.update_todo()
    assert (tmp_path / '.opt' / 'epoch_0.md').exists()
    assert (tmp_path / '.opt' / 'epoch_1.md').exists()
    assert (tmp_path / '.opt' / 'epoch_2.md').exists()
    texts = [item.text for item in opt._todo_items]
    assert 'Fix issue 0' in texts
    assert 'Fix issue 2' in texts

  def test_heterogeneous_multi_param_feedback(self, tmp_path):
    agent = _mock_agent()
    p1 = Parameter(requires_grad=True)
    p1.grad = TextGradient(attribution='text fix')
    p2 = Parameter(requires_grad=True)
    p2.grad = NumericGradient(value=0.42)
    opt = AgentOptimizer(
      agent,
      [p1, p2],
      context={'epoch': 0},
      feedback_dir=str(tmp_path / '.opt'),
    )
    path = opt.write_epoch_feedback(0)
    content = path.read_text(encoding='utf-8')
    assert 'text fix' in content
    assert 'gradient: 0.42' in content


# -- 4.8 (sub) Gradient todo_items -------------------------------------------


class TestGradientTodoItems:
  """Gradient.todo_items() and subclass overrides."""

  def test_gradient_todo_items_default_extracts_from_render(self):
    grad = NumericGradient(value=math.pi)
    items = grad.todo_items()
    assert isinstance(items, list)

  def test_text_gradient_todo_items_returns_attribution(self):
    grad = TextGradient(attribution='Fix error handling in parser module')
    items = grad.todo_items()
    assert items == ['Fix error handling in parser module']

  def test_text_gradient_todo_items_none_attribution(self):
    grad = TextGradient(text='improve accuracy')
    items = grad.todo_items()
    assert isinstance(items, list)

  def test_numeric_gradient_todo_items(self):
    grad = NumericGradient(value=0.5)
    items = grad.todo_items()
    assert isinstance(items, list)

  def test_custom_gradient_todo_items_override(self):
    @dataclass
    class _DomainGradient(Gradient):
      action: str = ''

      def render(self):
        return f'action: {self.action}'

      def accumulate(self, other):
        return _DomainGradient(action=f'{self.action}; {other.action}')

      def todo_items(self):
        return [self.action] if self.action else []

    grad = _DomainGradient(action='reduce latency by 50ms')
    assert grad.todo_items() == ['reduce latency by 50ms']

  def test_todo_items_returns_empty_for_no_actionable_content(self):
    @dataclass
    class _ShortGradient(Gradient):
      def render(self):
        return '# Header\nshort'

      def accumulate(self, other):
        return _ShortGradient()

    grad = _ShortGradient()
    assert grad.todo_items() == []


# -- 4.9 ClaudeCodeAgent cwd (BUG-054) ---------------------------------------


class TestClaudeCodeAgentCwd:
  """BUG-054: ClaudeCodeAgent cwd should track working_root."""

  def test_agent_cwd_equals_working_root_after_set_cwd(self):
    agent = ClaudeCodeAgent(cwd='/original')
    agent.set_cwd('/worktree/path')
    assert agent._cwd == '/worktree/path'

  def test_agent_cwd_uses_config_root_when_unbound(self):
    agent = ClaudeCodeAgent(cwd='/project/root')
    assert agent._cwd == '/project/root'

  def test_set_cwd_none_inherits_parent_cwd(self):
    agent = ClaudeCodeAgent(cwd='/some/path')
    agent.set_cwd(None)
    assert agent._cwd is None


# -- 4.10 load_state_dict + Store (BUG-056) ----------------------------------


class TestLoadStateDictPathParam:
  """BUG-056: Checkpoint resume should trigger store.checkout for PathParameters."""

  def test_restore_path_parameter_files_calls_checkout(self):
    mock_store = MagicMock()
    exp = Experiment('exp-1')
    exp.start()
    trainer = Trainer(store=mock_store, experiment=exp)
    mod = _StubModule()
    trainer._module = mod
    state = {
      'experiment': {
        'id': 'exp-1',
        'epoch': 2,
        'status': 'running',
        'hypothesis': None,
        'metrics': {},
        'notes': None,
        'created_at': '',
        'started_at': '',
        'completed_at': None,
        'failed_at': None,
        'cancelled_at': None,
        'error': None,
        'last_accepted_epoch': None,
        'strict_snapshot_after_complete': False,
      },
    }
    trainer._restore_path_parameter_files(state, mod)
    mock_store.checkout.assert_not_called()

  def test_restore_path_parameter_files_calls_checkout_with_path_params(self, tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'file.txt').write_text('original')

    mock_store = MagicMock()
    exp = Experiment('exp-1')
    exp.start()
    trainer = Trainer(store=mock_store, experiment=exp)

    class _PathModule(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.prompts = PathParameter(source=str(source))

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def validation_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return NoOpOptimizer([])

    mod = _PathModule()
    trainer._module = mod
    state = {
      'experiment': {
        'id': 'exp-1',
        'epoch': 2,
        'status': 'running',
        'hypothesis': None,
        'metrics': {},
        'notes': None,
        'created_at': '',
        'started_at': '',
        'completed_at': None,
        'failed_at': None,
        'cancelled_at': None,
        'error': None,
        'last_accepted_epoch': None,
        'strict_snapshot_after_complete': False,
      },
    }
    trainer._restore_path_parameter_files(state, mod)
    mock_store.checkout.assert_called_once_with('exp-1', 2, context='checkpoint resume epoch 2')


# -- 4.11 Orchestrator monitor / plateau (BUG-053) ---------------------------


class TestOrchestratorMonitor:
  """BUG-053: OrchestratorConfig.monitor defaults to None."""

  def test_orchestrator_monitor_none_with_plateau_raises(self):
    from autopilot.core.errors import ConfigError

    with pytest.raises(ConfigError, match='monitor is required'):
      OrchestratorConfig(monitor=None, plateau_window=3)

  def test_orchestrator_logs_info_when_plateau_disabled(self, caplog):
    orch = EpochOrchestrator(OrchestratorConfig(monitor=None, plateau_window=0))
    trainer = Trainer(dry_run=False)
    mod = _StubModule()
    trainer._module = mod
    config = _make_loop_config(max_epochs=1)
    with caplog.at_level(logging.INFO):
      orch.run(trainer, config)
    assert any('plateau detection is disabled' in r.message for r in caplog.records)

  def test_plateau_triggers_when_monitor_set_and_flat_metric_series(self):
    orch = EpochOrchestrator(
      OrchestratorConfig(monitor='accuracy', plateau_window=3, plateau_threshold=0.01)
    )
    orch._metric_history = [
      {'accuracy': 0.5},
      {'accuracy': 0.501},
      {'accuracy': 0.502},
    ]
    assert orch._detect_plateau(orch._metric_history) is True

  def test_plateau_disabled_when_plateau_window_zero(self):
    orch = EpochOrchestrator(OrchestratorConfig(monitor=None, plateau_window=0))
    orch._metric_history = [
      {'accuracy': 0.5},
      {'accuracy': 0.5},
      {'accuracy': 0.5},
    ]
    assert orch._detect_plateau(orch._metric_history) is False


# -- 4.13 Cross-integration / regression -------------------------------------


class TestCrossIntegration:
  """Cross-cutting integration tests."""

  def test_fit_loop_end_to_end_val_empty_dict_survives_to_experiment_complete(self):
    """BUG-059 + BUG-048 chain: empty val_metrics survives to experiment."""
    exp = Experiment('e2e-test')
    trainer = Trainer(experiment=exp)
    mod = _StubModule()
    trainer.fit(
      mod,
      train_dataloaders=DataLoader([EvalDatum(success=True)], batch_size=1),
      val_dataloaders=DataLoader([EvalDatum(success=True)], batch_size=1),
      max_epochs=1,
    )
    assert exp.status.value == 'completed'

  def test_orchestrator_stops_early_via_callback_stop_dict(self):
    """BUG-057 + orchestrator path."""

    class _StopEpoch1(Callback):
      def on_epoch_start(self, trainer, module, epoch: int):
        if epoch >= 1:
          return {'stop': True}
        return None

    orch = EpochOrchestrator()
    trainer = Trainer(callbacks=[_StopEpoch1()], loop=orch)
    mod = _StubModule()
    result = trainer.fit(
      mod,
      train_dataloaders=DataLoader([EvalDatum(success=True)], batch_size=1),
      max_epochs=5,
    )
    assert result['total_epochs'] == 1
    assert result['stop_reason'] == 'callback_stop'

  def test_trainer_experiment_metrics_use_val_prefixed_after_fit(self):
    """BUG-049 + BUG-048: experiment metrics use val-prefixed keys."""

    class _MetricModule(_StubModule):
      def __init__(self):
        super().__init__()
        self._met = _ReturnsMetric()

      def configure_optimizers(self):
        return NoOpOptimizer([])

    class _ReturnsMetric(Metric):
      higher_is_better: bool | None = True

      def update(self, datum):
        pass

      def compute(self):
        return {'score': 0.85}

    mod = _MetricModule()
    exp = Experiment('metric-test')
    trainer = Trainer(experiment=exp)
    trainer.fit(
      mod,
      train_dataloaders=DataLoader([EvalDatum(success=True)], batch_size=1),
      val_dataloaders=DataLoader([EvalDatum(success=True)], batch_size=1),
      max_epochs=1,
    )
    assert exp.status.value == 'completed'

  def test_agent_optimizer_context_auto_set_when_none(self):
    """BUG-052: AgentOptimizer context auto-wired from Trainer."""
    agent = _mock_agent()

    class _AgentModule(AutoPilotModule):
      def __init__(self):
        super().__init__()

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def validation_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return AgentOptimizer(agent, list(self.parameters()), agentic=False)

    mod = _AgentModule()
    exp = Experiment('ctx-test')
    trainer = Trainer(experiment=exp)
    trainer.fit(
      mod,
      train_dataloaders=DataLoader([EvalDatum(success=True)], batch_size=1),
      max_epochs=1,
    )

  def test_agent_optimizer_explicit_context_not_overwritten(self):
    agent = _mock_agent()
    explicit_ctx = {'epoch': 99, 'custom': True}

    class _AgentModule(AutoPilotModule):
      def __init__(self):
        super().__init__()

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def validation_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return AgentOptimizer(
          agent, list(self.parameters()), context=dict(explicit_ctx), agentic=False
        )

    mod = _AgentModule()
    trainer = Trainer()
    trainer.fit(
      mod,
      train_dataloaders=DataLoader([EvalDatum(success=True)], batch_size=1),
      max_epochs=1,
    )
    assert trainer.optimizer is not None
    assert trainer.optimizer.context['custom'] is True  # type: ignore[ty:unresolved-attribute]

  def test_trainer_auto_wires_feedback_dir_from_config(self, tmp_path):
    """BUG-003: Trainer auto-sets feedback_dir from config.root."""
    agent = _mock_agent()

    class _AgentModule(AutoPilotModule):
      def __init__(self):
        super().__init__()

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def validation_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return AgentOptimizer(agent, list(self.parameters()), agentic=False)

    mod = _AgentModule()
    config = Config(workspace=tmp_path)
    config.root = tmp_path
    exp = Experiment('feedback-wire')
    trainer = Trainer(config=config, experiment=exp)
    trainer.fit(
      mod,
      train_dataloaders=DataLoader([EvalDatum(success=True)], batch_size=1),
      max_epochs=1,
    )
    opt = trainer.optimizer
    assert opt is not None
    assert opt.feedback_dir == str(tmp_path / '.optimization')  # type: ignore[ty:unresolved-attribute]

  def test_trainer_does_not_overwrite_explicit_feedback_dir(self, tmp_path):
    """Explicit feedback_dir is not overwritten by Trainer auto-wiring."""
    agent = _mock_agent()
    explicit_dir = str(tmp_path / 'my_feedback')

    class _AgentModule(AutoPilotModule):
      def __init__(self):
        super().__init__()

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def validation_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return AgentOptimizer(
          agent, list(self.parameters()), agentic=False, feedback_dir=explicit_dir
        )

    mod = _AgentModule()
    project_root = tmp_path / 'project'
    project_root.mkdir()
    config = Config(workspace=tmp_path)
    config.root = project_root
    exp = Experiment('feedback-explicit')
    trainer = Trainer(config=config, experiment=exp)
    trainer.fit(
      mod,
      train_dataloaders=DataLoader([EvalDatum(success=True)], batch_size=1),
      max_epochs=1,
    )
    opt = trainer.optimizer
    assert opt is not None
    assert opt.feedback_dir == explicit_dir  # type: ignore[ty:unresolved-attribute]

  def test_metrics_comparator_import_propose_verify(self):
    """BUG-050: no circular import between propose and MetricsComparator."""
    from autopilot.cli.commands.propose import ProposeCommand
    from autopilot.core.comparison import MetricsComparator

    assert ProposeCommand is not None
    assert MetricsComparator is not None

  def test_val_metrics_empty_dict_asserts_equal_empty(self):
    """BUG-059: val_metrics={} key present AND value is {}."""
    loop = EpochLoop()
    trainer = Trainer(dry_run=False)
    mod = _StubModule()
    trainer._module = mod
    config = _make_loop_config(
      max_epochs=1,
      val_loader=DataLoader([EvalDatum(success=True)], batch_size=1),
    )
    result = loop.run(trainer, config)
    epoch = result['epochs'][0]
    assert 'val_metrics' in epoch
    assert epoch['val_metrics'] == {}

  def test_orchestrator_compatible_with_checkpoint_start_epoch(self):
    """BUG-058 + checkpoint: min_epoch wired from checkpoint resume."""
    orch = EpochOrchestrator()
    trainer = Trainer(dry_run=False, loop=orch)
    mod = _StubModule()
    trainer._module = mod
    config = _make_loop_config(min_epoch=2, max_epochs=4)
    result = orch.run(trainer, config)
    epochs = [e['epoch'] for e in result['epochs']]
    assert epochs == [2, 3]

  def test_complete_updates_forest_visible_metrics(self):
    """BUG-049 + BUG-048: experiment metrics visible through tree node."""
    exp = Experiment('forest-vis')
    mock_store = MagicMock()
    tree = Tree(name='test', store=mock_store)
    tree.add(Node(experiment=exp))
    trainer = Trainer(experiment=exp, tree=tree)
    mod = _StubModule()
    trainer.fit(
      mod,
      train_dataloaders=DataLoader([EvalDatum(success=True)], batch_size=1),
      max_epochs=1,
    )
    node = tree.get('forest-vis')
    assert node is not None
    assert node.experiment.status.value == 'completed'

  def test_update_todo_all_empty_todo_items(self):
    """BUG-076: when all todo_items() return [], todo stays empty."""
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = NumericGradient(value=0.1)
    opt = AgentOptimizer(agent, [p], context={'epoch': 0})
    opt.update_todo()
    assert len(opt._todo_items) == 0

  def test_context_missing_epoch_feedback_still_writes(self, tmp_path):
    """BUG-076: no epoch in context still produces valid file."""
    agent = _mock_agent()
    p = Parameter(requires_grad=True)
    p.grad = TextGradient(attribution='fix something')
    opt = AgentOptimizer(
      agent,
      [p],
      context={'metrics': {'acc': 0.5}},
      feedback_dir=str(tmp_path / '.opt'),
    )
    path = opt.write_epoch_feedback(0)
    assert path.exists()
    content = path.read_text(encoding='utf-8')
    assert 'acc: 0.5' in content

  def test_plateau_requires_min_delta_and_patience(self):
    """BUG-053: plateau only fires when within threshold over window."""
    orch = EpochOrchestrator(
      OrchestratorConfig(monitor='loss', plateau_window=3, plateau_threshold=0.001)
    )
    orch._metric_history = [
      {'loss': 1.0},
      {'loss': 0.5},
      {'loss': 0.1},
    ]
    assert orch._detect_plateau(orch._metric_history) is False
    orch._metric_history = [
      {'loss': 0.5},
      {'loss': 0.5001},
      {'loss': 0.4999},
    ]
    assert orch._detect_plateau(orch._metric_history) is True
