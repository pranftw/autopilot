"""Optimizer step full evidence tests.

Verifies that _finalize_epoch emits a context entry with
_type == 'optimizer_step' containing post-step parameter summaries
for each accepted epoch when the optimizer does not own step context.
"""

from autopilot.core.decision import DecisionEntry
from autopilot.core.experiment import Experiment
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from tests.doubles import DirectNumericLoss, NoopEvalModule, NoOpOptimizer


class SingleParamModule(NoopEvalModule):
  """Module with one parameter for evidence testing."""

  def __init__(self) -> None:
    super().__init__()
    self.weight = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.weight])

  def configure_optimizers(self):
    return NoOpOptimizer([self.weight])


class TwoParamModule(NoopEvalModule):
  """Module with two parameters."""

  def __init__(self) -> None:
    super().__init__()
    self.alpha = Parameter(requires_grad=True)
    self.beta = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.alpha, self.beta])

  def configure_optimizers(self):
    return NoOpOptimizer([self.alpha, self.beta])


def _make_trainer_and_fit(module, max_epochs=1):
  """Create trainer with experiment and run fit."""
  experiment = Experiment(experiment_id='test-exp', hypothesis='h')
  trainer = Trainer(experiment=experiment)
  trainer.fit(module, train_dataloaders=[[1]], max_epochs=max_epochs)
  return experiment


def test_optimizer_step_emits_context_with_param_summaries():
  """Optimizer step context entry has _type and param_summaries."""
  module = SingleParamModule()
  experiment = _make_trainer_and_fit(module)
  step_entries = [
    e
    for e in experiment.context_log
    if e.metadata.get('_type') == DecisionEntry.OPTIMIZER_STEP_TYPE
  ]
  assert len(step_entries) >= 1
  entry = step_entries[0]
  assert 'param_summaries' in entry.metadata
  assert isinstance(entry.metadata['param_summaries'], list)
  assert len(entry.metadata['param_summaries']) >= 1


def test_optimizer_step_evidence_skipped_for_agentic():
  """AgentOptimizer with agentic=True does not emit optimizer_step entries."""
  from unittest.mock import MagicMock, patch

  module = SingleParamModule()
  experiment = Experiment(experiment_id='agentic-exp', hypothesis='h')

  mock_opt = MagicMock()
  mock_opt.owns_step_gradient_context = True
  mock_opt.param_groups = [{'params': list(module.parameters()), 'lr': 1.0}]

  with patch.object(module, 'configure_optimizers', return_value=mock_opt):
    trainer = Trainer(experiment=experiment)
    trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  step_entries = [
    e
    for e in experiment.context_log
    if e.metadata.get('_type') == DecisionEntry.OPTIMIZER_STEP_TYPE
  ]
  assert len(step_entries) == 0


def test_optimizer_step_param_summaries_structure():
  """Each param summary has param_name, param_type, value_preview keys."""
  module = TwoParamModule()
  experiment = _make_trainer_and_fit(module)
  step_entries = [
    e
    for e in experiment.context_log
    if e.metadata.get('_type') == DecisionEntry.OPTIMIZER_STEP_TYPE
  ]
  assert len(step_entries) >= 1
  summaries = step_entries[0].metadata['param_summaries']
  for summary in summaries:
    assert 'param_name' in summary
    assert 'param_type' in summary
    assert 'value_preview' in summary


def test_optimizer_step_per_epoch():
  """Multi-epoch fit produces one optimizer_step entry per accepted epoch."""
  module = SingleParamModule()
  experiment = _make_trainer_and_fit(module, max_epochs=3)
  step_entries = [
    e
    for e in experiment.context_log
    if e.metadata.get('_type') == DecisionEntry.OPTIMIZER_STEP_TYPE
  ]
  assert len(step_entries) == 3
  epochs = [e.metadata['epoch'] for e in step_entries]
  assert epochs == [0, 1, 2]


def test_optimizer_step_gate_rejected_no_step_evidence():
  """When gate rejects, no optimizer_step context entry is emitted."""
  from autopilot.policy.gates import MinGate
  from autopilot.policy.quality_first import QualityFirstPolicy

  module = SingleParamModule()
  experiment = Experiment(experiment_id='gate-reject-exp', hypothesis='h')
  policy = QualityFirstPolicy(gates=[MinGate('nonexistent_metric', threshold=0.99)])
  trainer = Trainer(experiment=experiment, policy=policy)
  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  step_entries = [
    e
    for e in experiment.context_log
    if e.metadata.get('_type') == DecisionEntry.OPTIMIZER_STEP_TYPE
  ]
  assert len(step_entries) == 0


def test_value_preview_truncated():
  """value_preview is capped at 200 characters."""
  from autopilot.core.parameter import ScalarParameter
  from autopilot.core.trainer.journal import PARAM_SUMMARY_MAX_CHARS, build_param_summary_row

  param = ScalarParameter(value='x' * 500)
  row = build_param_summary_row('big_param', param)
  assert len(row['value_preview']) <= PARAM_SUMMARY_MAX_CHARS
