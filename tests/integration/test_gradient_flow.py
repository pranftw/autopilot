"""Integration tests for the full gradient pipeline.

Tests end-to-end wiring: Module -> JudgeLoss -> graph.backward() -> AccumulateGrad
-> param.grad -> AgentOptimizer.

Loss.backward() now seeds the computation graph via get_current_graph().backward()
instead of directly assigning param.grad. Tests that exercise backward() must use
Module.__call__ (which records the graph via ModuleCallOperator) so that data has
grad_fn.
"""

from autopilot.ai.agents.agent import AgentResult
from autopilot.ai.gradient import (
  CollationResult,
  ConcatCollator,
  GradientCollator,
  TextGradient,
)
from autopilot.ai.loss import JudgeLoss
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.core.callbacks.callback import Callback
from autopilot.core.gradient import Gradient, NumericGradient
from autopilot.core.loss import Loss
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import MagicMock

# helpers


def _loader(n: int) -> DataLoader:
  return DataLoader(
    [EvalDatum(feedback=f'fb_{i}', metadata={'i': i}) for i in range(n)],
    batch_size=1,
  )


class _DescParam(Parameter):
  """Parameter with a custom render() for test visibility."""

  def __init__(self, desc: str, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self._desc = desc

  def render(self) -> str:
    return self._desc


def _mock_agent(output: str = 'applied') -> MagicMock:
  agent = MagicMock()
  agent.run.return_value = AgentResult(output=output)
  agent.limiter = None
  return agent


@dataclass
class _PipelineRuleGrad(Gradient):
  """Semantic gradient carrying string issues for programmatic path tests."""

  issues: list[str] = field(default_factory=list)

  def accumulate(self, other: Gradient) -> Gradient:
    if not isinstance(other, _PipelineRuleGrad):
      msg = (
        f'Cannot accumulate {type(self).__name__} with {type(other).__name__}. '
        f'Insert a conversion operator to coerce types before fan-in.'
      )
      raise TypeError(msg)
    return _PipelineRuleGrad(issues=self.issues + other.issues)

  def render(self) -> str:
    return '\n'.join(self.issues)


class _PipelineRuleLoss(Loss):
  """Loss that aggregates feedback into a _PipelineRuleGrad seed."""

  def __init__(self, parameters: list[Parameter]) -> None:
    super().__init__(parameters)
    self._issues: list[str] = []

  def forward(self, data, targets=None):
    super().forward(data, targets)
    target = targets
    if isinstance(target, Datum) and target.items:
      target = target.items[0]
    if isinstance(target, EvalDatum) and not target.success:
      self._issues.append(target.feedback)

  def compute_seed_gradient(self):
    return _PipelineRuleGrad(issues=list(self._issues))

  def reset(self):
    super().reset()
    self._issues = []


class _PipelineRuleOpt(Optimizer):
  """Optimizer that records merged rule-gradient issues into a sink list."""

  def __init__(self, params: list[Parameter], applied_issues: list[list[str]]) -> None:
    super().__init__(params)
    self._applied_issues = applied_issues

  def step(self):
    self._applied_issues.extend(
      cast(_PipelineRuleGrad, param.grad).issues
      for param in self.parameters
      if param.requires_grad and param.grad is not None
    )


# test 1: judgeloss + collator -> agentoptimizer


class TestJudgeLossCollatorToAgentOptimizer:
  def test_full_pipeline(self):
    p1 = _DescParam('rules file', requires_grad=True)
    p2 = _DescParam('config file', requires_grad=True)

    collator = MagicMock(spec=GradientCollator)
    collator.collate.return_value = CollationResult(
      context='fix error handling',
      gradients={
        p1.id: TextGradient(attribution='add rules', severity=0.7),
        p2.id: TextGradient(attribution='update config', severity=0.3),
      },
    )

    judge = MagicMock()
    opt_agent = _mock_agent()

    class _Mod(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.loss = JudgeLoss(judge, collator, [p1, p2])
        self._opt = AgentOptimizer(opt_agent, [p1, p2], agentic=False)

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return self._opt

    mod = _Mod()
    trainer = Trainer(accumulate_grad_batches=3)
    trainer.fit(mod, train_dataloaders=_loader(3), max_epochs=1)

    collator.collate.assert_called_once()
    assert len(collator.collate.call_args[0][0]) == 3

    opt_agent.run.assert_called_once()
    prompt = opt_agent.run.call_args[0][0]
    assert 'rules file' in prompt
    assert 'config file' in prompt
    assert 'add rules' in prompt
    assert 'update config' in prompt


# test 2: programmatic gradient path (no llm)


class TestProgrammaticGradientPathNoLLM:
  def test_rule_gradient_no_collator(self):
    applied_issues: list[list[str]] = []

    p = Parameter(requires_grad=True)

    class _Mod(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.param = p
        self.loss = _PipelineRuleLoss([p])
        self._opt = _PipelineRuleOpt([p], applied_issues)

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return self._opt

    data = [
      EvalDatum(success=False, feedback='missing pattern A'),
      EvalDatum(success=True),
      EvalDatum(success=False, feedback='wrong category B'),
    ]
    mod = _Mod()
    trainer = Trainer(accumulate_grad_batches=3)
    trainer.fit(mod, train_dataloaders=DataLoader(data, batch_size=1), max_epochs=1)

    assert len(applied_issues) == 1
    assert 'missing pattern A' in applied_issues[0]
    assert 'wrong category B' in applied_issues[0]


# test 3: loss accumulation window


class TestLossAccumulationWindow:
  def test_backward_called_twice_for_6_batches_accum_3(self):
    collator = MagicMock(spec=GradientCollator)
    collator.collate.return_value = CollationResult(context='c', gradients={})

    judge = MagicMock()
    p = Parameter(requires_grad=True)
    loss = JudgeLoss(judge, collator, [p])

    class _Mod(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.param = p
        self.loss = loss
        self._opt = MagicMock()
        self._opt.step = MagicMock()
        self._opt.zero_grad = MagicMock()

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return self._opt

    mod = _Mod()
    trainer = Trainer(accumulate_grad_batches=3)
    trainer.fit(mod, train_dataloaders=_loader(6), max_epochs=1)

    assert collator.collate.call_count == 2
    first_call_feedback = collator.collate.call_args_list[0][0][0]
    second_call_feedback = collator.collate.call_args_list[1][0][0]
    assert len(first_call_feedback) == 3
    assert len(second_call_feedback) == 3


# test 4: collation context flows via callback


class _CollationContextCallback(Callback):
  def on_after_backward(self, trainer: Any, module: Any) -> None:
    loss_fn = next(
      (m for m in trainer.module.modules() if isinstance(m, Loss)),
      None,
    )
    if loss_fn and loss_fn.gradients:
      opt = trainer.optimizer
      if opt and hasattr(opt, 'update_context'):
        opt.update_context(collation_context=loss_fn.gradients.context)


class TestCollationContextFlowsToOptimizer:
  def test_context_via_callback(self):
    p = Parameter(requires_grad=True)
    collator = MagicMock(spec=GradientCollator)
    collator.collate.return_value = CollationResult(
      context='reduce false positives',
      gradients={p.id: TextGradient(attribution='fix rules')},
    )
    judge = MagicMock()
    opt_agent = _mock_agent()

    class _Mod(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.param = p
        self.loss = JudgeLoss(judge, collator, [p])
        self._opt = AgentOptimizer(opt_agent, [p], agentic=False)

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return self._opt

    mod = _Mod()
    cb = _CollationContextCallback()
    trainer = Trainer(callbacks=[cb], accumulate_grad_batches=1)
    trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=1)

    prompt = opt_agent.run.call_args[0][0]
    assert '## Overall Direction' in prompt
    assert 'reduce false positives' in prompt


# test 5: concatcollator end-to-end


class TestConcatCollatorEndToEnd:
  def test_real_concat_collator(self):
    p1 = Parameter(requires_grad=True)
    p2 = Parameter(requires_grad=True)
    judge = MagicMock()
    opt_agent = _mock_agent()

    class _Mod(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.loss = JudgeLoss(judge, ConcatCollator(), [p1, p2])
        self._opt = AgentOptimizer(opt_agent, [p1, p2], agentic=False)

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return self._opt

    mod = _Mod()
    trainer = Trainer(accumulate_grad_batches=3)
    trainer.fit(mod, train_dataloaders=_loader(3), max_epochs=1)

    opt_agent.run.assert_called_once()
    prompt = opt_agent.run.call_args[0][0]
    assert 'fb_0' in prompt
    assert 'fb_1' in prompt
    assert 'fb_2' in prompt

    assert p1.grad is None
    assert p2.grad is None


# test 6: concatcollator context flows via callback


class TestCollationContextWithRealConcatCollator:
  def test_real_concat_context_flows(self):
    p = Parameter(requires_grad=True)
    judge = MagicMock()
    opt_agent = _mock_agent()

    class _Mod(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.param = p
        self.loss = JudgeLoss(judge, ConcatCollator(), [p])
        self._opt = AgentOptimizer(opt_agent, [p], agentic=False)

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return self._opt

    mod = _Mod()
    cb = _CollationContextCallback()
    trainer = Trainer(callbacks=[cb], accumulate_grad_batches=2)
    trainer.fit(mod, train_dataloaders=_loader(2), max_epochs=1)

    prompt = opt_agent.run.call_args[0][0]
    assert '## Overall Direction' in prompt
    assert 'items evaluated' in prompt


# test 7: zero_grad -> backward -> step sequence


class TestZeroGradBackwardStepSequence:
  def test_full_sequence(self):
    p = Parameter(requires_grad=True)
    collator = MagicMock(spec=GradientCollator)
    collator.collate.return_value = CollationResult(
      context='ctx',
      gradients={p.id: TextGradient(attribution='fix')},
    )
    judge = MagicMock()
    opt_agent = _mock_agent()

    class _Mod(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.param = p
        self.loss = JudgeLoss(judge, collator, [p])
        self._opt = AgentOptimizer(opt_agent, [p], agentic=False)

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return self._opt

    mod = _Mod()
    trainer = Trainer(accumulate_grad_batches=1)
    trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=1)

    opt_agent.run.assert_called_once()
    prompt = opt_agent.run.call_args[0][0]
    assert 'What to change: fix' in prompt
    assert p.grad is None


# test 8: no callback means no collation context


class TestNoCallbackMeansNoCollationContext:
  def test_no_context_without_callback(self):
    p = Parameter(requires_grad=True)
    collator = MagicMock(spec=GradientCollator)
    collator.collate.return_value = CollationResult(
      context='should not appear',
      gradients={p.id: TextGradient(attribution='fix')},
    )
    judge = MagicMock()
    opt_agent = _mock_agent()

    class _Mod(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.param = p
        self.loss = JudgeLoss(judge, collator, [p])
        self._opt = AgentOptimizer(opt_agent, [p], agentic=False)

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return self._opt

    mod = _Mod()
    trainer = Trainer(accumulate_grad_batches=1)
    trainer.fit(mod, train_dataloaders=_loader(1), max_epochs=1)

    prompt = opt_agent.run.call_args[0][0]
    assert '## Overall Direction' not in prompt


# test 9: programmatic gradient, no collator, no agent


class TestProgrammaticGradientNoCollatorNoAgent:
  def test_numeric_gradient_flow(self):
    recorded_values: list[float] = []

    class _NumOpt(Optimizer):
      def step(self):
        recorded_values.extend(
          cast(NumericGradient, param.grad).value
          for param in self.parameters
          if param.requires_grad and param.grad is not None
        )

    class _NumLoss(Loss):
      def __init__(self, params):
        super().__init__(params)
        self._count = 0

      def forward(self, data, targets=None):
        super().forward(data, targets)
        self._count += 1

      def compute_seed_gradient(self):
        return NumericGradient(value=float(self._count))

      def reset(self):
        super().reset()
        self._count = 0

    p = Parameter(requires_grad=True)

    class _Mod(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.param = p
        self.loss = _NumLoss([p])
        self._opt = _NumOpt([p])

      def forward(self, batch):
        return batch

      def training_step(self, batch, batch_idx):
        return self(batch)

      def configure_optimizers(self):
        return self._opt

    mod = _Mod()
    trainer = Trainer(accumulate_grad_batches=2)
    result = trainer.fit(mod, train_dataloaders=_loader(4), max_epochs=1)

    assert result['total_epochs'] == 1
    assert len(recorded_values) == 2
    assert recorded_values[0] == 2.0
    assert recorded_values[1] == 2.0
