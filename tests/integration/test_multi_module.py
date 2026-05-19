"""Integration tests for the multi-module pipeline example.

Validates:
  1. Pipeline forward produces output with grad_fn (graph wiring)
  2. All parameters receive non-None gradients after backward
  3. CustomAttributionOperator transforms TextGradient with module_name
  4. run_trainer.py completes without exception (exit code 0)
  5. Backward visitation order: writer -> researcher -> planner
"""

from autopilot.ai.gradient import TextGradient
from autopilot.core.gradient import Gradient, NumericGradient
from autopilot.core.operator import Context
from autopilot.core.types import EvalDatum
from pathlib import Path
import subprocess
import sys

MULTI_MODULE_DIR = Path(__file__).resolve().parents[2] / 'examples' / 'multi_module'


def _with_example_path():
  """Ensure the multi_module example dir is on sys.path, return cleanup."""
  path_str = str(MULTI_MODULE_DIR)
  sys.path.insert(0, path_str)
  return path_str


def _import_example_modules():
  """Import example modules after path setup; returns classes and run_trainer.main."""
  from multi_module.loss import SimpleLoss
  from multi_module.module import AgentModule, CustomAttributionOperator, Pipeline
  from run_trainer import main

  return AgentModule, CustomAttributionOperator, Pipeline, SimpleLoss, main


class TestPipelineForward:
  """Pipeline()(EvalDatum(...)) runs forward; output has a grad_fn attribute."""

  def test_pipeline_forward_produces_grad_fn(self):
    _with_example_path()
    try:
      _, _, pipeline_cls, _, _ = _import_example_modules()
      pipeline = pipeline_cls()
      inp = EvalDatum(items=[], metadata={'task': 'test'})
      output = pipeline(inp)

      assert output is not None
      assert isinstance(output, EvalDatum)
      assert output.grad_fn is not None, (
        'Pipeline output should have grad_fn set by ModuleCallOperator'
      )
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))

  def test_pipeline_forward_metadata_traces_through(self):
    _with_example_path()
    try:
      _, _, pipeline_cls, _, _ = _import_example_modules()
      pipeline = pipeline_cls()
      inp = EvalDatum(items=[], metadata={'task': 'trace test'})
      output = pipeline(inp)

      assert output.metadata['agent'] == 'writer'
      assert output.metadata['stage'] == 'writer'
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))

  def test_agent_module_forward_produces_grad_fn(self):
    _with_example_path()
    try:
      agent_module_cls, _, _, _, _ = _import_example_modules()
      agent = agent_module_cls('test_agent')
      inp = EvalDatum(items=[], metadata={'task': 'unit'})
      output = agent(inp)

      assert output.grad_fn is not None
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))


class TestPipelineBackwardAllParamsGetGrad:
  """After loss.backward(), planner.prompt, researcher.prompt, and writer.prompt
  all receive non-None gradients."""

  def test_all_params_get_grad(self):
    _with_example_path()
    try:
      _, _, pipeline_cls, simple_loss_cls, _ = _import_example_modules()
      pipeline = pipeline_cls()
      loss = simple_loss_cls()

      inp = EvalDatum(items=[], metadata={'task': 'test', 'expected': 'comprehensive'})
      output = pipeline(inp)
      assert output.grad_fn is not None

      loss(output, inp)
      loss.backward()

      assert pipeline.planner.prompt.grad is not None, (
        'planner.prompt should have gradient after backward'
      )
      assert pipeline.researcher.prompt.grad is not None, (
        'researcher.prompt should have gradient after backward'
      )
      assert pipeline.writer.prompt.grad is not None, (
        'writer.prompt should have gradient after backward'
      )
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))

  def test_all_grads_are_gradient_instances(self):
    _with_example_path()
    try:
      _, _, pipeline_cls, simple_loss_cls, _ = _import_example_modules()
      pipeline = pipeline_cls()
      loss = simple_loss_cls()

      inp = EvalDatum(items=[], metadata={'task': 'check types'})
      output = pipeline(inp)
      loss(output, inp)
      loss.backward()

      for name, child in pipeline.named_children():
        assert isinstance(child.prompt.grad, Gradient), (
          f'{name}.prompt.grad should be a Gradient instance'
        )
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))

  def test_frozen_param_gets_no_grad(self):
    _with_example_path()
    try:
      _, _, pipeline_cls, simple_loss_cls, _ = _import_example_modules()
      pipeline = pipeline_cls()
      pipeline.researcher.prompt.requires_grad = False
      loss = simple_loss_cls()

      inp = EvalDatum(items=[], metadata={'task': 'frozen test'})
      output = pipeline(inp)
      loss(output, inp)
      loss.backward()

      assert pipeline.planner.prompt.grad is not None
      assert pipeline.researcher.prompt.grad is None, (
        'frozen parameter should not accumulate gradient'
      )
      assert pipeline.writer.prompt.grad is not None
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))


class TestCustomAttributionOperator:
  """CustomAttributionOperator backward produces transformed TextGradient."""

  def test_text_gradient_gets_module_name(self):
    _with_example_path()
    try:
      _, custom_attribution_cls, _, _, _ = _import_example_modules()
      ctx = Context()
      custom_attribution_cls.forward(ctx, EvalDatum(), module_name='writer')

      grad_in = TextGradient(text='improve output', severity=0.5)
      result = custom_attribution_cls.backward(ctx, grad_in)

      assert isinstance(result, tuple)
      assert len(result) == 1
      grad_out = result[0]
      assert isinstance(grad_out, TextGradient)
      assert 'writer' in grad_out.text
      assert 'Fix writer' in grad_out.text
      assert grad_out.attribution == 'Error attributed to writer'
      assert grad_out.severity == 0.5
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))

  def test_non_text_gradient_passes_through(self):
    _with_example_path()
    try:
      _, custom_attribution_cls, _, _, _ = _import_example_modules()
      ctx = Context()
      custom_attribution_cls.forward(ctx, EvalDatum(), module_name='planner')

      grad_in = NumericGradient(value=1.5)
      result = custom_attribution_cls.backward(ctx, grad_in)

      assert isinstance(result, tuple)
      assert len(result) == 1
      assert result[0] is grad_in
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))

  def test_none_direction_handled(self):
    _with_example_path()
    try:
      _, custom_attribution_cls, _, _, _ = _import_example_modules()
      ctx = Context()
      custom_attribution_cls.forward(ctx, EvalDatum(), module_name='researcher')

      grad_in = TextGradient(text=None, severity=0.3)
      result = custom_attribution_cls.backward(ctx, grad_in)

      grad_out = result[0]
      assert 'Fix researcher' in grad_out.text
      assert grad_out.attribution == 'Error attributed to researcher'
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))

  def test_apply_produces_datum_with_grad_fn(self):
    _with_example_path()
    try:
      _, custom_attribution_cls, _, _, _ = _import_example_modules()
      inp = EvalDatum(items=[], metadata={'test': True})
      output = custom_attribution_cls.apply(inp, module_name='writer')

      assert isinstance(output, EvalDatum)
      assert output.grad_fn is not None
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))

  def test_attribution_wired_in_pipeline_e2e(self):
    """CustomAttributionOperator is wired in Pipeline.forward and transforms
    gradients during end-to-end backward."""
    _with_example_path()
    try:
      _, _, pipeline_cls, simple_loss_cls, _ = _import_example_modules()
      pipeline = pipeline_cls()
      loss = simple_loss_cls()

      inp = EvalDatum(items=[], metadata={'task': 'attribution e2e'})
      output = pipeline(inp)
      loss(output, inp)
      loss.backward()

      for name in ['planner', 'researcher', 'writer']:
        child = getattr(pipeline, name)
        grad = child.prompt.grad
        assert grad is not None, f'{name}.prompt.grad missing'
        assert isinstance(grad, TextGradient)
        assert name in grad.text, f'{name} attribution not in gradient text: {grad.text}'
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))


class TestTrainingRuns:
  """run_trainer.py with --max-epochs 1 completes without exception."""

  def test_training_runs_in_process(self):
    _with_example_path()
    try:
      _, _, _, _, main = _import_example_modules()
      result = main(['--max-epochs', '1'])
      assert result is not None
      assert result['total_epochs'] == 1
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))

  def test_training_runs_subprocess(self):
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
      [
        sys.executable,
        '-c',
        (
          'import sys; sys.path.insert(0, "examples/multi_module"); '
          'from run_trainer import main; main(["--max-epochs", "1"])'
        ),
      ],
      capture_output=True,
      text=True,
      cwd=str(repo_root),
      check=False,
    )
    assert proc.returncode == 0, f'run_trainer.py failed with stderr:\n{proc.stderr}'
    assert 'Training complete' in proc.stdout

  def test_training_multiple_epochs(self):
    _with_example_path()
    try:
      _, _, _, _, main = _import_example_modules()
      result = main(['--max-epochs', '3'])
      assert result['total_epochs'] == 3
      assert len(result.get('epochs', [])) == 3
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))


class TestGradientFlowOrder:
  """Backward visitation order: writer -> researcher -> planner.

  Each module's backward_transform appends its name to a shared list;
  after loss.backward(), assert the list equals ['writer', 'researcher', 'planner'].
  """

  def test_backward_order_writer_researcher_planner(self):
    _with_example_path()
    try:
      _, _, pipeline_cls, simple_loss_cls, _ = _import_example_modules()
      pipeline = pipeline_cls()
      loss = simple_loss_cls()

      visit_order = []

      original_planner_bt = pipeline.planner.backward_transform
      original_researcher_bt = pipeline.researcher.backward_transform
      original_writer_bt = pipeline.writer.backward_transform

      def make_tracker(name, original):
        def tracked_backward_transform(ctx, grad_output):
          visit_order.append(name)
          return original(ctx, grad_output)

        return tracked_backward_transform

      pipeline.planner.backward_transform = make_tracker('planner', original_planner_bt)
      pipeline.researcher.backward_transform = make_tracker('researcher', original_researcher_bt)
      pipeline.writer.backward_transform = make_tracker('writer', original_writer_bt)

      inp = EvalDatum(items=[], metadata={'task': 'order test'})
      output = pipeline(inp)
      loss(output, inp)
      loss.backward()

      assert visit_order == ['writer', 'researcher', 'planner'], (
        f'Expected backward order [writer, researcher, planner], got {visit_order}'
      )
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))

  def test_backward_order_all_params_populated_after(self):
    _with_example_path()
    try:
      _, _, pipeline_cls, simple_loss_cls, _ = _import_example_modules()
      pipeline = pipeline_cls()
      loss = simple_loss_cls()

      inp = EvalDatum(items=[], metadata={'task': 'full flow'})
      output = pipeline(inp)
      loss(output, inp)
      loss.backward()

      for name in ['planner', 'researcher', 'writer']:
        child = getattr(pipeline, name)
        assert child.prompt.grad is not None, (
          f'{name}.prompt.grad should be populated after backward'
        )
    finally:
      sys.path.remove(str(MULTI_MODULE_DIR))
