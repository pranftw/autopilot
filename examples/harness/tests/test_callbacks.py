"""Tests for harness callbacks."""

from harness.callbacks import DeployCallback, MetricsWriterCallback, OptimizerContextCallback
from unittest.mock import MagicMock
import json


class TestMetricsWriterCallback:
  """Tests for MetricsWriterCallback."""

  def _make_trainer(self, tmp_path, experiment_id='exp-1'):
    """Build a mock trainer with config.experiment_path returning tmp_path."""
    trainer = MagicMock()
    trainer.experiment = MagicMock()
    trainer.experiment.id = experiment_id
    trainer.config = MagicMock()
    trainer.config.experiment_path.return_value = tmp_path
    return trainer

  def test_writes_json(self, tmp_path):
    cb = MetricsWriterCallback()
    trainer = self._make_trainer(tmp_path)
    result = MagicMock()
    result.metrics = {'task_success_rate': 0.75, 'tool_recall': 0.9}

    cb.on_epoch_end(trainer, MagicMock(), epoch=2, result=result)

    path = tmp_path / 'epoch_2_metrics.json'
    assert path.exists()
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data == {'task_success_rate': 0.75, 'tool_recall': 0.9}

  def test_experiment_path_called_with_slug(self, tmp_path):
    cb = MetricsWriterCallback()
    trainer = self._make_trainer(tmp_path, experiment_id='my-exp')
    result = MagicMock()
    result.metrics = {'acc': 0.5}

    cb.on_epoch_end(trainer, MagicMock(), epoch=0, result=result)

    trainer.config.experiment_path.assert_called_once_with(slug='my-exp')

  def test_no_result_no_file(self, tmp_path):
    cb = MetricsWriterCallback()
    trainer = self._make_trainer(tmp_path)

    cb.on_epoch_end(trainer, MagicMock(), epoch=0, result=None)

    assert not list(tmp_path.iterdir())

  def test_no_experiment_no_file(self, tmp_path):
    cb = MetricsWriterCallback()
    trainer = MagicMock()
    trainer.experiment = None
    result = MagicMock()
    result.metrics = {'x': 1.0}

    cb.on_epoch_end(trainer, MagicMock(), epoch=0, result=result)

    assert not list(tmp_path.iterdir())


class TestOptimizerContextCallback:
  """Tests for OptimizerContextCallback."""

  def _make_trainer_with_agent_optimizer(self):
    """Build a mock trainer whose optimizer is a mock AgentOptimizer."""
    from autopilot.ai.optimizer import AgentOptimizer

    trainer = MagicMock()
    opt = MagicMock(spec=AgentOptimizer)
    trainer.optimizer = opt
    return trainer, opt

  def test_on_epoch_start_updates_epoch(self):
    trainer, opt = self._make_trainer_with_agent_optimizer()
    cb = OptimizerContextCallback()

    cb.on_epoch_start(trainer, MagicMock(), epoch=3)

    opt.update_context.assert_called_once_with(epoch=3)

  def test_on_epoch_end_updates_metrics(self):
    trainer, opt = self._make_trainer_with_agent_optimizer()
    cb = OptimizerContextCallback()
    result = MagicMock()
    result.metrics = {'acc': 0.8, 'loss': 0.2}

    cb.on_epoch_end(trainer, MagicMock(), epoch=1, result=result)

    opt.update_context.assert_called_once_with(metrics={'acc': 0.8, 'loss': 0.2})

  def test_on_epoch_end_no_result_skips(self):
    trainer, opt = self._make_trainer_with_agent_optimizer()
    cb = OptimizerContextCallback()

    cb.on_epoch_end(trainer, MagicMock(), epoch=1, result=None)

    opt.update_context.assert_not_called()

  def test_non_agent_optimizer_skips(self):
    trainer = MagicMock()
    trainer.optimizer = MagicMock()  # not spec'd as AgentOptimizer
    cb = OptimizerContextCallback()

    # should not raise
    cb.on_epoch_start(trainer, MagicMock(), epoch=0)
    cb.on_epoch_end(trainer, MagicMock(), epoch=0, result=MagicMock())


class TestDeployCallback:
  """Tests for DeployCallback."""

  def test_logs_parameter_names(self, capsys):
    cb = DeployCallback()
    module = MagicMock()
    module.named_parameters.return_value = [
      ('system_prompt', MagicMock()),
      ('policies', MagicMock()),
      ('tools_code', MagicMock()),
    ]

    cb.on_fit_end(MagicMock(), module)

    captured = capsys.readouterr()
    assert '[deploy]' in captured.out
    assert 'system_prompt' in captured.out
    assert 'policies' in captured.out
    assert 'tools_code' in captured.out

  def test_empty_parameters(self, capsys):
    cb = DeployCallback()
    module = MagicMock()
    module.named_parameters.return_value = []

    cb.on_fit_end(MagicMock(), module)

    captured = capsys.readouterr()
    assert '[deploy]' in captured.out
    assert '[]' in captured.out
