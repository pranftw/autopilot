"""Tests for metric normalization in Trainer._complete_experiment_success.

Covers BUG-F-003: naive prefixing produced ``train_train_loss`` when upstream
metrics already included ``train_`` prefixes.  The fix uses
``strip_metric_prefix`` to strip existing prefixes before re-adding canonical
``train_*`` / ``val_*`` prefixes.

Regression IDs:
  - BUG-F-003: double metric prefix ``train_train_*``
  - FR-017: metric normalization consistency
"""

from autopilot.core.experiment import Experiment
from autopilot.core.metric_utils import strip_metric_prefix
from autopilot.core.trainer.trainer import Trainer


class TestStripMetricPrefix:
  """Unit tests for the module-level strip_metric_prefix helper."""

  def test_strips_train_prefix(self):
    """Keys starting with 'train_' are stripped."""
    base, prefix = strip_metric_prefix('train_loss')
    assert base == 'loss'
    assert prefix == 'train_'

  def test_strips_val_prefix(self):
    """Keys starting with 'val_' are stripped."""
    base, prefix = strip_metric_prefix('val_accuracy')
    assert base == 'accuracy'
    assert prefix == 'val_'

  def test_no_prefix_returns_empty(self):
    """Keys without recognized prefix return empty prefix."""
    base, prefix = strip_metric_prefix('accuracy')
    assert base == 'accuracy'
    assert not prefix

  def test_single_pass_train_train(self):
    """Only one level of prefix is stripped (no recursive strip)."""
    base, prefix = strip_metric_prefix('train_train_loss')
    assert base == 'train_loss'
    assert prefix == 'train_'

  def test_single_pass_val_val(self):
    """Only one level of val_ prefix is stripped."""
    base, prefix = strip_metric_prefix('val_val_f1')
    assert base == 'val_f1'
    assert prefix == 'val_'

  def test_empty_string(self):
    """Empty string is handled gracefully."""
    base, prefix = strip_metric_prefix('')
    assert not base
    assert not prefix

  def test_prefix_only(self):
    """A key that is just 'train_' strips to empty base."""
    base, prefix = strip_metric_prefix('train_')
    assert not base
    assert prefix == 'train_'


class TestTrainerFinalMetricsPrefixMerging:
  """BUG-F-003: _complete_experiment_success must not double-prefix metrics."""

  def test_final_metrics_no_double_train_prefix(self):
    """Train metrics with existing train_ prefix should not become train_train_.

    BUG-F-003: upstream metrics already include ``train_`` prefix; naive
    ``f'train_{key}'`` produced ``train_train_loss``.
    """
    exp = Experiment('test-exp')
    exp.start()
    trainer = Trainer(experiment=exp)
    trainer._experiment = exp
    loop_result = {
      'epochs': [
        {
          'epoch': 0,
          'metrics': {'train_loss': 0.1},
          'val_metrics': {'loss': 0.2},
        }
      ]
    }
    trainer._complete_experiment_success(loop_result)
    assert 'train_loss' in exp.metrics
    assert 'train_train_loss' not in exp.metrics
    assert exp.metrics['train_loss'] == 0.1
    assert 'val_loss' in exp.metrics
    assert exp.metrics['val_loss'] == 0.2

  def test_no_double_train_prefix(self):
    """Blanket assertion: no metric key contains 'train_train_' or 'val_val_'.

    BUG-F-003: covers all possible double-prefix combinations in a single
    sweep across both train and val metric dicts.
    """
    exp = Experiment('test-exp')
    exp.start()
    trainer = Trainer(experiment=exp)
    trainer._experiment = exp
    loop_result = {
      'epochs': [
        {
          'epoch': 0,
          'metrics': {'train_loss': 0.5, 'accuracy': 0.9},
          'val_metrics': {'val_f1': 0.8, 'recall': 0.7},
        }
      ]
    }
    trainer._complete_experiment_success(loop_result)
    all_keys = ''.join(exp.metrics.keys())
    assert 'train_train_' not in all_keys, (
      f'double train_ prefix found in metrics: {list(exp.metrics.keys())}'
    )
    assert 'val_val_' not in all_keys, (
      f'double val_ prefix found in metrics: {list(exp.metrics.keys())}'
    )

  def test_mixed_prefixed_and_bare_keys(self):
    """Hybrid dicts with mixed prefixed/bare keys collapse to canonical form.

    BUG-F-003: train dict has ``{'train_loss': 0.5, 'accuracy': 0.9}``
    and val dict has ``{'val_f1': 0.8, 'recall': 0.7}``.  Output should be
    ``train_loss``, ``train_accuracy``, ``val_f1``, ``val_recall``.
    """
    exp = Experiment('test-exp')
    exp.start()
    trainer = Trainer(experiment=exp)
    trainer._experiment = exp
    loop_result = {
      'epochs': [
        {
          'epoch': 0,
          'metrics': {'train_loss': 0.5, 'accuracy': 0.9},
          'val_metrics': {'val_f1': 0.8, 'recall': 0.7},
        }
      ]
    }
    trainer._complete_experiment_success(loop_result)
    assert exp.metrics == {
      'train_loss': 0.5,
      'train_accuracy': 0.9,
      'val_f1': 0.8,
      'val_recall': 0.7,
    }

  def test_metric_normalization_val_only_branch_unchanged(self):
    """When train is empty but val is present, val-only path unchanged.

    The ``val is not None`` but empty-train path should still return val
    metrics unprefixed (prior semantics preserved).
    """
    exp = Experiment('test-exp')
    exp.start()
    trainer = Trainer(experiment=exp)
    trainer._experiment = exp
    loop_result = {
      'epochs': [
        {
          'epoch': 0,
          'metrics': {},
          'val_metrics': {'accuracy': 0.95},
        }
      ]
    }
    trainer._complete_experiment_success(loop_result)
    assert exp.metrics == {'accuracy': 0.95}

  def test_bare_keys_still_prefixed_correctly(self):
    """Bare keys (no prefix) get canonical train_/val_ prefixes."""
    exp = Experiment('test-exp')
    exp.start()
    trainer = Trainer(experiment=exp)
    trainer._experiment = exp
    loop_result = {
      'epochs': [
        {
          'epoch': 0,
          'metrics': {'loss': 0.3, 'accuracy': 0.8},
          'val_metrics': {'loss': 0.2, 'accuracy': 0.9},
        }
      ]
    }
    trainer._complete_experiment_success(loop_result)
    assert exp.metrics == {
      'train_loss': 0.3,
      'train_accuracy': 0.8,
      'val_loss': 0.2,
      'val_accuracy': 0.9,
    }

  def test_val_prefixed_keys_not_double_prefixed(self):
    """Val metrics with existing val_ prefix should not become val_val_."""
    exp = Experiment('test-exp')
    exp.start()
    trainer = Trainer(experiment=exp)
    trainer._experiment = exp
    loop_result = {
      'epochs': [
        {
          'epoch': 0,
          'metrics': {'loss': 0.3},
          'val_metrics': {'val_accuracy': 0.9},
        }
      ]
    }
    trainer._complete_experiment_success(loop_result)
    assert 'val_accuracy' in exp.metrics
    assert 'val_val_accuracy' not in exp.metrics
    assert exp.metrics['val_accuracy'] == 0.9
