"""Tests for query --best metric resolution with prefix normalization.

Covers FR-017: ``resolve_metric_name`` handles legacy double-prefixed keys
(``train_train_accuracy``) by applying single-pass ``train_``/``val_`` strip
semantics to find matching metrics.

Also verifies consistency with ``experiment compare`` prefix normalization
(no regression in existing val-first preference).
"""

from autopilot.cli.commands.query import resolve_metric_name
from autopilot.core.experiment import Experiment
from autopilot.core.metric_utils import metric_base_name
from autopilot.core.node import Node


def _make_node(metrics: dict[str, float]) -> Node:
  """Create a minimal Node with the given metrics dict."""
  exp = Experiment(experiment_id='test-node')
  exp.start()
  exp.complete(metrics=metrics)
  return Node(experiment=exp)


class TestStripQueryMetricPrefix:
  """Unit tests for the single-pass strip helper in query.py."""

  def test_strip_train_prefix(self):
    assert metric_base_name('train_loss') == 'loss'

  def test_strip_val_prefix(self):
    assert metric_base_name('val_accuracy') == 'accuracy'

  def test_no_prefix_unchanged(self):
    assert metric_base_name('f1') == 'f1'

  def test_single_pass_only(self):
    """train_train_accuracy strips to train_accuracy, not accuracy."""
    assert metric_base_name('train_train_accuracy') == 'train_accuracy'

  def test_single_pass_val_val(self):
    """val_val_f1 strips to val_f1, not f1."""
    assert metric_base_name('val_val_f1') == 'val_f1'


class TestResolveMetricName:
  """Tests for resolve_metric_name with val-first prefix strategy."""

  def test_prefers_val_over_train_when_both_present(self):
    """Existing behavior: val_ variant is preferred over train_ variant."""
    node = _make_node({'val_accuracy': 0.9, 'train_accuracy': 0.8})
    result = resolve_metric_name([node], 'accuracy')
    assert result == 'val_accuracy'

  def test_falls_back_to_train_when_no_val(self):
    """When only train_ variant exists, resolve to it."""
    node = _make_node({'train_accuracy': 0.8})
    result = resolve_metric_name([node], 'accuracy')
    assert result == 'train_accuracy'

  def test_falls_back_to_bare_name(self):
    """When neither prefixed variant exists, fall back to bare name."""
    node = _make_node({'accuracy': 0.7})
    result = resolve_metric_name([node], 'accuracy')
    assert result == 'accuracy'

  def test_returns_original_when_nothing_matches(self):
    """When no variant exists, return original name."""
    node = _make_node({'loss': 0.5})
    result = resolve_metric_name([node], 'accuracy')
    assert result == 'accuracy'

  def test_query_best_resolves_train_train_metric(self):
    """Legacy double-prefixed key: train_train_accuracy resolved via single strip.

    BUG-F-003 / FR-017: nodes carry only ``train_train_accuracy`` (legacy).
    After one strip the base is ``train_accuracy``; requesting ``accuracy``
    should resolve to the actual key ``train_train_accuracy``.
    """
    node = _make_node({'train_train_accuracy': 0.85})
    result = resolve_metric_name([node], 'accuracy')
    assert result == 'train_train_accuracy'

  def test_resolves_double_prefixed_val(self):
    """Legacy val_val_ keys are resolved via single strip."""
    node = _make_node({'val_val_f1': 0.75})
    result = resolve_metric_name([node], 'f1')
    assert result == 'val_val_f1'

  def test_prefers_direct_match_over_stripped(self):
    """Direct val_/train_ candidates are preferred over stripped matches."""
    node = _make_node(
      {
        'val_accuracy': 0.9,
        'train_train_accuracy': 0.85,
      }
    )
    result = resolve_metric_name([node], 'accuracy')
    assert result == 'val_accuracy'

  def test_multiple_nodes_first_match_wins(self):
    """Searches across nodes; first matching node determines resolution."""
    node_a = _make_node({'loss': 0.5})
    node_b = _make_node({'val_accuracy': 0.8})
    result = resolve_metric_name([node_a, node_b], 'accuracy')
    assert result == 'val_accuracy'

  def test_query_compare_round_trip_prefixes(self):
    """Cross-check with experiment compare normalization.

    experiment compare strips one prefix level. Ensure resolve_metric_name
    handles the same keys consistently (no regression).
    """
    node = _make_node({'train_accuracy': 0.85, 'val_accuracy': 0.9})
    result = resolve_metric_name([node], 'accuracy')
    assert result == 'val_accuracy'
    node_train_only = _make_node({'train_loss': 0.3})
    result = resolve_metric_name([node_train_only], 'loss')
    assert result == 'train_loss'
