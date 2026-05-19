"""Tests for Google-style docstrings on core module public APIs.

Validates that:
- pydoc.render_doc completes without raising on key classes
- Docstrings contain expected sections (Args, Returns, Raises, Attributes)
- Epoch ordering in Trainer.fit reflects EpochLoop (validation before on_train_epoch_end)
"""

from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.module.module import Module
from autopilot.core.operator import Context, Operator, OperatorNode
from autopilot.core.query import QueryBuilder
from autopilot.core.store.base import Store
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.tree import Tree
import inspect
import pydoc
import pytest


class TestHelpCompletesWithoutRaising:
  """pydoc.render_doc on core classes must complete without raising."""

  def test_help_module(self):
    result = pydoc.render_doc(Module)
    assert 'Module' in result

  def test_help_autopilot_module(self):
    result = pydoc.render_doc(AutoPilotModule)
    assert 'AutoPilotModule' in result

  def test_help_trainer(self):
    result = pydoc.render_doc(Trainer)
    assert 'Trainer' in result

  def test_help_operator(self):
    result = pydoc.render_doc(Operator)
    assert 'Operator' in result

  def test_help_store(self):
    result = pydoc.render_doc(Store)
    assert 'Store' in result

  def test_help_query_builder(self):
    result = pydoc.render_doc(QueryBuilder)
    assert 'QueryBuilder' in result

  def test_help_tree(self):
    result = pydoc.render_doc(Tree)
    assert 'Tree' in result

  def test_help_metric(self):
    result = pydoc.render_doc(Metric)
    assert 'Metric' in result


class TestModuleDocstrings:
  """Module and AutoPilotModule docstring section probes."""

  def test_module_has_attributes_section(self):
    assert 'Attributes:' in (Module.__doc__ or '')

  def test_module_attributes_lists_training(self):
    assert 'training' in (Module.__doc__ or '')

  def test_register_forward_pre_hook_has_args_and_returns(self):
    doc = inspect.getdoc(Module.register_forward_pre_hook)
    assert doc is not None
    assert 'Args:' in doc
    assert 'Returns:' in doc

  def test_register_forward_hook_has_args_and_returns(self):
    doc = inspect.getdoc(Module.register_forward_hook)
    assert doc is not None
    assert 'Args:' in doc
    assert 'Returns:' in doc

  def test_configure_optimizers_has_returns_and_raises(self):
    doc = inspect.getdoc(AutoPilotModule.configure_optimizers)
    assert doc is not None
    assert 'Returns:' in doc
    assert 'Raises:' in doc

  def test_validation_step_has_args_and_raises(self):
    doc = inspect.getdoc(AutoPilotModule.validation_step)
    assert doc is not None
    assert 'Args:' in doc
    assert 'Raises:' in doc

  def test_test_step_has_raises(self):
    doc = inspect.getdoc(AutoPilotModule.test_step)
    assert doc is not None
    assert 'Raises:' in doc


class TestTrainerDocstrings:
  """Trainer docstring section probes."""

  def test_trainer_has_attributes_section(self):
    assert 'Attributes:' in (Trainer.__doc__ or '')

  def test_trainer_module_property_has_returns(self):
    doc = inspect.getdoc(Trainer.module.fget)
    assert doc is not None
    assert 'Returns:' in doc

  def test_trainer_fit_has_args(self):
    doc = inspect.getdoc(Trainer.fit)
    assert doc is not None
    assert 'Args:' in doc

  def test_trainer_fit_has_returns(self):
    doc = inspect.getdoc(Trainer.fit)
    assert doc is not None
    assert 'Returns:' in doc

  def test_trainer_fit_has_raises(self):
    doc = inspect.getdoc(Trainer.fit)
    assert doc is not None
    assert 'Raises:' in doc

  def test_trainer_fit_epoch_ordering_validation_before_epoch_end(self):
    """Trainer.fit docstring must show validation before on_train_epoch_end."""
    doc = inspect.getdoc(Trainer.fit) or ''
    val_pos = doc.find('validation')
    epoch_end_pos = doc.find('on_train_epoch_end')
    assert val_pos > 0
    assert epoch_end_pos > 0
    assert val_pos < epoch_end_pos


class TestOperatorDocstrings:
  """Operator/Context/OperatorNode docstring section probes."""

  def test_context_has_attributes(self):
    assert 'Attributes:' in (Context.__doc__ or '')

  def test_operator_node_has_attributes(self):
    assert 'Attributes:' in (OperatorNode.__doc__ or '')

  def test_operator_has_attributes(self):
    assert 'Attributes:' in (Operator.__doc__ or '')

  def test_operator_apply_has_args_returns_raises(self):
    doc = inspect.getdoc(Operator.apply)
    assert doc is not None
    assert 'Args:' in doc
    assert 'Returns:' in doc
    assert 'Raises:' in doc

  def test_context_save_for_backward_has_args(self):
    doc = inspect.getdoc(Context.save_for_backward)
    assert doc is not None
    assert 'Args:' in doc

  def test_operator_node_call_has_args_returns(self):
    doc = inspect.getdoc(OperatorNode.__call__)
    assert doc is not None
    assert 'Args:' in doc
    assert 'Returns:' in doc


class TestStoreDocstrings:
  """Store docstring section probes."""

  def test_store_has_attributes(self):
    assert 'Attributes:' in (Store.__doc__ or '')

  def test_store_has_lock_recovery(self):
    assert 'Lock recovery:' in (Store.__doc__ or '')

  def test_store_merge_analysis_has_returns(self):
    doc = inspect.getdoc(Store.merge_analysis)
    assert doc is not None
    assert 'Returns:' in doc

  def test_store_merge_preview_has_args(self):
    doc = inspect.getdoc(Store.merge_preview)
    assert doc is not None
    assert 'Args:' in doc

  def test_store_merge_apply_has_raises(self):
    doc = inspect.getdoc(Store.merge_apply)
    assert doc is not None
    assert 'Raises:' in doc

  def test_store_merge_preview_has_returns(self):
    doc = inspect.getdoc(Store.merge_preview)
    assert doc is not None
    assert 'Returns:' in doc


class TestQueryBuilderDocstrings:
  """QueryBuilder docstring section probes."""

  def test_query_builder_has_attributes(self):
    assert 'Attributes:' in (QueryBuilder.__doc__ or '')

  def test_query_builder_filter_has_args(self):
    doc = inspect.getdoc(QueryBuilder.filter)
    assert doc is not None
    assert 'Args:' in doc

  def test_query_builder_filter_has_returns(self):
    doc = inspect.getdoc(QueryBuilder.filter)
    assert doc is not None
    assert 'Returns:' in doc


class TestTreeDocstrings:
  """Tree docstring section probes."""

  def test_tree_has_attributes(self):
    assert 'Attributes:' in (Tree.__doc__ or '')

  def test_tree_mentions_experiment_dag(self):
    doc = Tree.__doc__ or ''
    assert 'experiment DAG' in doc

  def test_tree_disambiguates_git_tree(self):
    doc = Tree.__doc__ or ''
    assert 'not' in doc.lower()
    assert 'git' in doc.lower()


class TestMetricDocstrings:
  """Metric docstring section probes."""

  def test_metric_has_attributes(self):
    assert 'Attributes:' in (Metric.__doc__ or '')

  def test_metric_forward_has_raises(self):
    doc = inspect.getdoc(Metric.forward)
    assert doc is not None
    assert 'Raises:' in doc

  def test_metric_forward_raises_not_implemented(self):
    m = Metric()
    with pytest.raises(NotImplementedError):
      m.forward()

  def test_metric_forward_docstring_mentions_non_tensor(self):
    doc = inspect.getdoc(Metric.forward)
    assert doc is not None
    assert 'non-tensor' in doc
