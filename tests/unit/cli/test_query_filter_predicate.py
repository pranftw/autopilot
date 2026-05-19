"""Tests for experiment_attr_equals predicate builder in query command."""

from autopilot.cli.commands.query import experiment_attr_equals
from unittest.mock import Mock


class TestExperimentAttrEquals:
  def test_matching_attribute(self) -> None:
    node = Mock()
    node.experiment = Mock()
    node.experiment.hypothesis = 'baseline'
    predicate = experiment_attr_equals('hypothesis', 'baseline')
    assert predicate(node) is True

  def test_mismatched_attribute(self) -> None:
    node = Mock()
    node.experiment = Mock()
    node.experiment.hypothesis = 'improved'
    predicate = experiment_attr_equals('hypothesis', 'baseline')
    assert predicate(node) is False

  def test_missing_attribute_returns_none_string(self) -> None:
    node = Mock()
    node.experiment = Mock(spec=['id', 'status'])
    predicate = experiment_attr_equals('nonexistent', 'None')
    assert predicate(node) is True

  def test_missing_attribute_mismatch(self) -> None:
    node = Mock()
    node.experiment = Mock(spec=['id', 'status'])
    predicate = experiment_attr_equals('nonexistent', 'something')
    assert predicate(node) is False

  def test_numeric_attribute_stringified(self) -> None:
    node = Mock()
    node.experiment = Mock()
    node.experiment.epoch = 5
    assert experiment_attr_equals('epoch', '5')(node) is True
    assert experiment_attr_equals('epoch', '6')(node) is False

  def test_none_attribute_matches_none_string(self) -> None:
    node = Mock()
    node.experiment = Mock()
    node.experiment.error = None
    predicate = experiment_attr_equals('error', 'None')
    assert predicate(node) is True

  def test_boolean_attribute(self) -> None:
    node = Mock()
    node.experiment = Mock()
    node.experiment.active = True
    assert experiment_attr_equals('active', 'True')(node) is True
    assert experiment_attr_equals('active', 'False')(node) is False
