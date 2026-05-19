"""Tests for shared test doubles in tests/doubles.py."""

from autopilot.core.gradient import NumericGradient
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from tests.doubles import DirectNumericLoss, MockEvaluationOutput


def test_direct_numeric_loss_assigns_grad():
  """Verify DirectNumericLoss assigns NumericGradient(1.0) to requires_grad parameters."""
  p1 = Parameter(items=[])
  p1.requires_grad = True
  p2 = Parameter(items=[])
  p2.requires_grad = False
  loss = DirectNumericLoss([p1, p2])
  loss.backward()
  assert isinstance(p1.grad, NumericGradient)
  assert p1.grad.value == 1.0
  assert p2.grad is None


def test_direct_numeric_loss_empty_params():
  """DirectNumericLoss with no parameters does not error."""
  loss = DirectNumericLoss([])
  loss.backward()


def test_direct_numeric_loss_forward_returns_none():
  """DirectNumericLoss.forward() returns None (Loss is not a graph node)."""
  loss = DirectNumericLoss([])
  result = loss.forward(Datum())
  assert result is None


def test_mock_evaluation_output_info():
  """MockEvaluationOutput records info messages."""
  mock = MockEvaluationOutput()
  mock.info('hello')
  mock.info('world')
  assert mock.infos == ['hello', 'world']


def test_mock_evaluation_output_warn():
  """MockEvaluationOutput records warn messages."""
  mock = MockEvaluationOutput()
  mock.warn('caution')
  assert mock.warns == ['caution']


def test_mock_evaluation_output_result():
  """MockEvaluationOutput records result tuples with ok flag."""
  mock = MockEvaluationOutput()
  mock.result({'key': 'val'})
  mock.result({'err': True}, ok=False)
  assert mock.results == [({'key': 'val'}, True), ({'err': True}, False)]
