"""Tests for core API error messages and guided recovery (Plan 02).

Covers:
  - Parameter __init_subclass__ kwarg guard (BFR-02)
  - Loss.backward differentiated error paths (BFR-03)
  - Metric.update single-Datum arity enforcement (BFR-08 / FR-16)
"""

from autopilot.ai.parameter import PathParameter
from autopilot.core.gradient import Gradient
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.parameter import COMMON_WRONG_KWARGS, Parameter
from autopilot.core.types import Datum
from typing import Any
import pytest
import re

# ---------------------------------------------------------------------------
# 2.1 Parameter __init_subclass__ kwarg guard
# ---------------------------------------------------------------------------


class TestParameterKwargGuard:
  """Parameter rejects common misuse kwargs with tripartite guidance."""

  def test_parameter_value_kwarg_error(self):
    """Parameter(value='x') raises TypeError with PathParameter/ScalarParameter guidance."""
    with pytest.raises(TypeError, match='PathParameter') as exc_info:
      Parameter(value='x')  # type: ignore[ty:unknown-argument]
    msg = str(exc_info.value)
    assert 'ScalarParameter' in msg
    assert 'value' in msg

  def test_parameter_data_kwarg_error(self):
    """Parameter(data='x') raises TypeError mentioning subclassing."""
    with pytest.raises(TypeError, match='subclass Parameter') as exc_info:
      Parameter(data='x')  # type: ignore[ty:unknown-argument]
    msg = str(exc_info.value)
    assert 'data' in msg
    assert 'Datum subclass' in msg

  @pytest.mark.parametrize('kwarg', ['text', 'content', 'prompt'])
  def test_parameter_text_content_prompt_errors(self, kwarg: str):
    """Each banned kwarg triggers a TypeError with complete guidance."""
    with pytest.raises(TypeError) as exc_info:
      Parameter(**{kwarg: 'test'})  # type: ignore[ty:invalid-argument-type]
    msg = str(exc_info.value)
    assert kwarg in msg
    assert 'PathParameter' in msg
    assert 'ScalarParameter' in msg
    assert 'subclass Parameter' in msg

  def test_parameter_multiple_wrong_kwargs(self):
    """Multiple banned kwargs all listed in the error message."""
    with pytest.raises(TypeError) as exc_info:
      Parameter(value='x', data='y')  # type: ignore[ty:unknown-argument]
    msg = str(exc_info.value)
    assert 'data' in msg
    assert 'value' in msg

  def test_parameter_valid_construction(self):
    """Parameter() and Parameter(requires_grad=False) succeed without error."""
    p1 = Parameter()
    assert p1.requires_grad is True

    p2 = Parameter(requires_grad=False)
    assert p2.requires_grad is False

  def test_parameter_subclass_pathparameter_ok(self):
    """PathParameter accepts its own kwargs without triggering the guard."""
    pp = PathParameter(source='/tmp/test', pattern='*.txt')
    assert pp.source == '/tmp/test'
    assert pp.pattern == '*.txt'

  def test_parameter_subclass_custom_no_conflict(self):
    """Custom Parameter subclass with non-banned kwargs works fine."""

    class CustomParam(Parameter):
      def __init__(self, *, source: str = 'default', **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.custom_source = source

    cp = CustomParam(source='hello')
    assert cp.custom_source == 'hello'

  def test_common_wrong_kwargs_is_frozenset(self):
    """COMMON_WRONG_KWARGS is importable and contains expected entries."""
    assert isinstance(COMMON_WRONG_KWARGS, frozenset)
    assert {'value', 'data', 'text', 'content', 'prompt'} == COMMON_WRONG_KWARGS


# ---------------------------------------------------------------------------
# 2.2 Loss.backward differentiated errors
# ---------------------------------------------------------------------------


class _FwdBrokenLoss(Loss):
  """Loss subclass that overrides forward without calling super."""

  def forward(self, data: Datum, targets: Any | None = None) -> None:
    self._accumulated.append({'data': data, 'targets': targets})

  def compute_seed_gradient(self) -> Gradient:
    return Gradient()


class _MinimalLoss(Loss):
  """Loss subclass with compute_seed_gradient for testing no-forward path."""

  def compute_seed_gradient(self) -> Gradient:
    return Gradient()


class TestLossBackwardErrors:
  """Loss.backward produces distinct errors for different failure modes."""

  def test_loss_backward_no_forward_at_all(self):
    """Fresh Loss raises 'called without prior forward()' on backward."""
    loss = _MinimalLoss()
    with pytest.raises(RuntimeError, match='called without prior forward'):
      loss.backward()

  def test_loss_backward_override_without_super(self):
    """Override that skips super() produces guidance about super().forward."""
    loss = _FwdBrokenLoss()
    datum = Datum()
    loss.forward(datum, None)
    loss.forward(datum, None)

    with pytest.raises(RuntimeError) as exc_info:
      loss.backward()
    msg = str(exc_info.value)
    assert 'super().forward(data, targets)' in msg
    assert '2 batch(es) accumulated' in msg
    assert 'compute_seed_gradient()' in msg

  def test_loss_backward_single_batch_override_without_super(self):
    """Single batch case also reports correct count."""
    loss = _FwdBrokenLoss()
    loss.forward(Datum(), None)

    with pytest.raises(RuntimeError, match='1 batch\\(es\\) accumulated'):
      loss.backward()

  def test_loss_backward_no_grad_fn(self):
    """Datum without grad_fn after valid forward path raises grad_fn error."""
    loss = _MinimalLoss()
    datum = Datum()
    loss.forward(datum, None)

    with pytest.raises(RuntimeError, match='cannot backward: data has no grad_fn'):
      loss.backward()


# ---------------------------------------------------------------------------
# 2.3 Metric.update two-args arity error
# ---------------------------------------------------------------------------


class _ArityProbeMetric(Metric):
  """Minimal Metric for testing single-datum update contract."""

  def update(self, datum: Datum) -> None:
    return None

  def compute(self) -> dict[str, float]:
    return {'x': 1.0}


class TestMetricUpdateArity:
  """Metric.update(datum) rejects two positional args with a Python TypeError."""

  def test_metric_update_two_args_error(self):
    """Passing two Datum args to update raises TypeError about positional args."""
    metric = _ArityProbeMetric()
    with pytest.raises(TypeError) as exc_info:
      metric.update(Datum(), Datum())  # type: ignore[call-arg, ty:too-many-positional-arguments]
    msg = str(exc_info.value)
    assert re.search(r'takes.*positional', msg) or 'argument' in msg

  def test_metric_update_single_arg_ok(self):
    """Single-datum update works correctly."""
    metric = _ArityProbeMetric()
    metric.update(Datum())
    assert metric.update_count == 1
    result = metric.compute()
    assert result == {'x': 1.0}
