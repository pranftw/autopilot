"""Tests for BUG-009: explain() mislabels optional gate failures as required.

Covers:
- Required vs optional failure segmentation in explain() output.
- Optional-only failures use optional wording, never 'required gates failed'.
- Both categories shown separately when both fail.
- All-pass path unchanged.
- Idempotent output for repeated explain() calls.
"""

from autopilot.core.models import Result
from autopilot.core.types import GateResult
from autopilot.policy.gates import Gate, MinGate
from autopilot.policy.quality_first import QualityFirstPolicy
from typing import Any, cast


class _AlwaysFailGate(MinGate):
  """Gate that always returns FAIL for testing explain() segmentation."""

  def __init__(self, metric: str, *, required: bool = True) -> None:
    super().__init__(metric, threshold=0.0, required=required)

  def forward(self, result: Result) -> GateResult:
    return GateResult.FAIL


class _AlwaysPassGate(MinGate):
  """Gate that always returns PASSED for testing explain() all-pass path."""

  def __init__(self, metric: str, *, required: bool = True) -> None:
    super().__init__(metric, threshold=0.0, required=required)

  def forward(self, result: Result) -> GateResult:
    return GateResult.PASSED


class TestExplainSeparatesRequiredAndOptionalFailures:
  """Required accuracy fails, optional f1_score fails; each in its own segment."""

  def test_required_metric_in_required_segment_only(self) -> None:
    gates: list[Gate] = [
      _AlwaysFailGate('accuracy', required=True),
      _AlwaysFailGate('f1_score', required=False),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'accuracy': 0.3, 'f1_score': 0.2})
    text = policy.explain(result)
    required_part, _optional_part = text.split(';')
    assert 'required gates failed' in required_part
    assert 'accuracy' in required_part
    assert 'f1_score' not in required_part

  def test_optional_metric_in_optional_segment_only(self) -> None:
    gates: list[Gate] = [
      _AlwaysFailGate('accuracy', required=True),
      _AlwaysFailGate('f1_score', required=False),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'accuracy': 0.3, 'f1_score': 0.2})
    text = policy.explain(result)
    _required_part, optional_part = text.split(';')
    assert 'optional gates also failed' in optional_part
    assert 'f1_score' in optional_part
    assert 'accuracy' not in optional_part


class TestExplainListsAllRequiredFailures:
  """Two required gates fail; both listed under the required failure message."""

  def test_both_required_metrics_listed(self) -> None:
    gates: list[Gate] = [
      _AlwaysFailGate('accuracy', required=True),
      _AlwaysFailGate('precision', required=True),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'accuracy': 0.1, 'precision': 0.2})
    text = policy.explain(result)
    assert 'required gates failed' in text
    assert 'accuracy' in text
    assert 'precision' in text

  def test_no_optional_wording_when_only_required_fail(self) -> None:
    gates: list[Gate] = [
      _AlwaysFailGate('accuracy', required=True),
      _AlwaysPassGate('f1_score', required=False),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'accuracy': 0.1, 'f1_score': 0.9})
    text = policy.explain(result)
    assert 'required gates failed' in text
    assert 'optional' not in text


class TestExplainLabelsOptionalOnlyFailures:
  """Only optional gates fail; no 'required gates failed' wording appears."""

  def test_optional_only_uses_optional_wording(self) -> None:
    gates: list[Gate] = [
      _AlwaysPassGate('accuracy', required=True),
      _AlwaysFailGate('f1_score', required=False),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates), human_review_on_warn=False)
    result = Result(metrics={'accuracy': 0.9, 'f1_score': 0.2})
    text = policy.explain(result)
    assert 'optional gate(s) failed' in text
    assert 'f1_score' in text
    assert 'required gates failed' not in text

  def test_optional_only_with_human_review(self) -> None:
    gates: list[Gate] = [
      _AlwaysPassGate('accuracy', required=True),
      _AlwaysFailGate('f1_score', required=False),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates), human_review_on_warn=True)
    result = Result(metrics={'accuracy': 0.9, 'f1_score': 0.2})
    text = policy.explain(result)
    assert 'optional gate(s) failed' in text
    assert 'human review triggered' in text
    assert 'required gates failed' not in text

  def test_optional_only_without_human_review_no_review_suffix(self) -> None:
    gates: list[Gate] = [
      _AlwaysFailGate('recall', required=False),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates), human_review_on_warn=False)
    result = Result(metrics={'recall': 0.1})
    text = policy.explain(result)
    assert 'human review' not in text
    assert 'recall' in text


class TestExplainAllGatesPassed:
  """All gates pass -> 'all gates passed'."""

  def test_all_pass_message(self) -> None:
    gates: list[Gate] = [
      _AlwaysPassGate('accuracy', required=True),
      _AlwaysPassGate('f1_score', required=False),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'accuracy': 0.9, 'f1_score': 0.8})
    assert policy.explain(result) == 'all gates passed'

  def test_empty_gates_all_pass(self) -> None:
    policy = QualityFirstPolicy(gates=[])
    result = Result(metrics={'accuracy': 0.9})
    assert policy.explain(result) == 'all gates passed'

  def test_single_required_passes(self) -> None:
    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.5)])
    result = Result(metrics={'accuracy': 0.9})
    assert policy.explain(result) == 'all gates passed'


class TestExplainShowsBothFailureCategories:
  """Both required and optional fail; message includes both lists separately."""

  def test_both_categories_present(self) -> None:
    gates: list[Gate] = [
      _AlwaysFailGate('accuracy', required=True),
      _AlwaysFailGate('f1_score', required=False),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'accuracy': 0.1, 'f1_score': 0.2})
    text = policy.explain(result)
    assert 'required gates failed' in text
    assert 'optional gates also failed' in text

  def test_both_categories_stable_format(self) -> None:
    gates: list[Gate] = [
      _AlwaysFailGate('accuracy', required=True),
      _AlwaysFailGate('recall', required=True),
      _AlwaysFailGate('f1_score', required=False),
      _AlwaysFailGate('bleu', required=False),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={})
    text = policy.explain(result)
    assert 'accuracy' in text
    assert 'recall' in text
    assert 'f1_score' in text
    assert 'bleu' in text
    required_segment, optional_segment = text.split(';')
    assert 'accuracy' in required_segment
    assert 'recall' in required_segment
    assert 'f1_score' in optional_segment
    assert 'bleu' in optional_segment

  def test_semicolon_separates_categories(self) -> None:
    gates: list[Gate] = [
      _AlwaysFailGate('acc', required=True),
      _AlwaysFailGate('f1', required=False),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={})
    text = policy.explain(result)
    assert ';' in text
    parts = text.split(';')
    assert len(parts) == 2


class TestExplainIdempotent:
  """Same policy and result: explain(result) == explain(result)."""

  def test_repeated_explain_returns_same_string(self) -> None:
    gates: list[Gate] = [
      _AlwaysFailGate('accuracy', required=True),
      _AlwaysFailGate('f1_score', required=False),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'accuracy': 0.1, 'f1_score': 0.2})
    first = policy.explain(result)
    second = policy.explain(result)
    assert first == second

  def test_idempotent_after_forward(self) -> None:
    gates: list[Gate] = [
      _AlwaysFailGate('accuracy', required=True),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'accuracy': 0.1})
    policy.forward(result)
    first = policy.explain(result)
    second = policy.explain(result)
    assert first == second

  def test_idempotent_all_pass(self) -> None:
    gates: list[Gate] = [
      _AlwaysPassGate('accuracy', required=True),
    ]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'accuracy': 0.9})
    first = policy.explain(result)
    second = policy.explain(result)
    assert first == second
