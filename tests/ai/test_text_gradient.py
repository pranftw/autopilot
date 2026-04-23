"""Tests for TextGradient."""

from autopilot.ai.gradient import TextGradient
from autopilot.core.gradient import Gradient
from autopilot.core.types import Datum
from helpers import NumericGradient
import pytest


class TestTextGradientType:
  def test_is_gradient_subclass(self) -> None:
    assert isinstance(TextGradient(), Gradient)
    assert isinstance(TextGradient(), Datum)


class TestTextGradientAccumulate:
  def test_accumulate_merges_items(self) -> None:
    a = TextGradient(items=[Datum(feedback='f1')])
    b = TextGradient(items=[Datum(feedback='f2'), Datum(feedback='f3')])
    result = a.accumulate(b)
    assert len(result.items) == 3

  def test_accumulate_preserves_direction(self) -> None:
    a = TextGradient(direction='fix errors')
    b = TextGradient(direction='improve speed')
    result = a.accumulate(b)
    assert result.direction == 'fix errors'

  def test_accumulate_preserves_attribution(self) -> None:
    a = TextGradient(attribution='rewrite rules')
    b = TextGradient(attribution='add cases')
    result = a.accumulate(b)
    assert result.attribution == 'rewrite rules'

  def test_accumulate_takes_max_severity(self) -> None:
    a = TextGradient(severity=0.3)
    b = TextGradient(severity=0.8)
    result = a.accumulate(b)
    assert result.severity == 0.8

  def test_accumulate_merges_metadata(self) -> None:
    a = TextGradient(metadata={'a': 1, 'shared': 'old'})
    b = TextGradient(metadata={'b': 2, 'shared': 'new'})
    result = a.accumulate(b)
    assert result.metadata == {'a': 1, 'b': 2, 'shared': 'new'}

  def test_accumulate_both_none_direction(self) -> None:
    a = TextGradient(direction=None)
    b = TextGradient(direction=None)
    result = a.accumulate(b)
    assert result.direction is None

  def test_accumulate_preserves_items_order(self) -> None:
    a = TextGradient(items=[Datum(feedback='first')])
    b = TextGradient(items=[Datum(feedback='second')])
    result = a.accumulate(b)
    assert result.items[0].feedback == 'first'
    assert result.items[1].feedback == 'second'

  def test_accumulate_cross_type_raises(self) -> None:
    with pytest.raises(AttributeError):
      TextGradient().accumulate(NumericGradient())


class TestTextGradientRender:
  def test_render_with_attribution_and_items(self) -> None:
    g = TextGradient(
      attribution='fix the rules',
      items=[Datum(feedback='missing case'), Datum(feedback='wrong output')],
      severity=0.7,
    )
    output = g.render()
    assert 'What to change: fix the rules' in output
    assert 'Supporting evidence:' in output
    assert 'missing case' in output
    assert 'wrong output' in output
    assert 'Severity: 0.70' in output

  def test_render_empty(self) -> None:
    assert TextGradient().render() == ''

  def test_render_severity_only_when_positive(self) -> None:
    assert 'Severity' not in TextGradient(severity=0.0).render()
    assert 'Severity: 0.50' in TextGradient(severity=0.5).render()

  def test_render_direction_not_in_output(self) -> None:
    g = TextGradient(direction='improve everything')
    assert 'improve everything' not in g.render()

  def test_render_none_attribution(self) -> None:
    g = TextGradient(attribution=None)
    assert 'What to change:' not in g.render()

  def test_render_empty_attribution(self) -> None:
    g = TextGradient(attribution='')
    assert 'What to change:' not in g.render()


class TestTextGradientSerialization:
  def test_to_dict_includes_extra_fields(self) -> None:
    g = TextGradient(direction='dir', attribution='attr', severity=0.5)
    d = g.to_dict()
    assert d['direction'] == 'dir'
    assert d['attribution'] == 'attr'
    assert d['severity'] == 0.5

  def test_from_dict_roundtrip(self) -> None:
    g = TextGradient(
      direction='fix',
      attribution='rules',
      severity=0.8,
      items=[Datum(feedback='evidence')],
      metadata={'key': 'val'},
    )
    d = g.to_dict()
    g2 = TextGradient.from_dict(d)
    assert g2.direction == 'fix'
    assert g2.attribution == 'rules'
    assert g2.severity == 0.8
    assert len(g2.items) == 1
    assert g2.items[0].feedback == 'evidence'
    assert g2.metadata == {'key': 'val'}
    assert g2.id == g.id

  def test_from_dict_with_unknown_keys(self) -> None:
    d = {'direction': 'x', 'unknown_key': 'ignored'}
    g = TextGradient.from_dict(d)
    assert g.direction == 'x'

  def test_from_dict_missing_optional_keys(self) -> None:
    g = TextGradient.from_dict({})
    assert g.direction is None
    assert g.attribution is None
    assert g.severity == 0.0
