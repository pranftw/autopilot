"""Tests for TextGradient."""

from autopilot.ai.gradient import TextGradient
from autopilot.core.gradient import Gradient, NumericGradient
from autopilot.core.types import Datum, EvalDatum
from typing import Any, cast
import pytest


class TestTextGradientType:
  def test_is_gradient_subclass(self) -> None:
    assert isinstance(TextGradient(), Gradient)
    assert isinstance(TextGradient(), Datum)


class TestTextGradientAccumulate:
  def test_accumulate_merges_text(self) -> None:
    """Both text values present -> concatenated with '; '."""
    a = TextGradient(text='fix errors')
    b = TextGradient(text='improve speed')
    result = a.accumulate(b)
    assert result.text == 'fix errors; improve speed'

  def test_accumulate_merges_attribution(self) -> None:
    """Both attributions present -> concatenated with '; '."""
    a = TextGradient(attribution='rewrite rules')
    b = TextGradient(attribution='add cases')
    result = a.accumulate(b)
    assert result.attribution == 'rewrite rules; add cases'

  def test_accumulate_max_severity(self) -> None:
    a = TextGradient(severity=0.3)
    b = TextGradient(severity=0.7)
    result = a.accumulate(b)
    assert result.severity == 0.7

  def test_accumulate_merges_items(self) -> None:
    a = TextGradient(items=[EvalDatum(feedback='f1')])
    b = TextGradient(items=[EvalDatum(feedback='f2'), EvalDatum(feedback='f3')])
    result = a.accumulate(b)
    assert len(result.items) == 3

  def test_text_gradient_cross_type_raises(self) -> None:
    """TextGradient().accumulate(NumericGradient()) -> TypeError."""
    with pytest.raises(TypeError, match='Cannot accumulate'):
      TextGradient().accumulate(cast(Any, NumericGradient()))

  def test_text_gradient_cross_type_with_base_gradient(self) -> None:
    with pytest.raises(TypeError, match='Cannot accumulate'):
      TextGradient().accumulate(cast(Any, Gradient()))

  def test_accumulate_one_text_none(self) -> None:
    """One side text None -> non-None side wins."""
    a = TextGradient(text=None)
    b = TextGradient(text='fix it')
    result = a.accumulate(b)
    assert result.text == 'fix it'

  def test_accumulate_one_text_none_reversed(self) -> None:
    """Self has text, other is None -> self text preserved."""
    a = TextGradient(text='fix it')
    b = TextGradient(text=None)
    result = a.accumulate(b)
    assert result.text == 'fix it'

  def test_accumulate_both_text_none(self) -> None:
    """Both text None -> result text is None."""
    a = TextGradient(text=None)
    b = TextGradient(text=None)
    result = a.accumulate(b)
    assert result.text is None

  def test_accumulate_one_attribution_none(self) -> None:
    a = TextGradient(attribution=None)
    b = TextGradient(attribution='add cases')
    result = a.accumulate(b)
    assert result.attribution == 'add cases'

  def test_accumulate_both_attributions_none(self) -> None:
    a = TextGradient(attribution=None)
    b = TextGradient(attribution=None)
    result = a.accumulate(b)
    assert result.attribution is None

  def test_accumulate_preserves_items_order(self) -> None:
    a = TextGradient(items=[EvalDatum(feedback='first')])
    b = TextGradient(items=[EvalDatum(feedback='second')])
    result = a.accumulate(b)
    first_item = result.items[0]
    second_item = result.items[1]
    assert isinstance(first_item, EvalDatum)
    assert isinstance(second_item, EvalDatum)
    assert first_item.feedback == 'first'
    assert second_item.feedback == 'second'

  def test_accumulate_empty_items(self) -> None:
    a = TextGradient(items=[])
    b = TextGradient(items=[])
    result = a.accumulate(b)
    assert result.items == []

  def test_accumulate_severity_both_zero(self) -> None:
    a = TextGradient(severity=0.0)
    b = TextGradient(severity=0.0)
    result = a.accumulate(b)
    assert result.severity == 0.0

  def test_accumulate_full_merge(self) -> None:
    a = TextGradient(
      text='reduce errors',
      attribution='fix prompt template',
      severity=0.6,
      items=[EvalDatum(feedback='ev1')],
    )
    b = TextGradient(
      text='improve coverage',
      attribution='add edge cases',
      severity=0.9,
      items=[EvalDatum(feedback='ev2')],
    )
    result = a.accumulate(b)
    assert result.text == 'reduce errors; improve coverage'
    assert result.attribution == 'fix prompt template; add edge cases'
    assert result.severity == 0.9
    assert len(result.items) == 2


class TestTextGradientRender:
  def test_render_includes_text(self) -> None:
    g = TextGradient(text='improve accuracy', attribution='fix prompt')
    rendered = g.render()
    assert 'Text: improve accuracy' in rendered
    assert 'What to change: fix prompt' in rendered

  def test_render_text_only(self) -> None:
    g = TextGradient(text='reduce errors')
    rendered = g.render()
    assert 'Text: reduce errors' in rendered

  def test_render_no_text(self) -> None:
    g = TextGradient(attribution='fix prompt')
    rendered = g.render()
    assert 'Text:' not in rendered
    assert 'What to change: fix prompt' in rendered

  def test_render_all_fields(self) -> None:
    g = TextGradient(
      text='improve quality',
      attribution='rewrite intro',
      severity=0.8,
      items=[EvalDatum(feedback='too vague')],
    )
    rendered = g.render()
    assert 'Text: improve quality' in rendered
    assert 'What to change: rewrite intro' in rendered
    assert 'Severity: 0.80' in rendered
    assert 'too vague' in rendered

  def test_render_empty_gradient(self) -> None:
    g = TextGradient()
    assert not g.render()

  def test_render_severity_only_when_positive(self) -> None:
    assert 'Severity' not in TextGradient(severity=0.0).render()
    assert 'Severity: 0.50' in TextGradient(severity=0.5).render()

  def test_render_text_in_output(self) -> None:
    g = TextGradient(text='improve everything')
    assert 'Text: improve everything' in g.render()

  def test_render_none_attribution(self) -> None:
    g = TextGradient(attribution=None)
    assert 'What to change:' not in g.render()

  def test_render_empty_attribution(self) -> None:
    g = TextGradient(attribution='')
    assert 'What to change:' not in g.render()

  def test_render_with_eval_datum_evidence(self) -> None:
    g = TextGradient(
      attribution='fix the rules',
      items=[EvalDatum(feedback='missing case'), EvalDatum(feedback='wrong output')],
      severity=0.7,
    )
    output = g.render()
    assert 'Supporting evidence:' in output
    assert 'missing case' in output
    assert 'wrong output' in output

  def test_render_non_eval_datum_items_skipped(self) -> None:
    """Non-EvalDatum items are skipped in render (no getattr fallback)."""
    g = TextGradient(
      text='fix it',
      items=[Datum()],
      severity=0.5,
    )
    output = g.render()
    assert 'Text: fix it' in output
    assert 'Supporting evidence:' in output
    assert 'Severity: 0.50' in output

  def test_render_mixed_eval_and_plain_datum_items(self) -> None:
    """Only EvalDatum items contribute evidence lines."""
    g = TextGradient(
      items=[
        EvalDatum(feedback='real evidence'),
        Datum(),
        EvalDatum(feedback='more evidence'),
      ],
    )
    output = g.render()
    assert 'real evidence' in output
    assert 'more evidence' in output

  def test_render_error_message_fallback(self) -> None:
    """EvalDatum with error_message but no feedback still renders."""
    g = TextGradient(items=[EvalDatum(error_message='something broke')])
    output = g.render()
    assert 'something broke' in output

  def test_text_gradient_render_uses_text_heading(self) -> None:
    """Heading for self.text is 'Text:', not legacy 'Direction:'."""
    g = TextGradient(text='improve')
    assert 'Text: improve' in g.render()

  def test_text_gradient_render_no_direction_heading(self) -> None:
    """Legacy 'Direction:' heading must not appear in render output."""
    g = TextGradient(text='x')
    assert 'Direction:' not in g.render()

  def test_text_gradient_render_empty_text_no_heading(self) -> None:
    """Empty string text skipped by the 'if self.text:' guard."""
    g = TextGradient(text='')
    assert 'Text:' not in g.render()


class TestTextGradientSerialization:
  def test_to_dict_includes_extra_fields(self) -> None:
    g = TextGradient(text='dir', attribution='attr', severity=0.5)
    d = g.to_dict()
    assert d['text'] == 'dir'
    assert d['attribution'] == 'attr'
    assert d['severity'] == 0.5

  def test_text_gradient_to_dict_roundtrip(self) -> None:
    """Preserves text, attribution, severity through serialization."""
    g = TextGradient(
      text='fix',
      attribution='rules',
      severity=0.8,
      items=[EvalDatum(feedback='evidence')],
    )
    d = g.to_dict()
    g2 = TextGradient.from_dict(d)
    assert g2.text == 'fix'
    assert g2.attribution == 'rules'
    assert g2.severity == 0.8
    assert len(g2.items) == 1
    assert g2.id == g.id

  def test_from_dict_with_unknown_keys(self) -> None:
    d = {'text': 'x', 'unknown_key': 'ignored'}
    g = TextGradient.from_dict(d)
    assert g.text == 'x'

  def test_from_dict_missing_optional_keys(self) -> None:
    g = TextGradient.from_dict({})
    assert g.text is None
    assert g.attribution is None
    assert g.severity == 0.0
