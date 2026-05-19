"""Naming consistency tests for Dogfood V6 sub-plan 02.

Validates TextGradient rename (direction -> text), ContextLog.append overload,
and intentional API divergence documentation in CLAUDE.md.

Tests:
  2.1 TextGradient rename (1-4, 9, 12, 14)
  2.2 ContextLog.append overload (3-4, 10, 13)
  2.3 Intentional divergence documentation (5-8)
"""

from autopilot.ai.gradient import TextGradient
from autopilot.core.context import ContextEntry, ContextLog
from pathlib import Path
import pytest
import re

CLAUDE_MD = Path(__file__).resolve().parents[2] / 'CLAUDE.md'


# ---------------------------------------------------------------------------
# 2.1 TextGradient rename
# ---------------------------------------------------------------------------


def test_text_gradient_text_kwarg() -> None:
  """TextGradient(text='...') constructs and exposes .text."""
  tg = TextGradient(text='improve accuracy')
  assert tg.text == 'improve accuracy'
  assert tg.attribution is None
  assert tg.severity == 0.0


def test_text_gradient_direction_removed() -> None:
  """TextGradient(direction='...') raises TypeError."""
  with pytest.raises(TypeError):
    TextGradient(direction='old kwarg')


def test_text_gradient_direction_migration_message() -> None:
  """TypeError message contains migration guidance with text= and direction=."""
  with pytest.raises(TypeError, match='renamed') as exc_info:
    TextGradient(direction='old kwarg')
  msg = str(exc_info.value)
  assert "text='" in msg
  assert "direction='" in msg


def test_examples_compile_after_rename() -> None:
  """No stale TextGradient(direction= remains in examples/ source files."""
  examples_dir = Path(__file__).resolve().parents[2] / 'examples'
  if not examples_dir.exists():
    pytest.skip('examples directory not found')
  pattern = re.compile(r'TextGradient\([^)]*direction\s*=')
  stale: list[str] = []
  for py_file in examples_dir.rglob('*.py'):
    if '.venv' in py_file.parts or '__pycache__' in py_file.parts:
      continue
    content = py_file.read_text(encoding='utf-8')
    if pattern.search(content):
      stale.append(str(py_file.relative_to(examples_dir)))
  assert not stale, f'Stale TextGradient(direction=...) in examples: {stale}'


def test_text_gradient_roundtrip_after_rename() -> None:
  """TextGradient(text='feedback').to_dict() -> from_dict() round-trips text."""
  original = TextGradient(text='feedback', attribution='fix rules', severity=0.6)
  data = original.to_dict()
  assert 'text' in data
  restored = TextGradient.from_dict(data)
  assert restored.text == 'feedback'
  assert restored.attribution == 'fix rules'
  assert restored.severity == 0.6
  assert restored.id == original.id


def test_text_gradient_text_none() -> None:
  """TextGradient(text=None) is valid (matches str | None default)."""
  tg = TextGradient(text=None)
  assert tg.text is None


# ---------------------------------------------------------------------------
# 2.2 ContextLog.append overload
# ---------------------------------------------------------------------------


def test_context_log_append_entry_object() -> None:
  """log.append(entry) accepts ContextEntry and persists via record() path."""
  log = ContextLog()
  entry = ContextEntry.create('test reason', source='unit')
  result = log.append(entry)
  assert len(log) == 1
  assert log.entries[0] is entry
  assert result is entry


def test_context_log_append_kwargs_still_works() -> None:
  """log.append('reason', source='user', ...) still creates and stores an entry."""
  log = ContextLog()
  result = log.append('my reason', source='user', epoch=0)
  assert result is not None
  assert result.reason == 'my reason'
  assert result.source == 'user'
  assert result.epoch == 0
  assert len(log) == 1


def test_context_log_record_vs_append_both_work() -> None:
  """record(entry) and append(entry) both succeed for the same-style entry."""
  log = ContextLog()
  entry1 = ContextEntry.create('via record', source='test')
  log.record(entry1)

  entry2 = ContextEntry.create('via append', source='test')
  result = log.append(entry2)

  assert len(log) == 2
  assert log.entries[0] is entry1
  assert log.entries[1] is entry2
  assert result is entry2


def test_context_log_append_entry_kwargs_ignored() -> None:
  """log.append(entry, source='extra') silently ignores kwargs when entry is ContextEntry."""
  log = ContextLog()
  entry = ContextEntry.create('original', source='original_source')
  result = log.append(entry, source='extra', epoch=99)
  assert result is entry
  assert result.source == 'original_source'
  assert result.epoch is None
  assert len(log) == 1


# ---------------------------------------------------------------------------
# 2.3 Intentional divergence documentation
# ---------------------------------------------------------------------------


def test_conflict_entry_ancestor_documented() -> None:
  """CLAUDE.md states ancestor (not base) as the field name."""
  content = CLAUDE_MD.read_text(encoding='utf-8')
  assert 'ConflictEntry.ancestor' in content or 'ancestor' in content
  assert 'ConflictEntry.base' not in content


def test_delta_metric_documented() -> None:
  """CLAUDE.md states Delta.metric is the field (not metric_name)."""
  content = CLAUDE_MD.read_text(encoding='utf-8')
  assert 'Delta.metric' in content


def test_broadcast_returns_datum_documented() -> None:
  """CLAUDE.md states broadcast returns a Datum."""
  content = CLAUDE_MD.read_text(encoding='utf-8')
  assert 'broadcast' in content
  assert 'Datum' in content


def test_intentional_divergences_section_exists() -> None:
  """CLAUDE.md has a dedicated intentional API divergences section heading."""
  content = CLAUDE_MD.read_text(encoding='utf-8')
  assert '### Intentional API divergences' in content
