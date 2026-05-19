"""Tests for DiagnosticEntry dataclass and validation constants."""

from autopilot.core.diagnostic import (
  VALID_DIAGNOSTIC_CODES,
  VALID_REPAIR_ACTIONS,
  VALID_SEVERITY_LEVELS,
  DiagnosticEntry,
)
import pytest


def test_diagnostic_entry_roundtrip():
  """to_dict -> from_dict produces an identical entry."""
  entry = DiagnosticEntry(
    code='orphan_blob',
    severity='warning',
    path='/some/blob',
    message='orphan blob detected',
    repairable=True,
    repair_action='delete',
  )
  d = entry.to_dict()
  restored = DiagnosticEntry.from_dict(d)
  assert restored.code == entry.code
  assert restored.severity == entry.severity
  assert restored.path == entry.path
  assert restored.message == entry.message
  assert restored.repairable == entry.repairable
  assert restored.repair_action == entry.repair_action


def test_diagnostic_entry_constructor_invalid_type():
  """Invalid code, severity, or repair_action raises ValueError."""
  with pytest.raises(ValueError, match='invalid diagnostic code'):
    DiagnosticEntry(
      code='bogus',
      severity='error',
      path=None,
      message='bad code',
      repairable=False,
    )
  with pytest.raises(ValueError, match='invalid severity'):
    DiagnosticEntry(
      code='orphan_blob',
      severity='critical',
      path=None,
      message='bad severity',
      repairable=False,
    )
  with pytest.raises(ValueError, match='invalid repair_action'):
    DiagnosticEntry(
      code='orphan_blob',
      severity='warning',
      path=None,
      message='bad action',
      repairable=False,
      repair_action='nuke',
    )


def test_diagnostic_entry_from_dict_missing_key():
  """Missing required field raises during from_dict."""
  with pytest.raises((KeyError, TypeError)):
    DiagnosticEntry.from_dict({'code': 'orphan_blob'})


def test_diagnostic_entry_empty_message():
  """Empty message raises ValueError."""
  with pytest.raises(ValueError, match='message must not be empty; provide'):
    DiagnosticEntry(
      code='orphan_blob',
      severity='warning',
      path=None,
      message='',
      repairable=False,
    )


def test_diagnostic_repairable_without_action():
  """repairable=True with repair_action=None raises ValueError."""
  with pytest.raises(ValueError, match='repairable entry must specify repair_action; set'):
    DiagnosticEntry(
      code='orphan_blob',
      severity='warning',
      path=None,
      message='orphan blob',
      repairable=True,
      repair_action=None,
    )


def test_valid_constants_coverage():
  """Validation constants contain expected values."""
  assert 'orphan_blob' in VALID_DIAGNOSTIC_CODES
  assert 'manifest_error' in VALID_DIAGNOSTIC_CODES
  assert 'stale_lock' in VALID_DIAGNOSTIC_CODES
  assert 'broken_ref' in VALID_DIAGNOSTIC_CODES
  assert 'missing_blob' in VALID_DIAGNOSTIC_CODES
  assert 'reflog_gap' in VALID_DIAGNOSTIC_CODES
  assert {'error', 'warning', 'info'} == VALID_SEVERITY_LEVELS
  assert {'delete', 'reset', 'backfill'} == VALID_REPAIR_ACTIONS


def test_non_repairable_entry_valid():
  """Non-repairable entry with no repair_action is valid."""
  entry = DiagnosticEntry(
    code='missing_blob',
    severity='error',
    path='objects/ab/cdef',
    message='blob not found in object store',
    repairable=False,
  )
  assert entry.repair_action is None
  assert not entry.repairable
