"""Tests for parse_timestamp in tracking/io.py."""

from autopilot.tracking.io import parse_timestamp
from datetime import UTC, datetime, timedelta
import pytest


def test_parse_timestamp_utc_aware():
  """UTC-aware string parses to UTC-aware datetime."""
  result = parse_timestamp('2024-01-01T00:00:00+00:00')
  assert result.tzinfo is not None
  assert result == datetime(2024, 1, 1, tzinfo=UTC)


def test_parse_timestamp_naive_assumes_utc():
  """Naive string (no tzinfo) gets UTC tzinfo."""
  result = parse_timestamp('2024-06-15T12:30:00')
  assert result.tzinfo is not None
  assert result.tzinfo == UTC
  assert result == datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC)


def test_parse_timestamp_with_offset():
  """+05:30 offset is preserved correctly."""
  result = parse_timestamp('2024-03-10T15:45:00+05:30')
  assert result.tzinfo is not None
  expected_offset = timedelta(hours=5, minutes=30)
  assert result.utcoffset() == expected_offset


def test_parse_timestamp_preserves_microseconds():
  """Fractional seconds are preserved."""
  result = parse_timestamp('2024-01-01T00:00:00.123456+00:00')
  assert result.microsecond == 123456


def test_parse_timestamp_invalid_raises():
  """Invalid string raises ValueError."""
  with pytest.raises(ValueError, match='Invalid isoformat'):
    parse_timestamp('not-a-timestamp')
