"""Unit tests for Profiler and SimpleProfiler."""

from autopilot.core.profiler import Profiler, SimpleProfiler
from pathlib import Path
from typing import Any
import json
import pytest
import time

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TestPhilosophyAmendment:
  """Verify PHILOSOPHY.md contains the wall-clock profiler wording."""

  def test_philosophy_contains_wall_clock_profiler(self) -> None:
    text = (REPO_ROOT / 'PHILOSOPHY.md').read_text()
    assert 'Wall-clock profiler' in text


class TestSimpleProfilerStartStop:
  """Test start/stop records a positive duration_ms in describe()."""

  def test_simple_profiler_start_stop(self) -> None:
    profiler = SimpleProfiler()
    profiler.start('action_a')
    time.sleep(0.001)
    profiler.stop('action_a')
    result = profiler.describe()
    assert 'action_a' in result
    assert result['action_a']['count'] == 1
    assert result['action_a']['total_ms'] > 0
    assert result['action_a']['mean_ms'] > 0


class TestSimpleProfilerContextManager:
  """Test with profiler.profile('x') records one interval."""

  def test_simple_profiler_profile_context_manager(self) -> None:
    profiler = SimpleProfiler()
    with profiler.profile('my_action'):
      time.sleep(0.001)
    result = profiler.describe()
    assert 'my_action' in result
    assert result['my_action']['count'] == 1
    assert result['my_action']['total_ms'] > 0


class TestSimpleProfilerMultipleActions:
  """Test distinct actions accumulate separate keys in describe()."""

  def test_simple_profiler_multiple_actions(self) -> None:
    profiler = SimpleProfiler()
    with profiler.profile('a'):
      pass
    with profiler.profile('b'):
      pass
    with profiler.profile('a'):
      pass
    result = profiler.describe()
    assert 'a' in result
    assert 'b' in result
    assert result['a']['count'] == 2
    assert result['b']['count'] == 1


class TestSimpleProfilerDescribe:
  """Test describe() has count, total_ms, mean_ms per action."""

  def test_simple_profiler_describe(self) -> None:
    profiler = SimpleProfiler()
    with profiler.profile('x'):
      time.sleep(0.001)
    with profiler.profile('x'):
      time.sleep(0.001)
    result = profiler.describe()
    stats = result['x']
    assert 'count' in stats
    assert 'total_ms' in stats
    assert 'mean_ms' in stats
    assert stats['count'] == 2
    assert stats['total_ms'] >= stats['mean_ms']


class TestSimpleProfilerDescribeEmpty:
  """Test SimpleProfiler().describe() returns empty dict when no actions recorded."""

  def test_simple_profiler_describe_empty(self) -> None:
    profiler = SimpleProfiler()
    assert profiler.describe() == {}


class TestProfilerDescribeJson:
  """Test json.dumps(profiler.describe()) succeeds."""

  def test_profiler_describe_json(self) -> None:
    profiler = SimpleProfiler()
    with profiler.profile('json_test'):
      pass
    serialized = json.dumps(profiler.describe())
    parsed = json.loads(serialized)
    assert 'json_test' in parsed


class TestProfilerSubclass:
  """Test minimal Profiler subclass overriding start/stop/describe."""

  def test_profiler_subclass(self) -> None:
    calls: list[str] = []

    class MyProfiler(Profiler):
      def start(self, action: str) -> None:
        calls.append(f'start:{action}')

      def stop(self, action: str) -> None:
        calls.append(f'stop:{action}')

      def describe(self) -> dict[str, Any]:
        return {'calls': len(calls)}

    p = MyProfiler()
    with p.profile('test'):
      pass
    assert calls == ['start:test', 'stop:test']
    assert p.describe() == {'calls': 2}


class TestProfilerNestedActions:
  """Test start('a') twice without stop raises ValueError."""

  def test_profiler_nested_actions(self) -> None:
    profiler = SimpleProfiler()
    profiler.start('a')
    with pytest.raises(ValueError, match='already started'):
      profiler.start('a')


class TestProfilerStopWithoutStart:
  """Test stop('unknown') raises ValueError with guidance."""

  def test_profiler_stop_without_start(self) -> None:
    profiler = SimpleProfiler()
    with pytest.raises(ValueError, match='was never started'):
      profiler.stop('unknown')


class TestProfilerMismatchedAction:
  """Test start('a') then stop('b') raises ValueError."""

  def test_profiler_mismatched_action(self) -> None:
    profiler = SimpleProfiler()
    profiler.start('a')
    with pytest.raises(ValueError, match='was never started'):
      profiler.stop('b')


class TestSimpleProfilerDocstring:
  """Verify SimpleProfiler docstring accurately describes describe() output."""

  def test_simple_profiler_docstring_mentions_count(self) -> None:
    doc = SimpleProfiler.__doc__
    assert doc is not None
    assert 'count' in doc

  def test_simple_profiler_docstring_mentions_mean_ms(self) -> None:
    doc = SimpleProfiler.__doc__
    assert doc is not None
    assert 'mean_ms' in doc
