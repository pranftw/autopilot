"""Tests for AgentOptimizer anti-gaming path constraints (Plan 15, FR-009a)."""

from autopilot.ai.agents.agent import AgentResult
from autopilot.ai.gradient import TextGradient
from autopilot.ai.optimizer import (
  AgentOptimizer,
  _check_path_violations,
  _is_allowed,
  _is_forbidden,
  _list_files_under_parameters,
  _normalize_paths,
)
from autopilot.ai.parameter import PathParameter
from autopilot.core.errors import ConfigError
from autopilot.core.parameter import Parameter
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock
import pytest


def _mock_agent(output: str = 'done') -> MagicMock:
  agent = MagicMock()
  agent.run.return_value = AgentResult(output=output)
  agent.limiter = None
  return agent


# -- 2.1: Constructor and context wiring --


class TestPathConstraintsInContext:
  """Verify allowed_paths and forbidden_paths flow into build_context()."""

  def test_agent_optimizer_includes_path_constraints_in_context(self, tmp_path):
    """build_context() contains normalized allowed/forbidden path lists."""
    agent = _mock_agent()
    param = Parameter(requires_grad=True)
    allowed = [tmp_path / 'a', tmp_path / 'b']
    forbidden = [tmp_path / 'c']
    opt = AgentOptimizer(
      agent,
      [param],
      allowed_paths=allowed,
      forbidden_paths=forbidden,
    )
    ctx = opt.build_context()
    expected_allowed = sorted(PurePosixPath(p.resolve()).as_posix() for p in allowed)
    expected_forbidden = sorted(PurePosixPath(p.resolve()).as_posix() for p in forbidden)
    assert ctx['allowed_paths'] == expected_allowed
    assert ctx['forbidden_paths'] == expected_forbidden

  def test_agent_optimizer_empty_allowed_paths_means_unrestricted(self):
    """Empty allowed_paths means no allow restriction; validator never false-positives."""
    agent = _mock_agent()
    param = Parameter(requires_grad=True)
    opt = AgentOptimizer(agent, [param], allowed_paths=[], forbidden_paths=[])
    ctx = opt.build_context()
    assert ctx['allowed_paths'] == []
    assert ctx['forbidden_paths'] == []
    before: dict[str, float] = {}
    after = {'/any/path/at/all': 1.0}
    assert _check_path_violations(before, after, [], []) is None

  def test_none_paths_default_to_empty_lists(self):
    """None for both path lists results in empty lists in context."""
    agent = _mock_agent()
    param = Parameter(requires_grad=True)
    opt = AgentOptimizer(agent, [param])
    ctx = opt.build_context()
    assert ctx['allowed_paths'] == []
    assert ctx['forbidden_paths'] == []

  def test_context_keys_are_copies(self):
    """Mutation of returned context does not affect internal state."""
    agent = _mock_agent()
    param = Parameter(requires_grad=True)
    opt = AgentOptimizer(agent, [param], allowed_paths=['/a'], forbidden_paths=['/b'])
    ctx = opt.build_context()
    ctx['allowed_paths'].append('/hacked')
    ctx['forbidden_paths'].append('/hacked2')
    fresh = opt.build_context()
    assert '/hacked' not in fresh['allowed_paths']
    assert '/hacked2' not in fresh['forbidden_paths']


# -- 2.2: Post-step validation --


class TestPostStepValidation:
  """Verify post-step filesystem scanning and path violation detection."""

  def test_post_step_violation_raises(self, tmp_path):
    """Agent writing outside allowed prefix raises ConfigError.

    The parameter covers the whole workspace so the scanner detects
    the new file, which falls outside allowed_paths.
    """
    workspace = tmp_path / 'workspace'
    allowed_dir = workspace / 'a'
    allowed_dir.mkdir(parents=True)
    disallowed_dir = workspace / 'b'
    disallowed_dir.mkdir(parents=True)
    (allowed_dir / 'seed.txt').write_text('ok')

    agent = _mock_agent()

    def side_effect(*args, **kwargs):
      (disallowed_dir / 'x.txt').write_text('sneaky')
      return AgentResult(output='done')

    agent.run.side_effect = side_effect
    param = PathParameter(source=str(workspace), pattern='**/*', requires_grad=True)
    param.grad = TextGradient(attribution='fix')

    opt = AgentOptimizer(
      agent,
      [param],
      allowed_paths=[str(allowed_dir)],
      validate_paths_after_step=True,
      agentic=False,
      feedback_dir=str(tmp_path / 'feedback'),
    )

    with pytest.raises(ConfigError, match='post-step path violation') as exc_info:
      opt.step()
    assert 'b' in str(exc_info.value) or 'forbidden' in str(exc_info.value)

  def test_forbidden_over_allowed_overlap(self, tmp_path):
    """Forbidden prefix under an allowed parent still blocks the edit."""
    src = tmp_path / 'src'
    secrets = src / 'secrets'
    secrets.mkdir(parents=True)
    (src / 'main.py').write_text('ok')
    (secrets / 'existing.py').write_text('old')

    agent = _mock_agent()

    def side_effect(*args, **kwargs):
      (secrets / 'foo.py').write_text('leaked')
      return AgentResult(output='done')

    agent.run.side_effect = side_effect
    param = PathParameter(source=str(src), pattern='**/*', requires_grad=True)
    param.grad = TextGradient(attribution='fix')

    opt = AgentOptimizer(
      agent,
      [param],
      allowed_paths=[str(src)],
      forbidden_paths=[str(secrets)],
      validate_paths_after_step=True,
      agentic=False,
      feedback_dir=str(tmp_path / 'feedback'),
    )

    with pytest.raises(ConfigError, match='post-step path violation'):
      opt.step()

  def test_path_normalization_dot_segments(self, tmp_path):
    """./src/foo and src/foo normalize to the same logical key."""
    anchor = tmp_path / 'project'
    anchor.mkdir()
    result_dot = _normalize_paths(['./src/foo'], anchor)
    result_plain = _normalize_paths(['src/foo'], anchor)
    assert result_dot == result_plain

  def test_validation_disabled_by_default(self, tmp_path):
    """When validate_paths_after_step is False, no scanning occurs."""
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'f.txt').write_text('ok')

    agent = _mock_agent()

    def side_effect(*args, **kwargs):
      (tmp_path / 'rogue.txt').write_text('rogue')
      return AgentResult(output='done')

    agent.run.side_effect = side_effect
    param = PathParameter(source=str(src), pattern='**/*', requires_grad=True)
    param.grad = TextGradient(attribution='fix')

    opt = AgentOptimizer(
      agent,
      [param],
      allowed_paths=[str(src)],
      validate_paths_after_step=False,
      agentic=False,
    )
    opt.step()
    assert param.grad is None

  def test_valid_edit_passes_validation(self, tmp_path):
    """Editing within allowed paths does not raise."""
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'f.txt').write_text('original')

    agent = _mock_agent()

    def side_effect(*args, **kwargs):
      (src / 'f.txt').write_text('updated')
      return AgentResult(output='done')

    agent.run.side_effect = side_effect
    param = PathParameter(source=str(src), pattern='**/*', requires_grad=True)
    param.grad = TextGradient(attribution='fix')

    opt = AgentOptimizer(
      agent,
      [param],
      allowed_paths=[str(src)],
      validate_paths_after_step=True,
      agentic=False,
      feedback_dir=str(tmp_path / 'feedback'),
    )
    opt.step()
    assert param.grad is None

  def test_agentic_mode_post_step_validation(self, tmp_path):
    """Post-step validation works in agentic mode too."""
    workspace = tmp_path / 'workspace'
    src = workspace / 'src'
    bad = workspace / 'bad'
    src.mkdir(parents=True)
    bad.mkdir(parents=True)
    (src / 'f.txt').write_text('ok')

    agent = _mock_agent()

    def side_effect(*args, **kwargs):
      (bad / 'evil.txt').write_text('nope')
      return AgentResult(output='done')

    agent.run.side_effect = side_effect
    param = PathParameter(source=str(workspace), pattern='**/*', requires_grad=True)
    param.grad = TextGradient(attribution='fix')

    opt = AgentOptimizer(
      agent,
      [param],
      allowed_paths=[str(src)],
      validate_paths_after_step=True,
      agentic=True,
      feedback_dir=str(tmp_path / 'feedback'),
    )

    with pytest.raises(ConfigError, match='post-step path violation'):
      opt.step()


# -- Helper function unit tests --


class TestNormalizePaths:
  """Unit tests for _normalize_paths."""

  def test_none_returns_empty(self):
    assert _normalize_paths(None, Path('/anchor')) == []

  def test_empty_returns_empty(self):
    assert _normalize_paths([], Path('/anchor')) == []

  def test_absolute_paths_preserved(self, tmp_path):
    result = _normalize_paths([str(tmp_path / 'a')], tmp_path)
    expected = PurePosixPath((tmp_path / 'a').resolve()).as_posix()
    assert result == [expected]

  def test_relative_paths_resolved_against_anchor(self, tmp_path):
    result = _normalize_paths(['sub/dir'], tmp_path)
    expected = PurePosixPath((tmp_path / 'sub' / 'dir').resolve()).as_posix()
    assert result == [expected]

  def test_dot_segments_resolved(self, tmp_path):
    result1 = _normalize_paths(['./foo/bar'], tmp_path)
    result2 = _normalize_paths(['foo/bar'], tmp_path)
    assert result1 == result2

  def test_results_sorted(self):
    anchor = Path('/anchor')
    result = _normalize_paths(['/z', '/a', '/m'], anchor)
    assert result == sorted(result)

  def test_path_objects_accepted(self, tmp_path):
    result = _normalize_paths([Path('sub')], tmp_path)
    assert len(result) == 1


class TestIsForbiddenAndIsAllowed:
  """Unit tests for _is_forbidden and _is_allowed."""

  def test_exact_match_forbidden(self):
    assert _is_forbidden('/a/b', ['/a/b']) is True

  def test_prefix_match_forbidden(self):
    assert _is_forbidden('/a/b/c', ['/a/b']) is True

  def test_no_match_forbidden(self):
    assert _is_forbidden('/a/b', ['/c/d']) is False

  def test_partial_name_not_forbidden(self):
    assert _is_forbidden('/a/bcd', ['/a/b']) is False

  def test_exact_match_allowed(self):
    assert _is_allowed('/x/y', ['/x/y']) is True

  def test_prefix_match_allowed(self):
    assert _is_allowed('/x/y/z', ['/x/y']) is True

  def test_no_match_allowed(self):
    assert _is_allowed('/x/y', ['/other']) is False

  def test_partial_name_not_allowed(self):
    assert _is_allowed('/x/yz', ['/x/y']) is False


class TestCheckPathViolations:
  """Unit tests for _check_path_violations."""

  def test_no_changes_no_violation(self):
    snap = {'/a/b': 1.0}
    assert _check_path_violations(snap, snap, ['/a'], []) is None

  def test_new_file_outside_allowed(self):
    before: dict[str, float] = {}
    after = {'/other/x': 1.0}
    assert _check_path_violations(before, after, ['/a'], []) == '/other/x'

  def test_modified_file_in_forbidden(self):
    before = {'/a/secrets/key': 1.0}
    after = {'/a/secrets/key': 2.0}
    result = _check_path_violations(before, after, ['/a'], ['/a/secrets'])
    assert result == '/a/secrets/key'

  def test_forbidden_over_allowed(self):
    before: dict[str, float] = {}
    after = {'/src/secrets/f': 1.0}
    result = _check_path_violations(before, after, ['/src'], ['/src/secrets'])
    assert result == '/src/secrets/f'

  def test_empty_allowed_means_unrestricted(self):
    before: dict[str, float] = {}
    after = {'/anywhere/f': 1.0}
    assert _check_path_violations(before, after, [], []) is None

  def test_unchanged_file_not_checked(self):
    snap = {'/forbidden/f': 1.0}
    assert _check_path_violations(snap, snap, [], ['/forbidden']) is None


class TestListFilesUnderParameters:
  """Unit tests for _list_files_under_parameters."""

  def test_path_parameter_files_listed(self, tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.txt').write_text('hello')
    param = PathParameter(source=str(src), pattern='**/*')
    result = _list_files_under_parameters([param])
    assert len(result) == 1
    key = PurePosixPath((src / 'a.txt').resolve()).as_posix()
    assert key in result

  def test_plain_parameter_skipped(self):
    param = Parameter(requires_grad=True)
    result = _list_files_under_parameters([param])
    assert result == {}

  def test_nonexistent_source_returns_empty(self, tmp_path):
    param = PathParameter(source=str(tmp_path / 'missing'), pattern='**/*')
    result = _list_files_under_parameters([param])
    assert result == {}


class TestResolveAnchor:
  """Test _resolve_anchor uses config root or cwd."""

  def test_uses_cwd_when_no_config(self):
    agent = _mock_agent()
    param = Parameter(requires_grad=True)
    opt = AgentOptimizer(agent, [param])
    anchor = opt._resolve_anchor()
    assert anchor == Path.cwd().resolve()

  def test_uses_config_root_when_available(self, tmp_path):
    agent = _mock_agent()
    param = Parameter(requires_grad=True)
    config = MagicMock()
    config.root = str(tmp_path)
    opt = AgentOptimizer(agent, [param], context={'config': config}, allowed_paths=['sub'])
    ctx = opt.build_context()
    expected = PurePosixPath((tmp_path / 'sub').resolve()).as_posix()
    assert expected in ctx['allowed_paths']
