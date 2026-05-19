"""Tests for tree switch --bind flag (Plan 26 S2.1, updated for Plan 21).

Verifies that ``--bind`` with ``--no-checkout`` fails fast, and that
``--bind`` (with default checkout) invokes ``bind_path_parameters``.
Also tests the extracted ``bind_path_parameters`` helper independently.

Note: CLI integration tests for --bind mock ``_bind_parameters`` at the
instance level (after parser build) to avoid MagicMock attribute leaks
into argparse subcommand discovery.
"""

from autopilot.ai.environment import bind_path_parameters
from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import build_context
from autopilot.cli.main import AutoPilotCLI, build_parser
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import ConfigError
from autopilot.core.experiment import Experiment
from autopilot.core.module.module import Module
from autopilot.core.node import Node
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import contextlib
import io
import json
import pytest


def _run_full_cli(argv: list[str]) -> tuple[int, str, str]:
  """Run the full AutoPilotCLI with captured stdout/stderr."""
  cli = AutoPilotCLI()
  out = StringIO()
  err = StringIO()
  exit_code = 0
  with patch('sys.stdout', out), patch('sys.stderr', err):
    try:
      cli(argv=argv)
    except SystemExit as e:
      exit_code = int(e.code) if e.code is not None else 0
  return exit_code, out.getvalue(), err.getvalue()


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  """Workspace root directory."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def ws_with_snapshot(ws: Path) -> Path:
  """Workspace with two trees; 'alpha' HEAD has a store snapshot at epoch 0."""
  src_dir = ws / 'src'
  src_dir.mkdir()
  (src_dir / 'main.py').write_text('print("alpha")')

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  param = PathParameter(source=str(src_dir), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('exp-alpha', 0)

  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  exp_a = Experiment(experiment_id='exp-alpha', hypothesis='alpha test')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9})
  tree_a.add(Node(experiment=exp_a))
  tree_a.head = 'exp-alpha'

  tree_b = forest.create_tree('beta')
  exp_b = Experiment(experiment_id='exp-beta', hypothesis='beta test')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.8})
  tree_b.add(Node(experiment=exp_b))
  tree_b.head = 'exp-beta'

  forest.switch('beta')
  forest.save()
  return ws


class TestTreeSwitchBindRequiresCheckout:
  def test_bind_with_no_checkout_fails(self, ws: Path) -> None:
    """--bind with --no-checkout exits non-zero with guidance."""
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('a')
    forest.create_tree('b')
    forest.switch('b')
    forest.save()

    exit_code, stdout, stderr = _run_full_cli(
      [
        'tree',
        'switch',
        'a',
        '--no-checkout',
        '--bind',
        '--context',
        'test',
        '--workspace',
        str(ws),
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert '--bind requires checkout' in combined


class TestTreeSwitchCheckoutBind:
  def test_checkout_bind_invokes_bind(self, ws_with_snapshot: Path) -> None:
    """After default checkout + bind, _bind_parameters is called."""
    from autopilot.cli.commands.tree import TreeSwitch

    parser = build_parser()
    full_argv = [
      'tree',
      'switch',
      'alpha',
      '--bind',
      '--workspace',
      str(ws_with_snapshot),
      '--json',
      '--context',
      'test',
    ]
    parsed = parser.parse_args(full_argv)
    ctx = build_context(parsed)
    ctx.project = 'fake-project'

    buf = io.StringIO()
    with (
      patch('autopilot.ai.store.file_store.FileStore.checkout'),
      patch.object(TreeSwitch, '_bind_parameters', return_value=2) as mock_bind,
      contextlib.redirect_stdout(buf),
    ):
      parsed.handler(ctx, parsed)

    mock_bind.assert_called_once()
    result = json.loads(buf.getvalue().strip())
    assert result['ok'] is True
    assert result['result']['bind'] is True
    assert result['result']['bound_parameters'] == 2

  def test_checkout_bind_failure_calls_ctx_fail(self, ws_with_snapshot: Path) -> None:
    """If _bind_parameters triggers ctx.fail, the handler exits non-zero."""
    from autopilot.cli.commands.tree import TreeSwitch

    parser = build_parser()
    full_argv = [
      'tree',
      'switch',
      'alpha',
      '--bind',
      '--workspace',
      str(ws_with_snapshot),
      '--json',
      '--context',
      'test',
    ]
    parsed = parser.parse_args(full_argv)
    ctx = build_context(parsed)
    ctx.project = 'fake-project'

    with (
      patch('autopilot.ai.store.file_store.FileStore.checkout'),
      patch.object(
        TreeSwitch,
        '_bind_parameters',
        side_effect=SystemExit(1),
      ),
      pytest.raises(SystemExit),
    ):
      parsed.handler(ctx, parsed)

  def test_checkout_bind_without_project_fails(self, ws_with_snapshot: Path) -> None:
    """--bind without -p <project> exits non-zero with guidance."""
    with patch('autopilot.ai.store.file_store.FileStore.checkout'):
      exit_code, stdout, stderr = _run_full_cli(
        [
          'tree',
          'switch',
          'alpha',
          '--bind',
          '--context',
          'test',
          '--workspace',
          str(ws_with_snapshot),
        ]
      )

    assert exit_code == 1
    combined = stdout + stderr
    assert '--bind requires -p' in combined

  def test_bind_result_includes_bind_keys(self, ws_with_snapshot: Path) -> None:
    """JSON result includes bind=True and bound_parameters count."""
    from autopilot.cli.commands.tree import TreeSwitch

    parser = build_parser()
    full_argv = [
      'tree',
      'switch',
      'alpha',
      '--bind',
      '--workspace',
      str(ws_with_snapshot),
      '--json',
      '--context',
      'test',
    ]
    parsed = parser.parse_args(full_argv)
    ctx = build_context(parsed)
    ctx.project = 'fake-project'

    buf = io.StringIO()
    with (
      patch('autopilot.ai.store.file_store.FileStore.checkout'),
      patch.object(TreeSwitch, '_bind_parameters', return_value=1),
      contextlib.redirect_stdout(buf),
    ):
      parsed.handler(ctx, parsed)

    result = json.loads(buf.getvalue().strip())
    assert result['result']['bind'] is True
    assert result['result']['bound_parameters'] == 1

  def test_checkout_without_bind_has_no_bind_key(self, ws_with_snapshot: Path) -> None:
    """Without --bind, default checkout result has no bind key."""
    parser = build_parser()
    full_argv = [
      'tree',
      'switch',
      'alpha',
      '--workspace',
      str(ws_with_snapshot),
      '--json',
      '--context',
      'test',
    ]
    parsed = parser.parse_args(full_argv)
    ctx = build_context(parsed)

    buf = io.StringIO()
    with (
      patch('autopilot.ai.store.file_store.FileStore.checkout'),
      contextlib.redirect_stdout(buf),
    ):
      parsed.handler(ctx, parsed)

    result = json.loads(buf.getvalue().strip())
    assert 'bind' not in result['result']
    assert 'bound_parameters' not in result['result']


class TestBindPathParametersHelper:
  """Unit tests for the extracted bind_path_parameters helper."""

  def test_cwd_is_noop(self) -> None:
    """When wt_path == cwd, returns empty list without binding."""
    cwd = Path.cwd()
    module = Module()
    result = bind_path_parameters(module, cwd, cwd)
    assert result == []

  def test_binds_path_parameters(self, tmp_path: Path) -> None:
    """Binds PathParameter instances under config_root to worktree paths."""
    src = tmp_path / 'project' / 'prompts'
    src.mkdir(parents=True)
    (src / 'main.txt').write_text('hello')

    wt = tmp_path / 'worktree'
    wt.mkdir()

    class TestModule(Module):
      def __init__(self):
        super().__init__()
        self.prompts = PathParameter(source=str(src), pattern='*')

      def forward(self, data):
        return data

    module = TestModule()
    config_root = tmp_path / 'project'

    rebound = bind_path_parameters(module, config_root, wt)
    assert len(rebound) == 1
    assert rebound[0].working_root == str(wt / 'prompts')

    for p in rebound:
      p.unbind()

  def test_raises_config_error_outside_root(self, tmp_path: Path) -> None:
    """Raises ConfigError when PathParameter source is outside config_root."""

    class TestModule(Module):
      def __init__(self):
        super().__init__()
        self.prompts = PathParameter(source='/tmp/outside', pattern='*')

      def forward(self, data):
        return data

    module = TestModule()
    config_root = tmp_path / 'project'
    config_root.mkdir()
    wt = tmp_path / 'worktree'
    wt.mkdir()

    with pytest.raises(ConfigError, match='not under'):
      bind_path_parameters(module, config_root, wt)

  def test_unbinds_on_failure(self, tmp_path: Path) -> None:
    """When bind fails partway, already-bound PathParameters are unbound."""
    good_src = tmp_path / 'project' / 'good'
    good_src.mkdir(parents=True)
    (good_src / 'a.txt').write_text('hello')

    wt = tmp_path / 'worktree'
    wt.mkdir()

    class PartialModule(Module):
      def __init__(self):
        super().__init__()
        self.good = PathParameter(source=str(good_src), pattern='*')
        self.bad = PathParameter(source='/tmp/outside', pattern='*')

      def forward(self, data):
        return data

    module = PartialModule()
    config_root = tmp_path / 'project'
    worktree_good = str(wt / 'good')

    with pytest.raises(ConfigError, match='not under'):
      bind_path_parameters(module, config_root, wt)

    assert module.good.working_root != worktree_good
    assert module.good.working_root == str(good_src)

  def test_skips_non_path_parameters(self, tmp_path: Path) -> None:
    """Non-PathParameter parameters are not bound."""
    from autopilot.core.parameter import Parameter

    class MixedModule(Module):
      def __init__(self):
        super().__init__()
        self.scalar = Parameter()
        self.paths = PathParameter(
          source=str(tmp_path / 'project' / 'data'),
          pattern='*',
        )

      def forward(self, data):
        return data

    (tmp_path / 'project' / 'data').mkdir(parents=True)
    module = MixedModule()
    wt = tmp_path / 'worktree'
    wt.mkdir()

    rebound = bind_path_parameters(module, tmp_path / 'project', wt)
    assert len(rebound) == 1
    assert isinstance(rebound[0], PathParameter)

    for p in rebound:
      p.unbind()
