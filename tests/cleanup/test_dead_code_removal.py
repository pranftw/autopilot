"""Tests for sub-plan 01: dead code removal and dependency cleanup."""

from autopilot.ai.gradient import CollationResult, GradientCollator, TextGradient
from autopilot.ai.loss import JudgeLoss
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import AutoPilotError, ConfigError, ExperimentError
from autopilot.core.experiment import Experiment
from autopilot.core.graph import get_current_graph
from autopilot.core.module.module import Module
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from contextvars import copy_context
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock
import pytest
import subprocess
import zipfile


def test_cli_main_no_dead_code():
  import autopilot.cli.main as m

  assert not hasattr(m, '_dispatch')
  assert not hasattr(m, '_run_project_cli')
  assert callable(getattr(m, 'build_parser', None))
  assert callable(getattr(m, 'main', None))


def test_judge_loss_no_judge_storage():
  judge = MagicMock()
  collator = MagicMock(spec=GradientCollator)
  loss = JudgeLoss(judge, collator)
  assert not hasattr(loss, '_judge')


def test_judgeloss_training_path_unchanged():
  def run():
    g = get_current_graph()
    g.reset()
    g._freed = False

    class _Mod(Module):
      def __init__(self) -> None:
        super().__init__()
        self.w = Parameter(requires_grad=True)

      def forward(self, x: Datum) -> Datum:
        return Datum()

    m = _Mod()
    params = [m.w]
    gradients = {p.id: TextGradient(attribution=f'fix {p.id}') for p in params}
    collator = MagicMock(spec=GradientCollator)
    collator.collate.return_value = CollationResult(context='ctx', gradients=cast(Any, gradients))
    judge = MagicMock()
    loss = JudgeLoss(judge, collator, params)

    output = m(Datum())
    loss.forward(output)
    loss.backward()

    collator.collate.assert_called_once()
    assert m.w.grad is not None
    assert not hasattr(loss, '_judge')

  ctx = copy_context()
  ctx.run(run)


def test_config_error_raised(tmp_path: Path):
  c = AutoPilotConfig(workspace=tmp_path)
  with pytest.raises(ConfigError) as excinfo:
    c.init_project()
  assert 'no project set on config' in str(excinfo.value)
  assert isinstance(excinfo.value, AutoPilotError)


def test_experiment_error_raised():
  e = Experiment(experiment_id='x')
  e.start()
  e.complete()
  with pytest.raises(ExperimentError) as excinfo:
    e.complete()
  assert 'cannot complete' in str(excinfo.value)
  assert isinstance(excinfo.value, AutoPilotError)


@pytest.mark.parametrize(
  ('setup_status', 'method', 'required_substrings'),
  [
    ('running', 'start', ['cannot start', 'expected pending']),
    ('completed', 'start', ['cannot start', 'expected pending']),
    ('failed', 'start', ['cannot start', 'expected pending']),
    ('cancelled', 'start', ['cannot start', 'expected pending']),
    ('completed', 'complete', ['cannot complete', 'expected pending or running']),
    ('completed', 'cancel', ['cannot cancel', 'terminal']),
    ('failed', 'cancel', ['cannot cancel', 'terminal']),
    ('cancelled', 'cancel', ['cannot cancel', 'terminal']),
    ('pending', 'advance_epoch', ['cannot advance epoch', 'expected running']),
  ],
)
def test_experiment_lifecycle_transitions(
  setup_status: str, method: str, required_substrings: list[str]
):
  exp = Experiment(experiment_id='t')
  if setup_status == 'running':
    exp.start()
  elif setup_status == 'completed':
    exp.start()
    exp.complete()
  elif setup_status == 'failed':
    exp.start()
    exp.fail()
  elif setup_status == 'cancelled':
    exp.cancel()

  with pytest.raises(ExperimentError) as excinfo:
    getattr(exp, method)()

  msg = str(excinfo.value)
  for substring in required_substrings:
    assert substring in msg
  assert isinstance(excinfo.value, AutoPilotError)


def test_py_typed_exists():
  import importlib.util

  spec = importlib.util.find_spec('autopilot')
  assert spec is not None
  locations = spec.submodule_search_locations
  assert locations
  root = Path(locations[0])
  marker = root / 'py.typed'
  assert marker.is_file()
  assert marker.read_bytes() == b''


def test_uv_build_succeeds(tmp_path: Path):
  repo_root = Path(__file__).resolve().parents[2]
  dist_dir = repo_root / 'dist'
  if dist_dir.exists():
    import shutil

    shutil.rmtree(dist_dir)
  completed = subprocess.run(
    ['uv', 'build'],
    cwd=str(repo_root),
    capture_output=True,
    text=True,
    check=False,
  )
  assert completed.returncode == 0, f'uv build failed: {completed.stderr}'
  whls = list(dist_dir.glob('autopilot-*.whl'))
  assert len(whls) >= 1
  assert all(p.suffix == '.whl' for p in whls)


def test_uv_build_wheel_excludes_cursor():
  repo_root = Path(__file__).resolve().parents[2]
  dist_dir = repo_root / 'dist'
  whls = sorted(dist_dir.glob('autopilot-*.whl'))
  assert whls, 'no wheel found -- run test_uv_build_succeeds first'
  with zipfile.ZipFile(whls[-1]) as zf:
    names = zf.namelist()
  assert all(not n.startswith('.cursor/') and '/.cursor/' not in n for n in names)
  assert any(n.endswith('autopilot/py.typed') for n in names)
  assert sum(1 for n in names if n.endswith('py.typed')) == 1
