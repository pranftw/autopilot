"""End-to-end integration tests for the textmatch example.

Copies the textmatch example to tmp_path, runs the trainer, and verifies:
1. Exit code 0 as subprocess
2. 5 snapshots at epochs 0-4
3. Experiment status is completed
4. Second run creates new experiment_id via next_slug()
"""

from autopilot.tracking.io import read_json
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
TEXTMATCH_SRC = REPO_ROOT / 'examples' / 'textmatch'


def _copy_textmatch(tmp_path: Path) -> Path:
  """Copy textmatch example to tmp_path, rewriting pyproject.toml for the repo root."""
  dest = tmp_path / 'textmatch'
  shutil.copytree(
    TEXTMATCH_SRC,
    dest,
    ignore=shutil.ignore_patterns('.venv', '.store', '__pycache__', '*.pyc', '.ruff_cache'),
  )
  pyproject = dest / 'pyproject.toml'
  text = pyproject.read_text(encoding='utf-8')
  text = text.replace('path = "../.."', f'path = "{REPO_ROOT}"')
  pyproject.write_text(text, encoding='utf-8')
  return dest


def _cleanup_textmatch(original_path: list[str]) -> None:
  """Restore sys.path and remove textmatch modules from sys.modules."""
  sys.path[:] = original_path
  for name in list(sys.modules):
    if name.startswith('textmatch'):
      del sys.modules[name]


def _run_trainer_inprocess(dest: Path, max_epochs: int = 5):
  """Run the textmatch trainer in-process and return (trainer, store, result)."""
  original_path = sys.path[:]
  sys.path.insert(0, str(dest))

  import textmatch.data as data_mod
  import textmatch.module as module_mod
  import textmatch.trainer as trainer_mod

  try:
    store_path = dest / '.store'
    module = module_mod.TextMatchModule(str(dest / 'rules'))
    dm = data_mod.TextMatchDataModule(str(dest / 'datasets'))
    trainer, store = trainer_mod.build_trainer(module, store_path)
    result = trainer.fit(module, datamodule=dm, max_epochs=max_epochs)
    return trainer, store, result
  finally:
    _cleanup_textmatch(original_path)


def test_trainer_subprocess_exit_code_0(tmp_path: Path) -> None:
  """Running run_trainer.py as a subprocess completes with exit code 0."""
  dest = _copy_textmatch(tmp_path)
  result = subprocess.run(
    [sys.executable, 'run_trainer.py'],
    cwd=str(dest),
    capture_output=True,
    text=True,
    timeout=120,
    check=False,
  )
  assert result.returncode == 0, (
    f'run_trainer.py failed (exit {result.returncode})\n'
    f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
  )
  assert 'Done.' in result.stdout


def test_snapshots_created_5_epochs(tmp_path: Path) -> None:
  """Trainer.fit(max_epochs=5) creates 5 snapshots at epochs 0, 1, 2, 3, 4."""
  dest = _copy_textmatch(tmp_path)
  trainer, store, result = _run_trainer_inprocess(dest)

  experiment_id = trainer.experiment.id
  log_entries = store.log(experiment_id)
  assert len(log_entries) == 5
  assert [e.epoch for e in log_entries] == [0, 1, 2, 3, 4]
  for entry in log_entries:
    assert entry.file_count > 0

  for ep in result['epochs']:
    metrics = ep.get('metrics', {})
    accuracy = (
      metrics.get('val_accuracy') or metrics.get('train_accuracy') or metrics.get('accuracy', 0.0)
    )
    assert accuracy > 0.0, f'epoch {ep["epoch"]} had zero accuracy'


def test_experiment_status_completed(tmp_path: Path) -> None:
  """After successful training, experiment status is completed."""
  dest = _copy_textmatch(tmp_path)
  trainer, _store, result = _run_trainer_inprocess(dest)

  experiment = trainer.experiment
  assert experiment.status.value == 'completed'
  assert experiment.completed_at is not None
  assert result['total_epochs'] == 5


def test_second_run_creates_new_experiment_id(tmp_path: Path) -> None:
  """Running trainer twice creates two distinct experiments via next_slug()."""
  dest = _copy_textmatch(tmp_path)

  trainer1, _store1, _result1 = _run_trainer_inprocess(dest)
  exp_id_1 = trainer1.experiment.id

  trainer2, store2, _result2 = _run_trainer_inprocess(dest)
  exp_id_2 = trainer2.experiment.id

  assert exp_id_1 != exp_id_2
  assert exp_id_1 == 'run-1'
  assert exp_id_2 == 'run-2'

  refs = read_json(dest / '.store' / 'refs.json')
  assert isinstance(refs, dict)
  branches = refs.get('branches', {})
  assert exp_id_1 in branches
  assert exp_id_2 in branches

  log1 = store2.log(exp_id_1)
  log2 = store2.log(exp_id_2)
  assert len(log1) == 5
  assert len(log2) == 5
