"""Trainer builder for the harness example.

Provides ``build_trainer()`` which returns a fully-wired
``(Trainer, HarnessModule, HarnessDataModule)`` tuple for
``Trainer.fit(module, datamodule=dm, max_epochs=N)``.

Workspace layout:
  - ``root``: example project root (directory containing ``pyproject.toml``)
  - ``root / 'harness'``: Python package with prompts/, tools/, db/
  - ``root / '.autopilot' / 'store'``: FileStore content-addressed storage

The trainer uses ``EpochOrchestrator`` with plateau detection on
``task_success_rate`` (window 3, threshold 0.01, auto-rollback enabled).
Each experiment is stamped with a ``DatasetFingerprint`` from existing
train/val scenario files via ``experiment.dataset_meta``.

``max_epochs`` is intentionally NOT consumed here; callers pass it
directly to ``trainer.fit()``.

``use_judge`` controls whether the module uses ``JudgeLoss`` (``True``,
the default) or falls back to heuristic ``HarnessLoss``.  When ``None``,
defers to ``resolved_env.use_judge`` (environment preset).

``env`` optionally injects a specific ``EnvironmentConfig``; when
``None``, resolves from ``HARNESS_ENV`` (default: 'dev').

**Note:** ``DeployCallback`` is installed only by ``HarnessCLI``, not
by ``build_trainer()``.  Essential callbacks (``MetricsWriterCallback``,
``OptimizerContextCallback``) are always appended here for parity with
CLI runs.
"""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.ai.fingerprint import compute_fingerprint
from autopilot.ai.store.file_store import FileStore
from autopilot.core.callbacks.callback import Callback
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.config import AutoPilotConfig
from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.trainer.trainer import Trainer
from autopilot.policy.gates import Gate, MinGate
from autopilot.policy.quality_first import QualityFirstPolicy
from autopilot.tracking.io import read_json_dict
from harness import DEFAULT_MODEL
from harness.callbacks import (
  HarnessCostTrackerCallback,
  MetricsWriterCallback,
  OptimizerContextCallback,
)
from harness.data import HarnessDataModule
from harness.environments import EnvironmentConfig, get_environment_config
from harness.module import HarnessModule
from pathlib import Path
import os

TASK_SUCCESS_THRESHOLD = 0.3
TOOL_RECALL_THRESHOLD = 0.4
PLATEAU_WINDOW = 3
PLATEAU_THRESHOLD = 0.01


def _make_harness_orchestrator() -> EpochOrchestrator:
  """Create the harness EpochOrchestrator with plateau detection.

  Monitors ``task_success_rate`` with a plateau window of 3 epochs,
  0.01 relative improvement threshold, and auto-rollback on regression.

  Returns:
    Configured EpochOrchestrator for harness training.
  """
  return EpochOrchestrator(
    config=OrchestratorConfig(
      monitor='task_success_rate',
      auto_rollback=True,
      plateau_window=PLATEAU_WINDOW,
      plateau_threshold=PLATEAU_THRESHOLD,
    ),
  )


def next_slug(store_path: Path) -> str:
  """Allocate the next experiment slug from store refs.

  Parses numeric suffixes from ``harness-N`` branch names and returns
  ``harness-{max + 1}``.  Non-numeric ``harness-*`` branches are ignored.

  Args:
    store_path: Path to the FileStore root.

  Returns:
    A slug like ``'harness-1'``, ``'harness-2'``, etc.
  """
  refs_file = store_path / 'refs.json'
  if not refs_file.exists():
    return 'harness-1'
  refs = read_json_dict(refs_file, 'refs')
  branches = refs.get('branches', {})
  existing_nums = []
  for key in branches:
    if key.startswith('harness-'):
      suffix = key[len('harness-') :]
      if suffix.isdigit():
        existing_nums.append(int(suffix))
  if not existing_nums:
    return 'harness-1'
  return f'harness-{max(existing_nums) + 1}'


def build_trainer(
  root: Path,
  model: str = DEFAULT_MODEL,
  gates: list[Gate] | None = None,
  callbacks: list[Callback] | None = None,
  experiment_slug: str | None = None,
  use_judge: bool | None = None,
  env: EnvironmentConfig | None = None,
) -> tuple[Trainer, HarnessModule, HarnessDataModule]:
  """Build a fully-wired Trainer for the harness.

  Uses ``EpochOrchestrator`` with plateau detection on ``task_success_rate``
  and stamps each experiment with a ``DatasetFingerprint`` from existing
  scenario files (train.jsonl / val.jsonl).

  Always appends ``MetricsWriterCallback``, ``OptimizerContextCallback``,
  ``StoreCheckpointCallback``, and ``HarnessCostTrackerCallback`` after
  user-supplied callbacks.  ``DeployCallback`` is installed only by
  ``HarnessCLI``, not here.

  On ``optimize loop`` paths the framework injects its own base
  ``CostTrackerCallback``, so this standalone path is the only place
  ``HarnessCostTrackerCallback`` is registered.

  Args:
    root: Example project root (directory containing pyproject.toml).
    model: Model string forwarded to HarnessModule.
    gates: Optional QualityFirstPolicy gate list; defaults use
      task_success_rate and tool_recall.
    callbacks: Extra Trainer callbacks; essential callbacks are always
      appended after user-supplied ones.
    experiment_slug: Optional stable experiment id; otherwise allocated
      from store refs.
    use_judge: Whether the module should use ``JudgeLoss`` (``True``) or
      heuristic ``HarnessLoss`` (``False``).  ``None`` defers to
      ``resolved_env.use_judge``.
    env: Optional ``EnvironmentConfig`` override; when ``None``, resolved
      from ``HARNESS_ENV`` environment variable (default: 'dev').

  Returns:
    Tuple of (trainer, module, datamodule) for Trainer.fit(module, datamodule=...).
  """
  resolved_env = env or get_environment_config(os.environ.get('HARNESS_ENV', 'dev'))
  effective_use_judge = use_judge if use_judge is not None else resolved_env.use_judge

  harness_pkg = root / 'harness'
  harness_root = str(harness_pkg)

  module = HarnessModule(
    harness_root,
    model=model,
    use_judge=effective_use_judge,
    max_turns=resolved_env.max_turns,
  )
  datamodule = HarnessDataModule(str(harness_pkg / 'scenarios'))

  config = AutoPilotConfig(workspace=root)
  store_path = root / '.autopilot' / 'store'
  config.store_path = store_path
  store = FileStore(config)
  store.register_parameters(dict(module.named_parameters()))

  if gates is None:
    gates = [
      MinGate('task_success_rate', threshold=TASK_SUCCESS_THRESHOLD, required=True),
      MinGate('tool_recall', threshold=TOOL_RECALL_THRESHOLD, required=True),
    ]
  policy = QualityFirstPolicy(gates=gates)

  slug = experiment_slug or next_slug(store_path)
  experiment = AutoPilotExperiment(experiment_id=slug)
  experiment.store = store

  scenario_paths = [
    harness_pkg / 'scenarios' / 'train.jsonl',
    harness_pkg / 'scenarios' / 'val.jsonl',
  ]
  existing = [p for p in scenario_paths if p.exists()]
  fingerprint = compute_fingerprint(existing)
  experiment.dataset_meta = fingerprint.to_dict()

  cb_list = list(callbacks or [])
  cb_list.extend(
    [
      MetricsWriterCallback(),
      OptimizerContextCallback(),
    ]
  )
  cb_list.append(StoreCheckpointCallback())
  cb_list.append(HarnessCostTrackerCallback())

  orchestrator = _make_harness_orchestrator()
  trainer = Trainer(
    callbacks=cb_list,
    policy=policy,
    experiment=experiment,
    config=config,
    store=store,
    loop=orchestrator,
  )
  return trainer, module, datamodule
