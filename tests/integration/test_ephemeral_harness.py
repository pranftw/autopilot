"""Ephemeral workspace integration harness (Plan 11, section 2.6).

Exercises the full cold-start -> train -> snapshot -> resume flow in an
ephemeral directory created via ``tmp_path``.  Validates:

- Named parameter registration and snapshot round-trips
- Sampler-based DataLoader (no shuffle kwarg)
- Stage enum on DataModule
- Store snapshots persist across epochs
- Checkpoint save/resume via Trainer
- Experiment lifecycle (context manager)
"""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.callbacks.checkpoint import CheckpointCallback
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.checkpoint import JSONCheckpointIO
from autopilot.core.config import AutoPilotConfig
from autopilot.core.gradient import Gradient
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.optimizer import Optimizer
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule, Stage
from autopilot.data.dataset import Dataset
from autopilot.data.sampler import RandomSampler, SequentialSampler
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import random


class _InMemoryDataset(Dataset[Datum]):
  """Simple in-memory dataset of Datum instances."""

  def __init__(self, items: list[Any]) -> None:
    self._items: list[Datum] = items

  def __getitem__(self, index: int) -> Datum:
    return self._items[index]

  def __len__(self) -> int:
    return len(self._items)


@dataclass
class SimpleGradient(Gradient):
  """Test gradient carrying a numeric adjustment."""

  adjustment: float = 0.0

  def accumulate(self, other: 'Gradient') -> 'Gradient':
    if not isinstance(other, SimpleGradient):
      msg = f'expected SimpleGradient, got {type(other).__name__}'
      raise TypeError(msg)
    return SimpleGradient(adjustment=self.adjustment + other.adjustment)

  def render(self) -> str:
    return f'adjust by {self.adjustment}'


class CounterMetric(Metric):
  """Counts items processed."""

  higher_is_better = True

  def __init__(self):
    super().__init__()
    self.add_state('_count', 0)

  def update(self, datum: Datum) -> None:
    self._count += 1

  def compute(self) -> dict[str, float]:
    return {'count': float(self._count)}


class SimpleLoss(Loss):
  """Produces a SimpleGradient."""

  def forward(self, data: Datum, targets: Any = None) -> None:
    super().forward(data, targets)

  def compute_seed_gradient(self) -> SimpleGradient:
    return SimpleGradient(adjustment=0.1)


class SimpleOptimizer(Optimizer):
  """Appends a marker to PathParameter files on each step."""

  def __init__(self, params: list, config_path: Path | None = None):
    super().__init__(params)
    self._config_path = config_path

  def step(self) -> None:
    if self._config_path is not None and self._config_path.exists():
      content = json.loads(self._config_path.read_text())
      content['step_count'] = content.get('step_count', 0) + 1
      self._config_path.write_text(json.dumps(content, indent=2))


class SimpleModule(AutoPilotModule):
  """Minimal module with a PathParameter for file-based versioning."""

  def __init__(self, config_dir: str):
    super().__init__()
    self.config = PathParameter(source=config_dir, pattern='*.json')
    self.loss = SimpleLoss([self.config])
    self.metric = CounterMetric()
    self._config_dir = config_dir

  def forward(self, batch: Any) -> Datum:
    return batch

  def training_step(self, batch: Any, batch_idx: int) -> Datum:
    return self(batch)

  def validation_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    item = batch
    if isinstance(batch, Datum) and batch.items:
      first = batch.items[0]
      if isinstance(first, EvalDatum):
        item = first
    if not isinstance(item, EvalDatum):
      item = EvalDatum(success=True)
    return item

  def configure_optimizers(self) -> SimpleOptimizer:
    config_path = Path(self._config_dir) / 'config.json'
    return SimpleOptimizer(list(self.parameters()), config_path=config_path)


class SimpleDataModule(DataModule):
  """Minimal DataModule with Stage-aware setup."""

  def __init__(self, num_items: int = 5):
    super().__init__()
    self._num_items = num_items
    self._train_data: list[Datum] | None = None
    self._val_data: list[EvalDatum] | None = None
    self._stage: Stage | None = None

  def setup(self, stage: Stage) -> None:
    self._stage = stage
    if stage == Stage.fit:
      self._train_data = [Datum() for _ in range(self._num_items)]
      self._val_data = [EvalDatum(success=True) for _ in range(self._num_items)]

  def train_dataloader(self) -> DataLoader:
    ds = _InMemoryDataset(self._train_data or [])
    return DataLoader(ds, batch_size=1, sampler=SequentialSampler(ds))

  def val_dataloader(self) -> DataLoader:
    ds = _InMemoryDataset(self._val_data or [])
    return DataLoader(ds, batch_size=1, sampler=SequentialSampler(ds))

  def state_dict(self) -> dict[str, Any]:
    return {
      'num_items': self._num_items,
      'stage': self._stage.value if self._stage else None,
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    self._num_items = state['num_items']
    stage_val = state.get('stage')
    self._stage = Stage(stage_val) if stage_val is not None else None


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
  """Create workspace dirs and seed config file."""
  workspace = tmp_path / 'workspace'
  workspace.mkdir()
  config_dir = workspace / 'config'
  config_dir.mkdir()
  store_path = workspace / '.store'
  seed = {'version': 1, 'step_count': 0}
  (config_dir / 'config.json').write_text(json.dumps(seed, indent=2))
  return workspace, config_dir, store_path


def _resume_from_checkpoint(
  config: AutoPilotConfig,
  config_dir: Path,
  ckpt_dir: Path,
  ckpt_io: JSONCheckpointIO,
  ckpt_path: Path,
) -> AutoPilotExperiment:
  """Build a fresh trainer and resume from saved checkpoint."""
  module = SimpleModule(str(config_dir))
  store = FileStore(config)
  store.register_parameters(dict(module.named_parameters()))
  experiment = AutoPilotExperiment(experiment_id='ckpt-exp-resume')
  trainer = Trainer(
    callbacks=[
      StoreCheckpointCallback(),
      CheckpointCallback(directory=ckpt_dir, checkpoint_io=ckpt_io),
    ],
    experiment=experiment,
    store=store,
    accumulate_grad_batches=100,
  )
  dm = SimpleDataModule(num_items=3)
  trainer.fit(
    module,
    datamodule=dm,
    max_epochs=3,
    ckpt_path=ckpt_path,
    checkpoint_io=ckpt_io,
  )
  return experiment


class TestEphemeralIntegration:
  """Full cold-start -> train -> snapshot -> resume in ephemeral workspace."""

  def test_cold_start_train_snapshot(self, tmp_path: Path) -> None:
    """Train for 2 epochs, verify store snapshots and experiment metrics."""
    workspace, config_dir, store_path = _setup_workspace(tmp_path)

    module = SimpleModule(str(config_dir))
    config = AutoPilotConfig(workspace=workspace)
    config.store_path = store_path
    store = FileStore(config)
    store.register_parameters(dict(module.named_parameters()))

    experiment = AutoPilotExperiment(experiment_id='test-exp')
    trainer = Trainer(
      callbacks=[StoreCheckpointCallback()],
      experiment=experiment,
      store=store,
      accumulate_grad_batches=100,
    )

    dm = SimpleDataModule(num_items=3)
    trainer.fit(module, datamodule=dm, max_epochs=2)

    assert experiment.status == 'completed'
    assert experiment.metrics is not None

    log_entries = store.log('test-exp')
    assert len(log_entries) >= 2

  def test_checkpoint_save_and_resume(self, tmp_path: Path) -> None:
    """Save a checkpoint after training, then resume from it."""
    workspace, config_dir, store_path = _setup_workspace(tmp_path)
    ckpt_dir = workspace / 'checkpoints'
    ckpt_dir.mkdir()
    ckpt_io = JSONCheckpointIO()

    module = SimpleModule(str(config_dir))
    config = AutoPilotConfig(workspace=workspace)
    config.store_path = store_path
    store = FileStore(config)
    store.register_parameters(dict(module.named_parameters()))

    experiment = AutoPilotExperiment(experiment_id='ckpt-exp')
    trainer = Trainer(
      callbacks=[
        StoreCheckpointCallback(),
        CheckpointCallback(directory=ckpt_dir, checkpoint_io=ckpt_io),
      ],
      experiment=experiment,
      store=store,
      accumulate_grad_batches=100,
    )

    dm = SimpleDataModule(num_items=3)
    trainer.fit(module, datamodule=dm, max_epochs=2)

    ckpt_path = ckpt_dir / 'epoch-0001.json'
    assert ckpt_io.exists(ckpt_path)
    saved = ckpt_io.load(ckpt_path)
    assert 'experiment' in saved
    assert 'module' in saved
    assert 'callbacks' in saved

    resume_exp = _resume_from_checkpoint(config, config_dir, ckpt_dir, ckpt_io, ckpt_path)
    assert resume_exp.status == 'completed'

  def test_named_parameter_snapshot_roundtrip(self, tmp_path: Path) -> None:
    """Verify named parameters survive snapshot -> checkout cycle."""
    workspace, config_dir, store_path = _setup_workspace(tmp_path)

    module = SimpleModule(str(config_dir))
    config = AutoPilotConfig(workspace=workspace)
    config.store_path = store_path
    store = FileStore(config)
    store.register_parameters(dict(module.named_parameters()))

    (config_dir / 'config.json').write_text('{"version": 1, "step_count": 0}')
    store.snapshot('snap-exp', 0)

    (config_dir / 'config.json').write_text('{"version": 2, "step_count": 5}')
    store.snapshot('snap-exp', 1)

    store.checkout('snap-exp', 0)
    content = json.loads((config_dir / 'config.json').read_text())
    assert content['version'] == 1

    store.checkout('snap-exp', 1)
    content = json.loads((config_dir / 'config.json').read_text())
    assert content['version'] == 2

  def test_sampler_based_dataloader(self, tmp_path: Path) -> None:
    """DataLoader uses sampler protocol, not shuffle kwarg."""
    ds = _InMemoryDataset([Datum() for _ in range(10)])

    sequential_loader = DataLoader(ds, batch_size=2, sampler=SequentialSampler(ds))
    batches = list(sequential_loader)
    assert len(batches) == 5

    rng = random.Random(42)
    rng_sampler = RandomSampler(ds, generator=rng)
    random_loader = DataLoader(ds, batch_size=2, sampler=rng_sampler)
    random_batches = list(random_loader)
    assert len(random_batches) == 5

  def test_stage_enum_on_datamodule(self, tmp_path: Path) -> None:
    """DataModule.setup() accepts Stage enum, not strings."""
    dm = SimpleDataModule(num_items=4)
    dm.setup(Stage.fit)
    assert dm._stage == Stage.fit

    state = dm.state_dict()
    assert state['stage'] == 'fit'

    dm2 = SimpleDataModule()
    dm2.load_state_dict(state)
    assert dm2._num_items == 4
    assert dm2._stage == Stage.fit

  def test_experiment_context_manager(self, tmp_path: Path) -> None:
    """Experiment works as context manager with lifecycle guards."""
    exp = AutoPilotExperiment(experiment_id='ctx-exp')
    assert exp.status == 'pending'

    with exp:
      assert exp.status == 'running'
      exp.complete({'accuracy': 0.95})

    assert exp.status == 'completed'
    assert exp.metrics['accuracy'] == 0.95
