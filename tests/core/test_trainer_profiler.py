"""Integration tests for Trainer + Profiler."""

from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.profiler import Profiler, SimpleProfiler
from autopilot.core.trainer.trainer import Trainer
from autopilot.data.dataloader import DataLoader
from autopilot.data.dataset import Dataset
from pathlib import Path
from tests.doubles import DirectNumericLoss, NoopEvalModule, NoOpOptimizer
from typing import Any
import json
import pytest


class _TinyDataset(Dataset):
  """Minimal dataset for tests."""

  def __len__(self) -> int:
    return 3

  def __getitem__(self, index: int) -> dict[str, int]:
    return {'x': index}


class _ModuleWithOptimizer(NoopEvalModule):
  """Module that returns an optimizer from configure_optimizers."""

  def __init__(self) -> None:
    super().__init__()
    self.loss = DirectNumericLoss()
    self._opt = NoOpOptimizer([])

  def configure_optimizers(self):
    return self._opt


class TestTrainerWithProfiler:
  """Trainer(..., profiler=SimpleProfiler()): profiler is invoked during fit."""

  def test_trainer_with_profiler(self, tmp_path) -> None:
    profiler = SimpleProfiler()
    experiment = Experiment(experiment_id='prof-test')
    config = AutoPilotConfig(workspace=tmp_path)
    trainer = Trainer(
      experiment=experiment,
      config=config,
      profiler=profiler,
    )
    module = _ModuleWithOptimizer()
    train_loader = DataLoader(_TinyDataset(), batch_size=1)
    val_loader = DataLoader(_TinyDataset(), batch_size=1)

    trainer.fit(
      module,
      train_dataloaders=train_loader,
      val_dataloaders=val_loader,
      max_epochs=1,
    )

    result = profiler.describe()
    assert 'training_step' in result
    assert result['training_step']['count'] >= 1


class TestTrainerWithoutProfiler:
  """profiler=None: fit completes without AttributeError on profiler."""

  def test_trainer_without_profiler(self, tmp_path) -> None:
    experiment = Experiment(experiment_id='no-prof-test')
    config = AutoPilotConfig(workspace=tmp_path)
    trainer = Trainer(
      experiment=experiment,
      config=config,
    )
    module = _ModuleWithOptimizer()
    train_loader = DataLoader(_TinyDataset(), batch_size=1)

    result = trainer.fit(
      module,
      train_dataloaders=train_loader,
      max_epochs=1,
    )
    assert result is not None
    assert 'epochs' in result


class TestTrainerProfilerSectionCompleteness:
  """After fit, describe() contains at least 4 section keys (no store)."""

  def test_trainer_profiler_section_completeness(self, tmp_path) -> None:
    profiler = SimpleProfiler()
    experiment = Experiment(experiment_id='section-test')
    config = AutoPilotConfig(workspace=tmp_path)
    trainer = Trainer(
      experiment=experiment,
      config=config,
      profiler=profiler,
    )
    module = _ModuleWithOptimizer()
    train_loader = DataLoader(_TinyDataset(), batch_size=1)
    val_loader = DataLoader(_TinyDataset(), batch_size=1)

    trainer.fit(
      module,
      train_dataloaders=train_loader,
      val_dataloaders=val_loader,
      max_epochs=1,
    )

    result = profiler.describe()
    expected_sections = {'training_step', 'validation_step', 'backward', 'optimizer_step'}
    present = set(result.keys()) & expected_sections
    assert len(present) >= 4, f'expected >= 4 sections, got {present}'
    for section in present:
      assert result[section]['count'] > 0


class TestProfilerFailureIsolation:
  """Profiler raises once: training still completes (not aborted)."""

  def test_profiler_failure_isolation(self, tmp_path) -> None:
    calls: list[str] = []

    class BrokenProfiler(Profiler):
      def __init__(self) -> None:
        self._broken_once = False

      def start(self, action: str) -> None:
        if not self._broken_once:
          self._broken_once = True
          msg = 'profiler broken'
          raise RuntimeError(msg)
        calls.append(f'start:{action}')

      def stop(self, action: str) -> None:
        calls.append(f'stop:{action}')

      def describe(self) -> dict[str, Any]:
        return {}

    experiment = Experiment(experiment_id='isolation-test')
    config = AutoPilotConfig(workspace=tmp_path)
    trainer = Trainer(
      experiment=experiment,
      config=config,
      profiler=BrokenProfiler(),
    )
    module = _ModuleWithOptimizer()
    train_loader = DataLoader(_TinyDataset(), batch_size=1)

    result = trainer.fit(
      module,
      train_dataloaders=train_loader,
      max_epochs=1,
    )
    assert result is not None
    assert 'epochs' in result


class TestProfilerOutputWritten:
  """After fit, profiler_summary.json exists and parses to dict."""

  def test_profiler_output_written(self, tmp_path) -> None:
    profiler = SimpleProfiler()
    exp_id = 'output-test'
    config = AutoPilotConfig(workspace=tmp_path)
    experiment = Experiment(experiment_id=exp_id)
    trainer = Trainer(
      experiment=experiment,
      config=config,
      profiler=profiler,
    )
    module = _ModuleWithOptimizer()
    train_loader = DataLoader(_TinyDataset(), batch_size=1)
    val_loader = DataLoader(_TinyDataset(), batch_size=1)

    trainer.fit(
      module,
      train_dataloaders=train_loader,
      val_dataloaders=val_loader,
      max_epochs=1,
    )

    summary_path = config.experiment_path(slug=exp_id) / 'profiler_summary.json'
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert isinstance(data, dict)
    assert 'training_step' in data


class TestStoreSnapshotProfiled:
  """store_snapshot appears when StoreCheckpointCallback is active."""

  def test_store_snapshot_profiled(self, tmp_path) -> None:
    from autopilot.ai.parameter import PathParameter
    from autopilot.ai.store.file_store import FileStore

    src = tmp_path / 'src'
    src.mkdir()
    (src / 'main.py').write_text('print("hi")', encoding='utf-8')

    profiler = SimpleProfiler()
    exp_id = 'store-snap-test'
    config = AutoPilotConfig(workspace=tmp_path)
    param = PathParameter(source=str(src), pattern='*')
    store = FileStore(config)
    store.register_parameters({'source': param})

    experiment = Experiment(experiment_id=exp_id)
    trainer = Trainer(
      experiment=experiment,
      config=config,
      profiler=profiler,
      store=store,
      callbacks=[StoreCheckpointCallback()],
    )
    module = _ModuleWithOptimizer()
    train_loader = DataLoader(_TinyDataset(), batch_size=1)

    trainer.fit(
      module,
      train_dataloaders=train_loader,
      max_epochs=1,
    )

    result = profiler.describe()
    assert 'store_snapshot' in result
    assert result['store_snapshot']['count'] >= 1


class _FailOnFirstBatchModule(_ModuleWithOptimizer):
  """Module that raises on the first training batch."""

  def training_step(self, batch: Any, batch_idx: int) -> Any:
    msg = 'intentional training failure'
    raise RuntimeError(msg)


class TestProfilerSummaryWrittenOnFailure:
  """BUG-005: profiler summary must be written even when fit fails."""

  def test_profiler_summary_written_on_failure(self, tmp_path: Path) -> None:
    profiler = SimpleProfiler()
    exp_id = 'fail-prof-test'
    config = AutoPilotConfig(workspace=tmp_path)
    experiment = Experiment(experiment_id=exp_id)
    trainer = Trainer(
      experiment=experiment,
      config=config,
      profiler=profiler,
    )
    module = _FailOnFirstBatchModule()
    train_loader = DataLoader(_TinyDataset(), batch_size=1)

    with pytest.raises(RuntimeError, match='intentional training failure'):
      trainer.fit(module, train_dataloaders=train_loader, max_epochs=1)

    summary_path = config.experiment_path(slug=exp_id) / 'profiler_summary.json'
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert isinstance(data, dict)
    assert 'training_step' in data


class TestProfilerSummaryWrittenOnSuccess:
  """Regression: profiler summary still written on successful fit."""

  def test_profiler_summary_written_on_success(self, tmp_path: Path) -> None:
    profiler = SimpleProfiler()
    exp_id = 'success-prof-test'
    config = AutoPilotConfig(workspace=tmp_path)
    experiment = Experiment(experiment_id=exp_id)
    trainer = Trainer(
      experiment=experiment,
      config=config,
      profiler=profiler,
    )
    module = _ModuleWithOptimizer()
    train_loader = DataLoader(_TinyDataset(), batch_size=1)
    val_loader = DataLoader(_TinyDataset(), batch_size=1)

    trainer.fit(
      module,
      train_dataloaders=train_loader,
      val_dataloaders=val_loader,
      max_epochs=1,
    )

    summary_path = config.experiment_path(slug=exp_id) / 'profiler_summary.json'
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert 'training_step' in data
    assert 'validation_step' in data


class TestSanityCheckProfiledSection:
  """BUG-006: sanity check validation must appear as profiled section."""

  def test_sanity_check_profiled_section(self, tmp_path: Path) -> None:
    profiler = SimpleProfiler()
    experiment = Experiment(experiment_id='sanity-prof-test')
    config = AutoPilotConfig(workspace=tmp_path)
    trainer = Trainer(
      experiment=experiment,
      config=config,
      profiler=profiler,
      num_sanity_val_steps=2,
    )
    module = _ModuleWithOptimizer()
    train_loader = DataLoader(_TinyDataset(), batch_size=1)
    val_loader = DataLoader(_TinyDataset(), batch_size=1)

    trainer.fit(
      module,
      train_dataloaders=train_loader,
      val_dataloaders=val_loader,
      max_epochs=1,
    )

    result = profiler.describe()
    assert 'sanity_check' in result
    assert result['sanity_check']['count'] == 1


class TestNoSanitySectionWhenStepsZero:
  """No sanity_check section when num_sanity_val_steps=0."""

  def test_no_sanity_section_when_steps_zero(self, tmp_path: Path) -> None:
    profiler = SimpleProfiler()
    experiment = Experiment(experiment_id='no-sanity-prof-test')
    config = AutoPilotConfig(workspace=tmp_path)
    trainer = Trainer(
      experiment=experiment,
      config=config,
      profiler=profiler,
      num_sanity_val_steps=0,
    )
    module = _ModuleWithOptimizer()
    train_loader = DataLoader(_TinyDataset(), batch_size=1)
    val_loader = DataLoader(_TinyDataset(), batch_size=1)

    trainer.fit(
      module,
      train_dataloaders=train_loader,
      val_dataloaders=val_loader,
      max_epochs=1,
    )

    result = profiler.describe()
    assert 'sanity_check' not in result


class TestAllProfilerSectionsPresent:
  """Full fit with sanity steps: all expected sections present."""

  def test_all_profiler_sections_present(self, tmp_path: Path) -> None:
    profiler = SimpleProfiler()
    experiment = Experiment(experiment_id='all-sections-test')
    config = AutoPilotConfig(workspace=tmp_path)
    trainer = Trainer(
      experiment=experiment,
      config=config,
      profiler=profiler,
      num_sanity_val_steps=2,
    )
    module = _ModuleWithOptimizer()
    train_loader = DataLoader(_TinyDataset(), batch_size=1)
    val_loader = DataLoader(_TinyDataset(), batch_size=1)

    trainer.fit(
      module,
      train_dataloaders=train_loader,
      val_dataloaders=val_loader,
      max_epochs=1,
    )

    result = profiler.describe()
    expected = {'training_step', 'validation_step', 'backward', 'optimizer_step', 'sanity_check'}
    present = set(result.keys()) & expected
    assert present == expected, f'missing sections: {expected - present}'


class TestProfilerNoneNoCrashSanity:
  """profiler=None with sanity check: no crash."""

  def test_profiler_none_no_crash_sanity(self, tmp_path: Path) -> None:
    experiment = Experiment(experiment_id='none-prof-sanity')
    config = AutoPilotConfig(workspace=tmp_path)
    trainer = Trainer(
      experiment=experiment,
      config=config,
      num_sanity_val_steps=2,
    )
    module = _ModuleWithOptimizer()
    train_loader = DataLoader(_TinyDataset(), batch_size=1)
    val_loader = DataLoader(_TinyDataset(), batch_size=1)

    result = trainer.fit(
      module,
      train_dataloaders=train_loader,
      val_dataloaders=val_loader,
      max_epochs=1,
    )
    assert result is not None
