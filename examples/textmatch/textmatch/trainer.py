from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.ai.store.file_store import FileStore
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.config import AutoPilotConfig
from autopilot.core.models import Result
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import GateResult
from autopilot.policy.policy import Policy
from pathlib import Path
from textmatch.module import TextMatchModule
import json


class AccuracyPolicy(Policy):
  def __init__(self, threshold: float = 0.80):
    self._threshold = threshold

  def name(self) -> str:
    return 'AccuracyPolicy'

  def _resolve_accuracy(self, metrics: dict[str, float]) -> float:
    """Resolve accuracy from both prefixed and unprefixed metric keys."""
    if 'val_accuracy' in metrics:
      return metrics['val_accuracy']
    return metrics.get('accuracy', 0.0)

  def forward(self, result: Result) -> GateResult:
    accuracy = self._resolve_accuracy(result.metrics)
    if accuracy >= self._threshold:
      return GateResult.PASSED
    return GateResult.FAIL

  def explain(self, result: Result) -> str:
    accuracy = self._resolve_accuracy(result.metrics)
    return f'accuracy={accuracy:.2%}, threshold={self._threshold:.2%}'


def next_slug(store_path: Path) -> str:
  refs_file = store_path / 'refs.json'
  if not refs_file.exists():
    return 'run-1'
  refs = json.loads(refs_file.read_text(encoding='utf-8'))
  branches = refs.get('branches', {})
  existing = [k for k in branches if k.startswith('run-')]
  return f'run-{len(existing) + 1}'


def build_trainer(
  module: TextMatchModule,
  store_path: Path,
  dry_run: bool = False,
  threshold: float = 0.30,
  accumulate_grad_batches: int = 100,
  experiment_slug: str | None = None,
) -> tuple[Trainer, FileStore]:
  slug = experiment_slug or next_slug(store_path)
  config = AutoPilotConfig(workspace=store_path.parent)
  config.store_path = store_path
  store = FileStore(config)
  store.register_parameters(dict(module.named_parameters()))
  policy = AccuracyPolicy(threshold=threshold)
  experiment = AutoPilotExperiment(experiment_id=slug)
  trainer = Trainer(
    callbacks=[StoreCheckpointCallback()],
    policy=policy,
    experiment=experiment,
    store=store,
    dry_run=dry_run,
    accumulate_grad_batches=accumulate_grad_batches,
  )
  return trainer, store
