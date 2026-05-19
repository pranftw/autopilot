from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.ai.store.file_store import FileStore
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.config import AutoPilotConfig
from autopilot.core.models import Result
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import GateResult
from autopilot.policy.policy import Policy
from pathlib import Path
from protim.module import PromptModule
import json


class AccuracyPolicy(Policy):
  def __init__(self, threshold: float = 0.50):
    self._threshold = threshold

  def name(self) -> str:
    return 'AccuracyPolicy'

  def forward(self, result: Result) -> GateResult:
    accuracy = result.metrics.get('accuracy', 0.0)
    if accuracy >= self._threshold:
      return GateResult.PASSED
    return GateResult.FAIL

  def explain(self, result: Result) -> str:
    accuracy = result.metrics.get('accuracy', 0.0)
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
  module: PromptModule,
  store_path: Path,
  dry_run: bool = False,
) -> tuple[Trainer, FileStore]:
  slug = next_slug(store_path)
  config = AutoPilotConfig(workspace=store_path.parent)
  config.store_path = store_path
  store = FileStore(config)
  store.register_parameters(dict(module.named_parameters()))
  policy = AccuracyPolicy(threshold=0.50)
  experiment = AutoPilotExperiment(experiment_id=slug)
  trainer = Trainer(
    callbacks=[StoreCheckpointCallback()],
    policy=policy,
    experiment=experiment,
    store=store,
    dry_run=dry_run,
    accumulate_grad_batches=100,
  )
  return trainer, store
