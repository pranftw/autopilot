from autopilot.ai.store import FileStore
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.checkpoint import JSONCheckpoint
from autopilot.core.experiment import Experiment
from autopilot.core.logger import JSONLogger
from autopilot.core.models import Result
from autopilot.core.types import GateResult
from autopilot.core.trainer import Trainer
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
      return GateResult.PASS
    return GateResult.FAIL

  def explain(self, result: Result) -> str:
    accuracy = result.metrics.get('accuracy', 0.0)
    return f'accuracy={accuracy:.2%}, threshold={self._threshold:.2%}'


def next_slug(store_path: Path) -> str:
  refs_file = store_path / 'refs.json'
  if not refs_file.exists():
    return 'run-1'
  refs = json.loads(refs_file.read_text(encoding='utf-8'))
  existing = [k for k in refs if k.startswith('run-') and k != 'HEAD']
  return f'run-{len(existing) + 1}'


def build_trainer(
  module: PromptModule,
  store_path: Path,
  dry_run: bool = False,
  experiment_dir: Path | None = None,
) -> tuple[Trainer, FileStore]:
  slug = next_slug(store_path)
  store = FileStore(store_path, slug, list(module.parameters()))
  policy = AccuracyPolicy(threshold=0.50)
  exp_dir = experiment_dir or store_path / slug
  experiment = Experiment(
    exp_dir,
    slug=slug,
    logger=JSONLogger(exp_dir),
    checkpoint=JSONCheckpoint(),
    store=store,
  )
  trainer = Trainer(
    callbacks=[StoreCheckpointCallback()],
    policy=policy,
    experiment=experiment,
    dry_run=dry_run,
    accumulate_grad_batches=100,
  )
  return trainer, store
