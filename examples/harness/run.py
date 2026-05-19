"""Manual optimization loop for the agent harness (no Trainer).

Makes the training algorithm explicit: epochs, scenarios, forward, loss,
backward, optimizer step, store snapshot.

Requires configured credentials for the inference model (OPENROUTER_API_KEY)
and optimizer agent (ANTHROPIC_API_KEY). Set environment variables before running.

Usage:
  uv run python run.py --max-epochs 3
  uv run python run.py --max-epochs 5 --model openrouter:google/gemma-4-31b-it
  uv run python run.py --max-epochs 3 --no-judge
  uv run python run.py --max-epochs 3 --use-judge

Judge mode (default): uses JudgeLoss + AgentCollator + TextGradient.
Heuristic mode (--no-judge): uses HarnessLoss + HarnessGradient.
"""

from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from harness import DEFAULT_MODEL
from harness.data import HarnessDataModule
from harness.module import HarnessModule
from pathlib import Path
import argparse
import json
import sys


def example_dir() -> Path:
  """Return the example project root."""
  return Path(__file__).parent


def main(argv: list[str] | None = None) -> dict:
  """Run the manual optimization loop.

  Args:
    argv: Optional CLI arguments (defaults to sys.argv).

  Returns:
    Summary dict with per-epoch metrics.
  """
  parser = argparse.ArgumentParser(description='Harness manual optimization loop')
  parser.add_argument('--max-epochs', type=int, default=3)
  parser.add_argument('--scenario-dir', default=None, metavar='PATH')
  parser.add_argument('--model', default=DEFAULT_MODEL)
  parser.add_argument('--json', action='store_true', dest='use_json')
  parser.add_argument(
    '--use-judge',
    action='store_true',
    default=False,
    dest='use_judge',
    help='use JudgeLoss + AgentCollator + TextGradient (default)',
  )
  parser.add_argument(
    '--no-judge',
    action='store_true',
    default=False,
    dest='no_judge',
    help='use HarnessLoss + HarnessGradient heuristic path',
  )
  args = parser.parse_args(argv)

  if args.use_judge and args.no_judge:
    print('cannot pass both --use-judge and --no-judge', file=sys.stderr)
    sys.exit(2)

  use_judge: bool = not args.no_judge

  root = example_dir()
  harness_pkg = root / 'harness'
  harness_root = str(harness_pkg)
  scenario_dir = args.scenario_dir or str(harness_pkg / 'scenarios')

  module = HarnessModule(harness_root, model=args.model, use_judge=use_judge)
  datamodule = HarnessDataModule(scenario_dir)
  loss = module.loss_fn
  optimizer = module.configure_optimizers()
  metrics = module.metrics

  config = AutoPilotConfig(workspace=root)
  store_path = root / '.autopilot' / 'store'
  config.store_path = store_path
  store = FileStore(config)
  store.register_parameters(dict(module.named_parameters()))

  output: dict = {'total_epochs': args.max_epochs, 'epochs': []}

  train_loader = datamodule.train_dataloader()

  for epoch in range(args.max_epochs):
    metrics.reset()
    loss.reset()

    for batch_idx, batch in enumerate(train_loader):
      data = module.training_step(batch, batch_idx)
      loss(data)
      metrics.update(data)

    loss.backward()
    epoch_metrics = metrics.compute()

    optimizer.step()
    optimizer.zero_grad()

    store.snapshot('manual-run', epoch=epoch)

    output['epochs'].append(
      {
        'epoch': epoch,
        'metrics': epoch_metrics,
      }
    )

    if not args.use_json:
      print(f'Epoch {epoch}: {epoch_metrics}')

  if args.use_json:
    print(json.dumps(output, indent=2))
  elif not output['epochs']:
    print('No epochs run.')
  else:
    print(f'\nTotal epochs: {args.max_epochs}')
    print('Done.')

  return output


if __name__ == '__main__':
  main()
