"""Trainer.fit() entrypoint for the agent harness.

Delegates to ``build_trainer()`` and runs ``trainer.fit()`` with
``max_epochs``. Supports ``--json`` for structured output.

``--use-judge`` / ``--no-judge`` flags have parity with the CLI
(``autopilot -p harness optimize loop --use-judge``) and ``run.py``.
Default is judge mode when neither flag is passed.

Requires configured credentials for the inference model (OPENROUTER_API_KEY)
and optimizer agent (ANTHROPIC_API_KEY). Set environment variables before running.

Usage:
  uv run python run_trainer.py --max-epochs 3
  uv run python run_trainer.py --max-epochs 5 --env prod --json
  uv run python run_trainer.py --max-epochs 3 --no-judge
  uv run python run_trainer.py --max-epochs 3 --use-judge

JSON envelope shape:
  {
    "experiment": "<slug>",
    "total_epochs": N,
    "epochs": [{"epoch": 0, "metrics": {...}}, ...]
  }
"""

from harness import DEFAULT_MODEL
from harness.trainer import build_trainer
from pathlib import Path
import argparse
import json
import sys

ENV_MODELS = {
  'dev': DEFAULT_MODEL,
  'prod': DEFAULT_MODEL,
}


def example_dir() -> Path:
  """Return the example project root."""
  return Path(__file__).parent


def main(argv: list[str] | None = None) -> dict:
  """Run Trainer.fit() for the harness.

  Args:
    argv: Optional CLI arguments (defaults to sys.argv).

  Returns:
    Summary dict with experiment id and per-epoch metrics.
  """
  parser = argparse.ArgumentParser(description='Harness Trainer.fit()')
  parser.add_argument('--max-epochs', type=int, default=3)
  parser.add_argument(
    '--env',
    default='dev',
    choices=list(ENV_MODELS),
    help='Environment preset (controls default model)',
  )
  parser.add_argument('--experiment', default=None, metavar='SLUG')
  parser.add_argument('--json', action='store_true', dest='use_json')
  parser.add_argument(
    '--use-judge',
    action='store_true',
    default=False,
    dest='use_judge',
    help='force judge loss path (default when neither flag is passed)',
  )
  parser.add_argument(
    '--no-judge',
    action='store_true',
    default=False,
    dest='no_judge',
    help='use heuristic HarnessLoss path',
  )
  args = parser.parse_args(argv)

  if args.use_judge and args.no_judge:
    print('cannot pass both --use-judge and --no-judge', file=sys.stderr)
    sys.exit(2)

  use_judge: bool = not args.no_judge

  root = example_dir()
  model = ENV_MODELS[args.env]

  trainer, module, datamodule = build_trainer(
    root,
    model=model,
    experiment_slug=args.experiment,
    use_judge=use_judge,
  )

  experiment = trainer.experiment
  experiment.add_context(
    f'Chose max_epochs={args.max_epochs} with env={args.env}, '
    f'use_judge={use_judge} for this training run.',
    source='examples.harness.run_trainer',
    metadata={
      'argv_max_epochs': args.max_epochs,
      'argv_env': args.env,
      'use_judge': use_judge,
      'model': model,
    },
  )

  result = trainer.fit(module, datamodule=datamodule, max_epochs=args.max_epochs)
  experiment_id = experiment.id

  output: dict = {
    'experiment': experiment_id,
    'total_epochs': result['total_epochs'],
    'epochs': [],
  }
  for ep in result['epochs']:
    output['epochs'].append(
      {
        'epoch': ep['epoch'],
        'metrics': ep['metrics'],
      }
    )

  if args.use_json:
    print(json.dumps(output, indent=2))
  else:
    print(f'=== Harness: Trainer.fit() [experiment: {experiment_id}] ===\n')
    for ep in output['epochs']:
      print(f'  Epoch {ep["epoch"]}: {ep["metrics"]}')
    print(f'\nTotal epochs: {output["total_epochs"]}')
    print('\nDone.')

  return output


if __name__ == '__main__':
  main()
