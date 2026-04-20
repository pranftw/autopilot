"""Hyperparameter tracking and management.

First-class hyperparameter handling with versioning and locking.
"""

from autopilot.core.errors import ExperimentError
from autopilot.core.models import HyperparamSet
from pathlib import Path
from typing import Any
import json


def load_hyperparams(experiment_dir: Path) -> HyperparamSet:
  """Load hyperparams.json from an experiment directory."""
  path = experiment_dir / 'hyperparams.json'
  if not path.exists():
    return HyperparamSet()
  with open(path) as f:
    data = json.load(f)
  return HyperparamSet.from_dict(data)


def save_hyperparams(experiment_dir: Path, hparams: HyperparamSet) -> None:
  """Write hyperparams.json to an experiment directory."""
  path = experiment_dir / 'hyperparams.json'
  with open(path, 'w') as f:
    json.dump(hparams.to_dict(), f, indent=2)
    f.write('\n')


def update_hyperparams(
  experiment_dir: Path,
  updates: dict[str, Any],
) -> HyperparamSet:
  """Update hyperparameter values and bump version."""
  hparams = load_hyperparams(experiment_dir)
  if hparams.locked:
    raise ExperimentError('hyperparameters are locked; unlock before updating')
  hparams.values.update(updates)
  hparams.version += 1
  save_hyperparams(experiment_dir, hparams)
  return hparams


def lock_hyperparams(experiment_dir: Path) -> HyperparamSet:
  """Lock hyperparameters to prevent further changes."""
  hparams = load_hyperparams(experiment_dir)
  hparams.locked = True
  save_hyperparams(experiment_dir, hparams)
  return hparams


def validate_hyperparams_schema(
  hparams: HyperparamSet,
  schema: dict[str, Any],
) -> list[str]:
  """Validate hyperparameter values against a schema.

  Returns a list of validation error messages (empty if valid).
  Schema format: {param_name: {'type': str, 'required': bool}}
  """
  errors: list[str] = []
  for param_name, rules in schema.items():
    if rules.get('required') and param_name not in hparams.values:
      errors.append(f'missing required hyperparameter: {param_name}')
  for param_name in hparams.values:
    if param_name not in schema:
      errors.append(f'unknown hyperparameter: {param_name}')
  return errors
