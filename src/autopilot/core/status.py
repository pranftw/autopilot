"""Experiment status gathering from Forest and on-disk artifacts.

Resolves experiment state through ``Forest.query().filter(id=experiment_id).all()``
and merges on-disk artifacts (``SummaryArtifact``, ``RunStateArtifact``, epoch dirs)
from the workspace experiment directory.

Removed fields (formerly from Manifest): ``decision``, ``decision_reason`` --
policy gates own accept/reject decisions.
"""

from autopilot.core.artifacts.experiment import RunStateArtifact, SummaryArtifact
from autopilot.core.enums import Status
from autopilot.core.errors import TrackingError
from autopilot.core.forest import Forest
from pathlib import Path
from typing import Any


def get_experiment_status(forest: Forest, experiment_id: str) -> dict[str, Any]:
  """Gather experiment status from Forest-backed Experiment plus on-disk artifacts.

  Loads the node via ``forest.query().filter(id=experiment_id).all()``. When no
  node matches, raises ``TrackingError`` with an actionable message.

  Populates ``id`` and ``epoch`` from ``node.experiment``. Omits ``decision``
  and ``decision_reason`` (policy gates own accept/reject).

  Resolves artifact directory as ``forest.store.config.experiment_path(slug=experiment_id)``
  for ``SummaryArtifact``, ``RunStateArtifact``, and ``_scan_epoch_dirs``.

  When duplicate ids exist across trees, uses the first match.

  Args:
    forest: Loaded forest (e.g. from ``load_forest``).
    experiment_id: Experiment id / slug to resolve.

  Returns:
    Dict with trained epoch counts, experiment id/epoch, and stop metadata.

  Raises:
    TrackingError: When no node matches ``experiment_id`` in the forest.
  """
  nodes = forest.query().filter(id=experiment_id).all()
  if len(nodes) == 0:
    msg = f'experiment {experiment_id!r} not found in forest; register it in a tree first'
    raise TrackingError(msg)

  node = nodes[0]
  exp = node.experiment
  exp_dir = forest.store.config.experiment_path(slug=experiment_id)

  trained_epochs, _latest_epoch = _scan_epoch_dirs(exp_dir)

  result: dict[str, Any] = {
    'id': exp.id,
    'epoch': exp.epoch,
    'trained_epochs': trained_epochs,
  }

  summary = SummaryArtifact().read_raw(exp_dir)
  if isinstance(summary, dict):
    result['stop_reason'] = summary.get('stop_reason')
    result['last_good_epoch'] = summary.get('last_good_epoch')

  run_state = RunStateArtifact().read_raw(exp_dir)
  if isinstance(run_state, dict):
    if run_state['status'] == Status.running.value:
      result['stop_reason'] = 'crash'
    else:
      if 'stop_reason' not in result or result['stop_reason'] is None:
        result['stop_reason'] = run_state.get('stop_reason')
      if 'last_good_epoch' not in result or result['last_good_epoch'] is None:
        result['last_good_epoch'] = run_state.get('last_good_epoch')

  return result


def _scan_epoch_dirs(exp_dir: Path) -> tuple[int, int]:
  """Return (count, highest_epoch_number) from epoch dirs.

  Args:
    exp_dir: Experiment directory to scan.

  Returns:
    Tuple of (epoch dir count, highest epoch number found).
  """
  count = 0
  highest = 0
  if not exp_dir.exists():
    return 0, 0
  for child in exp_dir.iterdir():
    if child.is_dir() and child.name.startswith('epoch_'):
      try:
        num = int(child.name.split('_', 1)[1])
        count += 1
        highest = max(highest, num)
      except (ValueError, IndexError):
        pass
  return count, highest
