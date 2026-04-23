"""Result normalization across splits.

Thin wrappers around Artifact reads for experiment results and split summaries.
"""

from autopilot.core.artifacts.dataset import SplitSummaryArtifact
from autopilot.core.artifacts.experiment import ResultArtifact
from autopilot.core.errors import TrackingError
from pathlib import Path
from typing import Any


def load_result(experiment_dir: Path) -> dict[str, Any] | None:
  """Load result.json from an experiment directory, or None if missing."""
  return ResultArtifact().read_raw(experiment_dir)


def load_split_summary(experiment_dir: Path, split: str) -> dict[str, Any]:
  """Load a split summary from the experiment directory."""
  data = SplitSummaryArtifact(split).read_raw(experiment_dir)
  if data is None:
    raise TrackingError(f'summary not found: {experiment_dir / f"{split}_summary.json"}')
  return data
