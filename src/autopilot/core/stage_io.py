"""Stage artifact I/O -- thin composition of paths.py and tracking/io.py.

No JSON/JSONL logic of its own. All persistence delegates to tracking/io.py.
"""

from autopilot.core.paths import epoch_artifact
from autopilot.tracking.io import append_jsonl, atomic_write_json, read_json, read_jsonl
from pathlib import Path


def write_epoch_artifact(
  experiment_dir: Path,
  epoch: int,
  filename: str,
  data: dict,
) -> None:
  path = epoch_artifact(experiment_dir, epoch, filename)
  atomic_write_json(path, data)


def append_epoch_artifact(
  experiment_dir: Path,
  epoch: int,
  filename: str,
  record: dict,
) -> None:
  path = epoch_artifact(experiment_dir, epoch, filename)
  path.parent.mkdir(parents=True, exist_ok=True)
  append_jsonl(path, record)


def read_epoch_artifact(
  experiment_dir: Path,
  epoch: int,
  filename: str,
) -> dict | None:
  path = epoch_artifact(experiment_dir, epoch, filename)
  return read_json(path)


def read_epoch_artifact_lines(
  experiment_dir: Path,
  epoch: int,
  filename: str,
) -> list[dict]:
  path = epoch_artifact(experiment_dir, epoch, filename)
  return read_jsonl(path)


def write_experiment_artifact(
  experiment_dir: Path,
  filename: str,
  data: dict,
) -> None:
  atomic_write_json(experiment_dir / filename, data)


def append_experiment_artifact(
  experiment_dir: Path,
  filename: str,
  record: dict,
) -> None:
  path = experiment_dir / filename
  path.parent.mkdir(parents=True, exist_ok=True)
  append_jsonl(path, record)


def read_experiment_artifact(
  experiment_dir: Path,
  filename: str,
) -> dict | None:
  return read_json(experiment_dir / filename)


def read_experiment_artifact_lines(
  experiment_dir: Path,
  filename: str,
) -> list[dict]:
  return read_jsonl(experiment_dir / filename)
