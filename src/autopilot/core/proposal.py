"""Proposal I/O helper functions.

Pure functions for hypothesis log and verdict persistence.
"""

from autopilot.core.stage_io import (
  append_experiment_artifact,
  read_epoch_artifact,
  read_experiment_artifact_lines,
  write_epoch_artifact,
)
from autopilot.core.stage_models import ChangeProposal, ProposalVerdict
from pathlib import Path


def record_proposal(experiment_dir: Path, proposal: ChangeProposal) -> None:
  """Append to hypothesis_log.jsonl."""
  append_experiment_artifact(experiment_dir, 'hypothesis_log.jsonl', proposal.to_dict())


def read_proposals(experiment_dir: Path) -> list[ChangeProposal]:
  """Read all proposals from hypothesis_log.jsonl."""
  lines = read_experiment_artifact_lines(experiment_dir, 'hypothesis_log.jsonl')
  return [ChangeProposal.from_dict(line) for line in lines]


def record_verdict(experiment_dir: Path, epoch: int, verdict: ProposalVerdict) -> None:
  """Write proposal_verdict.json for an epoch."""
  write_epoch_artifact(experiment_dir, epoch, 'proposal_verdict.json', verdict.to_dict())


def read_verdict(experiment_dir: Path, epoch: int) -> ProposalVerdict | None:
  """Read proposal_verdict.json for an epoch."""
  data = read_epoch_artifact(experiment_dir, epoch, 'proposal_verdict.json')
  if data is None:
    return None
  return ProposalVerdict.from_dict(data)
