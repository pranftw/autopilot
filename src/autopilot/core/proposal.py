"""Proposal I/O helper functions.

Pure functions -- persistence via Artifact instances, not raw I/O.
Data models: ChangeProposal, ProposalVerdict, JudgeValidation.
"""

from autopilot.core.artifacts.dataset import ProposalLogArtifact
from autopilot.core.artifacts.epoch import VerdictArtifact
from autopilot.core.serialization import DictMixin
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ProposalVerdict(DictMixin):
  """Verification result for a ChangeProposal."""

  proposal_id: str | None = None
  items_tested: int = 0
  items_fixed: int = 0
  items_regressed: int = 0
  items_unchanged: int = 0
  verdict: str | None = None


@dataclass
class ChangeProposal(DictMixin):
  """Optimization proposal model."""

  proposal_id: str | None = None
  hypothesis: str | None = None
  target_node: str | None = None
  change_type: str | None = None
  expected_impact: str | None = None
  files_to_modify: list[str] = field(default_factory=list)
  epoch: int = 0
  status: str | None = None
  pre_snapshot_epoch: int = 0
  timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
  verification: ProposalVerdict | None = None

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'ChangeProposal':
    data = dict(data)
    v = data.get('verification')
    if v is not None and isinstance(v, dict):
      data['verification'] = ProposalVerdict.from_dict(v)
    else:
      data['verification'] = None
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class JudgeValidation(DictMixin):
  """Judge cross-validation result."""

  judge_id: str | None = None
  agreement_rate: float = 0.0
  disagreements: list[dict] = field(default_factory=list)
  anomalies: list[str] = field(default_factory=list)
  confidence: str | None = None


def record_proposal(experiment_dir: Path, proposal: ChangeProposal) -> None:
  """Append to hypothesis_log.jsonl."""
  ProposalLogArtifact().append(proposal.to_dict(), experiment_dir)


def read_proposals(experiment_dir: Path) -> list[ChangeProposal]:
  """Read all proposals from hypothesis_log.jsonl."""
  lines = ProposalLogArtifact().read_raw(experiment_dir)
  return [ChangeProposal.from_dict(line) for line in lines]


def record_verdict(experiment_dir: Path, epoch: int, verdict: ProposalVerdict) -> None:
  """Write proposal_verdict.json for an epoch."""
  VerdictArtifact().write(verdict.to_dict(), experiment_dir, epoch=epoch)


def read_verdict(experiment_dir: Path, epoch: int) -> ProposalVerdict | None:
  """Read proposal_verdict.json for an epoch."""
  data = VerdictArtifact().read_raw(experiment_dir, epoch=epoch)
  if data is None:
    return None
  return ProposalVerdict.from_dict(data)
