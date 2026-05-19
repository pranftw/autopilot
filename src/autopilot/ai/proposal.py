"""Proposal I/O helper functions.

Pure functions -- persistence via Artifact instances, not raw I/O.
Data models: ChangeProposal, ProposalVerdict, JudgeValidation.
"""

from autopilot.core.artifacts.dataset import ProposalLogArtifact
from autopilot.core.artifacts.epoch import VerdictArtifact
from autopilot.core.serialization import DictMixin
from autopilot.tracking.io import read_json_dict, utc_now_iso
from dataclasses import dataclass, field, fields
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
  timestamp: str = field(default_factory=utc_now_iso)
  verification: ProposalVerdict | None = None

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'ChangeProposal':
    """Deserialize nested verification and filtered proposal fields.

    Args:
      data: Raw mapping optionally containing a ``verification`` dict.

    Returns:
      :class:`ChangeProposal` with parsed :class:`ProposalVerdict` when present.
    """
    data = dict(data)
    verification = data.get('verification')
    if verification is not None and isinstance(verification, dict):
      data['verification'] = ProposalVerdict.from_dict(verification)
    else:
      data['verification'] = None
    names = {f.name for f in fields(cls)}
    return cls(**{k: val for k, val in data.items() if k in names})


@dataclass
class JudgeValidation(DictMixin):
  """Judge cross-validation result."""

  judge_id: str | None = None
  agreement_rate: float = 0.0
  disagreements: list[dict] = field(default_factory=list)
  anomalies: list[str] = field(default_factory=list)
  confidence: str | None = None


def record_proposal(path: Path, proposal: ChangeProposal) -> None:
  """Append to hypothesis_log.jsonl."""
  ProposalLogArtifact().append(proposal.to_dict(), path)


def read_proposals(path: Path) -> list[ChangeProposal]:
  """Read all proposals from hypothesis_log.jsonl.

  Returns:
    List of :class:`ChangeProposal` instances parsed from JSON lines.
  """
  lines = ProposalLogArtifact().read_raw(path)
  return [ChangeProposal.from_dict(line) for line in lines]


def record_verdict(path: Path, epoch: int, verdict: ProposalVerdict) -> None:
  """Write proposal_verdict.json for an epoch."""
  VerdictArtifact().write(verdict.to_dict(), path, epoch=epoch)


def read_verdict(path: Path, epoch: int) -> ProposalVerdict | None:
  """Read proposal_verdict.json for an epoch.

  Delegates to ``read_json_dict`` which raises ``TrackingError`` when the
  artifact is present but not a JSON object.

  Returns:
    Parsed verdict, or ``None`` when the artifact is missing.
  """
  resolved = VerdictArtifact().resolve_path(path, epoch=epoch)
  if not resolved.is_file():
    return None
  data = read_json_dict(resolved, 'proposal_verdict')
  return ProposalVerdict.from_dict(data)
