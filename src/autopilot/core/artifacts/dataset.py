"""Dataset-related artifact classes."""

from autopilot.core.artifacts.artifact import JSONArtifact, JSONLArtifact


class SplitSummaryArtifact(JSONArtifact):
  """Per-split evaluation summary ({split}_summary.json)."""

  def __init__(self, split: str) -> None:
    super().__init__(f'{split}_summary.json')


class ProposalLogArtifact(JSONLArtifact):
  """hypothesis_log.jsonl -- append-only proposal log at experiment scope."""

  def __init__(self) -> None:
    super().__init__('hypothesis_log.jsonl')
