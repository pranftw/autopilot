"""Dataset-related artifact classes."""

from autopilot.core.artifacts.artifact import JSONArtifact, JSONLArtifact


class SplitSummaryArtifact(JSONArtifact):
  """Per-split evaluation summary ({split}_summary.json)."""

  def __init__(self, split: str) -> None:
    """Create artifact named ``{split}_summary.json`` at default scope.

    Args:
      split: Split label embedded in the filename.
    """
    super().__init__(f'{split}_summary.json')


class ProposalLogArtifact(JSONLArtifact):
  """hypothesis_log.jsonl -- append-only proposal log at experiment scope."""

  def __init__(self) -> None:
    """Create the experiment-scoped hypothesis proposal log artifact."""
    super().__init__('hypothesis_log.jsonl')
