"""Epoch-scoped artifact classes."""

from autopilot.core.artifacts.artifact import JSONArtifact, JSONLArtifact


class MetricComparisonArtifact(JSONArtifact):
  """metric_comparison.json -- per-epoch comparison of candidate vs baseline."""

  def __init__(self) -> None:
    super().__init__('metric_comparison.json', scope='epoch')


class DataArtifact(JSONLArtifact):
  """data.jsonl -- per-epoch batch data."""

  def __init__(self) -> None:
    super().__init__('data.jsonl', scope='epoch')


class DiagnosesArtifact(JSONLArtifact):
  """trace_diagnoses.jsonl -- per-epoch failure diagnosis entries."""

  def __init__(self) -> None:
    super().__init__('trace_diagnoses.jsonl', scope='epoch')


class HeatmapArtifact(JSONArtifact):
  """node_heatmap.json -- per-epoch node error heatmap."""

  def __init__(self) -> None:
    super().__init__('node_heatmap.json', scope='epoch')


class VerdictArtifact(JSONArtifact):
  """proposal_verdict.json -- per-epoch proposal verdict."""

  def __init__(self) -> None:
    super().__init__('proposal_verdict.json', scope='epoch')
