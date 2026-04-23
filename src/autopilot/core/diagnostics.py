"""Diagnostics base class: separates analysis computation from persistence.

Extension points (override to customize):
  categorize(item)          -- assign a failure category string
  resolve_node(item)        -- extract node identity from an item
  is_failure(item)          -- determine if an item is a failure
  score_node(node, items)   -- compute per-node health score
  select_samples(items, limit) -- choose representative error messages
  analyze(data, epoch)      -- full pipeline, returns DiagnosticResult

I/O (uses owned artifacts, no raw writes):
  write(result)             -- persist a DiagnosticResult
  read_diagnoses(epoch)     -- read back diagnosis entries
  read_heatmap(epoch)       -- read back node heatmap
"""

from autopilot.core.artifacts.epoch import DiagnosesArtifact, HeatmapArtifact
from autopilot.core.artifacts.owner import ArtifactOwner
from autopilot.core.serialization import DictMixin
from pathlib import Path
import dataclasses


@dataclasses.dataclass
class DiagnosisEntry(DictMixin):
  """Single error category with count and sample errors."""

  category: str
  count: int
  sample_errors: list[str]


@dataclasses.dataclass
class NodeScore(DictMixin):
  """Per-node scoring: total attempts, failures, and error rate."""

  total: int
  failed: int
  error_rate: float


@dataclasses.dataclass
class DiagnosticResult:
  """Pure data output of analyze(). No I/O."""

  epoch: int
  diagnoses: list[DiagnosisEntry]
  heatmap: dict[str, NodeScore]


class Diagnostics(ArtifactOwner):
  """Base class for epoch diagnostic analysis.

  Composes ArtifactOwner for typed file I/O. Override analysis hooks
  to customize categorization, failure detection, node resolution, and scoring.
  """

  def __init__(self, experiment_dir: Path) -> None:
    self.__init_artifacts__()
    self._dir = experiment_dir
    self.diagnoses_artifact = DiagnosesArtifact()
    self.heatmap_artifact = HeatmapArtifact()

  def categorize(self, item: dict) -> str:
    """Assign a failure category. Override for domain-specific buckets."""
    return item.get('metadata', {}).get('category', 'uncategorized')

  def resolve_node(self, item: dict) -> str:
    """Extract node identity from an item. Override for custom node resolution."""
    return item.get('id') or item.get('metadata', {}).get('node', 'unknown')

  def is_failure(self, item: dict) -> bool:
    """Determine if an item is a failure. Override for custom failure criteria."""
    return not item.get('success', True) or bool(item.get('error_message'))

  def score_node(self, node: str, items: list[dict]) -> NodeScore:
    """Compute per-node health. Override for custom scoring (e.g. weighted)."""
    total = len(items)
    failed = sum(1 for i in items if self.is_failure(i))
    return NodeScore(
      total=total,
      failed=failed,
      error_rate=round(failed / total, 4) if total > 0 else 0.0,
    )

  def select_samples(self, items: list[dict], limit: int = 5) -> list[str]:
    """Choose representative error messages. Override for custom sampling."""
    samples: list[str] = []
    for item in items:
      if self.is_failure(item):
        msg = item.get('error_message') or 'failed (no message)'
        samples.append(msg)
        if len(samples) >= limit:
          break
    return samples

  def analyze(self, data: list[dict], epoch: int) -> DiagnosticResult:
    """Run full analysis. Returns pure data -- does NOT write anything."""
    by_category: dict[str, list[dict]] = {}
    by_node: dict[str, list[dict]] = {}

    for item in data:
      node = self.resolve_node(item)
      by_node.setdefault(node, []).append(item)
      if self.is_failure(item):
        cat = self.categorize(item)
        by_category.setdefault(cat, []).append(item)

    diagnoses = [
      DiagnosisEntry(
        category=cat,
        count=len(items),
        sample_errors=self.select_samples(items),
      )
      for cat, items in by_category.items()
    ]
    heatmap = {node: self.score_node(node, items) for node, items in by_node.items()}
    return DiagnosticResult(epoch=epoch, diagnoses=diagnoses, heatmap=heatmap)

  def write(self, result: DiagnosticResult) -> None:
    """Persist a DiagnosticResult via owned artifacts."""
    for entry in result.diagnoses:
      self.diagnoses_artifact.append(entry.to_dict(), self._dir, epoch=result.epoch)
    heatmap_dict = {node: score.to_dict() for node, score in result.heatmap.items()}
    self.heatmap_artifact.write(heatmap_dict, self._dir, epoch=result.epoch)

  def read_diagnoses(self, epoch: int) -> list[DiagnosisEntry]:
    """Read back diagnosis entries for an epoch."""
    raw = self.diagnoses_artifact.read(self._dir, epoch=epoch)
    return [DiagnosisEntry.from_dict(r) for r in raw]

  def read_heatmap(self, epoch: int) -> dict[str, NodeScore]:
    """Read back node heatmap for an epoch."""
    raw = self.heatmap_artifact.read(self._dir, epoch=epoch)
    if raw is None:
      return {}
    return {node: NodeScore.from_dict(score) for node, score in raw.items()}
