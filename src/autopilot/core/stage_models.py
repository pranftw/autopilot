"""Stage-related data models for optimization callbacks and orchestration.

Every field is a typed, queryable attribute. No free-form strings as
primary carriers of structured data. The optional `content` field on
MemoryRecord is a human-readable note, never the primary data carrier.
"""

from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any


@dataclass
class MemoryRecord:
  """Unit of storage for Memory. All fields typed and queryable."""

  epoch: int = 0
  outcome: str = ''
  content: str = ''
  category: str = ''
  node: str = ''
  strategy: str = ''
  metrics: dict[str, float] = field(default_factory=dict)
  timestamp: str = ''
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.timestamp:
      self.timestamp = datetime.now(timezone.utc).isoformat()

  def to_dict(self) -> dict[str, Any]:
    return {
      'epoch': self.epoch,
      'outcome': self.outcome,
      'content': self.content,
      'category': self.category,
      'node': self.node,
      'strategy': self.strategy,
      'metrics': self.metrics,
      'timestamp': self.timestamp,
      'metadata': self.metadata,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'MemoryRecord':
    data = dict(data)
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class TrendResult:
  """Return type of Memory.trends(). Metric trajectory analysis."""

  metric: str = ''
  direction: str = ''
  rate: float = 0.0
  projection: float = 0.0
  values: list[float] = field(default_factory=list)
  epochs: list[int] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      'metric': self.metric,
      'direction': self.direction,
      'rate': self.rate,
      'projection': self.projection,
      'values': self.values,
      'epochs': self.epochs,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'TrendResult':
    data = dict(data)
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class MemoryContext:
  """Return type of Memory.context(). Full decision context for the agent."""

  epoch: int = 0
  top_failures: list[MemoryRecord] = field(default_factory=list)
  strategies_tried: list[str] = field(default_factory=list)
  blocked: list[str] = field(default_factory=list)
  untried: list[str] = field(default_factory=list)
  trends: dict[str, TrendResult] = field(default_factory=dict)
  total_records: int = 0

  def to_dict(self) -> dict[str, Any]:
    return {
      'epoch': self.epoch,
      'top_failures': [r.to_dict() for r in self.top_failures],
      'strategies_tried': self.strategies_tried,
      'blocked': self.blocked,
      'untried': self.untried,
      'trends': {k: v.to_dict() for k, v in self.trends.items()},
      'total_records': self.total_records,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'MemoryContext':
    data = dict(data)
    data['top_failures'] = [MemoryRecord.from_dict(r) for r in data.get('top_failures', [])]
    data['trends'] = {k: TrendResult.from_dict(v) for k, v in data.get('trends', {}).items()}
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class BlockedStrategy:
  """Entry in strategy_blocklist.json."""

  strategy: str = ''
  reason: str = ''
  epoch_blocked: int = 0
  timestamp: str = ''

  def __post_init__(self) -> None:
    if not self.timestamp:
      self.timestamp = datetime.now(timezone.utc).isoformat()

  def to_dict(self) -> dict[str, Any]:
    return {
      'strategy': self.strategy,
      'reason': self.reason,
      'epoch_blocked': self.epoch_blocked,
      'timestamp': self.timestamp,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'BlockedStrategy':
    data = dict(data)
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class EpochMetrics:
  """Structured forward pass result per epoch."""

  epoch: int = 0
  split: str = ''
  total: int = 0
  passed: int = 0
  failed: int = 0
  accuracy: float = 0.0
  error_rate: float = 0.0
  latency_p95_ms: float = 0.0
  delta: dict[str, float] = field(default_factory=dict)
  gates: dict[str, str] = field(default_factory=dict)
  data_path: str = ''

  def to_dict(self) -> dict[str, Any]:
    return {
      'epoch': self.epoch,
      'split': self.split,
      'total': self.total,
      'passed': self.passed,
      'failed': self.failed,
      'accuracy': self.accuracy,
      'error_rate': self.error_rate,
      'latency_p95_ms': self.latency_p95_ms,
      'delta': self.delta,
      'gates': self.gates,
      'data_path': self.data_path,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'EpochMetrics':
    data = dict(data)
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class ProposalVerdict:
  """Verification result for a ChangeProposal."""

  proposal_id: str = ''
  items_tested: int = 0
  items_fixed: int = 0
  items_regressed: int = 0
  items_unchanged: int = 0
  verdict: str = ''

  def to_dict(self) -> dict[str, Any]:
    return {
      'proposal_id': self.proposal_id,
      'items_tested': self.items_tested,
      'items_fixed': self.items_fixed,
      'items_regressed': self.items_regressed,
      'items_unchanged': self.items_unchanged,
      'verdict': self.verdict,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'ProposalVerdict':
    data = dict(data)
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class ChangeProposal:
  """Optimization proposal model."""

  proposal_id: str = ''
  hypothesis: str = ''
  target_node: str = ''
  change_type: str = ''
  expected_impact: str = ''
  files_to_modify: list[str] = field(default_factory=list)
  epoch: int = 0
  status: str = ''
  pre_snapshot_epoch: int = 0
  timestamp: str = ''
  verification: ProposalVerdict | None = None

  def __post_init__(self) -> None:
    if not self.timestamp:
      self.timestamp = datetime.now(timezone.utc).isoformat()

  def to_dict(self) -> dict[str, Any]:
    result: dict[str, Any] = {
      'proposal_id': self.proposal_id,
      'hypothesis': self.hypothesis,
      'target_node': self.target_node,
      'change_type': self.change_type,
      'expected_impact': self.expected_impact,
      'files_to_modify': self.files_to_modify,
      'epoch': self.epoch,
      'status': self.status,
      'pre_snapshot_epoch': self.pre_snapshot_epoch,
      'timestamp': self.timestamp,
    }
    if self.verification is not None:
      result['verification'] = self.verification.to_dict()
    else:
      result['verification'] = None
    return result

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
class JudgeValidation:
  """Judge cross-validation result."""

  judge_id: str = ''
  agreement_rate: float = 0.0
  disagreements: list[dict] = field(default_factory=list)
  anomalies: list[str] = field(default_factory=list)
  confidence: str = ''

  def to_dict(self) -> dict[str, Any]:
    return {
      'judge_id': self.judge_id,
      'agreement_rate': self.agreement_rate,
      'disagreements': self.disagreements,
      'anomalies': self.anomalies,
      'confidence': self.confidence,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'JudgeValidation':
    data = dict(data)
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class RegressionAnalysis:
  """Result of compare_metrics() regression analysis."""

  epoch: int = 0
  overall_verdict: str = ''
  per_category_deltas: dict[str, float] = field(default_factory=dict)
  regressions: list[dict] = field(default_factory=list)
  improvements: list[dict] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      'epoch': self.epoch,
      'overall_verdict': self.overall_verdict,
      'per_category_deltas': self.per_category_deltas,
      'regressions': self.regressions,
      'improvements': self.improvements,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'RegressionAnalysis':
    data = dict(data)
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class CostEntry:
  """Per-epoch cost tracking."""

  epoch: int = 0
  wall_clock_s: float = 0.0
  api_calls: int = 0
  tokens_used: int = 0
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
      'epoch': self.epoch,
      'wall_clock_s': self.wall_clock_s,
      'api_calls': self.api_calls,
      'tokens_used': self.tokens_used,
      'metadata': self.metadata,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'CostEntry':
    data = dict(data)
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class ExperimentSummaryData:
  """Final experiment report."""

  slug: str = ''
  total_epochs: int = 0
  final_metrics: dict[str, float] = field(default_factory=dict)
  best_epoch: int = 0
  stop_reason: str | None = None
  last_good_epoch: int = 0
  promotions: list[dict] = field(default_factory=list)
  regressions: list[RegressionAnalysis] = field(default_factory=list)
  cost_total: CostEntry | None = None
  memory_entries: int = 0

  def to_dict(self) -> dict[str, Any]:
    result: dict[str, Any] = {
      'slug': self.slug,
      'total_epochs': self.total_epochs,
      'final_metrics': self.final_metrics,
      'best_epoch': self.best_epoch,
      'stop_reason': self.stop_reason,
      'last_good_epoch': self.last_good_epoch,
      'promotions': self.promotions,
      'regressions': [r.to_dict() for r in self.regressions],
      'cost_total': self.cost_total.to_dict() if self.cost_total else None,
      'memory_entries': self.memory_entries,
    }
    return result

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'ExperimentSummaryData':
    data = dict(data)
    data['regressions'] = [RegressionAnalysis.from_dict(r) for r in data.get('regressions', [])]
    ct = data.get('cost_total')
    if ct is not None and isinstance(ct, dict):
      data['cost_total'] = CostEntry.from_dict(ct)
    else:
      data['cost_total'] = None
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})
