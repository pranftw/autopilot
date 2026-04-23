"""Memory base class and FileMemory implementation.

Persistent learning memory across optimization epochs.
Like Logger and Checkpoint, Memory is a framework extension point.

Data models: MemoryRecord, TrendResult, MemoryContext, BlockedStrategy.
"""

from autopilot.core.errors import TrackingError
from autopilot.core.serialization import DictMixin
from autopilot.tracking.io import append_jsonl, atomic_write_json, read_json, read_jsonl
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import logging


@dataclass
class MemoryRecord(DictMixin):
  """Unit of storage for Memory. All fields typed and queryable."""

  epoch: int = 0
  outcome: str | None = None
  content: str | None = None
  category: str | None = None
  node: str | None = None
  strategy: str | None = None
  metrics: dict[str, float] = field(default_factory=dict)
  timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendResult(DictMixin):
  """Return type of Memory.trends(). Metric trajectory analysis."""

  metric: str | None = None
  direction: str | None = None
  rate: float = 0.0
  projection: float = 0.0
  values: list[float] = field(default_factory=list)
  epochs: list[int] = field(default_factory=list)


@dataclass
class MemoryContext(DictMixin):
  """Return type of Memory.context(). Full decision context for the agent."""

  epoch: int = 0
  top_failures: list[MemoryRecord] = field(default_factory=list)
  strategies_tried: list[str] = field(default_factory=list)
  blocked: list[str] = field(default_factory=list)
  untried: list[str] = field(default_factory=list)
  trends: dict[str, TrendResult] = field(default_factory=dict)
  total_records: int = 0

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'MemoryContext':
    data = dict(data)
    data['top_failures'] = [MemoryRecord.from_dict(r) for r in data.get('top_failures', [])]
    data['trends'] = {k: TrendResult.from_dict(v) for k, v in data.get('trends', {}).items()}
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class BlockedStrategy(DictMixin):
  """Entry in strategy_blocklist.json."""

  strategy: str | None = None
  reason: str | None = None
  epoch_blocked: int = 0
  timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


logger = logging.getLogger(__name__)


class Memory:
  """Persistent learning memory across optimization epochs.

  Abstract base. Override all methods for custom storage backends.
  All data is structured -- MemoryRecord is the unit of storage.

  Core operations:
    learn(epoch, outcome, *, content, category, node, strategy, metrics, metadata)
    recall(**filters)           -- query with filters: category, node, outcome,
                                   strategy, epoch, epoch_min, epoch_max
    trends(metric, window=5)    -- compute metric trajectory (direction, rate, projection)
    context(epoch)              -- full decision context for the agent
    block_strategy(strategy)    -- mark a strategy as blocked
    is_strategy_blocked(strategy) -- check if blocked
    blocked_strategies()        -- list all blocked strategy names

  Trend direction detection:
    >60% positive diffs = 'improving', >60% negative = 'declining',
    max diff < 0.01 = 'plateau', else 'oscillating'.

  Integration: MemoryCallback (core/callbacks/memory.py) wires Memory into
  the training loop. AgentOptimizer.update_context() accepts memory context.
  The blocklist flow: MemoryCallback.on_before_optimizer_step syncs blocked
  strategies from Memory to the Optimizer.

  Built-in subclass: FileMemory (below) using knowledge_base.jsonl +
  strategy_blocklist.json.
  """

  def learn(
    self,
    epoch: int,
    outcome: str,
    *,
    content: str | None = None,
    category: str | None = None,
    node: str | None = None,
    strategy: str | None = None,
    metrics: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
  ) -> None:
    pass

  def recall(self, **filters: Any) -> list[MemoryRecord]:
    return []

  def trends(self, metric: str | None = None, window: int = 5) -> TrendResult:
    return TrendResult()

  def context(self, epoch: int) -> MemoryContext:
    return MemoryContext(epoch=epoch)

  def block_strategy(self, strategy: str, *, reason: str | None = None, epoch: int = 0) -> None:
    pass

  def is_strategy_blocked(self, strategy: str) -> bool:
    return False

  def blocked_strategies(self) -> list[str]:
    return []

  def state_dict(self) -> dict:
    return {}

  def load_state_dict(self, state_dict: dict) -> None:
    pass


class FileMemory(Memory):
  """File-backed Memory using knowledge_base.jsonl + strategy_blocklist.json.

  File formats:
    knowledge_base.jsonl      -- one MemoryRecord.to_dict() per line, append-only
    strategy_blocklist.json   -- JSON array of BlockedStrategy.to_dict(), atomic-written

  Gotchas:
    - _load_records() reads the full JSONL on every recall/trends/context call.
      No in-memory cache. For long experiments this could be slow.
    - blocked_strategies() returns list[str] (names only), not list[BlockedStrategy].
    - learn() requires outcome as a positional argument; all other fields are keyword-only.
    - The strategy field on MemoryRecord and trainer.fit_context['strategy'] are the
      linkage point for the blocklist system. If ctx['strategy'] is not set, the
      blocklist has no effect.
  """

  def __init__(self, experiment_dir: Path) -> None:
    self._dir = Path(experiment_dir)
    self._kb_path = self._dir / 'knowledge_base.jsonl'
    self._blocklist_path = self._dir / 'strategy_blocklist.json'

  def learn(
    self,
    epoch: int,
    outcome: str,
    *,
    content: str | None = None,
    category: str | None = None,
    node: str | None = None,
    strategy: str | None = None,
    metrics: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
  ) -> None:
    record = MemoryRecord(
      epoch=epoch,
      outcome=outcome,
      content=content,
      category=category,
      node=node,
      strategy=strategy,
      metrics=dict(metrics) if metrics else {},
      metadata=dict(metadata) if metadata else {},
    )
    append_jsonl(self._kb_path, record.to_dict())

  def recall(self, **filters: Any) -> list[MemoryRecord]:
    records = self._load_records()
    return self._apply_filters(records, filters)

  def trends(self, metric: str | None = None, window: int = 5) -> TrendResult:
    records = self._load_records()
    if not records or not metric:
      return TrendResult(metric=metric or '')

    values: list[float] = []
    epochs: list[int] = []
    for r in records:
      if metric in r.metrics:
        values.append(r.metrics[metric])
        epochs.append(r.epoch)

    if not values:
      return TrendResult(metric=metric)

    windowed_values = values[-window:]
    windowed_epochs = epochs[-window:]

    direction = self._detect_direction(windowed_values)
    rate = self._compute_rate(windowed_values)
    projection = windowed_values[-1] + rate if windowed_values else 0.0

    return TrendResult(
      metric=metric,
      direction=direction,
      rate=rate,
      projection=projection,
      values=windowed_values,
      epochs=windowed_epochs,
    )

  def context(self, epoch: int) -> MemoryContext:
    records = self._load_records()
    failures = [r for r in records if r.outcome == 'failed']
    top_failures = failures[-5:]
    strategies_tried = list({r.strategy for r in records if r.strategy})
    blocked = self.blocked_strategies()

    trends: dict[str, TrendResult] = {}
    all_metrics = set()
    for r in records:
      all_metrics.update(r.metrics.keys())
    for m in all_metrics:
      trends[m] = self.trends(metric=m)

    return MemoryContext(
      epoch=epoch,
      top_failures=top_failures,
      strategies_tried=strategies_tried,
      blocked=blocked,
      untried=[],
      trends=trends,
      total_records=len(records),
    )

  def block_strategy(self, strategy: str, *, reason: str | None = None, epoch: int = 0) -> None:
    blocklist = self._load_blocklist()
    existing = {b.strategy for b in blocklist}
    if strategy in existing:
      return
    entry = BlockedStrategy(strategy=strategy, reason=reason, epoch_blocked=epoch)
    blocklist.append(entry)
    self._save_blocklist(blocklist)

  def is_strategy_blocked(self, strategy: str) -> bool:
    blocklist = self._load_blocklist()
    return any(b.strategy == strategy for b in blocklist)

  def blocked_strategies(self) -> list[str]:
    blocklist = self._load_blocklist()
    return [b.strategy for b in blocklist]

  def state_dict(self) -> dict:
    records = self._load_records()
    blocklist = self._load_blocklist()
    return {
      'records': [r.to_dict() for r in records],
      'blocklist': [b.to_dict() for b in blocklist],
    }

  def load_state_dict(self, state_dict: dict) -> None:
    records = state_dict.get('records', [])
    blocklist = state_dict.get('blocklist', [])
    self._dir.mkdir(parents=True, exist_ok=True)
    if self._kb_path.exists():
      self._kb_path.unlink()
    for r in records:
      append_jsonl(self._kb_path, r)
    self._save_blocklist([BlockedStrategy.from_dict(b) for b in blocklist])

  def _load_records(self) -> list[MemoryRecord]:
    raw = read_jsonl(self._kb_path, strict=False)
    return [MemoryRecord.from_dict(r) for r in raw]

  def _load_blocklist(self) -> list[BlockedStrategy]:
    try:
      data = read_json(self._blocklist_path)
    except TrackingError:
      logger.warning('corrupt blocklist at %s, treating as empty', self._blocklist_path)
      return []
    if data is None:
      return []
    if not isinstance(data, list):
      logger.warning('invalid blocklist format at %s', self._blocklist_path)
      return []
    result: list[BlockedStrategy] = []
    for item in data:
      if isinstance(item, dict):
        result.append(BlockedStrategy.from_dict(item))
    return result

  def _save_blocklist(self, blocklist: list[BlockedStrategy]) -> None:
    payload = [b.to_dict() for b in blocklist]
    atomic_write_json(self._blocklist_path, payload)

  def _apply_filters(
    self,
    records: list[MemoryRecord],
    filters: dict[str, Any],
  ) -> list[MemoryRecord]:
    result = records
    if 'category' in filters and filters['category']:
      result = [r for r in result if r.category == filters['category']]
    if 'node' in filters and filters['node']:
      result = [r for r in result if r.node == filters['node']]
    if 'outcome' in filters and filters['outcome']:
      result = [r for r in result if r.outcome == filters['outcome']]
    if 'strategy' in filters and filters['strategy']:
      result = [r for r in result if r.strategy == filters['strategy']]
    if 'epoch' in filters and filters['epoch']:
      result = [r for r in result if r.epoch == filters['epoch']]
    if 'epoch_min' in filters and filters['epoch_min']:
      result = [r for r in result if r.epoch >= filters['epoch_min']]
    if 'epoch_max' in filters and filters['epoch_max']:
      result = [r for r in result if r.epoch <= filters['epoch_max']]
    return result

  def _detect_direction(self, values: list[float]) -> str:
    if len(values) < 2:
      return 'plateau'
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    positive = sum(1 for d in diffs if d > 0.01)
    negative = sum(1 for d in diffs if d < -0.01)
    total = len(diffs)
    if positive > total * 0.6:
      return 'improving'
    if negative > total * 0.6:
      return 'declining'
    max_diff = max(abs(d) for d in diffs) if diffs else 0
    if max_diff < 0.01:
      return 'plateau'
    return 'oscillating'

  def _compute_rate(self, values: list[float]) -> float:
    if len(values) < 2:
      return 0.0
    return (values[-1] - values[0]) / (len(values) - 1)
