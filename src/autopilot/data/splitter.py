"""Incremental train/val/test splitter for ID-stable corpus growth.

Assigns string item IDs to named splits (e.g. ``train``, ``val``, ``test``)
based on configurable ratios. Existing assignments are never reshuffled when
new items arrive -- only new IDs are placed into splits.

Ratio validation: ``abs(sum(ratios.values()) - 1.0) <= 1e-6``.

RNG: uses ``random.Random(seed)`` for deterministic, reproducible placement.
``extend`` consumes RNG state deterministically from ``(seed, sorted new ids)``
ordering.

Placement rationale: splitting by string IDs is orthogonal to index-based
``Sampler`` (``data/sampler.py``). It operates at the corpus level, not the
batch level.
"""

from autopilot.core.serialization import DictMixin
from dataclasses import dataclass, field
import random

_RATIO_TOLERANCE = 1e-6
VALIDATION_ERROR_SAMPLE_SIZE = 5
OVERLAP_DISPLAY_LIMIT = 3


def _validate_ratios(ratios: dict[str, float]) -> None:
  """Raise ``ValueError`` when ratios do not sum to 1.0 within tolerance.

  Args:
    ratios: Split name to ratio mapping.

  Raises:
    ValueError: When ratios are empty or do not sum to 1.0 within 1e-6.
  """
  if not ratios:
    msg = 'ratios must be non-empty'
    raise ValueError(msg)
  total = sum(ratios.values())
  if abs(total - 1.0) > _RATIO_TOLERANCE:
    msg = f'ratios must sum to 1.0 (tolerance {_RATIO_TOLERANCE}), got {total:.10f} from {ratios}'
    raise ValueError(msg)


@dataclass
class SplitAssignment(DictMixin):
  """Mapping from item ID to split name.

  ``assignments`` maps each item ID (string) to its split name (e.g.
  ``'train'``, ``'val'``). ``ratios`` and ``seed`` are preserved across
  ``extend`` calls for reproducibility.

  Validation: assignment values must be keys of ``ratios``. Construction
  via ``from_dict`` does not validate; use ``validate()`` explicitly.
  """

  assignments: dict[str, str] = field(default_factory=dict)
  ratios: dict[str, float] = field(default_factory=dict)
  seed: int = 42

  def validate(self) -> None:
    """Check that all assignment values are valid split names.

    Raises:
      ValueError: When any assignment value is not a key in ``ratios``.
    """
    valid_splits = set(self.ratios)
    for item_id, split_name in self.assignments.items():
      if split_name not in valid_splits:
        msg = (
          f'item {item_id!r} assigned to split {split_name!r} '
          f'which is not in ratios keys {sorted(valid_splits)}'
        )
        raise ValueError(msg)

  def validate_universe(self, universe: set[str]) -> None:
    """Check that all assigned IDs exist in the current universe.

    Detects "removed items" -- IDs present in ``assignments`` but absent
    from the current corpus. Raises ``ValueError`` when any assigned ID
    is not in ``universe``.

    Args:
      universe: Complete set of valid item IDs in the current corpus.

    Raises:
      ValueError: When any assigned item ID is not in ``universe``.
    """
    assigned = set(self.assignments)
    removed = assigned - universe
    if removed:
      sample = sorted(removed)[:VALIDATION_ERROR_SAMPLE_SIZE]
      msg = (
        f'assignment references {len(removed)} item(s) not in current universe: '
        f'{sample}; remove stale IDs or rebuild assignments'
      )
      raise ValueError(msg)


class IncrementalSplitter:
  """Grows splits without reshuffling existing assignments.

  Uses ``random.Random(seed)`` for deterministic placement. Ratios must
  sum to 1.0 within tolerance 1e-6.

  Typical usage::

      splitter = IncrementalSplitter({'train': 0.8, 'val': 0.1, 'test': 0.1})
      assignment = splitter.fit(['a', 'b', 'c', 'd', 'e'])
      assignment = splitter.extend(assignment, ['f', 'g'])
  """

  def __init__(
    self,
    ratios: dict[str, float],
    seed: int = 42,
  ) -> None:
    """Create a splitter with fixed ratios and seed.

    Args:
      ratios: Split name to target ratio mapping. Must sum to 1.0.
      seed: RNG seed for deterministic placement.
    """
    _validate_ratios(ratios)
    self._ratios = dict(ratios)
    self._seed = seed

  def fit(self, item_ids: list[str]) -> SplitAssignment:
    """Assign all items to splits based on ratios.

    Empty ``item_ids`` returns ``SplitAssignment(assignments={}, ...)``
    with the splitter's ``ratios`` and ``seed``.

    Args:
      item_ids: Unique item identifiers to assign.

    Returns:
      ``SplitAssignment`` with all items placed.

    Raises:
      ValueError: When ``item_ids`` contains duplicates.
    """
    if len(item_ids) != len(set(item_ids)):
      msg = 'item_ids contains duplicates'
      raise ValueError(msg)
    if not item_ids:
      return SplitAssignment(
        assignments={},
        ratios=dict(self._ratios),
        seed=self._seed,
      )
    return self._assign(item_ids)

  def extend(
    self,
    existing: SplitAssignment,
    new_item_ids: list[str],
  ) -> SplitAssignment:
    """Add new items without changing existing assignments.

    Args:
      existing: Prior assignment to preserve.
      new_item_ids: Fresh item IDs to place. Must not overlap with existing
        or contain duplicates.

    Returns:
      New ``SplitAssignment`` combining existing and new placements.

    Raises:
      ValueError: When ``new_item_ids`` contains duplicates, overlaps with
        existing assignments, or existing assignments reference invalid splits.
    """
    existing.validate()
    if not new_item_ids:
      return SplitAssignment(
        assignments=dict(existing.assignments),
        ratios=dict(self._ratios),
        seed=self._seed,
      )
    new_set = set(new_item_ids)
    if len(new_item_ids) != len(new_set):
      msg = 'new_item_ids contains duplicates'
      raise ValueError(msg)
    overlap = new_set & set(existing.assignments)
    if overlap:
      sample = sorted(overlap)[:OVERLAP_DISPLAY_LIMIT]
      msg = f'new_item_ids overlap with existing assignments: {sample}'
      raise ValueError(msg)
    new_assignment = self._assign(new_item_ids)
    merged = dict(existing.assignments)
    merged.update(new_assignment.assignments)
    return SplitAssignment(
      assignments=merged,
      ratios=dict(self._ratios),
      seed=self._seed,
    )

  def _assign(self, item_ids: list[str]) -> SplitAssignment:
    """Deterministically assign item IDs to splits.

    Uses a seeded RNG with sorted IDs for reproducible placement.
    Split names are expanded into a pool proportional to ratios, then
    shuffled and assigned round-robin.

    Args:
      item_ids: Non-empty list of unique item IDs.

    Returns:
      ``SplitAssignment`` with all items placed.
    """
    rng = random.Random(self._seed)
    sorted_ids = sorted(item_ids)
    split_names = list(self._ratios)
    split_counts = self._compute_split_counts(len(sorted_ids))
    pool: list[str] = []
    for name in split_names:
      pool.extend([name] * split_counts[name])
    rng.shuffle(pool)
    assignments = dict(zip(sorted_ids, pool, strict=True))
    return SplitAssignment(
      assignments=assignments,
      ratios=dict(self._ratios),
      seed=self._seed,
    )

  def _compute_split_counts(self, total: int) -> dict[str, int]:
    """Compute per-split counts that sum to total, respecting ratios.

    Uses largest-remainder method for fair rounding.

    Args:
      total: Total number of items to distribute.

    Returns:
      Split name to count mapping summing to ``total``.
    """
    split_names = list(self._ratios)
    raw = {name: self._ratios[name] * total for name in split_names}
    floors = {name: int(raw[name]) for name in split_names}
    remainders = {name: raw[name] - floors[name] for name in split_names}
    allocated = sum(floors.values())
    deficit = total - allocated
    by_remainder = sorted(split_names, key=lambda n: -remainders[n])
    for i in range(deficit):
      floors[by_remainder[i]] += 1
    return floors
