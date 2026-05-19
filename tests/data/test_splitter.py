"""Tests for IncrementalSplitter and SplitAssignment."""

from autopilot.data.splitter import IncrementalSplitter, SplitAssignment
import pytest


class TestIncrementalSplitterFit:
  """Tests for IncrementalSplitter.fit."""

  def test_approximate_ratio_counts(self) -> None:
    """fit distributes items approximately per ratios for N=100."""
    splitter = IncrementalSplitter({'train': 0.8, 'val': 0.1, 'test': 0.1}, seed=42)
    ids = [f'item_{i}' for i in range(100)]
    assignment = splitter.fit(ids)
    counts = _count_splits(assignment)
    assert abs(counts['train'] - 80) <= 2
    assert abs(counts['val'] - 10) <= 2
    assert abs(counts['test'] - 10) <= 2

  def test_deterministic_across_runs(self) -> None:
    """Same seed produces identical assignments."""
    splitter = IncrementalSplitter({'train': 0.7, 'val': 0.3}, seed=99)
    ids = [f'doc_{i}' for i in range(50)]
    a1 = splitter.fit(ids)
    a2 = splitter.fit(ids)
    assert a1.assignments == a2.assignments

  def test_empty_item_ids(self) -> None:
    """Empty item_ids returns empty assignments with preserved ratios/seed."""
    splitter = IncrementalSplitter({'train': 0.8, 'val': 0.2}, seed=7)
    assignment = splitter.fit([])
    assert assignment.assignments == {}
    assert assignment.ratios == {'train': 0.8, 'val': 0.2}
    assert assignment.seed == 7

  def test_duplicate_item_ids_raises(self) -> None:
    """Duplicate item IDs in fit raises ValueError."""
    splitter = IncrementalSplitter({'train': 0.5, 'val': 0.5})
    with pytest.raises(ValueError, match='duplicates'):
      splitter.fit(['a', 'b', 'a'])


class TestIncrementalSplitterExtend:
  """Tests for IncrementalSplitter.extend."""

  def test_prior_assignments_unchanged(self) -> None:
    """All prior assignments remain unchanged after extend."""
    splitter = IncrementalSplitter({'train': 0.8, 'val': 0.2}, seed=42)
    original = splitter.fit([f'x_{i}' for i in range(20)])
    extended = splitter.extend(original, [f'y_{i}' for i in range(10)])
    for item_id, split in original.assignments.items():
      assert extended.assignments[item_id] == split

  def test_new_ids_distributed_per_ratios(self) -> None:
    """New IDs distributed approximately per ratios."""
    splitter = IncrementalSplitter({'train': 0.8, 'val': 0.2}, seed=42)
    existing = splitter.fit([f'old_{i}' for i in range(100)])
    extended = splitter.extend(existing, [f'new_{i}' for i in range(100)])
    new_only = {k: v for k, v in extended.assignments.items() if k.startswith('new_')}
    counts: dict[str, int] = {}
    for split in new_only.values():
      counts[split] = counts.get(split, 0) + 1
    assert abs(counts.get('train', 0) - 80) <= 2
    assert abs(counts.get('val', 0) - 20) <= 2

  def test_duplicate_new_id_raises(self) -> None:
    """Duplicate new ID raises ValueError."""
    splitter = IncrementalSplitter({'train': 0.5, 'val': 0.5})
    existing = splitter.fit(['a', 'b'])
    with pytest.raises(ValueError, match='duplicates'):
      splitter.extend(existing, ['c', 'd', 'c'])

  def test_overlap_with_existing_raises(self) -> None:
    """new_item_ids overlapping with existing raises ValueError."""
    splitter = IncrementalSplitter({'train': 0.5, 'val': 0.5})
    existing = splitter.fit(['a', 'b', 'c'])
    with pytest.raises(ValueError, match='overlap'):
      splitter.extend(existing, ['c', 'd'])

  def test_empty_new_item_ids_identity(self) -> None:
    """extend with empty new_item_ids returns identical assignments."""
    splitter = IncrementalSplitter({'train': 0.6, 'val': 0.4})
    existing = splitter.fit(['a', 'b', 'c'])
    extended = splitter.extend(existing, [])
    assert extended.assignments == existing.assignments


class TestSplitAssignment:
  """Tests for SplitAssignment round-trip and validation."""

  def test_round_trip_dict_equality(self) -> None:
    """to_dict -> from_dict preserves all fields."""
    sa = SplitAssignment(
      assignments={'a': 'train', 'b': 'val'},
      ratios={'train': 0.8, 'val': 0.2},
      seed=42,
    )
    restored = SplitAssignment.from_dict(sa.to_dict())
    assert restored.assignments == sa.assignments
    assert restored.ratios == sa.ratios
    assert restored.seed == sa.seed

  def test_strict_validation_invalid_split(self) -> None:
    """Assignment with split not in ratios raises ValueError on validate."""
    sa = SplitAssignment(
      assignments={'a': 'train', 'b': 'unknown'},
      ratios={'train': 0.8, 'val': 0.2},
      seed=42,
    )
    with pytest.raises(ValueError, match='unknown'):
      sa.validate()

  def test_removed_items_raises_value_error(self) -> None:
    """validate_universe raises ValueError when assigned IDs are absent from universe."""
    sa = SplitAssignment(
      assignments={'a': 'train', 'b': 'val', 'c': 'train'},
      ratios={'train': 0.8, 'val': 0.2},
      seed=42,
    )
    with pytest.raises(ValueError, match='not in current universe'):
      sa.validate_universe({'a', 'c'})


class TestIncrementalSplitterValidation:
  """Tests for ratio validation."""

  def test_invalid_ratios_raises(self) -> None:
    """Ratios not summing to 1.0 raises ValueError."""
    with pytest.raises(ValueError, match=r'sum to 1\.0'):
      IncrementalSplitter({'train': 0.5, 'val': 0.3})

  def test_valid_ratios_within_tolerance(self) -> None:
    """Ratios summing to 1.0 within tolerance are accepted."""
    splitter = IncrementalSplitter(
      {'train': 0.7, 'val': 0.2, 'test': 0.1},
    )
    assignment = splitter.fit(['a'])
    assert len(assignment.assignments) == 1


def _count_splits(assignment: SplitAssignment) -> dict[str, int]:
  """Count items per split in an assignment."""
  counts: dict[str, int] = {}
  for split in assignment.assignments.values():
    counts[split] = counts.get(split, 0) + 1
  return counts
