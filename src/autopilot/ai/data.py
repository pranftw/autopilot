"""Stratified splitting and slot planning."""

from autopilot.ai.evaluation.schemas import DataItem, VarDef
from autopilot.data.dataset import ListDataset
from collections.abc import Callable
from pydantic import BaseModel
from typing import TypeVar
import random

T = TypeVar('T', bound=BaseModel)


def split_names_and_normalized_ratios(
  ratios: dict[str, float],
) -> tuple[list[str], dict[str, float]]:
  """Extract ordered split names and normalize ratios to sum to 1.

  Args:
    ratios: Mapping of split name to unnormalized weight.

  Returns:
    Tuple of (ordered split names, normalized ratio dict).

  Raises:
    ValueError: If ratios sum to zero or negative.
  """
  split_names = list(ratios)
  total = sum(ratios.values())
  if total <= 0:
    msg = f'ratios must sum to a positive value, got sum={total}'
    raise ValueError(
      msg,
    )
  norm = {name: ratios[name] / total for name in split_names}
  return split_names, norm


def _allocate_group_counts(
  n: int,
  split_names: list[str],
  norm: dict[str, float],
) -> dict[str, int]:
  """Assign integer counts per split for n items in one stratification group.

  Returns:
    Mapping from split name to allocated count for that split.
  """
  k = len(split_names)
  if n == 0:
    return dict.fromkeys(split_names, 0)
  if n < k:
    winner = max(split_names, key=lambda s: norm[s])
    return {name: (n if name == winner else 0) for name in split_names}
  counts: dict[str, int] = {}
  used = 0
  for name in split_names[:-1]:
    c = int(n * norm[name])
    counts[name] = c
    used += c
  counts[split_names[-1]] = n - used
  return counts


class StratifiedSplitter:
  """Split a dataset into train/val/test with matched distributions."""

  def __init__(self, ratios: dict[str, float], key_fn: Callable[[DataItem], str], seed: int):
    """Configure split ratios, stratification key function, and RNG seed.

    Args:
      ratios: Split names mapped to non-negative weights (need not sum to 1).
      key_fn: Function producing a stratification key for each item.
      seed: Seed for shuffling within each stratum.
    """
    self._ratios = ratios
    self._key_fn = key_fn
    self._seed = seed

  def _distribute_group(
    self,
    indices: list[int],
    rng: random.Random,
    split_names: list[str],
    norm: dict[str, float],
    split_to_indices: dict[str, list[int]],
  ) -> None:
    """Shuffle a stratum group and distribute indices across splits.

    Args:
      indices: Item indices belonging to one stratification group.
      rng: Seeded RNG for shuffling.
      split_names: Ordered split names.
      norm: Normalized ratio weights per split.
      split_to_indices: Accumulator mapping split name to collected indices.
    """
    local = list(indices)
    rng.shuffle(local)
    counts = _allocate_group_counts(len(local), split_names, norm)
    offset = 0
    for name in split_names:
      take = counts[name]
      split_to_indices[name].extend(local[offset : offset + take])
      offset += take

  def split(self, dataset: ListDataset[DataItem[T]]) -> dict[str, ListDataset[DataItem[T]]]:
    """Split dataset with matched distributions across all splits.

    Groups items by key_fn, then within each group distributes items
    across splits according to ratios. Uses seeded RNG for reproducibility.

    Returns:
      Mapping from split name to a :class:`ListDataset` with ``split`` field set.
    """
    split_names, norm = split_names_and_normalized_ratios(self._ratios)
    rng = random.Random(self._seed)

    groups: dict[str, list[int]] = {}
    for idx in range(len(dataset)):
      key = self._key_fn(dataset[idx])
      groups.setdefault(key, []).append(idx)

    split_to_indices: dict[str, list[int]] = {name: [] for name in split_names}
    for indices in groups.values():
      self._distribute_group(indices, rng, split_names, norm, split_to_indices)

    result: dict[str, ListDataset[DataItem[T]]] = {}
    for name in split_names:
      items = [dataset[idx].model_copy(update={'split': name}) for idx in split_to_indices[name]]
      result[name] = ListDataset(items)
    return result


class SlotPlanner:
  """Built-in slot planner using variable definitions with weighted distributions."""

  def __init__(self, variable_defs: dict[str, VarDef], seed: int):
    """Initialize planner from variable definitions and RNG seed.

    Args:
      variable_defs: Names mapped to :class:`VarDef` with choices and weights.
      seed: Seed for weighted random choices.
    """
    self._vars = variable_defs
    self._seed = seed
    self._rng = random.Random(seed)

  def weighted_pick(self, var: VarDef) -> tuple[str, dict | None]:
    """Pick a choice from a VarDef with its optional metadata.

    Returns:
      Tuple of chosen value string and optional metadata dict (or ``None``).
    """
    choice = self._rng.choices(var.choices, weights=var.distribution, k=1)[0]
    if var.metadata is None:
      return (choice, None)
    idx = var.choices.index(choice)
    meta = var.metadata[idx] if idx < len(var.metadata) else None
    return (choice, meta)

  def create_slots(self, total_count: int, id_prefix: str | None = None) -> list[dict]:
    """Generate slots with seeded weighted sampling.

    Returns:
      List of slot dicts including ``id`` and per-variable sampled fields.
    """
    prefix = id_prefix if id_prefix is not None else 'ITEM'
    slots: list[dict] = []
    for i in range(total_count):
      slot: dict = {'id': f'{prefix}_{i:06d}'}
      for var_name, var in self._vars.items():
        choice, meta = self.weighted_pick(var)
        slot[var_name] = choice
        if meta is not None:
          slot.update(meta)
      slots.append(slot)
    return slots
