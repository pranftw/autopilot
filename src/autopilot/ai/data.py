"""Stratified splitting and slot planning."""

from autopilot.ai.models import DataItem, VarDef
from autopilot.data.dataset import ListDataset
from pydantic import BaseModel
from typing import Callable, TypeVar
import random

T = TypeVar('T', bound=BaseModel)


def _split_names_and_normalized_ratios(
  ratios: dict[str, float],
) -> tuple[list[str], dict[str, float]]:
  split_names = list(ratios.keys())
  total = sum(ratios.values())
  if total <= 0:
    raise ValueError('ratios must sum to a positive value')
  norm = {name: ratios[name] / total for name in split_names}
  return split_names, norm


def _allocate_group_counts(
  n: int,
  split_names: list[str],
  norm: dict[str, float],
) -> dict[str, int]:
  """Assign integer counts per split for n items in one stratification group."""
  k = len(split_names)
  if n == 0:
    return {name: 0 for name in split_names}
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
    self._ratios = ratios
    self._key_fn = key_fn
    self._seed = seed

  def split(self, dataset: ListDataset[DataItem[T]]) -> dict[str, ListDataset[DataItem[T]]]:
    """Split dataset with matched distributions across all splits.

    Groups items by key_fn, then within each group distributes items
    across splits according to ratios. Uses seeded RNG for reproducibility.
    """
    split_names, norm = _split_names_and_normalized_ratios(self._ratios)
    rng = random.Random(self._seed)

    groups: dict[str, list[int]] = {}
    for idx in range(len(dataset)):
      item = dataset[idx]
      key = self._key_fn(item)
      groups.setdefault(key, []).append(idx)

    split_to_indices: dict[str, list[int]] = {name: [] for name in split_names}

    for _key, indices in groups.items():
      local = list(indices)
      rng.shuffle(local)
      n = len(local)
      counts = _allocate_group_counts(n, split_names, norm)
      offset = 0
      for name in split_names:
        take = counts[name]
        chunk = local[offset : offset + take]
        offset += take
        split_to_indices[name].extend(chunk)

    result: dict[str, ListDataset[DataItem[T]]] = {}
    for name in split_names:
      out_items: list[DataItem[T]] = []
      for idx in split_to_indices[name]:
        base = dataset[idx]
        out_items.append(base.model_copy(update={'split': name}))
      result[name] = ListDataset(out_items)
    return result


class SlotPlanner:
  """Built-in slot planner using vars with weighted distributions."""

  def __init__(self, vars: dict[str, VarDef], seed: int):
    self._vars = vars
    self._seed = seed
    self._rng = random.Random(seed)

  def weighted_pick(self, var: VarDef) -> tuple[str, dict | None]:
    """Pick a choice from a VarDef with its optional metadata."""
    choice = self._rng.choices(var.choices, weights=var.distribution, k=1)[0]
    if var.metadata is None:
      return (choice, None)
    idx = var.choices.index(choice)
    meta = var.metadata[idx] if idx < len(var.metadata) else None
    return (choice, meta)

  def create_slots(self, total_count: int, id_prefix: str | None = None) -> list[dict]:
    """Generate slots with seeded weighted sampling."""
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
