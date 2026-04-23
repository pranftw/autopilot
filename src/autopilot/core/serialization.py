"""Shared serialization mixin for dataclasses.

DictMixin provides mechanical to_dict/from_dict for any dataclass.
Handles nested DictMixin instances, lists of DictMixin, and dicts
with DictMixin values recursively.

Override from_dict on specific classes that need custom nested
deserialization (e.g. MemoryContext, ExperimentSummaryData).
"""

from dataclasses import fields
from typing import Any, Self


def _serialize(value: Any) -> Any:
  if isinstance(value, DictMixin):
    return value.to_dict()
  if isinstance(value, list):
    return [_serialize(v) for v in value]
  if isinstance(value, dict):
    return {k: _serialize(v) for k, v in value.items()}
  return value


class DictMixin:
  """Mixin for dataclasses: adds generic to_dict/from_dict."""

  def to_dict(self) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for f in fields(self):
      result[f.name] = _serialize(getattr(self, f.name))
    return result

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> Self:
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})
