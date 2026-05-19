"""Shared serialization mixin for dataclasses.

DictMixin provides mechanical to_dict/from_dict for any dataclass.
Handles nested DictMixin instances, lists of DictMixin, dicts
with DictMixin values, and enum serialization via .value recursively.

Override from_dict on specific classes that need custom nested
deserialization (e.g. MemoryContext, ExperimentSummaryData).
"""

from dataclasses import fields
from enum import Enum
from typing import Any, Self
import dataclasses


def serialize(value: Any) -> Any:
  """Recursively convert a value to a JSON-safe representation.

  Handles DictMixin, Enum, list, and dict types. Scalars pass through.

  Args:
    value: The value to serialize.

  Returns:
    A JSON-serializable equivalent of ``value``.
  """
  if isinstance(value, DictMixin):
    return value.to_dict()
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, list):
    return [serialize(v) for v in value]
  if isinstance(value, dict):
    return {k: serialize(v) for k, v in value.items()}
  return value


def coerce_none_to_empty(data: dict[str, Any], field_defaults: dict[str, type]) -> dict[str, Any]:
  """Replace None values with empty containers for dict/list typed fields.

  Args:
    data: Input mapping possibly containing explicit nulls.
    field_defaults: Field name to factory type (e.g. list, dict) for empty values.

  Returns:
    Copy of data with None replaced by factory() for listed keys.
  """
  result = dict(data)
  for key, factory in field_defaults.items():
    if key in result and result[key] is None:
      result[key] = factory()
  return result


class DictMixin:
  """Mixin for dataclasses: adds generic to_dict/from_dict.

  to_dict() recursively serializes nested DictMixin and enum values.
  from_dict() is a thin cls(**data) constructor that does NOT reverse this:
  nested DictMixin fields become plain dicts, enums become strings. Classes
  requiring full round-trip fidelity must override from_dict() (see
  SnapshotManifest, DiffResult for examples).
  """

  @staticmethod
  def _coerce_none_to_empty(
    data: dict[str, Any], field_defaults: dict[str, type]
  ) -> dict[str, Any]:
    """Delegate to module-level :func:`coerce_none_to_empty`.

    Args:
      data: Raw dict for coercion.
      field_defaults: Field factories for None substitution.

    Returns:
      Coerced dict safe for dataclass construction.
    """
    return coerce_none_to_empty(data, field_defaults)

  def to_dict(self) -> dict[str, Any]:
    """Serialize this dataclass instance to nested dicts and plain values.

    Returns:
      Mapping of field names to serialized values.
    """
    result: dict[str, Any] = {}
    if not dataclasses.is_dataclass(self):
      return result
    for f in fields(self):
      result[f.name] = serialize(getattr(self, f.name))
    return result

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> Self:
    """Instantiate this dataclass from keys present in data only.

    Args:
      data: Flat mapping of field names to values.

    Returns:
      New instance constructed from intersecting keys.
    """
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})
