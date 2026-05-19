"""Tree entry for the experiment DAG.

Node holds real Python objects: an Experiment, an optional parent Node,
and an optional baseline Node. Node's identity is node.experiment.id.

Extensible via subclassing for domain-specific attributes.
Serialization converts objects to IDs on save (to_dict), resolves
IDs back to objects on load (from_dict with resolver).

Subclass example::

  @dataclass
  class TranslationNode(Node):
    source_lang: str | None = None
    target_lang: str | None = None
"""

from autopilot.core.experiment import Experiment
from autopilot.core.serialization import DictMixin
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Self, cast


@dataclass
class Node(DictMixin):
  """Tree entry holding real Python objects.

  Attributes:
    experiment: the actual Experiment object (not a string ID)
    parent: the actual parent Node (or None for root)
    baseline: the actual baseline Node (or None for fresh start)
    deployed_as: optional deployment label (forest-wide unique)

  Identity is node.experiment.id. Serialization converts objects
  to IDs on save, resolves IDs back to objects on load.
  """

  experiment: Experiment
  parent: 'Node | None' = None
  baseline: 'Node | None' = None
  deployed_as: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Serialize Node. Experiment/parent/baseline become ID strings.

    Includes 'type' field for subclass resolution on from_dict.
    Extra dataclass fields on subclasses are serialized normally.

    Returns:
      Mapping suitable for ``from_dict`` with a resolver.
    """
    result: dict[str, Any] = {
      'type': type(self).__qualname__,
      'experiment': self.experiment.id,
      'parent': self.parent.experiment.id if self.parent is not None else None,
      'baseline': self.baseline.experiment.id if self.baseline is not None else None,
    }
    base_names = {'experiment', 'parent', 'baseline'}
    for f in fields(self):
      if f.name not in base_names:
        result[f.name] = getattr(self, f.name)
    return result

  @classmethod
  def from_dict(  # type: ignore[ty:invalid-method-override]  # extra parameters vs DictMixin.from_dict; do not widen base
    cls,
    data: dict[str, Any],
    resolver: Callable[[str], 'Experiment | Node'],
    node_types: dict[str, type['Node']] | None = None,
  ) -> Self:
    """Deserialize Node. Resolver hydrates IDs back to objects.

    Args:
      data: dict from to_dict()
      resolver: callable that takes an ID string and returns either
        an Experiment (for 'experiment' key) or a Node (for 'parent'/'baseline')
      node_types: optional mapping of type qualname -> Node subclass.
        If None, instantiates base Node. If provided and the type
        string is missing, raises KeyError.

    Returns:
      Concrete ``Node`` subclass instance reconstructed from data.

    Raises:
      KeyError: When ``node_types`` is provided but ``data['type']`` is unknown.
    """
    type_str = data.get('type')

    if node_types is not None:
      if type_str not in node_types:
        msg = f'unknown node type: {type_str!r}'
        raise KeyError(msg)
      target_cls = node_types[type_str]
    else:
      target_cls = cls

    experiment_raw = resolver(data['experiment'])
    experiment: Experiment = cast(Experiment, experiment_raw)

    parent: Node | None = None
    if data.get('parent') is not None:
      parent = cast('Node | None', resolver(data['parent']))

    baseline: Node | None = None
    if data.get('baseline') is not None:
      baseline = cast('Node | None', resolver(data['baseline']))

    extra_fields = {f.name for f in fields(target_cls)} - {'experiment', 'parent', 'baseline'}
    extra_kwargs = {k: v for k, v in data.items() if k in extra_fields}

    return cast(
      Self,
      target_cls(
        experiment=experiment,
        parent=parent,
        baseline=baseline,
        **extra_kwargs,
      ),
    )
