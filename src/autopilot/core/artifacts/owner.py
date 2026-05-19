"""ArtifactOwner mixin: auto-registers Artifact attributes into _artifacts dict.

Like Module.__setattr__ for Parameter. Any class that owns typed
file artifacts mixes this in and calls init_artifacts() in __init__.
"""

from autopilot.core.artifacts.artifact import Artifact
from typing import Any


class ArtifactOwner:
  """Mixin: auto-registers Artifact attributes into _artifacts dict."""

  _artifacts: dict[str, 'Artifact']

  def init_artifacts(self) -> None:
    """Initialize the internal artifact registry before attributes are set."""
    super().__setattr__('_artifacts', {})

  def __setattr__(self, name: str, value: Any) -> None:
    """Set attribute and register Artifact values in the owner map.

    Args:
      name: Attribute name.
      value: New value; Artifact instances are tracked under name.
    """
    artifacts = self.__dict__.get('_artifacts')
    if artifacts is not None and isinstance(value, Artifact):
      artifacts[name] = value
    object.__setattr__(self, name, value)

  @property
  def artifacts(self) -> dict[str, 'Artifact']:
    """Shallow copy of registered artifacts keyed by attribute name.

    Returns:
      Dict mapping attribute names to Artifact instances.
    """
    return dict(self._artifacts)
