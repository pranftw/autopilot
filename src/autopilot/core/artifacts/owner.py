"""ArtifactOwner mixin: auto-registers Artifact attributes into _artifacts dict.

Like Module.__setattr__ for Parameter. Any class that owns typed
file artifacts mixes this in and calls __init_artifacts__() in __init__.
"""

from autopilot.core.artifacts.artifact import Artifact
from typing import Any


class ArtifactOwner:
  """Mixin: auto-registers Artifact attributes into _artifacts dict."""

  def __init_artifacts__(self) -> None:
    object.__setattr__(self, '_artifacts', {})

  def __setattr__(self, name: str, value: Any) -> None:
    artifacts = self.__dict__.get('_artifacts')
    if artifacts is not None and isinstance(value, Artifact):
      artifacts[name] = value
    object.__setattr__(self, name, value)

  @property
  def artifacts(self) -> dict[str, 'Artifact']:
    return dict(self._artifacts)
