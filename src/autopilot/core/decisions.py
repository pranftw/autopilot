"""Decision queries for experiment lifecycle."""

from autopilot.core.models import Manifest


def is_decided(manifest: Manifest) -> bool:
  """Whether a decision has been recorded."""
  return bool(manifest.decision)
