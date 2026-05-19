"""Tests for the convenience gradient re-export module.

Verifies that ``autopilot.gradients`` re-exports ``Gradient``,
``NumericGradient``, and ``TextGradient`` as identity-preserving
references to the canonical layer modules.
"""

from autopilot.ai.gradient import TextGradient as AITextGradient
from autopilot.core.gradient import Gradient as CoreGradient
from autopilot.core.gradient import NumericGradient as CoreNumericGradient
from autopilot.gradients import Gradient, NumericGradient, TextGradient


class TestGradientsReexportModule:
  """Import and type checks for the re-export module."""

  def test_gradients_reexport_module(self) -> None:
    """All three gradient classes importable from autopilot.gradients."""
    assert isinstance(Gradient, type)
    assert isinstance(NumericGradient, type)
    assert isinstance(TextGradient, type)

  def test_gradients_reexport_identity(self) -> None:
    """Re-exported classes are the exact same objects as canonical originals."""
    assert Gradient is CoreGradient
    assert NumericGradient is CoreNumericGradient
    assert TextGradient is AITextGradient
