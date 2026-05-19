"""Convenience re-exports for gradient types.

All gradient classes available from one import path::

    from autopilot.gradients import Gradient, NumericGradient, TextGradient

Original layer-specific import paths remain valid (additive only, no breaking change).
"""

from autopilot.ai.gradient import TextGradient
from autopilot.core.gradient import Gradient, NumericGradient

__all__ = ['Gradient', 'NumericGradient', 'TextGradient']
