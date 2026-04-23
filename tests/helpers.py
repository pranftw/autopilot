"""Shared test utilities for gradient tests."""

from autopilot.core.gradient import Gradient
from dataclasses import dataclass


@dataclass
class NumericGradient(Gradient):
  value: float = 0.0

  def accumulate(self, other: 'NumericGradient') -> 'NumericGradient':
    return NumericGradient(value=self.value + other.value)

  def render(self) -> str:
    return f'gradient: {self.value}'
