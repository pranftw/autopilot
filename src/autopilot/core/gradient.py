"""Gradient base class. Extends Datum like Parameter extends Datum.

Gradient is the structured feedback type assigned to Parameter.grad by
Loss.backward(). Unlike numeric tensors, gradients here carry semantic
information about WHERE and WHAT to fix.
"""

from autopilot.core.types import Datum
from dataclasses import dataclass


@dataclass
class Gradient(Datum):
  """Base gradient type. Assigned as Parameter.grad by Loss.backward().

  Subclass and override:
    accumulate(other) -> Gradient  -- combine two gradients (for grad accumulation)
    render() -> str                -- describe for prompt inclusion (AgentOptimizer reads this)

  AccumulateGrad (in core/graph.py) calls accumulate() during backward traversal.
  AgentOptimizer calls render() to build the optimization prompt.

  Built-in subclass: TextGradient (ai/gradient.py) for LLM-oriented gradients
  with direction, attribution, severity, and evidence items.
  """

  def accumulate(self, other: 'Gradient') -> 'Gradient':
    raise NotImplementedError

  def render(self) -> str:
    raise NotImplementedError
