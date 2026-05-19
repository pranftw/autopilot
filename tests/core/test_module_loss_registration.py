"""Verify Loss auto-registers in _modules via Module inheritance.

F-008/BUG-015 was a stale finding: Loss already inherits from Module
(core/loss.py: class Loss(Module):), so assignment via __setattr__
registers it in _modules without any special-case branch.
"""

from autopilot.core.loss import Loss
from autopilot.core.module.module import Module
from autopilot.core.types import Datum


class StubLoss(Loss):
  """Minimal loss for registration verification."""

  def forward(self, data: Datum, targets: Datum | None = None) -> None:
    pass

  def backward(self) -> None:
    pass


class ModuleWithLoss(Module):
  """Module that holds a Loss attribute."""

  def __init__(self) -> None:
    super().__init__()
    self.loss = StubLoss()

  def forward(self, *args, **kwargs) -> Datum:
    return Datum(items=[])


def test_loss_registers_in_modules_via_module_inheritance():
  """Loss assigned as attribute is discoverable via module.modules()."""
  m = ModuleWithLoss()
  all_modules = list(m.modules())
  assert m.loss in all_modules


def test_loss_discoverable_via_named_modules():
  """Loss appears in named_modules() with its attribute name."""
  m = ModuleWithLoss()
  named = dict(m.named_modules())
  assert 'loss' in named
  assert named['loss'] is m.loss
