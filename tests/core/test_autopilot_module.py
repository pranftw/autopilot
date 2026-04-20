"""Tests for AutoPilotModule (LightningModule pattern)."""

from autopilot.core.models import Datum
from autopilot.core.module import AutoPilotModule, Module
from autopilot.core.parameter import Parameter
import pytest


class _ChildModule(Module):
  def forward(self, batch: object) -> Datum:
    return Datum(success=True)


class _ConcreteAutoPilot(AutoPilotModule):
  def forward(self, batch: object) -> Datum:
    return Datum(success=True, metrics={'ok': 1.0})

  def training_step(self, batch: object) -> Datum:
    return self.forward(batch)

  def validation_step(self, batch: object) -> Datum:
    return self.forward(batch)

  def test_step(self, batch: object) -> Datum:
    return self.forward(batch)

  def configure_optimizers(self) -> str:
    return 'mock_optimizer'


class TestAutoPilotModuleInheritance:
  def test_is_module_subclass(self) -> None:
    mod = AutoPilotModule()
    assert isinstance(mod, Module)

  def test_forward_inherited_from_module(self) -> None:
    mod = _ConcreteAutoPilot()
    result = mod.forward(None)
    assert isinstance(result, Datum)
    assert result.success is True

  def test_children_and_parameters_inherited(self) -> None:
    mod = _ConcreteAutoPilot()
    mod.child = _ChildModule()
    mod.param = Parameter()
    assert len(list(mod.children())) == 1
    assert len(list(mod.parameters())) == 1

  def test_train_eval_inherited(self) -> None:
    mod = _ConcreteAutoPilot()
    mod.child = _ChildModule()
    mod.eval()
    assert mod.training is False
    assert mod.child.training is False
    mod.train()
    assert mod.training is True


class TestAutoPilotModuleStepMethods:
  def test_training_step_raises_not_implemented(self) -> None:
    mod = AutoPilotModule()
    with pytest.raises(NotImplementedError):
      mod.training_step(None)

  def test_validation_step_raises_not_implemented(self) -> None:
    mod = AutoPilotModule()
    with pytest.raises(NotImplementedError):
      mod.validation_step(None)

  def test_test_step_raises_not_implemented(self) -> None:
    mod = AutoPilotModule()
    with pytest.raises(NotImplementedError):
      mod.test_step(None)

  def test_configure_optimizers_raises_not_implemented(self) -> None:
    mod = AutoPilotModule()
    with pytest.raises(NotImplementedError):
      mod.configure_optimizers()

  def test_subclass_training_step(self) -> None:
    mod = _ConcreteAutoPilot()
    result = mod.training_step(None)
    assert result.success is True

  def test_subclass_validation_step(self) -> None:
    mod = _ConcreteAutoPilot()
    result = mod.validation_step(None)
    assert result.success is True

  def test_subclass_test_step(self) -> None:
    mod = _ConcreteAutoPilot()
    result = mod.test_step(None)
    assert result.success is True

  def test_subclass_configure_optimizers(self) -> None:
    mod = _ConcreteAutoPilot()
    assert mod.configure_optimizers() == 'mock_optimizer'


class TestAutoPilotModuleLifecycle:
  def test_lifecycle_hooks_are_noop_by_default(self) -> None:
    mod = AutoPilotModule()
    mod.setup()
    mod.teardown()
    mod.on_train_start()
    mod.on_train_end()
    mod.on_validation_start()
    mod.on_validation_end()

  def test_trainer_property_default_none(self) -> None:
    mod = AutoPilotModule()
    assert mod.trainer is None

  def test_trainer_property_set_externally(self) -> None:
    mod = AutoPilotModule()
    mock_trainer = object()
    mod._trainer = mock_trainer
    assert mod.trainer is mock_trainer
