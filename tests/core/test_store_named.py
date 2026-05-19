"""Tests for Store ABC registration contract and config-only constructor."""

from autopilot.core.parameter import Parameter
from autopilot.core.store.base import Store
from typing import Any, cast
import pytest


class TestStoreBaseRegisterParameters:
  """Store base class contract for register_parameters."""

  def test_store_init_config_only(self) -> None:
    with pytest.raises(NotImplementedError):
      Store(cast(Any, None))

  def test_store_register_parameters_exists(self) -> None:
    assert hasattr(Store, 'register_parameters')
    with pytest.raises(NotImplementedError):
      Store.register_parameters(cast(Any, None), {'p': Parameter()})
