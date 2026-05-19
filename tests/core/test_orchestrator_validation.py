"""Tests for OrchestratorConfig validation (plan 08).

Validates that ``plateau_window > 0`` with ``monitor is None`` raises
``ConfigError`` at construction time, while ``plateau_window == 0`` without
a monitor is valid.
"""

from autopilot.core.errors import ConfigError
from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
import pytest


class TestOrchestratorMonitorValidation:
  """Tests 9-11: OrchestratorConfig plateau_window + monitor validation."""

  def test_orchestrator_monitor_none_with_plateau_window_raises(self) -> None:
    """plateau_window=3, monitor=None -> ConfigError."""
    with pytest.raises(ConfigError, match='monitor is required'):
      OrchestratorConfig(plateau_window=3, monitor=None)

  def test_orchestrator_monitor_none_without_plateau_passes(self) -> None:
    """plateau_window=0, monitor=None -> valid construction."""
    config = OrchestratorConfig(plateau_window=0, monitor=None)
    assert config.monitor is None
    assert config.plateau_window == 0
    orch = EpochOrchestrator(config)
    assert orch._config.monitor is None

  def test_orchestrator_monitor_set_with_plateau_works(self) -> None:
    """monitor='val_accuracy', plateau_window=3 -> valid construction."""
    config = OrchestratorConfig(monitor='val_accuracy', plateau_window=3)
    assert config.monitor == 'val_accuracy'
    assert config.plateau_window == 3
    orch = EpochOrchestrator(config)
    assert orch._config.monitor == 'val_accuracy'

  def test_error_message_is_actionable(self) -> None:
    """Error message names both fields and suggests prefixed metric keys."""
    with pytest.raises(ConfigError, match='plateau_window') as exc_info:
      OrchestratorConfig(plateau_window=5, monitor=None)
    msg = str(exc_info.value)
    assert 'val_accuracy' in msg or 'train_loss' in msg
    assert 'plateau_window=0' in msg

  def test_default_orchestrator_no_config_uses_plateau_zero(self) -> None:
    """EpochOrchestrator() without config defaults to plateau_window=0."""
    orch = EpochOrchestrator()
    assert orch._config.plateau_window == 0
    assert orch._config.monitor is None

  def test_plateau_window_one_requires_monitor(self) -> None:
    """Edge case: plateau_window=1 with no monitor raises."""
    with pytest.raises(ConfigError, match='monitor is required'):
      OrchestratorConfig(plateau_window=1)
