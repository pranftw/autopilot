"""Tests for dogfood-v3 core hardening fixes (plan 02).

Covers:
  - ISSUE-004: Tightened TypeError heuristic in _process_batch (inspect-based)
  - ISSUE-005: Narrowed _handle_rollback exception clause
  - ISSUE-006: MergeAnalysisResult.from_dict raises on unknown classification
"""

from autopilot.core.errors import ExperimentError, OrchestratorError, StoreError
from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.store.types import MergeAnalysisResult, MergeClassification
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from tests.doubles import DirectNumericLoss, NoOpOptimizer
from unittest.mock import MagicMock
import pytest


class _MissingBatchIdxModule(AutoPilotModule):
  """Module whose training_step is missing the batch_idx parameter."""

  def __init__(self) -> None:
    super().__init__()
    self.param = Parameter()
    self.loss = DirectNumericLoss([self.param])

  def forward(self, batch: Datum) -> Datum:
    return batch

  def training_step(self, batch: Datum) -> Datum:  # ty: ignore[invalid-method-override]
    return self(batch)

  def configure_optimizers(self):
    return NoOpOptimizer([self.param])


class _InternalTypeErrorModule(AutoPilotModule):
  """Module with correct signature that raises TypeError internally."""

  def __init__(self) -> None:
    super().__init__()
    self.param = Parameter()
    self.loss = DirectNumericLoss([self.param])

  def forward(self, batch: Datum) -> Datum:
    return batch

  def training_step(self, batch: Datum, batch_idx: int) -> Datum:
    msg = 'unsupported operand type(s)'
    raise TypeError(msg)

  def configure_optimizers(self):
    return NoOpOptimizer([self.param])


class _SingleItemDataModule(DataModule):
  """DataModule yielding a single EvalDatum sample."""

  def train_dataloader(self) -> DataLoader:
    return DataLoader(
      [EvalDatum(metadata={'idx': 0}, success=True)],
      batch_size=1,
    )


# ---------------------------------------------------------------------------
# ISSUE-004 tests: Epoch loop TypeError handling
# ---------------------------------------------------------------------------


class TestProcessBatchTypeErrorHeuristic:
  """ISSUE-004: inspect-based TypeError heuristic in _process_batch."""

  def test_training_step_missing_batch_idx_guidance(self) -> None:
    """Module with training_step(self, batch) gets guidance about batch_idx."""
    module = _MissingBatchIdxModule()
    trainer = Trainer()
    with pytest.raises(TypeError, match='Add batch_idx') as exc_info:
      trainer.fit(module, datamodule=_SingleItemDataModule(), max_epochs=1)

    assert exc_info.value.__cause__ is not None

  def test_training_step_internal_type_error_propagates(self) -> None:
    """Module with correct signature that raises TypeError internally propagates unchanged."""
    module = _InternalTypeErrorModule()
    trainer = Trainer()
    with pytest.raises(TypeError, match='unsupported operand type') as exc_info:
      trainer.fit(module, datamodule=_SingleItemDataModule(), max_epochs=1)

    assert 'batch_idx' not in str(exc_info.value)


# ---------------------------------------------------------------------------
# ISSUE-005 tests: Orchestrator rollback exception narrowing
# ---------------------------------------------------------------------------


class TestHandleRollbackExceptionNarrowing:
  """ISSUE-005: _handle_rollback catches only StoreError and ExperimentError."""

  def _make_orchestrator(self) -> EpochOrchestrator:
    """Create orchestrator with auto_rollback enabled and a prior good epoch."""
    config = OrchestratorConfig(auto_rollback=True, plateau_window=0)
    orch = EpochOrchestrator(config)
    orch._last_good_epoch = 0
    return orch

  def test_handle_rollback_store_error_wrapped(self) -> None:
    """StoreError from rollback is wrapped in OrchestratorError."""
    orch = self._make_orchestrator()
    experiment = MagicMock()
    experiment.store = True
    experiment.rollback.side_effect = StoreError('checkout failed')

    with pytest.raises(OrchestratorError, match='rollback to epoch 0 failed') as exc_info:
      orch._handle_rollback(experiment, 1)

    assert isinstance(exc_info.value.__cause__, StoreError)

  def test_handle_rollback_experiment_error_wrapped(self) -> None:
    """ExperimentError from rollback is wrapped in OrchestratorError."""
    orch = self._make_orchestrator()
    experiment = MagicMock()
    experiment.store = True
    experiment.rollback.side_effect = ExperimentError('lifecycle error')

    with pytest.raises(OrchestratorError, match='rollback to epoch 0 failed') as exc_info:
      orch._handle_rollback(experiment, 1)

    assert isinstance(exc_info.value.__cause__, ExperimentError)

  def test_handle_rollback_value_error_propagates(self) -> None:
    """ValueError from rollback propagates unwrapped (not OrchestratorError)."""
    orch = self._make_orchestrator()
    experiment = MagicMock()
    experiment.store = True
    experiment.rollback.side_effect = ValueError('programming bug')

    with pytest.raises(ValueError, match='programming bug'):
      orch._handle_rollback(experiment, 1)


# ---------------------------------------------------------------------------
# ISSUE-006 tests: MergeAnalysisResult classification
# ---------------------------------------------------------------------------


def _base_merge_dict(**overrides: object) -> dict[str, object]:
  """Build a minimal MergeAnalysisResult dict with overrides."""
  base: dict[str, object] = {
    'can_fast_forward': False,
    'has_conflicts': False,
    'conflict_count': 0,
    'ancestor_epoch': None,
    'classification': MergeClassification.up_to_date.value,
  }
  base.update(overrides)
  return base


class TestMergeAnalysisResultClassification:
  """ISSUE-006: from_dict raises StoreError on unknown classification."""

  def test_merge_analysis_round_trip_all_classifications(self) -> None:
    """Round-trip with each MergeClassification value."""
    for member in MergeClassification:
      original = MergeAnalysisResult(
        can_fast_forward=True,
        has_conflicts=False,
        conflict_count=0,
        ancestor_epoch=5,
        classification=member,
      )
      serialized = original.to_dict()
      restored = MergeAnalysisResult.from_dict(serialized)
      assert restored == original
      assert restored.classification is member

  def test_merge_analysis_unknown_classification_raises(self) -> None:
    """Bogus classification string raises StoreError."""
    data = _base_merge_dict(classification='bogus')
    with pytest.raises(StoreError, match='unknown merge classification') as exc_info:
      MergeAnalysisResult.from_dict(data)

    assert 'bogus' in str(exc_info.value)

  def test_merge_analysis_legacy_unknown_raises(self) -> None:
    """Legacy 'unknown' classification raises StoreError (no silent fallback)."""
    data = _base_merge_dict(classification='unknown')
    with pytest.raises(StoreError, match='unknown merge classification'):
      MergeAnalysisResult.from_dict(data)

  def test_merge_analysis_missing_classification_defaults(self) -> None:
    """Missing classification key defaults to up_to_date."""
    data = _base_merge_dict()
    del data['classification']
    result = MergeAnalysisResult.from_dict(data)
    assert result.classification is MergeClassification.up_to_date
