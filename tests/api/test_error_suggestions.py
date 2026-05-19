"""Error-suggestion tests for Dogfood V4 sub-plan 02.

Validates that common API misuse triggers actionable ``TypeError`` /
``ValueError`` messages with did-you-mean hints, and that correct usage
does not trip suggestion branches.

Tests:
  4.1 Constructor suggestions (1-8)
  4.2 Runtime messages (9-11)
  4.3 Training loop (12-13)
  4.4 Documentation and graph (14-15)
  4.5 Safety (16-17)
"""

from autopilot.ai.gradient import TextGradient
from autopilot.core.callbacks.checkpoint import CheckpointCallback
from autopilot.core.callbacks.diagnostics import DiagnosticsCallback
from autopilot.core.context import ContextEntry, ContextLog
from autopilot.core.diagnostics import Diagnostics
from autopilot.core.graph import get_current_graph
from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.module.module import Module
from autopilot.core.ops import select
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from autopilot.policy.gates import BudgetGate, MonotonicGate, RangeGate
from contextvars import copy_context
from pathlib import Path
from tests.doubles import NoopEvalModule
from unittest.mock import MagicMock
import pytest

# ---------------------------------------------------------------------------
# 4.1 Constructor suggestions
# ---------------------------------------------------------------------------


def test_text_gradient_direction_kwarg_suggests_text() -> None:
  """TextGradient(direction='x') raises TypeError mentioning 'text'."""
  with pytest.raises(TypeError, match='text'):
    TextGradient(direction='foo')


def test_range_gate_low_high_suggests_min_max() -> None:
  """RangeGate(..., low=..., high=...) raises TypeError with min/max hints."""
  with pytest.raises(TypeError, match='min_value') as exc_info:
    RangeGate('accuracy', 0.5, 0.9, low=0.4)  # type: ignore[call-arg]
  assert 'low' in str(exc_info.value)

  with pytest.raises(TypeError, match='max_value') as exc_info:
    RangeGate('accuracy', 0.5, 0.9, high=1.0)  # type: ignore[call-arg]
  assert 'high' in str(exc_info.value)


def test_range_gate_low_high_only_suggests_min_max() -> None:
  """RangeGate('m', low=0, high=1) (no positional bounds) raises TypeError with hint."""
  with pytest.raises(TypeError, match='min_value') as exc_info:
    RangeGate('m', low=0, high=1)  # type: ignore[call-arg]
  assert 'low' in str(exc_info.value)


def test_range_gate_omitted_bounds() -> None:
  """RangeGate('m') raises TypeError mentioning both min_value and max_value."""
  with pytest.raises(TypeError, match='min_value') as exc_info:
    RangeGate('m')
  assert 'max_value' in str(exc_info.value)


def test_range_gate_positional_still_works() -> None:
  """RangeGate('acc', 0.0, 1.0) positional succeeds."""
  gate = RangeGate('acc', 0.0, 1.0)
  assert gate.min_value == 0.0
  assert gate.max_value == 1.0


def test_range_gate_min_greater_than_max() -> None:
  """RangeGate('m', 1.0, 0.0) raises ValueError."""
  with pytest.raises(ValueError, match='min_value'):
    RangeGate('m', 1.0, 0.0)


def test_budget_gate_budget_suggests_max_usd() -> None:
  """BudgetGate(max_usd=5.0, budget=10.0) raises TypeError with max_usd hint."""
  with pytest.raises(TypeError, match='max_usd') as exc_info:
    BudgetGate(max_usd=5.0, budget=10.0)  # type: ignore[call-arg]
  assert 'budget' in str(exc_info.value)


def test_budget_gate_budget_only_suggests_max_usd() -> None:
  """BudgetGate(budget=50.0) (no max_usd) raises TypeError with hint."""
  with pytest.raises(TypeError, match='max_usd') as exc_info:
    BudgetGate(budget=50.0)  # type: ignore[call-arg]
  assert 'budget' in str(exc_info.value)


def test_budget_gate_omitted_suggests_max_usd() -> None:
  """BudgetGate() raises TypeError mentioning max_usd."""
  with pytest.raises(TypeError, match='max_usd'):
    BudgetGate()


def test_budget_gate_positional_still_works() -> None:
  """BudgetGate(50.0) positional succeeds."""
  gate = BudgetGate(50.0)
  assert gate._max_usd == 50.0


def test_checkpoint_callback_no_args_mentions_directory(tmp_path: Path) -> None:
  """CheckpointCallback() without directory raises TypeError naming directory."""
  with pytest.raises(TypeError, match='directory'):
    CheckpointCallback()


def test_checkpoint_callback_none_directory() -> None:
  """CheckpointCallback(directory=None) raises TypeError naming directory."""
  with pytest.raises(TypeError, match='directory'):
    CheckpointCallback(directory=None)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_diagnostics_callback_no_args_mentions_diagnostics() -> None:
  """DiagnosticsCallback() without diagnostics raises TypeError naming Diagnostics."""
  with pytest.raises(TypeError, match='Diagnostics'):
    DiagnosticsCallback()


def test_range_gate_correct_kwarg_no_error() -> None:
  """Valid RangeGate construction does not raise."""
  gate = RangeGate('accuracy', min_value=0.5, max_value=0.9)
  assert gate.min_value == 0.5
  assert gate.max_value == 0.9


def test_budget_gate_correct_kwarg_no_error() -> None:
  """Valid BudgetGate construction does not raise."""
  gate = BudgetGate(max_usd=50.0)
  assert gate._max_usd == 50.0


def test_checkpoint_callback_correct_args_no_error(tmp_path: Path) -> None:
  """CheckpointCallback(tmp_path) with valid directory does not raise."""
  cb = CheckpointCallback(tmp_path)
  assert cb.directory == tmp_path


# ---------------------------------------------------------------------------
# 4.2 Runtime messages
# ---------------------------------------------------------------------------


def test_select_out_of_range_shows_arity() -> None:
  """select(datum, bad_index) raises IndexError with count context."""
  d = Datum(items=[Datum(), Datum()])
  with pytest.raises(IndexError, match='select') as exc_info:
    select(d, 5)
  msg = str(exc_info.value)
  assert '5' in msg
  assert '2' in msg


def test_monotonic_gate_invalid_direction_lists_valid() -> None:
  """Invalid direction= raises ValueError listing valid options."""
  with pytest.raises(ValueError, match='non_decreasing') as exc_info:
    MonotonicGate('accuracy', direction='increasing')
  assert 'non_increasing' in str(exc_info.value)


def test_context_log_append_entry_delegates_to_record() -> None:
  """append(ContextEntry) delegates to record() and returns the entry."""
  log = ContextLog()
  entry = ContextEntry.create('test reason')
  result = log.append(entry)
  assert result is entry
  assert len(log) == 1


# ---------------------------------------------------------------------------
# 4.3 Training loop
# ---------------------------------------------------------------------------


class _NoneReturningModule(AutoPilotModule):
  """Stub module that returns None from training_step."""

  def forward(self, *args, **kwargs) -> Datum:
    return Datum()

  def training_step(self, batch, batch_idx):
    return None

  def configure_optimizers(self):
    return None


def test_training_step_none_error_message() -> None:
  """Stub module returning None triggers TypeError with guidance."""
  loop = EpochLoop()
  module = _NoneReturningModule()
  trainer = MagicMock()
  trainer.module = module
  trainer.dispatch_callbacks = MagicMock()

  with pytest.raises(TypeError, match=r'training_step.*returned None'):
    loop._process_batch(
      trainer,
      module,
      batch_idx=0,
      batch=Datum(),
      is_last=True,
      loss_fn=None,
      optimizer=None,
      metrics={},
      accumulate=1,
    )


def test_training_step_valid_datum_no_error() -> None:
  """Module returning valid Datum does not raise."""
  loop = EpochLoop()
  module = NoopEvalModule()
  trainer = MagicMock()
  trainer.module = module
  trainer.dispatch_callbacks = MagicMock()

  loop._process_batch(
    trainer,
    module,
    batch_idx=0,
    batch=Datum(),
    is_last=True,
    loss_fn=None,
    optimizer=None,
    metrics={},
    accumulate=1,
  )


# ---------------------------------------------------------------------------
# 4.4 Documentation and graph
# ---------------------------------------------------------------------------


def test_forward_direct_call_guidance_documented() -> None:
  """CLAUDE.md contains forward-vs-__call__ guidance phrases."""
  path = Path(__file__).resolve().parents[2] / 'CLAUDE.md'
  content = path.read_text(encoding='utf-8')
  required_phrases = [
    'Module.__call__',
    'forward()',
    'ModuleCallOperator',
    'bypasses graph recording',
  ]
  for phrase in required_phrases:
    assert phrase in content, f'CLAUDE.md missing required phrase: {phrase!r}'


def test_forward_via_call_records_graph() -> None:
  """module(datum) records an operator node; module.forward(datum) does not."""

  def run():
    class M(Module):
      def __init__(self):
        super().__init__()
        self.w = Parameter(requires_grad=True)

      def forward(self, x):
        return Datum()

    g = get_current_graph()
    g.reset()
    g._freed = False

    m = M()
    result_call = m(Datum())
    assert result_call.grad_fn is not None, (
      'module(datum) should record graph via ModuleCallOperator'
    )

    g.reset()
    g._freed = False
    result_forward = m.forward(Datum())
    assert result_forward.grad_fn is None, 'module.forward(datum) should bypass graph recording'

  ctx = copy_context()
  ctx.run(run)


# ---------------------------------------------------------------------------
# 4.5 Safety
# ---------------------------------------------------------------------------


def test_text_gradient_correct_kwarg_no_error() -> None:
  """TextGradient with text= succeeds without error."""
  tg = TextGradient(text='improve accuracy')
  assert tg.text == 'improve accuracy'


def test_error_message_no_false_trigger(tmp_path: Path) -> None:
  """Combined valid instantiations do not raise suggestion errors."""
  rg = RangeGate('accuracy', min_value=0.5, max_value=0.9)
  assert rg.min_value == 0.5

  bg = BudgetGate(max_usd=100.0)
  assert bg._max_usd == 100.0

  tg = TextGradient(text='test content')
  assert tg.text == 'test content'

  mg = MonotonicGate('loss', direction='non_increasing')
  assert mg._direction == 'non_increasing'

  cb = CheckpointCallback(tmp_path)
  assert cb.directory == tmp_path

  diag = Diagnostics(path=tmp_path)
  dc = DiagnosticsCallback(diag)
  assert dc._diagnostics is diag
