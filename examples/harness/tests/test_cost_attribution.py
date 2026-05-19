"""Tests for cost attribution data flow.

Verifies the end-to-end pipeline:
  ``result.usage()`` -> ``ConversationResult`` -> ``EvalDatum.metadata``
  -> harness sum metrics -> ``Result.metrics`` -> ``HarnessCostTrackerCallback``
  -> ``CostEntry.api_calls`` / ``CostEntry.tokens_used``.

No live LLM calls -- all token counts are synthetic.
"""

from autopilot.core.callbacks.cost import CostEntry, CostTrackerCallback
from autopilot.core.types import EvalDatum
from harness.agent import ConversationResult
from harness.callbacks import HarnessCostTrackerCallback
from harness.evaluator import EvaluationResult
from harness.metrics import (
  HarnessMetrics,
  TotalApiCalls,
  TotalInputTokens,
  TotalOutputTokens,
)
from unittest.mock import MagicMock, patch
import pytest


def _ok_result(**overrides):
  """Build a non-errored EvaluationResult with perfect scores by default."""
  defaults = {
    'task_success': True,
    'tool_recall': 1.0,
    'tool_precision': 1.0,
    'tool_argument_accuracy': 1.0,
    'communication_recall': 1.0,
    'policy_compliance': 1.0,
    'turns': 1,
    'errored': False,
  }
  defaults.update(overrides)
  return EvaluationResult(**defaults)


class TestConversationResultTokenFields:
  """Section 4.1: ConversationResult exposes token/API fields."""

  def test_defaults_are_zero(self):
    """Default ConversationResult has zero token and API fields."""
    cr = ConversationResult()
    assert cr.input_tokens == 0
    assert cr.output_tokens == 0
    assert cr.api_calls == 0

  def test_custom_values(self):
    """Token fields accept arbitrary int values."""
    cr = ConversationResult(input_tokens=150, output_tokens=80, api_calls=3)
    assert cr.input_tokens == 150
    assert cr.output_tokens == 80
    assert cr.api_calls == 3

  def test_field_types(self):
    """Field annotations exist and defaults are int."""
    cr = ConversationResult()
    assert isinstance(cr.input_tokens, int)
    assert isinstance(cr.output_tokens, int)
    assert isinstance(cr.api_calls, int)


class TestEvalDatumMetadataIncludesTokens:
  """Section 4.1: HarnessModule.forward copies token counts to EvalDatum.metadata."""

  def test_metadata_keys_present(self, tmp_path):
    """Patched forward produces EvalDatum with token metadata keys."""
    conv_result = ConversationResult(
      trajectory=[{'role': 'assistant', 'content': 'hi', 'turn': 0}],
      tool_calls=[],
      turns=1,
      input_tokens=200,
      output_tokens=100,
      api_calls=5,
    )
    eval_result = _ok_result()

    harness_pkg = tmp_path / 'harness'
    for sub in ('prompts', 'tools', 'db'):
      (harness_pkg / sub).mkdir(parents=True)
    (harness_pkg / 'prompts' / 'system_prompt.md').write_text('prompt', encoding='utf-8')
    (harness_pkg / 'prompts' / 'policies.md').write_text('policy', encoding='utf-8')
    (harness_pkg / 'tools' / 'retail_tools.py').write_text('', encoding='utf-8')
    (harness_pkg / 'db' / 'retail.json').write_text(
      '{"customers": [], "orders": [], "products": []}', encoding='utf-8'
    )

    with (
      patch('harness.module.HarnessAgent') as mock_agent_cls,
      patch('harness.module.ConversationEvaluator') as mock_eval_cls,
      patch('harness.module.load_tools', return_value={}),
    ):
      mock_agent_cls.return_value.run_conversation.return_value = conv_result
      mock_eval_cls.evaluate.return_value = eval_result

      from harness.module import HarnessModule

      module = HarnessModule(str(harness_pkg), use_judge=False)
      object.__setattr__(module, '_agent', mock_agent_cls.return_value)

      scenario = {'initial_message': 'hi', 'task_id': 't0'}
      batch = EvalDatum(success=False, metadata=scenario)
      result = module.forward(batch)

    assert result.metadata['input_tokens'] == 200
    assert result.metadata['output_tokens'] == 100
    assert result.metadata['api_calls'] == 5

  def test_error_path_omits_tokens(self, tmp_path):
    """Error path metadata does not include token keys."""
    harness_pkg = tmp_path / 'harness'
    for sub in ('prompts', 'tools', 'db'):
      (harness_pkg / sub).mkdir(parents=True)
    (harness_pkg / 'prompts' / 'system_prompt.md').write_text('prompt', encoding='utf-8')
    (harness_pkg / 'prompts' / 'policies.md').write_text('policy', encoding='utf-8')
    (harness_pkg / 'tools' / 'retail_tools.py').write_text('', encoding='utf-8')
    (harness_pkg / 'db' / 'retail.json').write_text(
      '{"customers": [], "orders": [], "products": []}', encoding='utf-8'
    )

    with (
      patch('harness.module.HarnessAgent') as mock_agent_cls,
      patch('harness.module.load_tools', return_value={}),
    ):
      from harness.module import HarnessModule

      module = HarnessModule(str(harness_pkg), use_judge=False)
      mock_agent = mock_agent_cls.return_value
      mock_agent.run_conversation.side_effect = RuntimeError('boom')
      object.__setattr__(module, '_agent', mock_agent)

      scenario = {'initial_message': 'hi', 'task_id': 't0'}
      batch = EvalDatum(success=False, metadata=scenario)
      result = module.forward(batch)

    assert 'input_tokens' not in result.metadata
    assert 'output_tokens' not in result.metadata
    assert 'api_calls' not in result.metadata


class TestTokenSumMetrics:
  """Section 4.2: individual sum metrics accumulate from EvalDatum.metadata."""

  def _datum(self, input_tokens=0, output_tokens=0, api_calls=0):
    """Build an EvalDatum with token metadata and a valid eval_result."""
    return EvalDatum(
      success=True,
      metadata={
        'eval_result': _ok_result().to_dict(),
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'api_calls': api_calls,
      },
    )

  def test_total_input_tokens(self):
    """Accumulates input_tokens across datums."""
    m = TotalInputTokens()
    m.update(self._datum(input_tokens=100))
    m.update(self._datum(input_tokens=250))
    assert m.compute() == {'total_input_tokens': 350.0}

  def test_total_output_tokens(self):
    """Accumulates output_tokens across datums."""
    m = TotalOutputTokens()
    m.update(self._datum(output_tokens=50))
    m.update(self._datum(output_tokens=75))
    assert m.compute() == {'total_output_tokens': 125.0}

  def test_total_api_calls(self):
    """Accumulates api_calls across datums."""
    m = TotalApiCalls()
    m.update(self._datum(api_calls=3))
    m.update(self._datum(api_calls=7))
    assert m.compute() == {'total_api_calls': 10.0}

  def test_missing_keys_treated_as_zero(self):
    """Datums without token metadata keys contribute zero."""
    m = TotalInputTokens()
    d = EvalDatum(
      success=True,
      metadata={'eval_result': _ok_result().to_dict()},
    )
    m.update(d)
    assert m.compute() == {'total_input_tokens': 0.0}

  def test_none_metadata_treated_as_zero(self):
    """Datum with None metadata contributes zero."""
    m = TotalApiCalls()
    d = EvalDatum(success=True, metadata=None)
    m.update(d)
    assert m.compute() == {'total_api_calls': 0.0}

  def test_higher_is_better_false(self):
    """All cost metrics have higher_is_better=False."""
    assert TotalInputTokens().higher_is_better is False
    assert TotalOutputTokens().higher_is_better is False
    assert TotalApiCalls().higher_is_better is False

  def test_empty_compute_warns(self):
    """Compute with no updates warns and returns zero."""
    m = TotalInputTokens()
    with pytest.warns(UserWarning):
      result = m.compute()
    assert result == {'total_input_tokens': 0.0}


class TestHarnessMetricsTokenKeys:
  """HarnessMetrics collection includes token sum metric keys."""

  def test_collection_includes_token_keys(self):
    """compute() returns all twelve keys including token sums."""
    coll = HarnessMetrics()
    d = EvalDatum(
      success=True,
      metadata={
        'eval_result': _ok_result().to_dict(),
        'input_tokens': 100,
        'output_tokens': 50,
        'api_calls': 3,
      },
    )
    coll.update(d)
    out = coll.compute()
    assert 'total_input_tokens' in out
    assert 'total_output_tokens' in out
    assert 'total_api_calls' in out
    assert out['total_input_tokens'] == 100.0
    assert out['total_output_tokens'] == 50.0
    assert out['total_api_calls'] == 3.0

  def test_collection_twelve_keys(self):
    """HarnessMetrics produces exactly twelve distinct keys."""
    coll = HarnessMetrics()
    d = EvalDatum(
      success=True,
      metadata={
        'eval_result': _ok_result().to_dict(),
        'input_tokens': 10,
        'output_tokens': 5,
        'api_calls': 1,
      },
    )
    coll.update(d)
    out = coll.compute()
    expected_count = 12
    assert len(out) == expected_count


class TestHarnessCostTrackerCallback:
  """Section 4.2: HarnessCostTrackerCallback.measure() populates CostEntry."""

  def test_is_subclass_of_cost_tracker(self):
    """HarnessCostTrackerCallback inherits from CostTrackerCallback."""
    assert issubclass(HarnessCostTrackerCallback, CostTrackerCallback)

  def test_measure_train_only(self):
    """Train-only metrics (no val_ prefix) populate api_calls and tokens_used."""
    cb = HarnessCostTrackerCallback()
    result = MagicMock()
    result.metrics = {
      'total_input_tokens': 500.0,
      'total_output_tokens': 300.0,
      'total_api_calls': 10.0,
      'task_success_rate': 0.8,
    }
    entry = cb.measure(epoch=0, elapsed=5.0, result=result)
    assert isinstance(entry, CostEntry)
    assert entry.api_calls == 10
    assert entry.tokens_used == 800
    assert entry.epoch == 0

  def test_measure_train_and_val(self):
    """Both train and val_ prefixed metrics combine into totals."""
    cb = HarnessCostTrackerCallback()
    result = MagicMock()
    result.metrics = {
      'total_input_tokens': 400.0,
      'total_output_tokens': 200.0,
      'total_api_calls': 8.0,
      'val_total_input_tokens': 100.0,
      'val_total_output_tokens': 50.0,
      'val_total_api_calls': 3.0,
    }
    entry = cb.measure(epoch=1, elapsed=10.0, result=result)
    assert entry.api_calls == 11
    assert entry.tokens_used == 750

  def test_measure_no_result(self):
    """No result yields default CostEntry (zeros)."""
    cb = HarnessCostTrackerCallback()
    entry = cb.measure(epoch=0, elapsed=1.0, result=None)
    assert entry.api_calls == 0
    assert entry.tokens_used == 0

  def test_measure_empty_metrics(self):
    """Empty metrics dict yields zero api_calls and tokens_used."""
    cb = HarnessCostTrackerCallback()
    result = MagicMock()
    result.metrics = {}
    entry = cb.measure(epoch=0, elapsed=2.0, result=result)
    assert entry.api_calls == 0
    assert entry.tokens_used == 0

  def test_measure_preserves_wall_clock(self):
    """Wall clock from super().measure() is preserved."""
    cb = HarnessCostTrackerCallback()
    result = MagicMock()
    result.metrics = {'total_input_tokens': 10.0}
    entry = cb.measure(epoch=2, elapsed=7.123, result=result)
    assert entry.wall_clock_s == 7.123

  def test_measure_float_metric_values(self):
    """Float metric values are safely coerced to int."""
    cb = HarnessCostTrackerCallback()
    result = MagicMock()
    result.metrics = {
      'total_input_tokens': 99.0,
      'total_output_tokens': 51.0,
      'total_api_calls': 4.0,
    }
    entry = cb.measure(epoch=0, elapsed=1.0, result=result)
    assert entry.api_calls == 4
    assert entry.tokens_used == 150

  def test_measure_metadata_copied(self):
    """Metrics are also available in CostEntry.metadata via super()."""
    cb = HarnessCostTrackerCallback()
    result = MagicMock()
    result.metrics = {
      'task_success_rate': 0.9,
      'total_input_tokens': 100.0,
    }
    entry = cb.measure(epoch=0, elapsed=1.0, result=result)
    assert entry.metadata['task_success_rate'] == 0.9


class TestCostDataFlowEndToEnd:
  """Section 4.2: end-to-end from fabricated EvalDatum through callback."""

  def _make_datum(self, input_tokens, output_tokens, api_calls):
    """Build a realistic EvalDatum with token metadata."""
    return EvalDatum(
      success=True,
      metadata={
        'eval_result': _ok_result().to_dict(),
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'api_calls': api_calls,
      },
    )

  def test_metrics_to_callback(self):
    """Metrics from datums flow into HarnessCostTrackerCallback correctly."""
    coll = HarnessMetrics()
    datums = [
      self._make_datum(100, 50, 3),
      self._make_datum(200, 75, 5),
    ]
    for d in datums:
      coll.update(d)
    computed = coll.compute()

    result = MagicMock()
    result.metrics = computed

    cb = HarnessCostTrackerCallback()
    entry = cb.measure(epoch=0, elapsed=1.0, result=result)

    assert entry.api_calls == 8
    assert entry.tokens_used == 425

  def test_metrics_to_callback_train_and_val(self):
    """Simulated train + val metrics flow through callback correctly."""
    train_metrics = HarnessMetrics()
    val_metrics = HarnessMetrics()

    train_datums = [
      self._make_datum(100, 50, 3),
      self._make_datum(200, 75, 5),
    ]
    val_datums = [
      self._make_datum(80, 40, 2),
    ]

    for d in train_datums:
      train_metrics.update(d)
    for d in val_datums:
      val_metrics.update(d)

    train_computed = train_metrics.compute()
    val_computed = val_metrics.compute()

    merged = dict(train_computed)
    for key, value in val_computed.items():
      merged[f'val_{key}'] = value

    result = MagicMock()
    result.metrics = merged

    cb = HarnessCostTrackerCallback()
    entry = cb.measure(epoch=0, elapsed=1.0, result=result)

    assert entry.api_calls == 10
    assert entry.tokens_used == 545

  def test_zero_usage_datums(self):
    """Datums with zero token counts produce zero cost totals."""
    coll = HarnessMetrics()
    coll.update(self._make_datum(0, 0, 0))
    computed = coll.compute()

    result = MagicMock()
    result.metrics = computed

    cb = HarnessCostTrackerCallback()
    entry = cb.measure(epoch=0, elapsed=1.0, result=result)

    assert entry.api_calls == 0
    assert entry.tokens_used == 0


class TestBuildTrainerIncludesCostCallback:
  """Standalone build_trainer() includes HarnessCostTrackerCallback."""

  def test_cost_callback_in_trainer(self, tmp_path):
    """build_trainer registers HarnessCostTrackerCallback on the trainer."""
    harness_pkg = tmp_path / 'harness'
    for sub in ('prompts', 'tools', 'db', 'scenarios'):
      (harness_pkg / sub).mkdir(parents=True)
    (harness_pkg / 'prompts' / 'system_prompt.md').write_text('prompt', encoding='utf-8')
    (harness_pkg / 'prompts' / 'policies.md').write_text('policy', encoding='utf-8')
    (harness_pkg / 'tools' / 'retail_tools.py').write_text('', encoding='utf-8')
    (harness_pkg / 'db' / 'retail.json').write_text(
      '{"customers": [], "orders": [], "products": []}', encoding='utf-8'
    )
    (harness_pkg / 'scenarios' / 'train.jsonl').write_text('', encoding='utf-8')
    (harness_pkg / 'scenarios' / 'val.jsonl').write_text('', encoding='utf-8')

    from harness.trainer import build_trainer

    trainer, _, _ = build_trainer(tmp_path, use_judge=False)
    cost_cbs = [cb for cb in trainer.callbacks if isinstance(cb, HarnessCostTrackerCallback)]
    assert len(cost_cbs) == 1

  def test_no_duplicate_cost_callbacks(self, tmp_path):
    """build_trainer does not produce duplicate CostTrackerCallback instances."""
    harness_pkg = tmp_path / 'harness'
    for sub in ('prompts', 'tools', 'db', 'scenarios'):
      (harness_pkg / sub).mkdir(parents=True)
    (harness_pkg / 'prompts' / 'system_prompt.md').write_text('prompt', encoding='utf-8')
    (harness_pkg / 'prompts' / 'policies.md').write_text('policy', encoding='utf-8')
    (harness_pkg / 'tools' / 'retail_tools.py').write_text('', encoding='utf-8')
    (harness_pkg / 'db' / 'retail.json').write_text(
      '{"customers": [], "orders": [], "products": []}', encoding='utf-8'
    )
    (harness_pkg / 'scenarios' / 'train.jsonl').write_text('', encoding='utf-8')
    (harness_pkg / 'scenarios' / 'val.jsonl').write_text('', encoding='utf-8')

    from harness.trainer import build_trainer

    trainer, _, _ = build_trainer(tmp_path, use_judge=False)
    cost_cbs = [cb for cb in trainer.callbacks if isinstance(cb, CostTrackerCallback)]
    assert len(cost_cbs) == 1
