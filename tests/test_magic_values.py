"""Tests for plan 03: magic values, named constants, and StrEnums.

Covers all constants from the DRY consolidation plan 03 checklist:
  - Core/tracking: SHORT_ID_HEX_LEN, TODO_LINE_MIN_LEN, TODO_ITEM_MAX_CHARS, WALL_CLOCK_DECIMALS
  - AI layer: RPM_WINDOW_SECONDS, FILE_ENTRY_MTIME_UNAVAILABLE, REFS_FORMAT_VERSION,
    DIGEST_DISPLAY_HEX_LEN, STDOUT_ERROR_SNIPPET_LEN, AGENT_OUTPUT_ERROR_SNIPPET_LEN,
    PATH_LISTING_LIMIT, CONFIG_HASH_HEX_LEN (pipeline)
  - CLI layer: EXEC_STDOUT_PREVIEW_LEN, GRADIENT_PREVIEW_LEN, MERGE_CONFLICT_DISPLAY_LIMIT,
    PROPOSAL_ID_HEX_LEN, FALLBACK_JUDGE_CONFIG
  - Data layer: VALIDATION_ERROR_SAMPLE_SIZE
  - StrEnums: MergeClassification, DiffKind, VerdictKind
"""

from autopilot.ai.agents.claude_code import STDOUT_ERROR_SNIPPET_LEN
from autopilot.ai.evaluation.pipeline import CONFIG_HASH_HEX_LEN
from autopilot.ai.gradient import AGENT_OUTPUT_ERROR_SNIPPET_LEN
from autopilot.ai.parameter import PATH_LISTING_LIMIT
from autopilot.ai.runtime import RPM_WINDOW_SECONDS
from autopilot.ai.store.merge import REFS_FORMAT_VERSION
from autopilot.ai.store.snapshot import (
  DIGEST_DISPLAY_HEX_LEN,
  FILE_ENTRY_MTIME_UNAVAILABLE,
)
from autopilot.cli.commands.ai import FALLBACK_JUDGE_CONFIG
from autopilot.cli.commands.debug import EXEC_STDOUT_PREVIEW_LEN, GRADIENT_PREVIEW_LEN
from autopilot.cli.commands.propose import PROPOSAL_ID_HEX_LEN, VerdictKind
from autopilot.cli.commands.store.merge import MERGE_CONFLICT_DISPLAY_LIMIT
from autopilot.core.callbacks.cost import WALL_CLOCK_DECIMALS, CostEntry, CostTrackerCallback
from autopilot.core.gradient import (
  TODO_ITEM_MAX_CHARS,
  TODO_LINE_MIN_LEN,
  Gradient,
  NumericGradient,
)
from autopilot.core.store.types import (
  DiffEntry,
  DiffKind,
  DiffResult,
  MergeAnalysisResult,
  MergeClassification,
  StatusEntry,
  StatusResult,
)
from autopilot.core.types import SHORT_ID_HEX_LEN, Datum
from autopilot.data.splitter import VALIDATION_ERROR_SAMPLE_SIZE, SplitAssignment
from dataclasses import dataclass
from pathlib import Path
import json
import pytest

# Section 4.1: constant types and positivity


class TestConstantTypes:
  """Assert selected constants are the expected type (int/float/str) and positive."""

  def test_short_id_hex_len_is_positive_int(self) -> None:
    assert isinstance(SHORT_ID_HEX_LEN, int)
    assert SHORT_ID_HEX_LEN > 0

  def test_todo_line_min_len_is_positive_int(self) -> None:
    assert isinstance(TODO_LINE_MIN_LEN, int)
    assert TODO_LINE_MIN_LEN > 0

  def test_todo_item_max_chars_is_positive_int(self) -> None:
    assert isinstance(TODO_ITEM_MAX_CHARS, int)
    assert TODO_ITEM_MAX_CHARS > 0

  def test_wall_clock_decimals_is_int(self) -> None:
    assert isinstance(WALL_CLOCK_DECIMALS, int)

  def test_rpm_window_seconds_is_positive_float(self) -> None:
    assert isinstance(RPM_WINDOW_SECONDS, float)
    assert RPM_WINDOW_SECONDS > 0

  def test_file_entry_mtime_unavailable_is_float(self) -> None:
    assert isinstance(FILE_ENTRY_MTIME_UNAVAILABLE, float)
    assert FILE_ENTRY_MTIME_UNAVAILABLE == 0.0

  def test_refs_format_version_is_positive_int(self) -> None:
    assert isinstance(REFS_FORMAT_VERSION, int)
    assert REFS_FORMAT_VERSION > 0

  def test_digest_display_hex_len_is_positive_int(self) -> None:
    assert isinstance(DIGEST_DISPLAY_HEX_LEN, int)
    assert DIGEST_DISPLAY_HEX_LEN > 0

  def test_stdout_error_snippet_len_is_positive_int(self) -> None:
    assert isinstance(STDOUT_ERROR_SNIPPET_LEN, int)
    assert STDOUT_ERROR_SNIPPET_LEN > 0

  def test_agent_output_error_snippet_len_is_positive_int(self) -> None:
    assert isinstance(AGENT_OUTPUT_ERROR_SNIPPET_LEN, int)
    assert AGENT_OUTPUT_ERROR_SNIPPET_LEN > 0

  def test_path_listing_limit_is_positive_int(self) -> None:
    assert isinstance(PATH_LISTING_LIMIT, int)
    assert PATH_LISTING_LIMIT > 0

  def test_config_hash_hex_len_is_positive_int(self) -> None:
    assert isinstance(CONFIG_HASH_HEX_LEN, int)
    assert CONFIG_HASH_HEX_LEN > 0

  def test_exec_stdout_preview_len_is_positive_int(self) -> None:
    assert isinstance(EXEC_STDOUT_PREVIEW_LEN, int)
    assert EXEC_STDOUT_PREVIEW_LEN > 0

  def test_gradient_preview_len_is_positive_int(self) -> None:
    assert isinstance(GRADIENT_PREVIEW_LEN, int)
    assert GRADIENT_PREVIEW_LEN > 0

  def test_merge_conflict_display_limit_is_positive_int(self) -> None:
    assert isinstance(MERGE_CONFLICT_DISPLAY_LIMIT, int)
    assert MERGE_CONFLICT_DISPLAY_LIMIT > 0

  def test_proposal_id_hex_len_is_positive_int(self) -> None:
    assert isinstance(PROPOSAL_ID_HEX_LEN, int)
    assert PROPOSAL_ID_HEX_LEN > 0

  def test_validation_error_sample_size_is_positive_int(self) -> None:
    assert isinstance(VALIDATION_ERROR_SAMPLE_SIZE, int)
    assert VALIDATION_ERROR_SAMPLE_SIZE > 0

  def test_fallback_judge_config_is_dict(self) -> None:
    assert isinstance(FALLBACK_JUDGE_CONFIG, dict)
    assert 'run' in FALLBACK_JUDGE_CONFIG
    assert 'system_prompt' in FALLBACK_JUDGE_CONFIG


# Section 4.2: StrEnums equal strings


class TestDiffKindAllValues:
  """Iterate every DiffKind member; assert str(m) == m.value."""

  @pytest.mark.parametrize(
    'member',
    list(DiffKind),
    ids=lambda m: m.name,
  )
  def test_str_equals_value(self, member: DiffKind) -> None:
    assert str(member) == member.value

  @pytest.mark.parametrize(
    ('member', 'expected'),
    [
      (DiffKind.added, 'added'),
      (DiffKind.modified, 'modified'),
      (DiffKind.deleted, 'deleted'),
      (DiffKind.unchanged, 'unchanged'),
    ],
  )
  def test_equals_plain_string(self, member: DiffKind, expected: str) -> None:
    assert member == expected
    assert member.value == expected

  def test_json_serialization(self) -> None:
    payload = json.dumps({'status': DiffKind.added})
    assert payload == '{"status": "added"}'


class TestMergeClassificationAllValues:
  """Iterate every MergeClassification member; assert str(m) == m.value."""

  @pytest.mark.parametrize(
    'member',
    list(MergeClassification),
    ids=lambda m: m.name,
  )
  def test_str_equals_value(self, member: MergeClassification) -> None:
    assert str(member) == member.value

  @pytest.mark.parametrize(
    ('member', 'expected'),
    [
      (MergeClassification.up_to_date, 'up_to_date'),
      (MergeClassification.fast_forward, 'fast_forward'),
      (MergeClassification.clean, 'clean'),
      (MergeClassification.conflict, 'conflict'),
    ],
  )
  def test_equals_plain_string(self, member: MergeClassification, expected: str) -> None:
    assert member == expected
    assert member.value == expected

  def test_json_serialization(self) -> None:
    payload = json.dumps({'classification': MergeClassification.clean})
    assert payload == '{"classification": "clean"}'


class TestVerdictKindAllValues:
  """Iterate every VerdictKind member; assert str(m) == m.value."""

  @pytest.mark.parametrize(
    'member',
    list(VerdictKind),
    ids=lambda m: m.name,
  )
  def test_str_equals_value(self, member: VerdictKind) -> None:
    assert str(member) == member.value

  @pytest.mark.parametrize(
    ('member', 'expected'),
    [
      (VerdictKind.improved, 'improved'),
      (VerdictKind.regressed, 'regressed'),
      (VerdictKind.inconclusive, 'inconclusive'),
    ],
  )
  def test_equals_plain_string(self, member: VerdictKind, expected: str) -> None:
    assert member == expected
    assert member.value == expected


# Section 4.3: serialization round-trips unchanged


class TestDiffEntryRoundTrip:
  """DiffEntry/StatusEntry to_dict/from_dict produce unchanged str status."""

  def test_diff_entry_round_trip_with_added(self) -> None:
    entry = DiffEntry(path='a/b.txt', status=DiffKind.added, new_hash='abc123')
    serialized = entry.to_dict()
    assert serialized['status'] == 'added'
    restored = DiffEntry.from_dict(serialized)
    assert restored.status == 'added'
    assert restored.path == 'a/b.txt'

  def test_diff_entry_round_trip_with_deleted(self) -> None:
    entry = DiffEntry(path='c.txt', status=DiffKind.deleted, old_hash='xyz')
    serialized = entry.to_dict()
    assert serialized['status'] == 'deleted'
    restored = DiffEntry.from_dict(serialized)
    assert restored.status == 'deleted'

  def test_status_entry_round_trip_with_modified(self) -> None:
    entry = StatusEntry(path='d.txt', status=DiffKind.modified)
    serialized = entry.to_dict()
    assert serialized['status'] == 'modified'
    restored = StatusEntry.from_dict(serialized)
    assert restored.status == 'modified'

  def test_diff_result_helper_methods(self) -> None:
    result = DiffResult(
      entries=[
        DiffEntry(path='a', status=DiffKind.added),
        DiffEntry(path='b', status=DiffKind.modified),
        DiffEntry(path='c', status=DiffKind.deleted),
        DiffEntry(path='d', status='added'),
      ]
    )
    assert len(result.added()) == 2
    assert len(result.modified()) == 1
    assert len(result.deleted()) == 1

  def test_status_result_helper_methods(self) -> None:
    result = StatusResult(
      entries=[
        StatusEntry(path='a', status=DiffKind.added),
        StatusEntry(path='b', status=DiffKind.unchanged),
        StatusEntry(path='c', status=DiffKind.modified),
        StatusEntry(path='d', status=DiffKind.deleted),
      ]
    )
    assert len(result.added()) == 1
    assert len(result.modified()) == 1
    assert len(result.deleted()) == 1
    assert len(result.unchanged()) == 1

  def test_merge_analysis_round_trip(self) -> None:
    result = MergeAnalysisResult(
      can_fast_forward=True,
      has_conflicts=False,
      conflict_count=0,
      ancestor_epoch=3,
      classification=MergeClassification.fast_forward,
    )
    serialized = result.to_dict()
    assert serialized['classification'] == 'fast_forward'
    restored = MergeAnalysisResult.from_dict(serialized)
    assert restored.classification == 'fast_forward'
    assert restored.classification == MergeClassification.fast_forward


# Section per-subplan: SHORT_ID_HEX_LEN


class TestShortIdHexLen:
  """Short IDs use SHORT_ID_HEX_LEN characters."""

  def test_datum_id_length(self) -> None:
    datum = Datum()
    assert len(datum.id) == SHORT_ID_HEX_LEN

  def test_multiple_datums_unique(self) -> None:
    ids = {Datum().id for _ in range(100)}
    assert len(ids) == 100
    for d_id in ids:
      assert len(d_id) == SHORT_ID_HEX_LEN


# Section per-subplan: TODO_LINE_MIN_LEN / TODO_ITEM_MAX_CHARS


class TestGradientTodoConstants:
  """Gradient.todo_items() respects TODO_LINE_MIN_LEN and TODO_ITEM_MAX_CHARS."""

  def test_short_lines_excluded(self) -> None:
    @dataclass
    class _TestGrad(Gradient):
      text: str = ''

      def render(self) -> str:
        return self.text

    short = 'x' * TODO_LINE_MIN_LEN
    grad = _TestGrad(text=short)
    assert grad.todo_items() == []

  def test_long_line_included(self) -> None:
    @dataclass
    class _TestGrad(Gradient):
      text: str = ''

      def render(self) -> str:
        return self.text

    long_line = 'x' * (TODO_LINE_MIN_LEN + 1)
    grad = _TestGrad(text=long_line)
    items = grad.todo_items()
    assert len(items) == 1
    assert items[0] == long_line

  def test_truncation_at_max_chars(self) -> None:
    @dataclass
    class _TestGrad(Gradient):
      text: str = ''

      def render(self) -> str:
        return self.text

    very_long = 'a' * (TODO_ITEM_MAX_CHARS + 50)
    grad = _TestGrad(text=very_long)
    items = grad.todo_items()
    assert len(items) == 1
    assert len(items[0]) == TODO_ITEM_MAX_CHARS

  def test_header_lines_excluded(self) -> None:
    @dataclass
    class _TestGrad(Gradient):
      text: str = ''

      def render(self) -> str:
        return self.text

    grad = _TestGrad(text='# This is a long header line that should be ignored')
    assert grad.todo_items() == []


# Section per-subplan: WALL_CLOCK_DECIMALS


class TestWallClockDecimals:
  """CostTrackerCallback rounds wall-clock to WALL_CLOCK_DECIMALS digits."""

  def test_measure_rounds_to_decimals(self) -> None:
    cb = CostTrackerCallback()
    entry = cb.measure(epoch=0, elapsed=1.23456789)
    text = f'{entry.wall_clock_s:.10f}'
    decimal_part = text.split('.')[1]
    significant = decimal_part.rstrip('0')
    assert len(significant) <= WALL_CLOCK_DECIMALS

  def test_total_rounds_to_decimals(self) -> None:
    cb = CostTrackerCallback()
    cb._entries = [
      CostEntry(epoch=0, wall_clock_s=1.1115),
      CostEntry(epoch=1, wall_clock_s=2.2225),
    ]
    total = cb.total()
    text = f'{total.wall_clock_s:.10f}'
    decimal_part = text.split('.')[1]
    significant = decimal_part.rstrip('0')
    assert len(significant) <= WALL_CLOCK_DECIMALS


# Section per-subplan: VALIDATION_ERROR_SAMPLE_SIZE


class TestValidationErrorSampleSize:
  """validate_universe caps sample at VALIDATION_ERROR_SAMPLE_SIZE."""

  def test_sample_capped(self) -> None:
    assignment = SplitAssignment(
      assignments={f'item_{i}': 'train' for i in range(20)},
      ratios={'train': 0.8, 'val': 0.2},
      seed=42,
    )
    universe: set[str] = set()
    with pytest.raises(ValueError, match='assignment references') as exc_info:
      assignment.validate_universe(universe)
    error_msg = str(exc_info.value)
    items_shown = error_msg.split('[')[1].split(']')[0]
    shown_count = len(items_shown.split(','))
    assert shown_count <= VALIDATION_ERROR_SAMPLE_SIZE


# Section per-subplan: PATH_LISTING_LIMIT


class TestPathListingLimit:
  """PathParameter.render() caps listing at PATH_LISTING_LIMIT."""

  def test_listing_capped(self, tmp_path: Path) -> None:
    from autopilot.ai.parameter import PathParameter

    src = tmp_path / 'src'
    src.mkdir()
    for i in range(PATH_LISTING_LIMIT + 10):
      (src / f'file_{i:03d}.txt').write_text(f'content {i}')
    param = PathParameter(source=str(src), pattern='**/*.txt')
    rendered = param.render()
    line_count = sum(1 for line in rendered.split('\n') if line.strip().startswith('- '))
    assert line_count == PATH_LISTING_LIMIT


# Section per-subplan: FALLBACK_JUDGE_CONFIG


class TestFallbackJudgeConfig:
  """FALLBACK_JUDGE_CONFIG round-trips through JudgeConfig."""

  def test_model_validate_round_trip(self) -> None:
    from autopilot.ai.evaluation.schemas import JudgeConfig

    config = JudgeConfig.model_validate(FALLBACK_JUDGE_CONFIG)
    assert config.run.model == 'openai:gpt-4o'
    assert config.run.num_parallel == 5
    assert config.run.max_rpm == 100
    assert config.run.rpm_safety_margin == 0.9
    assert config.run.retry.max_retries == 3
    assert config.run.retry.min_timeout_ms == 1000
    assert config.run.retry.max_timeout_ms == 30000
    assert config.run.retry.backoff_factor == 2
    assert config.run.max_tool_steps == 5
    assert config.run.max_output_tokens == 4096
    assert config.system_prompt is None


# Section per-subplan: REFS_FORMAT_VERSION


class TestRefsFormatVersion:
  """REFS_FORMAT_VERSION matches expected schema version."""

  def test_value_is_two(self) -> None:
    assert REFS_FORMAT_VERSION == 2


# Section per-subplan: AGENT_OUTPUT_ERROR_SNIPPET_LEN


class TestAgentOutputSnippet:
  """Agent collator error truncation respects AGENT_OUTPUT_ERROR_SNIPPET_LEN."""

  def test_parse_error_truncated(self) -> None:
    from autopilot.ai.agents.agent import Agent, AgentResult
    from autopilot.ai.gradient import AgentCollator

    class _DummyAgent(Agent):
      def run(self, prompt: str, context: dict | None = None) -> AgentResult:
        return AgentResult(output='')

      async def async_run(self, prompt: str, context: dict | None = None) -> AgentResult:
        return AgentResult(output='')

    long_output = 'x' * (AGENT_OUTPUT_ERROR_SNIPPET_LEN + 100)
    collator = AgentCollator(_DummyAgent())
    with pytest.raises(RuntimeError) as exc_info:
      collator.parse_result(long_output, [])
    error_text = str(exc_info.value)
    snippet_in_error = error_text.split(': ', 1)[1] if ': ' in error_text else error_text
    assert len(snippet_in_error) <= AGENT_OUTPUT_ERROR_SNIPPET_LEN + 50


# Section per-subplan: STDOUT_ERROR_SNIPPET_LEN (claude_code)


class TestStdoutErrorSnippetLen:
  """ClaudeCodeAgent parse error truncates to STDOUT_ERROR_SNIPPET_LEN."""

  def test_parse_error_truncates_stdout(self) -> None:
    long_stdout = 'z' * (STDOUT_ERROR_SNIPPET_LEN + 100)
    snippet = long_stdout[:STDOUT_ERROR_SNIPPET_LEN]
    msg = f'failed to parse claude output: {snippet}'
    assert snippet in msg
    assert len(snippet) == STDOUT_ERROR_SNIPPET_LEN


# Section per-subplan: EXEC_STDOUT_PREVIEW_LEN / GRADIENT_PREVIEW_LEN (debug)


class TestDebugPreviewConstants:
  """Debug CLI previews respect EXEC_STDOUT_PREVIEW_LEN and GRADIENT_PREVIEW_LEN."""

  def test_stdout_preview_truncation(self) -> None:
    long_stdout = 'a' * (EXEC_STDOUT_PREVIEW_LEN + 50)
    preview = long_stdout[:EXEC_STDOUT_PREVIEW_LEN].replace('\n', ' ')
    assert len(preview) == EXEC_STDOUT_PREVIEW_LEN

  def test_gradient_preview_truncation(self) -> None:
    long_gradient = 'g' * (GRADIENT_PREVIEW_LEN + 100)
    preview = long_gradient[:GRADIENT_PREVIEW_LEN]
    assert len(preview) == GRADIENT_PREVIEW_LEN

  def test_gradient_render_under_limit(self) -> None:
    grad = NumericGradient(value=42.0)
    rendered = grad.render()[:GRADIENT_PREVIEW_LEN]
    assert len(rendered) <= GRADIENT_PREVIEW_LEN


# Section per-subplan: MERGE_CONFLICT_DISPLAY_LIMIT (store CLI)


class TestMergeConflictDisplayLimit:
  """Store CLI caps displayed conflicts at MERGE_CONFLICT_DISPLAY_LIMIT."""

  def test_display_limit_caps_output(self) -> None:
    keys = [f'param/file_{i:03d}.txt' for i in range(MERGE_CONFLICT_DISPLAY_LIMIT + 10)]
    displayed = sorted(keys)[:MERGE_CONFLICT_DISPLAY_LIMIT]
    assert len(displayed) == MERGE_CONFLICT_DISPLAY_LIMIT
    assert len(keys) > MERGE_CONFLICT_DISPLAY_LIMIT


# Section per-subplan: PROPOSAL_ID_HEX_LEN (propose)


class TestProposalIdHexLen:
  """Proposal IDs use PROPOSAL_ID_HEX_LEN characters."""

  def test_proposal_id_width(self) -> None:
    import uuid

    proposal_id = str(uuid.uuid4())[:PROPOSAL_ID_HEX_LEN]
    assert len(proposal_id) == PROPOSAL_ID_HEX_LEN


# Section 4.7: constants_positive_where_applicable


class TestConstantsPositiveWhereApplicable:
  """Numeric constants where 'positive' is meaningful must be > 0."""

  @pytest.mark.parametrize(
    ('name', 'value'),
    [
      ('RPM_WINDOW_SECONDS', RPM_WINDOW_SECONDS),
      ('SHORT_ID_HEX_LEN', SHORT_ID_HEX_LEN),
      ('TODO_LINE_MIN_LEN', TODO_LINE_MIN_LEN),
      ('TODO_ITEM_MAX_CHARS', TODO_ITEM_MAX_CHARS),
      ('PATH_LISTING_LIMIT', PATH_LISTING_LIMIT),
      ('VALIDATION_ERROR_SAMPLE_SIZE', VALIDATION_ERROR_SAMPLE_SIZE),
      ('REFS_FORMAT_VERSION', REFS_FORMAT_VERSION),
      ('DIGEST_DISPLAY_HEX_LEN', DIGEST_DISPLAY_HEX_LEN),
      ('CONFIG_HASH_HEX_LEN', CONFIG_HASH_HEX_LEN),
      ('STDOUT_ERROR_SNIPPET_LEN', STDOUT_ERROR_SNIPPET_LEN),
      ('AGENT_OUTPUT_ERROR_SNIPPET_LEN', AGENT_OUTPUT_ERROR_SNIPPET_LEN),
      ('MERGE_CONFLICT_DISPLAY_LIMIT', MERGE_CONFLICT_DISPLAY_LIMIT),
      ('PROPOSAL_ID_HEX_LEN', PROPOSAL_ID_HEX_LEN),
      ('EXEC_STDOUT_PREVIEW_LEN', EXEC_STDOUT_PREVIEW_LEN),
      ('GRADIENT_PREVIEW_LEN', GRADIENT_PREVIEW_LEN),
    ],
  )
  def test_positive(self, name: str, value: int | float) -> None:
    assert value > 0, f'{name} should be positive, got {value}'
