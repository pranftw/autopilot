"""Tests for evaluation tier scenario JSONL files and judge config JSON files.

Validates:
  - smoke.jsonl: 5-10 lines, valid JSON, required keys present.
  - regression.jsonl: >=3 lines, valid JSON, required keys.
  - safety.jsonl: >=2 lines, valid JSON, required keys, string task_ids.
  - judge_config.json, judge_smoke.json, judge_regression.json, judge_safety.json:
    valid JSON; ``JudgeConfig.model_validate`` succeeds; ``run.model`` non-empty;
    ``custom['scenario_input']`` ends in ``.jsonl``.
"""

from autopilot.ai.evaluation.schemas import JudgeConfig
from pathlib import Path
import json
import pytest

HARNESS_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = HARNESS_ROOT / 'harness' / 'scenarios'
CONFIGS_DIR = HARNESS_ROOT / 'configs'

REQUIRED_SCENARIO_KEYS = frozenset(
  {
    'task_id',
    'initial_message',
    'user_instructions',
    'evaluation_criteria',
  }
)

JUDGE_CONFIG_FILES = (
  'judge_config.json',
  'judge_smoke.json',
  'judge_regression.json',
  'judge_safety.json',
)

TIER_SCENARIO_PAIRINGS = {
  'judge_config.json': 'harness/scenarios/val.jsonl',
  'judge_smoke.json': 'harness/scenarios/smoke.jsonl',
  'judge_regression.json': 'harness/scenarios/regression.jsonl',
  'judge_safety.json': 'harness/scenarios/safety.jsonl',
}

TIER_NAMES = {
  'judge_config.json': 'full',
  'judge_smoke.json': 'smoke',
  'judge_regression.json': 'regression',
  'judge_safety.json': 'safety',
}

MIN_SMOKE_LINES = 5
MAX_SMOKE_LINES = 10
MIN_REGRESSION_LINES = 3
MIN_SAFETY_LINES = 2


def _load_jsonl(path: Path) -> list[dict]:
  """Load a JSONL file and return a list of parsed dicts."""
  entries = []
  text = path.read_text(encoding='utf-8')
  for line in text.splitlines():
    stripped = line.strip()
    if stripped:
      entries.append(json.loads(stripped))
  return entries


# -- scenario JSONL contracts -------------------------------------------------


class TestSmokeScenarios:
  """Contract tests for harness/scenarios/smoke.jsonl."""

  def test_smoke_scenarios_exist(self) -> None:
    """smoke.jsonl exists, has 5-10 lines, each parses and has required keys."""
    path = SCENARIOS_DIR / 'smoke.jsonl'
    assert path.exists(), f'missing {path}'
    entries = _load_jsonl(path)
    assert MIN_SMOKE_LINES <= len(entries) <= MAX_SMOKE_LINES, (
      f'expected {MIN_SMOKE_LINES}-{MAX_SMOKE_LINES} entries, got {len(entries)}'
    )
    for i, entry in enumerate(entries):
      missing = REQUIRED_SCENARIO_KEYS - set(entry.keys())
      assert not missing, f'line {i}: missing keys {missing}'


class TestRegressionScenarios:
  """Contract tests for harness/scenarios/regression.jsonl."""

  def test_regression_scenarios_exist(self) -> None:
    """regression.jsonl exists, has >=3 lines, each parses and has required keys."""
    path = SCENARIOS_DIR / 'regression.jsonl'
    assert path.exists(), f'missing {path}'
    entries = _load_jsonl(path)
    assert len(entries) >= MIN_REGRESSION_LINES, (
      f'expected >={MIN_REGRESSION_LINES} entries, got {len(entries)}'
    )
    for i, entry in enumerate(entries):
      missing = REQUIRED_SCENARIO_KEYS - set(entry.keys())
      assert not missing, f'line {i}: missing keys {missing}'


class TestSafetyScenarios:
  """Contract tests for harness/scenarios/safety.jsonl."""

  def test_safety_scenarios_exist(self) -> None:
    """safety.jsonl exists, has >=2 lines, each parses and has required keys."""
    path = SCENARIOS_DIR / 'safety.jsonl'
    assert path.exists(), f'missing {path}'
    entries = _load_jsonl(path)
    assert len(entries) >= MIN_SAFETY_LINES, (
      f'expected >={MIN_SAFETY_LINES} entries, got {len(entries)}'
    )
    for i, entry in enumerate(entries):
      missing = REQUIRED_SCENARIO_KEYS - set(entry.keys())
      assert not missing, f'line {i}: missing keys {missing}'
      assert isinstance(entry['task_id'], str), (
        f'line {i}: task_id must be str, got {type(entry["task_id"]).__name__}'
      )


# -- judge config contracts ---------------------------------------------------


class TestJudgeConfigs:
  """Contract tests for configs/judge_*.json files."""

  def test_judge_configs_valid_json(self) -> None:
    """All four judge config files exist and parse as valid JSON."""
    for name in JUDGE_CONFIG_FILES:
      path = CONFIGS_DIR / name
      assert path.exists(), f'missing {path}'
      data = json.loads(path.read_text(encoding='utf-8'))
      assert isinstance(data, dict), f'{name}: expected dict, got {type(data).__name__}'

  def test_judge_configs_have_required_fields(self) -> None:
    """JudgeConfig validates; run.model non-empty; custom.scenario_input matches tier."""
    for name in JUDGE_CONFIG_FILES:
      path = CONFIGS_DIR / name
      data = json.loads(path.read_text(encoding='utf-8'))
      config = JudgeConfig.model_validate(data)
      assert config.run.model, f'{name}: run.model is empty'
      custom = data['custom']
      assert isinstance(custom, dict), f'{name}: custom must be a dict'
      scenario_input = custom['scenario_input']
      assert isinstance(scenario_input, str) and scenario_input.endswith('.jsonl'), (
        f'{name}: custom.scenario_input must be a non-empty string ending in .jsonl, '
        f'got {scenario_input!r}'
      )

  @pytest.mark.parametrize('name', JUDGE_CONFIG_FILES)
  def test_judge_config_scenario_pairing(self, name: str) -> None:
    """Each config's custom.scenario_input matches the expected tier file."""
    path = CONFIGS_DIR / name
    data = json.loads(path.read_text(encoding='utf-8'))
    expected_input = TIER_SCENARIO_PAIRINGS[name]
    assert data['custom']['scenario_input'] == expected_input, (
      f'{name}: expected scenario_input={expected_input!r}, '
      f'got {data["custom"]["scenario_input"]!r}'
    )

  @pytest.mark.parametrize('name', JUDGE_CONFIG_FILES)
  def test_judge_config_tier_and_rubric(self, name: str) -> None:
    """Each config has custom.tier matching expected name and rubric_dimensions list."""
    path = CONFIGS_DIR / name
    data = json.loads(path.read_text(encoding='utf-8'))
    custom = data['custom']
    expected_tier = TIER_NAMES[name]
    assert custom['tier'] == expected_tier, (
      f'{name}: expected tier={expected_tier!r}, got {custom["tier"]!r}'
    )
    dims = custom['rubric_dimensions']
    assert isinstance(dims, list) and len(dims) > 0, (
      f'{name}: rubric_dimensions must be a non-empty list, got {dims!r}'
    )

  def test_safety_config_has_system_prompt(self) -> None:
    """Safety config should have a non-null system_prompt for strict scoring."""
    path = CONFIGS_DIR / 'judge_safety.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    config = JudgeConfig.model_validate(data)
    assert config.system_prompt is not None, (
      'judge_safety.json should have a system_prompt for safety-focused evaluation'
    )
    assert 'policy_compliance' in config.system_prompt


class TestSmokeSubset:
  """Verify that smoke.jsonl task_ids are a subset of val.jsonl task_ids."""

  def test_smoke_is_subset_of_val(self):
    smoke_ids = {e['task_id'] for e in _load_jsonl(SCENARIOS_DIR / 'smoke.jsonl')}
    val_ids = {e['task_id'] for e in _load_jsonl(SCENARIOS_DIR / 'val.jsonl')}
    assert smoke_ids <= val_ids, f'smoke task_ids not in val: {smoke_ids - val_ids}'
