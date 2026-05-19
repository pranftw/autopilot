"""Integration audit tests for plan 11 (dry-run, no API calls).

Verifies checklist items A through N and end-to-end construction of
both judge and heuristic modes without live LLM calls.
"""

from autopilot.ai.gradient import AgentCollator
from autopilot.ai.loss import JudgeLoss
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.loops.orchestrator import EpochOrchestrator
from harness.agent import ConversationResult
from harness.agents import PydanticAgent
from harness.callbacks import (
  HarnessCostTrackerCallback,
  MetricsWriterCallback,
  OptimizerContextCallback,
)
from harness.cli import HarnessCLI
from harness.judge import HarnessJudge
from harness.loss import HarnessLoss
from harness.module import HarnessModule
from harness.trainer import build_trainer
from pathlib import Path
import json
import pytest

STUB_TOOLS_CODE = '''\
def calculate(ctx, expression):
  """A stub tool."""
  return str(expression)
'''


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
  """Create a minimal project tree for build_trainer and CLI tests."""
  root = tmp_path / 'project'
  root.mkdir()
  harness = root / 'harness'
  harness.mkdir()
  prompts = harness / 'prompts'
  prompts.mkdir()
  (prompts / 'system_prompt.md').write_text('You are a test agent.', encoding='utf-8')
  (prompts / 'policies.md').write_text('Be polite.', encoding='utf-8')
  tools = harness / 'tools'
  tools.mkdir()
  (tools / 'retail_tools.py').write_text(STUB_TOOLS_CODE, encoding='utf-8')
  db_dir = harness / 'db'
  db_dir.mkdir()
  db_data = {
    'users': [{'user_id': 'u1', 'name': 'Test', 'email': 'test@test.com'}],
    'orders': [],
    'products': [],
  }
  (db_dir / 'retail.json').write_text(json.dumps(db_data), encoding='utf-8')
  scenarios = harness / 'scenarios'
  scenarios.mkdir()
  record = {
    'task_id': 't0',
    'initial_message': 'hello',
    'user_instructions': {
      'reason_for_call': 'help',
      'known_info': {},
      'task_instructions': 'do something',
    },
    'evaluation_criteria': {
      'expected_actions': [{'tool': 'calculate', 'args': {'expression': '1+1'}}],
      'communicate_info': ['result is 2'],
      'nl_assertions': ['agent was helpful'],
    },
  }
  for split in ('train.jsonl', 'val.jsonl', 'test.jsonl'):
    (scenarios / split).write_text(json.dumps(record) + '\n', encoding='utf-8')
  return root


# -- Checklist A: Imports resolve --


class TestChecklistAImports:
  """All public entrypoints import without error."""

  def test_pydantic_agent_import(self):
    from harness.agents import PydanticAgent

    assert PydanticAgent is not None

  def test_harness_judge_import(self):
    from harness.judge import HarnessVerdict as HV

    assert HarnessJudge is not None
    assert HV is not None

  def test_judge_loss_import(self):
    from autopilot.ai.loss import JudgeLoss

    assert JudgeLoss is not None

  def test_agent_collator_import(self):
    from autopilot.ai.gradient import AgentCollator

    assert AgentCollator is not None

  def test_harness_module_import(self):
    from harness.module import HarnessModule

    assert HarnessModule is not None

  def test_harness_cli_import(self):
    from harness.cli import HarnessCLI

    assert HarnessCLI is not None

  def test_epoch_orchestrator_import(self):
    from autopilot.core.loops.orchestrator import EpochOrchestrator

    assert EpochOrchestrator is not None

  def test_compute_fingerprint_import(self):
    from autopilot.ai import fingerprint as fp_mod

    assert fp_mod.compute_fingerprint is not None


# -- Checklist B: Both module modes --


class TestChecklistBModuleModes:
  """HarnessModule constructs in both judge and heuristic modes."""

  def test_judge_mode_loss_type(self, project_root: Path):
    module = HarnessModule(str(project_root / 'harness'), use_judge=True)
    assert isinstance(module.loss_fn, JudgeLoss)

  def test_heuristic_mode_loss_type(self, project_root: Path):
    module = HarnessModule(str(project_root / 'harness'), use_judge=False)
    assert isinstance(module.loss_fn, HarnessLoss)

  def test_judge_mode_has_judge(self, project_root: Path):
    module = HarnessModule(str(project_root / 'harness'), use_judge=True)
    assert module._judge is not None
    assert isinstance(module._judge, HarnessJudge)

  def test_heuristic_mode_no_judge(self, project_root: Path):
    module = HarnessModule(str(project_root / 'harness'), use_judge=False)
    assert module._judge is None


# -- Checklist C: CLI judge wired --


class TestChecklistCCLIJudge:
  """HarnessCLI has a live HarnessJudge and generator is None."""

  def test_cli_judge_is_harness_judge(self):
    cli = HarnessCLI()
    assert cli.judge is not None
    assert isinstance(cli.judge, HarnessJudge)

  def test_cli_generator_is_none(self):
    cli = HarnessCLI()
    assert cli.generator is None


# -- Checklist D: EpochOrchestrator in trainer --


class TestChecklistDOrchestrator:
  """build_trainer installs EpochOrchestrator as the trainer loop."""

  def test_trainer_loop_is_orchestrator(self, project_root: Path):
    trainer, _, _ = build_trainer(project_root)
    assert isinstance(trainer.loop, EpochOrchestrator)


# -- Checklist E: DatasetFingerprint stamped --


class TestChecklistEFingerprint:
  """build_trainer stamps experiment.dataset_meta with fingerprint data."""

  def test_dataset_meta_populated(self, project_root: Path):
    trainer, _, _ = build_trainer(project_root)
    meta = trainer.experiment.dataset_meta
    assert isinstance(meta, dict)
    assert len(meta) > 0

  def test_dataset_meta_has_expected_keys(self, project_root: Path):
    trainer, _, _ = build_trainer(project_root)
    meta = trainer.experiment.dataset_meta
    assert 'bundle_hash' in meta
    assert 'row_count' in meta


# -- Checklist F: Scenario files --


class TestChecklistFScenarios:
  """Required scenario JSONL files exist under harness/scenarios/."""

  @pytest.fixture
  def harness_scenarios(self) -> Path:
    return Path(__file__).resolve().parent.parent / 'harness' / 'scenarios'

  @pytest.mark.parametrize(
    'filename',
    [
      'train.jsonl',
      'val.jsonl',
      'smoke.jsonl',
      'regression.jsonl',
      'safety.jsonl',
    ],
  )
  def test_scenario_file_exists(self, harness_scenarios: Path, filename: str):
    assert (harness_scenarios / filename).exists(), f'missing {filename}'

  @pytest.mark.parametrize(
    'filename',
    [
      'train.jsonl',
      'val.jsonl',
      'smoke.jsonl',
      'regression.jsonl',
      'safety.jsonl',
    ],
  )
  def test_scenario_file_valid_jsonl(self, harness_scenarios: Path, filename: str):
    path = harness_scenarios / filename
    lines = path.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) > 0, f'{filename} is empty'
    for line in lines:
      json.loads(line)


# -- Checklist G: Judge configs --


class TestChecklistGConfigs:
  """Judge config files parse and contain expected structure."""

  @pytest.fixture
  def configs_dir(self) -> Path:
    return Path(__file__).resolve().parent.parent / 'configs'

  @pytest.mark.parametrize(
    'filename',
    [
      'judge_config.json',
      'judge_smoke.json',
      'judge_regression.json',
      'judge_safety.json',
    ],
  )
  def test_config_parses(self, configs_dir: Path, filename: str):
    path = configs_dir / filename
    assert path.exists(), f'missing {filename}'
    data = json.loads(path.read_text(encoding='utf-8'))
    assert isinstance(data, dict)


# -- Checklist H: AGENT_GUIDE.md sections --


class TestChecklistHAgentGuide:
  """AGENT_GUIDE.md has required sections."""

  @pytest.fixture
  def agent_guide(self) -> str:
    path = Path(__file__).resolve().parent.parent / 'AGENT_GUIDE.md'
    return path.read_text(encoding='utf-8')

  def test_lifecycle_section(self, agent_guide: str):
    assert 'Experiment lifecycle' in agent_guide or 'experiment lifecycle' in agent_guide.lower()

  def test_operational_recipes(self, agent_guide: str):
    assert 'Operational Recipes' in agent_guide or 'Recipe:' in agent_guide

  def test_decision_guide(self, agent_guide: str):
    assert 'Decision Guide' in agent_guide or 'Decision Rubric' in agent_guide


# -- Checklist I: README.md --


class TestChecklistIReadme:
  """README.md has architecture summary and primitives table."""

  @pytest.fixture
  def readme(self) -> str:
    path = Path(__file__).resolve().parent.parent / 'README.md'
    return path.read_text(encoding='utf-8')

  def test_architecture_section(self, readme: str):
    assert '## Architecture' in readme or '## architecture' in readme.lower()

  def test_primitives_table(self, readme: str):
    assert 'Framework Primitives' in readme or 'Harness Role' in readme


# -- Checklist J: Cost attribution --


class TestChecklistJCost:
  """ConversationResult feeds token fields into EvalDatum.metadata."""

  def test_conversation_result_has_token_fields(self):
    result = ConversationResult(
      input_tokens=100,
      output_tokens=50,
      api_calls=3,
    )
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.api_calls == 3

  def test_cost_tracker_callback_exists(self):
    cb = HarnessCostTrackerCallback()
    assert hasattr(cb, 'measure')


# -- Checklist K: max_turns wired --


class TestChecklistKMaxTurns:
  """EnvironmentConfig.max_turns flows into HarnessModule."""

  def test_max_turns_forwarded_in_build_trainer(self, project_root: Path):
    from harness.environments import EnvironmentConfig

    env = EnvironmentConfig(model='test', max_epochs=1, max_turns=7, use_judge=False)
    _, module, _ = build_trainer(project_root, env=env)
    assert module._max_turns == 7


# -- Checklist L: build_trainer callback parity --


class TestChecklistLCallbacks:
  """build_trainer includes essential callbacks."""

  def test_has_metrics_writer(self, project_root: Path):
    trainer, _, _ = build_trainer(project_root)
    cb_types = [type(cb) for cb in trainer.callbacks]
    assert MetricsWriterCallback in cb_types

  def test_has_optimizer_context(self, project_root: Path):
    trainer, _, _ = build_trainer(project_root)
    cb_types = [type(cb) for cb in trainer.callbacks]
    assert OptimizerContextCallback in cb_types

  def test_has_store_checkpoint(self, project_root: Path):
    trainer, _, _ = build_trainer(project_root)
    cb_types = [type(cb) for cb in trainer.callbacks]
    assert StoreCheckpointCallback in cb_types

  def test_has_cost_tracker(self, project_root: Path):
    trainer, _, _ = build_trainer(project_root)
    cb_types = [type(cb) for cb in trainer.callbacks]
    assert HarnessCostTrackerCallback in cb_types


# -- Checklist M: TOOL_NAMES coupling documented --


class TestChecklistMToolNames:
  """TOOL_NAMES coupling is documented."""

  def test_tool_loader_docstring_mentions_tool_names(self):
    import harness.tool_loader as tl

    assert 'TOOL_NAMES' in (tl.__doc__ or '')

  def test_agent_guide_mentions_tool_names(self):
    path = Path(__file__).resolve().parent.parent / 'AGENT_GUIDE.md'
    content = path.read_text(encoding='utf-8')
    assert 'TOOL_NAMES' in content


# -- Checklist N: BUG-001 documented --


class TestChecklistNBug001:
  """BUG-001 PathParameter str source workaround documented."""

  def test_learnings_documents_bug001(self):
    path = Path(__file__).resolve().parent.parent / 'AUTOPILOT_LEARNINGS.md'
    content = path.read_text(encoding='utf-8')
    assert 'BUG-001' in content

  def test_module_uses_str_source(self):
    """HarnessModule passes str to PathParameter source (not Path)."""
    import inspect

    source = inspect.getsource(HarnessModule.__init__)
    assert "f'{root}" in source or 'str(root)' in source or "source=f'" in source


# -- Section 2.6: End-to-end walkthrough (dry-run) --


class TestIntegrationJudgeMode:
  """Full pipeline construction with use_judge=True (no API calls)."""

  def test_module_judge_mode_construction(self, project_root: Path):
    module = HarnessModule(str(project_root / 'harness'), use_judge=True)
    assert isinstance(module.loss_fn, JudgeLoss)
    assert module._judge is not None
    assert isinstance(module._judge, HarnessJudge)

  def test_cli_judge_wired(self):
    cli = HarnessCLI()
    assert isinstance(cli.judge, HarnessJudge)
    assert cli.generator is None

  def test_trainer_judge_mode(self, project_root: Path):
    trainer, module, dm = build_trainer(project_root, use_judge=True)
    assert isinstance(trainer.loop, EpochOrchestrator)
    assert isinstance(module.loss_fn, JudgeLoss)
    assert module._judge is not None

    cb_types = [type(cb) for cb in trainer.callbacks]
    assert MetricsWriterCallback in cb_types
    assert OptimizerContextCallback in cb_types
    assert StoreCheckpointCallback in cb_types

    meta = trainer.experiment.dataset_meta
    assert isinstance(meta, dict)
    assert len(meta) > 0

  def test_judge_loss_collator_wired(self, project_root: Path):
    module = HarnessModule(str(project_root / 'harness'), use_judge=True)
    assert isinstance(module.loss_fn._collator, AgentCollator)
    assert isinstance(module.loss_fn._collator._agent, PydanticAgent)


class TestIntegrationHeuristicMode:
  """Full pipeline construction with use_judge=False (no API calls)."""

  def test_module_heuristic_mode_construction(self, project_root: Path):
    module = HarnessModule(str(project_root / 'harness'), use_judge=False)
    assert isinstance(module.loss_fn, HarnessLoss)
    assert module._judge is None

  def test_trainer_heuristic_mode(self, project_root: Path):
    trainer, module, dm = build_trainer(project_root, use_judge=False)
    assert isinstance(trainer.loop, EpochOrchestrator)
    assert isinstance(module.loss_fn, HarnessLoss)
    assert module._judge is None

    cb_types = [type(cb) for cb in trainer.callbacks]
    assert MetricsWriterCallback in cb_types
    assert OptimizerContextCallback in cb_types
    assert StoreCheckpointCallback in cb_types

  def test_trainer_still_has_fingerprint(self, project_root: Path):
    trainer, _, _ = build_trainer(project_root, use_judge=False)
    meta = trainer.experiment.dataset_meta
    assert isinstance(meta, dict)
    assert len(meta) > 0
