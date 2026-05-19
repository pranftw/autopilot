"""Dogfood V6 final verification audit integration tests (sub-plan 08).

Nine contract tests covering:
  1. select clean-break: no legacy ``select(0,`` pattern in src/ or tests/
  2. TextGradient rename: no ``direction=`` kwarg in gradient.py
  3. Profiler smoke: Trainer(profiler=SimpleProfiler()) minimal fit
  4. Resume tokens: fit(ckpt_path='last') and fit(ckpt_path='best') resolve
  5. Full test count: collected test count >= baseline
  6. CLAUDE.md / AGENTS.md byte-parity
  7. OnExceptionCallback smoke: crash checkpoint created on fit failure
  8. Reflog expire smoke: store reflog expire --older-than 30d --dry-run JSON
  9. Tag verify smoke: store tag verify returns valid JSON with verified field
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.callbacks.checkpoint import CheckpointCallback
from autopilot.core.callbacks.on_exception import OnExceptionCallback
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.profiler import SimpleProfiler
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.tree import Node
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
from tests.integration.doubles import minimal_trainer_stack
import contextlib
import json
import re

REPO_ROOT = Path(__file__).resolve().parents[2]

SELECT_OLD_PATTERN = re.compile(r'select\(\s*0\s*,')

MINIMUM_TEST_COUNT = 800


def _build_workspace_with_store(
  tmp_path: Path,
) -> tuple[AutoPilotConfig, PathParameter, FileStore, FileForest]:
  """Build a workspace with config, store, and forest for smoke tests.

  Returns:
    Tuple of (config, path_param, store, forest).
  """
  workspace = tmp_path / 'workspace'
  workspace.mkdir()
  params_dir = workspace / 'params'
  params_dir.mkdir()
  (params_dir / 'seed.txt').write_text('seed', encoding='utf-8')

  config = AutoPilotConfig(workspace=workspace)
  path_param = PathParameter(source=str(params_dir), pattern='*.txt')
  store = FileStore(config)
  store.register_parameters({'source': path_param})
  forest = FileForest(store)
  return config, path_param, store, forest


def _seed_experiment(
  forest: FileForest,
  experiment_id: str,
  hypothesis: str,
) -> tuple[Experiment, Node]:
  """Create and add an experiment to the active tree.

  Returns:
    Tuple of (experiment, node).
  """
  tree = forest.active
  assert tree is not None, 'no active tree; call create_tree/switch first'
  exp = Experiment(experiment_id=experiment_id, hypothesis=hypothesis)
  node = Node(experiment=exp)
  tree.add(node)
  forest.save()
  return exp, node


CLEAN_BREAK_ALLOWED_FILES = frozenset(
  {
    'tests/core/test_operator_convention.py',
    'tests/api/test_api_contracts.py',
    'tests/integration/test_repo_contract_and_profiler_smoke.py',
  }
)


class TestSelectCleanBreakNoOldPattern:
  """Scan src/ Python sources for legacy select(0, calls (clean break)."""

  def test_select_clean_break_no_old_pattern(self) -> None:
    """No production code should use the old select(0, pattern."""
    violations: list[str] = []
    for py_file in (REPO_ROOT / 'src').rglob('*.py'):
      text = py_file.read_text(encoding='utf-8')
      for line_no, line in enumerate(text.splitlines(), start=1):
        if SELECT_OLD_PATTERN.search(line):
          violations.append(f'{py_file.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}')
    assert not violations, 'Legacy select(0, pattern found in src/:\n' + '\n'.join(violations)

  def test_select_clean_break_tests_only_migration(self) -> None:
    """Test files using select(0, must be migration-error test files only."""
    violations: list[str] = []
    for py_file in (REPO_ROOT / 'tests').rglob('*.py'):
      rel = str(py_file.relative_to(REPO_ROOT))
      if rel in CLEAN_BREAK_ALLOWED_FILES:
        continue
      text = py_file.read_text(encoding='utf-8')
      for line_no, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith('#') or line.lstrip().startswith('"""'):
          continue
        if SELECT_OLD_PATTERN.search(line):
          violations.append(f'{rel}:{line_no}: {line.strip()}')
    assert not violations, 'Legacy select(0, in non-migration test files:\n' + '\n'.join(violations)


class TestTextGradientRenameComplete:
  """Ensure direction= kwarg does not appear in gradient.py source."""

  def test_text_gradient_rename_complete(self) -> None:
    """src/autopilot/ai/gradient.py must not use direction= as a kwarg definition.

    Excludes lines where direction= appears only inside string literals
    (e.g. migration error messages) or string-key checks.
    """
    gradient_py = REPO_ROOT / 'src' / 'autopilot' / 'ai' / 'gradient.py'
    text = gradient_py.read_text(encoding='utf-8')
    direction_kwarg = re.compile(r'(?<!\w)direction\s*=')
    string_context = re.compile(r'["\'].*direction.*["\']')
    matches: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
      if direction_kwarg.search(line):
        stripped = line.strip()
        if string_context.search(stripped):
          continue
        if stripped.startswith('#'):
          continue
        matches.append(f'  line {line_no}: {stripped}')
    assert not matches, 'TextGradient rename incomplete -- direction= still used:\n' + '\n'.join(
      matches
    )


class TestProfilerSmoke:
  """Trainer(profiler=SimpleProfiler()) minimal fit completes without error."""

  def test_profiler_smoke(self, tmp_path: Path) -> None:
    """Profiler records timing data during a minimal fit run."""
    config, path_param, store, forest = _build_workspace_with_store(tmp_path)
    tree = forest.create_tree('main')
    forest.switch('main')
    exp, _ = _seed_experiment(forest, 'profiler-exp', 'profiler smoke')

    module, datamodule = minimal_trainer_stack(path_param)
    profiler = SimpleProfiler()
    trainer = Trainer(
      config=config,
      experiment=exp,
      store=store,
      tree=tree,
      profiler=profiler,
    )
    trainer.fit(module, datamodule=datamodule, max_epochs=1)

    summary = profiler.describe()
    assert len(summary) > 0, 'profiler should have recorded at least one section'
    assert any(key in summary for key in ('training_step', 'backward', 'optimizer_step')), (
      f'expected training sections in profiler, got: {list(summary.keys())}'
    )


def _first_fit_with_checkpoint(
  tmp_path: Path,
  token: str,
  accuracy: float,
) -> tuple[AutoPilotConfig, PathParameter, CheckpointCallback]:
  """Run first fit with a CheckpointCallback, return state for resume.

  Args:
    tmp_path: Pytest temp directory.
    token: Resume token label for experiment naming.
    accuracy: Accuracy for the FixedAccuracyMetric.

  Returns:
    Tuple for the second fit call (config, param, ckpt_cb).
  """
  config, path_param, _store, _forest = _build_workspace_with_store(tmp_path)

  ckpt_dir = tmp_path / 'checkpoints'
  ckpt_dir.mkdir()
  ckpt_cb = CheckpointCallback(directory=ckpt_dir, monitor='AccuracyMetric')

  exp = Experiment(experiment_id=f'{token}-exp', hypothesis=f'{token} smoke')
  module, datamodule = minimal_trainer_stack(path_param, accuracy=accuracy)
  trainer = Trainer(config=config, experiment=exp, callbacks=[ckpt_cb])
  trainer.fit(module, datamodule=datamodule, max_epochs=1)
  return config, path_param, ckpt_cb


def _resume_fit(
  config: AutoPilotConfig,
  path_param: PathParameter,
  ckpt_cb: CheckpointCallback,
  token: str,
  accuracy: float,
) -> Trainer:
  """Resume from a checkpoint token after the first fit.

  Args:
    config: Workspace configuration.
    path_param: Path parameter for the module.
    ckpt_cb: CheckpointCallback from the first fit.
    token: Resume token ('last' or 'best').
    accuracy: Accuracy for the FixedAccuracyMetric.

  Returns:
    The Trainer instance after fit completes.
  """
  exp2 = Experiment(experiment_id=f'{token}-exp-2', hypothesis=f'{token} resume')
  module2, dm2 = minimal_trainer_stack(path_param, accuracy=accuracy)
  trainer2 = Trainer(config=config, experiment=exp2, callbacks=[ckpt_cb])
  trainer2.fit(module2, datamodule=dm2, max_epochs=2, ckpt_path=token)
  return trainer2


class TestResumeTokens:
  """fit(ckpt_path='last') and fit(ckpt_path='best') resolve correctly."""

  def test_resume_token_last(self, tmp_path: Path) -> None:
    """ckpt_path='last' resolves to the last-saved checkpoint."""
    config, path_param, ckpt_cb = _first_fit_with_checkpoint(
      tmp_path,
      'last',
      accuracy=0.5,
    )
    assert ckpt_cb.last_checkpoint_path is not None
    assert ckpt_cb.last_checkpoint_path.exists()
    trainer = _resume_fit(config, path_param, ckpt_cb, 'last', accuracy=0.5)
    assert trainer is not None
    assert trainer.current_epoch == 1

  def test_resume_token_best(self, tmp_path: Path) -> None:
    """ckpt_path='best' resolves to the best-monitored checkpoint."""
    config, path_param, ckpt_cb = _first_fit_with_checkpoint(
      tmp_path,
      'best',
      accuracy=0.9,
    )
    assert ckpt_cb.best_checkpoint_path is not None
    assert ckpt_cb.best_checkpoint_path.exists()
    trainer = _resume_fit(config, path_param, ckpt_cb, 'best', accuracy=0.9)
    assert trainer is not None
    assert trainer.current_epoch == 1


class TestFullTestCount:
  """Collected test count >= documented baseline.

  Maintenance note: update MINIMUM_TEST_COUNT when the suite grows
  substantially (e.g. after a new master plan). The current baseline
  reflects the V6 final state.
  """

  def test_full_test_count(self) -> None:
    """Total test count meets minimum baseline by scanning test files in-process."""
    test_dir = REPO_ROOT / 'tests'
    test_pattern = re.compile(r'^\s*def (test_\w+)\(')
    count = 0
    for py_file in test_dir.rglob('test_*.py'):
      text = py_file.read_text(encoding='utf-8')
      for line in text.splitlines():
        if test_pattern.match(line):
          count += 1
    assert count >= MINIMUM_TEST_COUNT, (
      f'Test function count {count} below minimum baseline {MINIMUM_TEST_COUNT}. '
      f'Expected at least {MINIMUM_TEST_COUNT} test functions after V6.'
    )


class TestClaudeMdAgentsMdIdenticalV6:
  """CLAUDE.md and AGENTS.md must be byte-for-byte identical."""

  def test_claude_md_agents_md_identical(self) -> None:
    """Read both files and assert string equality."""
    claude_md = (REPO_ROOT / 'CLAUDE.md').read_text(encoding='utf-8')
    agents_md = (REPO_ROOT / 'AGENTS.md').read_text(encoding='utf-8')
    assert claude_md == agents_md, (
      'CLAUDE.md and AGENTS.md are not identical. '
      'CLAUDE.md is canonical; AGENTS.md must mirror it byte-for-byte.'
    )


class FailingModule(AutoPilotModule):
  """Module that raises on training_step for crash testing."""

  def __init__(self, pp: PathParameter) -> None:
    super().__init__()
    self.param = pp

  def forward(self, batch: Datum) -> Datum:
    return batch

  def training_step(self, batch: Datum, batch_idx: int) -> Datum:
    msg = 'intentional crash'
    raise RuntimeError(msg)

  def configure_optimizers(self):
    return None


class OneItemDM(DataModule):
  """DataModule with one train batch for crash testing."""

  def train_dataloader(self) -> DataLoader:
    return DataLoader([EvalDatum(success=True)], batch_size=1)


class TestOnExceptionCallbackSmoke:
  """OnExceptionCallback triggers on fit failure, crash checkpoint exists."""

  def test_on_exception_callback_smoke(self, tmp_path: Path) -> None:
    """Crash checkpoint is written when fit raises."""
    config, path_param, store, forest = _build_workspace_with_store(tmp_path)
    tree = forest.create_tree('main')
    forest.switch('main')
    exp, _ = _seed_experiment(forest, 'crash-exp', 'crash test')

    crash_dir = tmp_path / 'crash'
    crash_dir.mkdir()
    on_exc_cb = OnExceptionCallback(directory=crash_dir)

    module = FailingModule(path_param)
    datamodule = OneItemDM()
    trainer = Trainer(
      config=config,
      experiment=exp,
      store=store,
      tree=tree,
      callbacks=[on_exc_cb],
    )

    with contextlib.suppress(RuntimeError):
      trainer.fit(module, datamodule=datamodule, max_epochs=1)

    crash_path = crash_dir / 'crash_checkpoint.json'
    assert crash_path.exists(), (
      f'OnExceptionCallback did not create crash checkpoint at {crash_path}'
    )
    crash_content = crash_path.read_text(encoding='utf-8')
    crash_data = json.loads(crash_content)
    assert isinstance(crash_data, dict), (
      f'Crash checkpoint should be a JSON dict, got {type(crash_data).__name__}'
    )
    assert len(crash_data) > 0, 'Crash checkpoint dict should not be empty'


class TestReflogExpireSmoke:
  """store reflog expire --older-than 30d --dry-run produces valid JSON."""

  def test_reflog_expire_smoke(self, tmp_path: Path) -> None:
    """Dry-run expire returns a valid JSON envelope."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    src = ws / 'src'
    src.mkdir()
    (src / 'main.py').write_text('print("hello")\n')

    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    param = PathParameter(source=str(src), pattern='**/*.py')
    store = FileStore(config)
    store.register_parameters({'source': param})
    forest = FileForest(store)
    forest.save()

    store.snapshot('exp-1', 0)

    result = run_cli(
      ws,
      ['store', 'reflog', 'expire', '--older-than', '30d', '--dry-run'],
    )
    assert result['ok'] is True
    assert 'result' in result
    assert 'expired_count' in result['result']


class TestTagVerifySmoke:
  """store tag verify returns valid JSON with verified field."""

  def test_tag_verify_smoke(self, tmp_path: Path) -> None:
    """Tag verify on a valid tag returns verified=True."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    src = ws / 'src'
    src.mkdir()
    (src / 'main.py').write_text('print("hello")\n')

    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    param = PathParameter(source=str(src), pattern='**/*.py')
    store = FileStore(config)
    store.register_parameters({'source': param})
    forest = FileForest(store)
    forest.save()

    store.snapshot('exp-1', 0)
    store.tag('v1.0', 'exp-1', 0, context='release')

    result = run_cli_no_context(ws, ['store', 'tag', 'verify', 'v1.0'])
    assert result['ok'] is True
    assert 'result' in result
    assert 'verified' in result['result']
    assert result['result']['verified'] is True
