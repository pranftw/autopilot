"""Documentation presence tests.

Lightweight assertions that critical operational guidance is present in
CLAUDE.md / AGENTS.md. Prevents accidental deletion of key documentation
added during the dogfood documentation plan (plan 14), audit fixes (plan 03),
and final docs wave (plan 24).

Plan 17 (API Documentation Sync) adds constructor docstring alignment checks
for critical public classes (section 2.4). These verify that documented
``Args:`` parameter names are a subset of actual ``inspect.signature``
parameters (allowing ``self``, ``*args``, ``**kwargs`` variance).
"""

from pathlib import Path
import inspect
import pytest
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAUDE_MD = (REPO_ROOT / 'CLAUDE.md').read_text()
AGENTS_MD = (REPO_ROOT / 'AGENTS.md').read_text()


# --- existing tests ---


def test_claude_md_documents_checkpoint_callback():
  """CheckpointCallback explicit-attach guidance must be present."""
  assert 'CheckpointCallback' in CLAUDE_MD
  assert 'explicitly attached' in CLAUDE_MD


def test_claude_md_documents_cost_tracker():
  """CostTrackerCallback explicit-attach guidance must be present."""
  assert 'CostTrackerCallback' in CLAUDE_MD
  assert 'cost_summary.json' in CLAUDE_MD


def test_agents_md_documents_accumulate_grad_batches():
  """Cost control / accumulate_grad_batches guidance must be present."""
  assert 'accumulate_grad_batches' in AGENTS_MD


def test_claude_md_documents_cli_command_matrix():
  """CLI command matrix section must be present."""
  assert '## CLI command matrix' in CLAUDE_MD


def test_claude_md_documents_operational_guidance():
  """Operational guidance section must be present."""
  assert '## Operational guidance' in CLAUDE_MD


def test_claude_md_documents_merge_conflict_keys():
  """Merge conflict key documentation must be present."""
  assert 'manifest-relative paths' in CLAUDE_MD


def test_claude_md_documents_pythonunbuffered():
  """PYTHONUNBUFFERED streaming guidance must be present."""
  assert 'PYTHONUNBUFFERED' in CLAUDE_MD


def test_claude_md_documents_execution_tracking_scope():
  """Execution tracking scope (non-CLI not logged) must be documented."""
  assert 'executions.jsonl' in CLAUDE_MD
  assert 'autopilot execute' in CLAUDE_MD


# --- 2.1 CLI matrix accuracy: tree add -> tree create ---


def test_no_tree_add_in_matrix():
  """Neither CLAUDE.md nor AGENTS.md contains `tree add` as a matrix row."""
  assert '| `tree add`' not in CLAUDE_MD
  assert '| `tree add`' not in AGENTS_MD


def test_tree_create_in_matrix():
  """CLAUDE.md contains `tree create` in the CLI matrix."""
  assert 'tree create' in CLAUDE_MD


# --- 2.2 missing commands in CLI matrix ---


def test_matrix_contains_notes_commands():
  """`experiment notes show` and `experiment notes write` appear in the matrix."""
  assert '| `experiment notes show`' in CLAUDE_MD
  assert '| `experiment notes write`' in CLAUDE_MD


def test_matrix_contains_store_diff():
  """`store diff` appears in the CLI matrix."""
  assert '| `store diff`' in CLAUDE_MD


def test_matrix_contains_prune_orphans_separate():
  """`debug store prune-orphans` appears as a separate row with context required."""
  matrix = _extract_section(CLAUDE_MD, '## CLI command matrix')
  for line in matrix.splitlines():
    if '`debug store prune-orphans`' in line:
      assert '| Yes |' in line
      break
  else:
    msg = 'debug store prune-orphans not found as separate matrix row'
    raise AssertionError(msg)


def test_matrix_contains_project_doctor():
  """`project doctor` appears in the CLI matrix."""
  assert '| `project doctor`' in CLAUDE_MD


def test_matrix_contains_all_new_commands():
  """All newly added commands appear in the CLI matrix."""
  required = [
    'propose revert',
    'dataset seed',
    'dataset split',
    'store create',
    'store worktree list',
    'store worktree create',
    'store promote',
  ]
  for cmd in required:
    assert f'| `{cmd}`' in CLAUDE_MD, f'{cmd} missing from CLI matrix'


# --- 2.3 deduplication ---


def test_no_duplicate_dataset_fingerprint():
  """The DatasetFingerprint auto-attaches sentence appears exactly once."""
  search = 'auto-attaches `DataModule.dataset_fingerprint`'
  assert CLAUDE_MD.count(search) == 1


def test_no_duplicate_friction_003():
  """FRICTION-003 appears exactly once in the design principles section."""
  section = _extract_section(CLAUDE_MD, '## Design principles')
  assert section.count('FRICTION-003') == 1


def test_no_duplicate_reset_branch():
  """`Store.reset_branch` appears exactly once in the design principles section."""
  section = _extract_section(CLAUDE_MD, '## Design principles')
  assert section.count('Store.reset_branch') == 1


# --- 2.4 merge-preview cache ---


def test_merge_preview_cache_documented():
  """`merge-preview` ephemeral cache behavior is documented."""
  section = _extract_section(CLAUDE_MD, '### Merge operations')
  assert 'merge-preview' in section
  assert 'ephemeral' in section


# --- 2.5 workspace doctor ---


def test_workspace_doctor_documented():
  """`workspace doctor` assumptions are documented with API-only mention."""
  assert 'API-only workspaces' in CLAUDE_MD
  found = False
  for line in CLAUDE_MD.splitlines():
    if 'workspace doctor' in line and 'API-only' in line:
      found = True
      break
  assert found, 'workspace doctor and API-only not in same line'


# --- 2.6 relationship note ---


def test_doc_relationship_noted():
  """Both files note the CLAUDE.md/AGENTS.md relationship."""
  assert 'AGENTS.md mirrors' in CLAUDE_MD
  assert 'AGENTS.md mirrors' in AGENTS_MD


# --- 4.4 file consistency ---


def test_claude_and_agents_identical():
  """CLAUDE.md and AGENTS.md must have identical content."""
  assert CLAUDE_MD == AGENTS_MD


# --- plan 24: new commands in CLI matrix ---


def test_matrix_contains_lifecycle_commands():
  """Plan 01 lifecycle commands appear in the CLI matrix."""
  required = [
    'experiment complete',
    'experiment fail',
    'experiment cancel',
    'experiment impact',
  ]
  for cmd in required:
    assert f'| `{cmd}`' in CLAUDE_MD, f'{cmd} missing from CLI matrix'


def test_matrix_contains_store_doctor():
  """`store doctor` appears in the CLI matrix as read-only."""
  matrix = _extract_section(CLAUDE_MD, '## CLI command matrix')
  for line in matrix.splitlines():
    if '`store doctor`' in line:
      assert 'No' in line.split('|')[2], 'store doctor should be non-mutating'
      break
  else:
    msg = 'store doctor not found in CLI matrix'
    raise AssertionError(msg)


def test_matrix_contains_workspace_status():
  """`workspace status` appears in the CLI matrix."""
  assert '| `workspace status`' in CLAUDE_MD


def test_matrix_contains_track():
  """`track` appears in the CLI matrix as mutating."""
  matrix = _extract_section(CLAUDE_MD, '## CLI command matrix')
  for line in matrix.splitlines():
    if '`track`' in line:
      assert '| Yes |' in line
      break
  else:
    msg = 'track not found in CLI matrix'
    raise AssertionError(msg)


# --- plan 24: MonotonicGate / BudgetGate documentation ---


def test_monotonic_gate_documented():
  """MonotonicGate is documented in the policy section of CLAUDE.md."""
  assert 'MonotonicGate' in CLAUDE_MD
  assert 'non_decreasing' in CLAUDE_MD
  assert 'non_increasing' in CLAUDE_MD


def test_budget_gate_documented():
  """BudgetGate is documented in the policy section of CLAUDE.md."""
  assert 'BudgetGate' in CLAUDE_MD
  assert 'cost_usd' in CLAUDE_MD


def test_budget_gate_opt_in_wording():
  """CostTracker docs reference BudgetGate as opt-in."""
  assert 'BudgetGate' in CLAUDE_MD
  assert 'opt-in' in CLAUDE_MD.lower()


# --- plan 24: track command documentation ---


def test_track_command_documented():
  """The track command is documented in the agent execution interface."""
  section = _extract_section(CLAUDE_MD, '## Agent execution interface')
  assert 'autopilot track' in section
  assert 'shell=False' in section or 'REMAINDER' in section


# --- plan 24: strip_metric_prefix documented ---


def test_strip_metric_prefix_documented():
  """strip_metric_prefix is mentioned in CLAUDE.md."""
  assert 'strip_metric_prefix' in CLAUDE_MD


# --- plan 24: EpochOrchestrator plateau docs ---


def test_orchestrator_plateau_documented():
  """EpochOrchestrator plateau metric matching is documented."""
  assert 'OrchestratorConfig' in CLAUDE_MD or 'plateau detection' in CLAUDE_MD
  assert 'val_accuracy' in CLAUDE_MD


# --- plan 24: prev metric injection documented ---


def test_prev_metric_injection_documented():
  """_prev_ metric injection is documented in CLAUDE.md."""
  assert '_prev_' in CLAUDE_MD


# --- plan 24: template uses Stage enum ---


def test_project_template_uses_stage_enum():
  """Project data template uses Stage enum, not string."""
  template_path = REPO_ROOT / 'src' / 'autopilot' / 'templates' / 'project' / 'data.py'
  content = template_path.read_text()
  assert 'stage: Stage' in content
  assert 'stage: str' not in content


def test_workspace_template_data_stage_enum():
  """BUG-DFV1-005: workspace-root template must use Stage enum."""
  ws_data = REPO_ROOT / 'templates' / 'project' / 'data.py'
  if not ws_data.exists():
    pytest.skip('workspace template not present')
  content = ws_data.read_text()
  assert 'stage: Stage' in content, 'workspace template must use Stage enum'
  assert 'stage: str' not in content, 'workspace template must not use str for stage'


# --- plan 24: template compilation ---


def test_project_templates_compile():
  """All project template Python files compile after placeholder substitution."""
  template_dir = REPO_ROOT / 'src' / 'autopilot' / 'templates' / 'project'
  for py_file in sorted(template_dir.glob('*.py')):
    source = py_file.read_text()
    filled = source.replace('{name}', 'Example')
    compile(filled, str(py_file), 'exec')


# --- plan 24: context exemptions aligned with CLI matrix ---


def test_exempt_set_aligned_with_matrix_read_only():
  """Every read-only command in the CLI matrix should be in _BASE_CONTEXT_EXEMPT."""
  from autopilot.cli.command import _BASE_CONTEXT_EXEMPT

  matrix = _extract_section(CLAUDE_MD, '## CLI command matrix')
  read_only_in_matrix = []
  for line in matrix.splitlines():
    if '|' not in line or 'Command' in line or '---' in line:
      continue
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 4:
      continue
    cmd_cell = parts[1]
    mutating_cell = parts[2]
    if mutating_cell != 'No':
      continue
    raw = cmd_cell.strip('` ')
    if '(' in raw:
      continue
    read_only_in_matrix.append(raw)

  assert len(read_only_in_matrix) > 10, 'too few read-only commands parsed'
  for cmd in read_only_in_matrix:
    top = cmd.split()[0] if cmd else ''
    in_set = cmd in _BASE_CONTEXT_EXEMPT or top in _BASE_CONTEXT_EXEMPT
    assert in_set, f'read-only command {cmd!r} from CLI matrix is not in _BASE_CONTEXT_EXEMPT'


# --- BUG-DFV1-003: ai commands in CLI matrix ---


def test_ai_commands_in_claude_matrix():
  """BUG-DFV1-003: ai command group appears in CLAUDE.md matrix."""
  assert 'ai generate run' in CLAUDE_MD
  assert 'ai judge run' in CLAUDE_MD
  assert 'ai judge summarize' in CLAUDE_MD


# --- BUG-DFV1-004: optimize subcommands in CLI matrix ---


def test_optimize_subcommands_in_claude_matrix():
  """BUG-DFV1-004: optimize subcommands appear in CLAUDE.md matrix."""
  assert 'optimize train' in CLAUDE_MD
  assert 'optimize preflight' in CLAUDE_MD
  assert 'optimize loop' in CLAUDE_MD


# --- BUG-DFV1-008: store merge in CLI matrix ---


def test_store_merge_in_claude_matrix():
  """BUG-DFV1-008: store merge appears in CLAUDE.md matrix."""
  assert '| `store merge`' in CLAUDE_MD


# --- BUG-DFV1-009: architecture cli bullet completeness ---


def test_architecture_cli_bullet_complete():
  """BUG-DFV1-009: architecture cli bullet lists all top-level commands."""
  section = _extract_section(CLAUDE_MD, '## Architecture')
  required = [
    'dataset',
    'propose',
    'store',
    'report',
    'policy',
    'status',
    'diagnose',
    'trace',
    'track',
  ]
  for cmd in required:
    assert cmd in section, f'{cmd} missing from architecture cli bullet'


# --- BUG-DFV1-010: store doctor orphan behavior ---


def test_store_doctor_orphan_behavior_documented():
  """BUG-DFV1-010: store doctor healthy=True with orphans is documented."""
  assert 'healthy=True' in CLAUDE_MD
  assert 'orphan blobs exist' in CLAUDE_MD


# --- BUG-DFV1-011/012: umbrella rows expanded ---


def test_policy_subcommands_in_matrix():
  """BUG-DFV1-011: policy subcommands appear in matrix."""
  assert '| `policy check`' in CLAUDE_MD
  assert '| `policy explain`' in CLAUDE_MD


def test_diagnose_subcommands_in_matrix():
  """BUG-DFV1-012: diagnose subcommands appear in matrix."""
  assert '| `diagnose run`' in CLAUDE_MD
  assert '| `diagnose heatmap`' in CLAUDE_MD


def test_trace_subcommands_in_matrix():
  """BUG-DFV1-012: trace subcommands appear in matrix."""
  assert '| `trace collect`' in CLAUDE_MD
  assert '| `trace inspect`' in CLAUDE_MD


def test_report_subcommands_in_matrix():
  """BUG-DFV1-012: report subcommands appear in matrix."""
  assert '| `report compare`' in CLAUDE_MD
  assert '| `report summary`' in CLAUDE_MD


# --- BUG-DFV1-002: fail accepts pending documented ---


def test_fail_accepts_pending_documented():
  """BUG-DFV1-002: fail() accepting pending status is documented."""
  assert '`complete` and `fail` accept both `pending` and `running`' in CLAUDE_MD


# --- CLAUDE.md and AGENTS.md byte-identity ---


def test_claude_agents_byte_identical():
  """CLAUDE.md and AGENTS.md must be byte-identical."""
  claude = REPO_ROOT / 'CLAUDE.md'
  agents = REPO_ROOT / 'AGENTS.md'
  assert claude.read_bytes() == agents.read_bytes()


# --- plan 17: constructor docstring argument alignment ---

_ARGS_BLOCK_RE = re.compile(r'^(\s*)Args:\s*$', re.MULTILINE)

_GOOGLE_SECTIONS = frozenset(
  ('Returns', 'Raises', 'Yields', 'Note', 'Example', 'Attributes', 'Methods')
)


def _parse_docstring_args(docstring: str) -> set[str]:
  """Extract parameter names from a Google-style ``Args:`` block.

  Returns a set of names documented in the first ``Args:`` section.
  Stops at the next Google-style section (Returns, Raises, etc.) at the
  same indentation level, or at a less-indented non-blank line.
  """
  match = _ARGS_BLOCK_RE.search(docstring)
  if match is None:
    return set()
  args_indent = len(match.group(1))
  arg_line_re = re.compile(r'^' + r'\s' * (args_indent + 2) + r'(\w+)\s*(?:\(|:)')
  rest = docstring[match.end() :]
  names: set[str] = set()
  for line in rest.split('\n'):
    stripped = line.strip()
    if not stripped:
      continue
    indent = len(line) - len(line.lstrip())
    if indent <= args_indent and stripped.rstrip(':') in _GOOGLE_SECTIONS:
      break
    if indent <= args_indent and stripped:
      break
    arg_match = arg_line_re.match(line)
    if arg_match:
      names.add(arg_match.group(1))
  return names


def _get_init_params(cls: type) -> set[str]:
  """Return non-self parameter names from ``cls.__init__`` signature."""
  sig = inspect.signature(cls.__init__)
  return {
    p.name
    for p in sig.parameters.values()
    if p.name != 'self'
    and p.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
  }


_INIT_ALLOWLIST: dict[type, set[str]] = {}


def _check_args_subset(cls: type) -> None:
  """Assert documented Args names are a subset of actual __init__ params."""
  doc = cls.__init__.__doc__
  if doc is None:
    return
  documented = _parse_docstring_args(doc) - _INIT_ALLOWLIST.get(cls, set())
  actual = _get_init_params(cls)
  extra = documented - actual
  assert not extra, (
    f'{cls.__name__}.__init__ documents Args {sorted(extra)} not in signature {sorted(actual)}'
  )


_CRITICAL_CLASSES: list[type] = []


def _load_critical_classes() -> list[type]:
  """Load critical classes once for constructor alignment tests."""
  if _CRITICAL_CLASSES:
    return _CRITICAL_CLASSES
  from autopilot.ai.loss import JudgeLoss
  from autopilot.ai.optimizer import AgentOptimizer
  from autopilot.ai.store.file_store import FileStore
  from autopilot.core.comparison import ComparatorMetric, MetricsComparator
  from autopilot.core.config import AutoPilotConfig, Config
  from autopilot.core.context import ContextEntry, ContextLog
  from autopilot.core.experiment import Experiment
  from autopilot.core.gradient import Gradient, NumericGradient
  from autopilot.core.loss import Loss
  from autopilot.core.metric import Metric
  from autopilot.core.module.autopilot_module import AutoPilotModule
  from autopilot.core.module.module import Module
  from autopilot.core.optimizer import Optimizer
  from autopilot.core.parameter import Parameter
  from autopilot.core.scheduler import LambdaScheduler, Scheduler
  from autopilot.core.snapshot import ParameterSchema, SnapshotManifest
  from autopilot.core.trainer.trainer import Trainer
  from autopilot.core.types import Datum, EvalDatum
  from autopilot.policy.gates import BudgetGate, MonotonicGate
  from autopilot.policy.quality_first import QualityFirstPolicy

  classes = [
    Datum,
    EvalDatum,
    Parameter,
    Gradient,
    NumericGradient,
    Module,
    AutoPilotModule,
    Loss,
    Metric,
    Optimizer,
    Scheduler,
    LambdaScheduler,
    Trainer,
    Config,
    AutoPilotConfig,
    Experiment,
    ContextEntry,
    ContextLog,
    ComparatorMetric,
    MetricsComparator,
    ParameterSchema,
    SnapshotManifest,
    BudgetGate,
    MonotonicGate,
    QualityFirstPolicy,
    FileStore,
    JudgeLoss,
    AgentOptimizer,
  ]
  _CRITICAL_CLASSES.extend(classes)
  return _CRITICAL_CLASSES


@pytest.mark.parametrize(
  'cls',
  _load_critical_classes(),
  ids=lambda c: c.__name__,
)
def test_constructor_args_match_signature(cls):
  """Documented Args names in __init__ must be a subset of actual params."""
  _check_args_subset(cls)


# --- plan 17: intentional non-features documented ---


def test_module_docstring_mentions_no_log():
  """Module module docstring should mention no module.log() by design."""
  from autopilot.core.module import module as mod

  doc = mod.__doc__ or ''
  assert 'log()' in doc or 'log_dict()' in doc, (
    'module.py module docstring should mention intentional absence of module.log()'
  )


def test_module_docstring_mentions_no_gradient_clipping():
  """Module module docstring should mention no gradient clipping by design."""
  from autopilot.core.module import module as mod

  doc = mod.__doc__ or ''
  assert 'clipping' in doc.lower() or 'Gradient clipping' in doc, (
    'module.py module docstring should mention gradient clipping is undefined'
  )


def test_datum_items_documented_as_list():
  """Datum class docstring must describe items as list[Datum], not a mapping."""
  from autopilot.core.types import Datum

  doc = Datum.__doc__ or ''
  assert 'list[Datum]' in doc or 'list' in doc, (
    'Datum docstring must clarify items is a list, not a mapping'
  )


# --- helpers ---


def _extract_section(content: str, heading: str) -> str:
  """Extract text between a heading and the next heading at the same level."""
  level = len(heading) - len(heading.lstrip('#'))
  pattern = rf'^{re.escape(heading)}$'
  match = re.search(pattern, content, re.MULTILINE)
  assert match is not None, f'heading {heading!r} not found'
  start = match.end()
  next_heading = re.compile(rf'^#{{{level}}} ', re.MULTILINE)
  next_match = next_heading.search(content, start)
  end = next_match.start() if next_match else len(content)
  return content[start:end]
