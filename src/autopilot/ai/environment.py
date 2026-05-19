"""Worktree-based isolation environment for experiments.

IsolatedEnvironment creates disposable worktree directories with:
- Symlinks for ALL non-parameter files (including venvs, node_modules)
- Copies of parameter files from Store snapshots (isolated per experiment)
- Ignore patterns for exclusion (fnmatch-style)

Constructor requires ``ignore_patterns``, ``symlink_as_unit``, and
``core_files`` as explicit tuples -- the framework supplies no default
isolation policy. Callers must provide all three at every construction site.

Worktree rules live on the environment instance, not on Config.

Symlinks are recreated fresh each setup to ensure sync with project state.
During optimization, the agent modifies files IN the worktree directory.

PathParameter bind/unbind lifecycle: owned by Trainer._run_fit_loop.
IsolatedEnvironment.activate() creates the worktree and returns its path;
it does NOT call PathParameter.bind() or unbind(). The Trainer uses the
returned path to bind each PathParameter and unbinds in a finally block.

Limitations of fnmatch patterns: no ** recursive glob, no ! negation.
Users adding custom patterns must be aware of fnmatch semantics.

Snapshot fallback: when an experiment has no snapshot yet (latest_epoch < 0),
the parent branch's snapshot is used if available. Binary file content that
cannot be decoded as UTF-8 gets a ``<binary: N bytes>`` placeholder string.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.core.config import AutoPilotConfig
from autopilot.core.environment import Environment, WorktreeStore
from autopilot.core.errors import ConfigError, StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.module.module import Module
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
import shutil


def _matches_ignore(name: str, patterns: tuple[str, ...] | list[str]) -> bool:
  """Check if a file/directory name matches any ignore pattern.

  Returns:
    True when ``name`` matches any fnmatch pattern in ``patterns``.
  """
  return any(fnmatch(name, pat) or fnmatch(name + '/', pat) for pat in patterns)


def collect_parameter_files(module: Module) -> set[Path]:
  """Collect all parameter file paths from the module tree.

  Only PathParameter has matched_files() and source attributes.

  Returns:
    Set of resolved paths for all files owned by :class:`PathParameter` leaves.
  """
  param_files: set[Path] = set()
  for param in module.parameters():
    if isinstance(param, PathParameter):
      param_files.update(f.resolve() for f in param.matched_files())
  return param_files


def build_param_content_map(
  module: Module,
  snapshot_content: dict[str, str],
) -> dict[Path, str]:
  """Build a mapping from resolved file path to snapshot content.

  Walks the module's named parameters and matches snapshot keys
  (name/relative_key) back to absolute paths using each PathParameter's
  source attribute.

  Returns:
    Map from absolute file path to text content for worktree materialization.
  """
  result: dict[Path, str] = {}
  for param_name, param in module.named_parameters():
    if not isinstance(param, PathParameter):
      continue
    source = Path(param.source).expanduser()
    for key, text in snapshot_content.items():
      name_part, _, rel_key = key.partition('/')
      if name_part != param_name:
        continue
      if source.is_file() and rel_key == '.':
        result[source.resolve()] = text
      elif source.is_dir():
        result[(source / rel_key).resolve()] = text
      elif not source.exists() and rel_key == '.':
        result[source.resolve()] = text
  return result


def bind_path_parameters(module: Module, config_root: Path, wt_path: Path) -> list[PathParameter]:
  """Bind PathParameters to worktree-relative paths.

  Scans the module for :class:`PathParameter` instances and rebinds each to
  the corresponding path under ``wt_path`` by computing its relative position
  from ``config_root``.

  When ``wt_path`` equals ``Path.cwd()`` (the ``LocalEnvironment`` / None case),
  returns immediately without modifying any PathParameter. This matches
  Trainer semantics -- when no IsolatedEnvironment is configured, PathParameters
  retain their canonical ``source`` paths.

  Used by ``Trainer._run_fit_loop`` and ``tree switch --bind``.

  Args:
    module: Module whose parameters are scanned for PathParameter instances.
    config_root: Project root against which parameter sources are relativized.
    wt_path: Worktree path from environment activation.

  Returns:
    List of PathParameters that were bound (caller must unbind in finally).

  Raises:
    ConfigError: When a PathParameter source is outside config_root.
  """
  rebound: list[PathParameter] = []
  if wt_path == Path.cwd():
    return rebound
  try:
    for _name, param in module.named_parameters():
      if not isinstance(param, PathParameter):
        continue
      try:
        rel = Path(param.source).relative_to(config_root)
      except ValueError as exc:
        msg = (
          f'PathParameter source {param.source!r} is not under '
          f'config root {str(config_root)!r}; cannot compute worktree-relative path'
        )
        raise ConfigError(msg) from exc
      param.bind(str(wt_path / rel))
      rebound.append(param)
  except ConfigError:
    for p in rebound:
      p.unbind()
    raise
  return rebound


@dataclass(frozen=True)
class WorktreeWalkContext:
  """Immutable bundle of paths and maps shared across a worktree directory walk.

  Attributes:
    project_root: Absolute project root being mirrored into the worktree.
    wt_path: Root of the isolated worktree directory.
    param_files: Resolved paths that belong to parameters (copy/snapshot behavior).
    ignore_patterns: fnmatch patterns for excluded names.
    unit_dirs: Directory basenames symlinked as a single unit.
    param_content_map: Snapshot text keyed by resolved parameter paths.
  """

  project_root: Path
  wt_path: Path
  param_files: set[Path]
  ignore_patterns: tuple[str, ...] | list[str]
  unit_dirs: set[str]
  param_content_map: dict[Path, str]


class IsolatedEnvironment(Environment):
  """Worktree-based isolation with symlinks for non-params, copies for params.

  Implements the full worktree-building algorithm: symlinks for non-parameter
  files and directories, copies for parameter files with snapshot content.

  All three tuple parameters (``ignore_patterns``, ``symlink_as_unit``,
  ``core_files``) are required -- the framework supplies no default isolation
  policy. Callers must supply explicit tuples tailored to their project layout.
  """

  def __init__(
    self,
    config: AutoPilotConfig,
    ignore_patterns: tuple[str, ...],
    symlink_as_unit: tuple[str, ...],
    core_files: tuple[str, ...],
  ) -> None:
    """Initialize isolated environment.

    Args:
      config: AutoPilotConfig for path resolution (``config.root`` as project root).
      ignore_patterns: Glob patterns skipped during walk (fnmatch semantics).
        No framework-provided defaults; callers must supply explicitly.
      symlink_as_unit: Directory basenames symlinked as one unit.
        No framework-provided defaults; callers must supply explicitly.
      core_files: Files symlinked into worktree even when snapshot is empty.
        No framework-provided defaults; callers must supply explicitly.
    """
    self.config = config
    self.ignore_patterns = ignore_patterns
    self.symlink_as_unit = symlink_as_unit
    self.core_files = core_files
    self._store: WorktreeStore | None = None

  def setup(self, experiment: Experiment, store: WorktreeStore, module: Module) -> Path:
    """Create worktree and return its path.

    Saves ``self._store = store`` before ``store.create_worktree``, then
    builds the worktree with symlinks for non-params and copies for params.

    Args:
      experiment: Experiment whose branch provides the snapshot.
      store: Store for worktree creation and snapshot access.
      module: Module tree used to discover parameter file paths.

    Returns:
      Absolute path to the worktree directory.
    """
    self._store = store
    wt_path = store.create_worktree(experiment.id)

    param_files = collect_parameter_files(module)
    snapshot_content = self._get_snapshot_content(experiment.id)
    param_content_map = build_param_content_map(module, snapshot_content)

    self._build_worktree(
      project_root=self.config.root,
      wt_path=wt_path,
      param_files=param_files,
      ignore_patterns=self.ignore_patterns,
      unit_dirs=set(self.symlink_as_unit),
      core_files=set(self.core_files),
      param_content_map=param_content_map,
    )

    return wt_path

  def teardown(self, experiment: Experiment) -> None:
    """Remove worktree via ``self._store.remove_worktree(experiment.id)``.

    Args:
      experiment: Experiment whose worktree should be removed.
    """
    if self._store is not None:
      self._store.remove_worktree(experiment.id)

  def _get_snapshot_content(self, experiment_id: str) -> dict[str, str]:
    """Load parameter file content from the store snapshot.

    Returns a dict mapping 'param_name/relative_path' -> file content.
    When branching from a non-stabilized parent, falls back to the
    parent's Store snapshot if this experiment has no snapshot yet.
    Returns empty dict if no snapshot exists.

    Args:
      experiment_id: Experiment whose snapshot to load.

    Returns:
      Snapshot text map for the experiment's latest available epoch, or empty.

    Raises:
      RuntimeError: When store has not been initialized via ``setup()``.
      StoreError: When ``branches`` in refs is not a dict.
    """
    if self._store is None:
      msg = 'store is not initialized; call setup() before _get_snapshot_content()'
      raise RuntimeError(msg)
    refs = self._store.load_refs()
    branches = refs.get('branches', {})
    if not isinstance(branches, dict):
      msg = (
        f'expected branches to be a dict, got {type(branches).__name__}. '
        'Store refs metadata may be corrupted.'
      )
      raise StoreError(msg)
    if experiment_id not in branches:
      return {}
    branch = branches[experiment_id]
    latest_epoch = branch['latest_epoch']
    if latest_epoch < 0:
      parent_id = branch.get('parent_id')
      parent_epoch = branch.get('parent_epoch')
      if parent_id and parent_epoch is not None:
        return self._load_snapshot_content(parent_id, parent_epoch)
      return {}
    return self._load_snapshot_content(experiment_id, latest_epoch)

  def _load_snapshot_content(self, experiment_id: str, epoch: int) -> dict[str, str]:
    """Read all file contents from a specific snapshot.

    Args:
      experiment_id: Experiment branch whose snapshot to read.
      epoch: Epoch number within the branch.

    Returns:
      Map from store snapshot keys to UTF-8 text (binary entries get placeholders).

    Raises:
      RuntimeError: When store has not been initialized via ``setup()``.
      StoreError: When snapshot load or object reads fail at the store layer.
    """
    if self._store is None:
      msg = 'store is not initialized; call setup() before _load_snapshot_content()'
      raise RuntimeError(msg)
    try:
      snap = self._store.load_snapshot(experiment_id, epoch)
      content: dict[str, str] = {}
      for key, entry in snap.entries.items():
        data = self._store.read_object(entry.digest)
        try:
          content[key] = data.decode('utf-8')
        except UnicodeDecodeError:
          content[key] = f'<binary: {len(data)} bytes>'
    except StoreError as exc:
      msg = f'{experiment_id}: {exc}'
      raise StoreError(msg) from exc
    else:
      return content

  def _build_worktree(
    self,
    project_root: Path,
    wt_path: Path,
    param_files: set[Path],
    ignore_patterns: tuple[str, ...] | list[str],
    unit_dirs: set[str],
    core_files: set[str],
    param_content_map: dict[Path, str],
  ) -> None:
    """Walk project root and populate the worktree.

    Args:
      project_root: Absolute project root being mirrored.
      wt_path: Root of the isolated worktree directory.
      param_files: Resolved paths that belong to parameters.
      ignore_patterns: fnmatch patterns for excluded names.
      unit_dirs: Directory basenames symlinked as a single unit.
      core_files: Files always symlinked even for empty snapshots.
      param_content_map: Snapshot text keyed by resolved parameter paths.

    Raises:
      ConfigError: When ``project_root`` does not exist.
    """
    if not project_root.exists():
      msg = (
        f'project_root does not exist: {project_root}. '
        'Ensure the path is correct and the directory has been created.'
      )
      raise ConfigError(msg)

    for core_file in core_files:
      src = project_root / core_file
      dst = wt_path / core_file
      if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)

    ctx = WorktreeWalkContext(
      project_root=project_root,
      wt_path=wt_path,
      param_files=param_files,
      ignore_patterns=ignore_patterns,
      unit_dirs=unit_dirs,
      param_content_map=param_content_map,
    )
    self._walk_and_link(ctx=ctx, current_dir=project_root)

  def _walk_and_link(
    self,
    ctx: WorktreeWalkContext,
    current_dir: Path,
  ) -> None:
    """Recursively walk and create symlinks/copies in worktree.

    Args:
      ctx: Immutable context with project root, worktree path, and parameter sets.
      current_dir: Directory to iterate; entries are linked or recursed into.

    Raises:
      PermissionError: When directory listing is denied by the OS.
    """
    try:
      entries = sorted(current_dir.iterdir(), key=lambda p: p.name)
    except PermissionError as exc:
      msg = (
        f'permission denied reading {current_dir}; fix directory permissions or exclude this path'
      )
      raise PermissionError(msg) from exc

    store_path = self.config.store_path

    for entry in entries:
      name = entry.name

      if _matches_ignore(name, ctx.ignore_patterns):
        continue

      if entry.is_dir() and entry == store_path:
        continue

      rel = entry.relative_to(ctx.project_root)
      wt_target = ctx.wt_path / rel

      if entry.is_symlink():
        self._link_entry(
          entry,
          entry.resolve(),
          wt_target,
          ctx.param_files,
          ctx.param_content_map,
        )
      elif entry.is_dir():
        self._handle_directory_entry(entry, name, wt_target, ctx)
      elif entry.is_file():
        self._link_entry(
          entry,
          entry.resolve(),
          wt_target,
          ctx.param_files,
          ctx.param_content_map,
        )

  def _link_entry(
    self,
    entry: Path,
    resolved: Path,
    wt_target: Path,
    param_files: set[Path],
    param_content_map: dict[Path, str],
  ) -> None:
    """Copy parameter snapshot content or symlink a file/symlink entry into the worktree.

    When ``resolved`` is a parameter path, writes snapshot text, copies from disk if
    missing from the map, or copies the entry as a fallback. Otherwise creates a
    symlink to ``resolved`` when the worktree target does not exist.

    Args:
      entry: Source path (preserved for ``shutil.copy2`` semantics).
      resolved: Canonical resolved path for parameter checks and symlink targets.
      wt_target: Destination path inside the worktree.
      param_files: Set of resolved paths owned by parameters.
      param_content_map: Snapshot text keyed by resolved parameter paths.
    """
    if resolved in param_files:
      wt_target.parent.mkdir(parents=True, exist_ok=True)
      # BUG-004: core_files may have symlinked this path earlier;
      # writing through the symlink would mutate the canonical source.
      if wt_target.is_symlink():
        wt_target.unlink()
      content = param_content_map.get(resolved)
      if content is not None:
        wt_target.write_text(content, encoding='utf-8')
      else:
        shutil.copy2(str(entry), str(wt_target))
    elif not wt_target.exists():
      wt_target.parent.mkdir(parents=True, exist_ok=True)
      wt_target.symlink_to(resolved)

  def _handle_directory_entry(
    self,
    entry: Path,
    name: str,
    wt_target: Path,
    ctx: WorktreeWalkContext,
  ) -> None:
    """Symlink a unit directory wholesale or recurse into a normal directory.

    Unit directories (per ``ctx.unit_dirs``) become a single symlink to the
    project entry; other directories are created in the worktree and walked
    recursively.

    Args:
      entry: Directory path under the project root.
      name: Base name of ``entry`` (for unit-dir lookup).
      wt_target: Corresponding worktree directory path.
      ctx: Shared walk context (roots, parameter maps, ignore and unit settings).
    """
    if name in ctx.unit_dirs:
      if not wt_target.exists():
        wt_target.parent.mkdir(parents=True, exist_ok=True)
        wt_target.symlink_to(entry.resolve())
      return
    wt_target.mkdir(parents=True, exist_ok=True)
    self._walk_and_link(ctx=ctx, current_dir=entry)
