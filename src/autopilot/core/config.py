"""Unified configuration and path resolution.

FilePath descriptor: auto-registering path descriptor with format string
parameterization via string.Formatter.parse(). Static paths return Path,
parameterized paths return a callable.

Config base class: workspace/project wiring, root property, injectable
``environment: Environment`` for execution isolation, stabilize,
init_workspace/init_project. All path computation goes through Config --
no standalone paths.py.

AutoPilotConfig: concrete Config with all .autopilot/ layout paths as
FilePath attributes and workspace/project init.
Config is a subclassable extension point alongside Module, Experiment.

Stabilize snapshot ordering: ``_latest_snapshot_file`` uses numeric epoch
extraction from ``epoch_<n>.json`` filenames (regex ``^epoch_(digit+).json$``).
Only files matching the pattern participate; directories and non-epoch files
are silently ignored. This prevents the lexicographic ordering bug where
``epoch_9`` sorts after ``epoch_11`` (BUG-034).

Multi-project stabilize caveat (BUG-045): when multiple projects share the
same workspace, ``stabilize()`` writes files into the project root based on
``original_path`` from the manifest. A second stabilize from a different
project can overwrite artifacts produced by the first if both projects
emit files to the same ``original_path`` destinations. Use distinct
``original_path`` layouts per project. A ``--parameter-prefix`` CLI flag
is available via ``--parameter-prefix``.
"""

from autopilot.core.environment import Environment, LocalEnvironment
from autopilot.core.errors import ConfigError
from autopilot.core.snapshot import SnapshotManifest
from autopilot.tracking.io import read_json_dict
from pathlib import Path
from typing import Any
import re
import shutil
import string

_formatter = string.Formatter()

EPOCH_SNAPSHOT_RE = re.compile(r'^epoch_(\d+)\.json$')


def _latest_snapshot_file(snapshots_dir: Path) -> Path | None:
  """Return the stabilize snapshot path with the greatest numeric epoch suffix.

  Scans ``snapshots_dir`` for files matching ``epoch_<int>.json`` and returns
  the one with the highest numeric epoch. Directories, non-matching filenames,
  and files with extra suffixes (e.g. ``epoch_001.backup.json``) are ignored.

  Args:
    snapshots_dir: Directory containing per-epoch snapshot JSON files.

  Returns:
    Path to the latest snapshot file, or ``None`` when no valid files exist.
  """
  best_epoch = -1
  best_path: Path | None = None
  if not snapshots_dir.exists():
    return None
  for path in snapshots_dir.iterdir():
    if not path.is_file():
      continue
    match = EPOCH_SNAPSHOT_RE.match(path.name)
    if match is None:
      continue
    epoch = int(match.group(1))
    if epoch > best_epoch:
      best_epoch = epoch
      best_path = path
  return best_path


class FilePath:
  """Path descriptor with auto-registration and format string support.

  Static paths (no format fields) resolve to Path on access.
  Parameterized paths (with {field} in name) return a callable
  that takes keyword arguments and returns a Path.

  Auto-registers into owner.file_paths via __set_name__.
  Override via __set__ writes to instance __dict__.
  """

  def __init__(self, parent: str, name: str) -> None:
    """Create a descriptor bound to a parent attribute and path template.

    Args:
      parent: Attribute name on the owner holding the base ``Path`` (or callable).
      name: Path fragment or ``str.format`` template for this segment.
    """
    self._parent = parent
    self._name = name
    self._attr = ''
    self._fields = tuple(
      field_name for _, field_name, _, _ in _formatter.parse(name) if field_name is not None
    )

  def __set_name__(self, owner: type, name: str) -> None:
    """Register this descriptor on ``owner`` and record the attribute name."""
    self._attr = name
    typed_owner: Any = owner
    if not hasattr(owner, 'file_paths'):
      typed_owner.file_paths = {}
    typed_owner.file_paths[name] = self

  def __get__(self, obj: Any, objtype: type | None = None) -> Any:
    """Resolve to a ``Path``, callable resolver, or this descriptor on the class.

    Returns:
      This descriptor when ``obj`` is ``None``; otherwise a ``Path`` or a callable
      that formats parameterized segments when ``kwargs`` are supplied.
    """
    if obj is None:
      return self
    if self._attr in obj.__dict__:
      return obj.__dict__[self._attr]
    if self._fields:

      def resolve(**kwargs: Any) -> Path:
        parent_val = getattr(obj, self._parent)
        if callable(parent_val):
          parent_val = parent_val(**kwargs)
        return parent_val / self._name.format(**kwargs)

      return resolve
    parent_val = getattr(obj, self._parent)
    if callable(parent_val):

      def resolve(**kwargs: Any) -> Path:
        return parent_val(**kwargs) / self._name

      return resolve
    return parent_val / self._name

  def __set__(self, obj: Any, value: Any) -> None:
    """Override the resolved path by storing a concrete value on the instance."""
    obj.__dict__[self._attr] = value


class Config:
  """Abstract configuration base.

  Provides workspace/project wiring, root property with override,
  injectable ``environment: Environment`` for execution isolation,
  and extension points for workspace init, project init, and stabilization.

  Subclass and add FilePath descriptors for custom layout.
  __init_subclass__ merges file_paths across the MRO.
  """

  def __init__(
    self,
    workspace: Path,
    project: str | None = None,
    environment: Environment | None = None,
  ) -> None:
    """Wire workspace root, optional project slug, and environment.

    Args:
      workspace: Filesystem root for this workspace.
      project: Optional project directory name under ``projects_path``.
      environment: Execution isolation strategy. Defaults to ``LocalEnvironment()``.
    """
    self.workspace = workspace
    self.project = project
    self.environment = environment if environment is not None else LocalEnvironment()

  file_paths: dict[str, 'FilePath']

  def __init_subclass__(cls, **kwargs: Any) -> None:
    """Merge ``file_paths`` from the MRO so subclasses accumulate descriptors."""
    super().__init_subclass__(**kwargs)
    merged: dict[str, FilePath] = {}
    for base in reversed(cls.__mro__):
      if hasattr(base, 'file_paths'):
        base_paths: Any = base.file_paths
        paths: dict[str, FilePath] = base_paths
        merged.update(paths)
    cls.file_paths = merged

  @property
  def root(self) -> Path:
    """Project root: ``projects_path / project`` when set, else ``autopilot_path``.

    Returns:
      Resolved ``Path`` for artifact layout.
    """
    if 'root' in self.__dict__:
      return self.__dict__['root']
    self_any: Any = self
    if self.project:
      return self_any.projects_path / self.project
    return self_any.autopilot_path

  @root.setter
  def root(self, value: Path) -> None:
    """Override the computed root with an explicit path."""
    self.__dict__['root'] = value

  def stabilize(self, experiment_id: str, parameter_prefix: str | None = None) -> list[Path]:
    """Copy stable artifacts from store back into the workspace.

    Args:
      experiment_id: Experiment branch to stabilize.
      parameter_prefix: When set, only entries whose manifest key starts
        with this prefix are copied. Default (None): all parameters.

    Raises:
      NotImplementedError: On ``Config``; subclasses must implement.
    """
    raise NotImplementedError

  def init_workspace(self) -> None:
    """Create top-level workspace directories.

    Raises:
      NotImplementedError: On ``Config``; subclasses must implement.
    """
    raise NotImplementedError

  def init_project(self) -> None:
    """Create project-scoped directories.

    Raises:
      NotImplementedError: On ``Config``; subclasses must implement.
    """
    raise NotImplementedError


class AutoPilotConfig(Config):
  """Concrete Config with all .autopilot/ layout paths.

  All paths are FilePath descriptors. Static paths resolve to Path,
  parameterized paths (with {slug}, {epoch}, {experiment_id}) return
  callables. Execution isolation is configured via the inherited
  ``environment`` parameter (defaults to ``LocalEnvironment``).

  Store layout:
    - Without project: ``workspace/.autopilot/store/``
    - With project: ``workspace/.autopilot/projects/<project>/store/``
  """

  autopilot_path = FilePath('workspace', '.autopilot')
  projects_path = FilePath('autopilot_path', 'projects')
  experiments_path = FilePath('root', 'experiments')
  experiment_path = FilePath('experiments_path', '{slug}')
  epoch_path = FilePath('experiment_path', 'epoch_{epoch}')
  datasets_path = FilePath('root', 'datasets')
  records_path = FilePath('root', 'records')
  executions_path = FilePath('root', 'executions.jsonl')
  cli_file = FilePath('root', 'cli.py')
  store_path = FilePath('root', 'store')
  objects_path = FilePath('store_path', 'objects')
  snapshots_path = FilePath('store_path', 'snapshots')
  worktrees_path = FilePath('store_path', 'worktrees')
  forest_file = FilePath('store_path', 'forest.json')
  refs_file = FilePath('store_path', 'refs.json')
  store_experiments_path = FilePath('store_path', 'experiments')
  store_experiment_path = FilePath('store_experiments_path', '{experiment_id}')
  store_epoch_path = FilePath('store_experiment_path', 'epoch_{epoch}')
  result_file = FilePath('store_epoch_path', 'result.json')
  templates_path = FilePath('workspace', 'templates')

  def init_workspace(self) -> None:
    """Create the autopilot directory structure.

    Creates: .autopilot/, .autopilot/projects/, .autopilot/experiments/,
    .autopilot/records/, .autopilot/datasets/.
    All directories are created with parents=True, exist_ok=True.
    """
    self.autopilot_path.mkdir(parents=True, exist_ok=True)
    self.projects_path.mkdir(parents=True, exist_ok=True)
    self.experiments_path.mkdir(parents=True, exist_ok=True)
    self.records_path.mkdir(parents=True, exist_ok=True)
    self.datasets_path.mkdir(parents=True, exist_ok=True)

  def init_project(self) -> None:
    """Create the project directory under projects_path.

    Raises:
      ConfigError: When ``project`` is not set on this config.
    """
    if not self.project:
      msg = f'no project set on config (workspace={self.workspace!s})'
      raise ConfigError(
        msg,
      )
    project_dir = self.projects_path / self.project
    project_dir.mkdir(parents=True, exist_ok=True)

  def stabilize(self, experiment_id: str, parameter_prefix: str | None = None) -> list[Path]:
    """Copy parameter files from the experiment's latest store snapshot to project root.

    Uses numeric epoch extraction (``_latest_snapshot_file``) to find the
    highest-numbered ``epoch_<n>.json`` snapshot, avoiding the lexicographic
    ordering bug where ``epoch_9`` sorts after ``epoch_11`` (BUG-034).

    When multiple projects share a workspace, stabilize writes files based on
    ``original_path`` from the manifest. A second project stabilizing to the
    same destinations will overwrite the first (BUG-045 documented hazard).
    Use ``--parameter-prefix`` to scope which parameters are copied.

    Args:
      experiment_id: Experiment branch to stabilize.
      parameter_prefix: When set, only entries whose manifest key starts
        with ``<prefix>/`` are copied. Default (None): all parameters.

    Returns:
      Paths copied into the workspace, or an empty list when no snapshot exists.

    Raises:
      KeyError: When a manifest entry lacks ``original_path``.
    """
    snapshots_dir = self.snapshots_path / experiment_id
    latest_snapshot = _latest_snapshot_file(snapshots_dir)
    if latest_snapshot is None:
      return []
    data = read_json_dict(latest_snapshot, 'snapshot')
    manifest = SnapshotManifest.from_dict(data)
    copied: list[Path] = []
    objects_dir = self.objects_path
    prefix_filter = f'{parameter_prefix}/' if parameter_prefix is not None else None
    for key, entry in manifest.entries.items():
      if prefix_filter is not None and not key.startswith(prefix_filter):
        continue
      obj_hash = entry.digest
      if not obj_hash:
        continue
      original = entry.original_path
      if original is None:
        msg = f'manifest entry missing original_path for key {key!r}'
        raise KeyError(msg)
      src = objects_dir / obj_hash[:2] / obj_hash[2:]
      dst = self.workspace / original
      if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        copied.append(dst)
    return copied
