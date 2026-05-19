"""PathParameter: file-system parameters for optimizer scoping.

Declares which files/directories the optimizer is allowed to modify.
Supports ephemeral working_root via bind/unbind for worktree isolation.

Symlinks whose resolved target falls outside the parameter root are skipped
during snapshot with a warning log. Broken symlinks are also skipped with a
warning. ``logging.warning`` (stdlib logging) is used for observability;
``import warnings`` / ``warnings.warn()`` remain prohibited per CLAUDE.md.
"""

from autopilot.core.errors import StoreError
from autopilot.core.parameter import Parameter
from autopilot.core.snapshot import ParameterSchemaEntry
from autopilot.core.types import Datum
from pathlib import Path
from typing import Any
import logging

PATH_LISTING_LIMIT = 20

logger = logging.getLogger(__name__)


class PathParameter(Parameter):
  """File-system parameter declaring mutable scope.

  source path is expanded via expanduser() in snapshot(), restore(), and
  matched_files(). Binary files (those that raise UnicodeDecodeError) raise
  ``StoreError`` during snapshot. Constructor uses str | None defaults (not
  empty-string defaults) for optional string fields.

  source: canonical path on disk (directory or file), always serialized
  pattern: glob for which files within source are mutable
  working_root: ephemeral runtime path (via bind/unbind) for worktree
    isolation; never serialized. All I/O (matched_files, render, snapshot,
    restore) routes through working_root when bound.

  Example::

    class MyModule(Module):
      def __init__(self):
        super().__init__()
        self.prompts = PathParameter(source='~/project/prompts', pattern='**/*.md')
        self.config = PathParameter(source='~/project/config', pattern='*.tf')
  """

  source: str
  pattern: str = '**/*'
  _working_root: str | None = None

  def __init__(
    self,
    source: str,
    pattern: str = '**/*',
    **kwargs: Any,
  ) -> None:
    """Create a path-scoped parameter with optional extra :class:`Parameter` fields.

    Args:
      source: Root file or directory path (may use ``~``).
      pattern: Glob pattern relative to ``source`` for matched files.
      **kwargs: Forwarded to :class:`Parameter` (items, requires_grad, etc.).
    """
    super().__init__(**kwargs)
    object.__setattr__(self, 'source', source)
    object.__setattr__(self, 'pattern', pattern)
    object.__setattr__(self, '_working_root', None)

  @property
  def working_root(self) -> str:
    """Effective root for all I/O.

    Returns ``_working_root`` when bound to a worktree, else ``source``.
    """
    return self._working_root if self._working_root is not None else self.source

  def bind(self, root: str) -> None:
    """Bind to a worktree path for isolated I/O.

    Called by Trainer._run_fit_loop after environment activation.

    Warning:
      Not thread-safe. Concurrent use across Trainer instances sharing
      the same Module is unsupported. Trainer owns bind/unbind lifecycle.

    Args:
      root: Absolute worktree path to use as the effective root.
    """
    self._working_root = root

  def unbind(self) -> None:
    """Reset to canonical source. Called by Trainer in finally block.

    Warning:
      Not thread-safe. Concurrent use across Trainer instances sharing
      the same Module is unsupported. Trainer owns bind/unbind lifecycle.
    """
    self._working_root = None

  def matched_files(self) -> list[Path]:
    """List files matching the pattern within working_root.

    Filters with ``is_file()`` to prevent ``IsADirectoryError`` on
    ``**/*`` patterns (BUG-006).

    Returns:
      Sorted absolute paths, or empty list when working_root does not exist.
    """
    source_path = Path(self.working_root).expanduser()
    if not source_path.exists():
      return []
    if source_path.is_file():
      return [source_path]
    return sorted(p for p in source_path.glob(self.pattern) if p.is_file())

  def _log_skipped_symlink(self, path: Path, root: Path, reason: str) -> None:
    """Emit a warning when a symlink candidate is excluded from snapshot.

    Args:
      path: Symlink path under the parameter root.
      root: Resolved parameter root directory.
      reason: Short reason token (``'outside_root'`` or ``'broken'``).
    """
    rel = path.relative_to(root) if path.is_relative_to(root) else path
    logger.warning(
      'PathParameter snapshot skipped symlink %s (%s)',
      rel,
      reason,
    )

  def _warn_broken_symlinks(self, root: Path) -> None:
    """Scan for broken symlinks matching the pattern and log warnings.

    Broken symlinks are excluded by ``matched_files()`` (which filters
    with ``is_file()``), so they never reach ``_is_within_root()``. This
    method runs a separate glob pass to catch and warn about them.

    Args:
      root: Resolved parameter root directory.
    """
    source_path = Path(self.working_root).expanduser()
    if not source_path.exists() or source_path.is_file():
      return
    for p in source_path.glob(self.pattern):
      if p.is_symlink() and not p.exists():
        self._log_skipped_symlink(p, root, 'broken')

  def _is_within_root(self, path: Path, root: Path) -> bool:
    """Check whether a path's resolved target stays within root.

    Broken symlinks and symlinks whose target resolves outside ``root``
    return False and emit a warning for observability. Regular files and
    symlinks within ``root`` return True.

    Args:
      path: Candidate file path (may be a symlink).
      root: Resolved parameter root directory.

    Returns:
      True when the path is safe to snapshot.
    """
    if path.is_symlink():
      try:
        resolved = path.resolve(strict=True)
      except OSError:
        self._log_skipped_symlink(path, root, 'broken')
        return False
      try:
        resolved.relative_to(root)
      except ValueError:
        self._log_skipped_symlink(path, root, 'outside_root')
        return False
    return True

  def render(self) -> str:
    """Render a short list of editable files for prompts.

    Shows working_root paths so the agent edits the right files.

    Returns:
      Multi-line description, or empty string when no files match.
    """
    files = self.matched_files()
    if not files:
      return ''
    parts = [f'Editable files ({self.working_root}):']
    parts.extend(f'  - {f}' for f in files[:PATH_LISTING_LIMIT])
    return '\n'.join(parts)

  def snapshot(self) -> dict[str, str]:
    """Read all matched text files under working_root.

    Captures worktree state when bound (BUG-003 fix). Symlinks whose
    resolved target falls outside the parameter root are skipped with a
    warning log (no exfiltration). Broken symlinks are also skipped with
    a warning. Symlinks whose target stays within root are followed and
    their content is captured.

    Returns:
      Mapping from paths relative to working_root to UTF-8 text.

    Raises:
      StoreError: When a binary file matches the pattern and cannot be versioned.
    """
    result: dict[str, str] = {}
    root = Path(self.working_root).expanduser().resolve()
    self._warn_broken_symlinks(root)
    for f in self.matched_files():
      if not self._is_within_root(f, root):
        continue
      key = str(f.relative_to(Path(self.working_root).expanduser()))
      try:
        result[key] = f.read_text(encoding='utf-8')
      except UnicodeDecodeError as exc:
        msg = (
          f'binary file {f} matched by pattern {self.pattern!r} cannot be versioned;'
          f' exclude binary files from the pattern or use a dedicated binary store'
        )
        raise StoreError(msg) from exc
    return result

  def restore(self, content: dict[str, str]) -> None:
    """Write text files under working_root from a prior :meth:`snapshot`.

    Writes to worktree when bound (BUG-005 fix).

    Note:
      File permissions are NOT preserved. Files are written with default
      ``write_text()`` mode (typically 0644). Executable scripts requiring
      ``chmod +x`` must be re-permissioned after checkout. Symlinks are
      restored as regular files containing the target's content at snapshot
      time -- the symlink itself is not recreated.

    Args:
      content: Relative path keys with UTF-8 text values from :meth:`snapshot`.

    Raises:
      ValueError: If any key is absolute, empty, or resolves outside the
        parameter root (path traversal protection).
    """
    root = Path(self.working_root).expanduser().resolve()
    for rel_path, text in content.items():
      if not rel_path or rel_path.startswith('/'):
        msg = (
          f'PathParameter.restore() rejected absolute or empty key: {rel_path!r}. '
          f'Manifest keys must be relative paths within the parameter root.'
        )
        raise ValueError(msg)
      target = (root / rel_path).resolve()
      try:
        target.relative_to(root)
      except ValueError:
        msg = (
          f'PathParameter.restore() path traversal blocked: {rel_path!r} resolves to '
          f'{target} which is outside parameter root {root}. '
          f'Check manifest entries for ".." segments or symlink escapes.'
        )
        raise ValueError(msg) from None
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_text(text, encoding='utf-8')

  def schema_entry(self) -> ParameterSchemaEntry:
    """Return schema metadata including source and pattern.

    Returns:
      Schema entry with filesystem provenance fields populated.
    """
    return ParameterSchemaEntry(
      name='',
      type_name=type(self).__name__,
      source=self.source,
      pattern=self.pattern,
    )

  def load_from_dict(self, data: dict[str, Any]) -> None:
    """Apply serialized state into this live PathParameter instance.

    Restores ``requires_grad`` via base class, then reapplies file payloads
    when the dict carries snapshot-shaped content (BUG-007 fix).

    Args:
      data: Dict produced by ``to_dict()`` / ``state_dict()``.
    """
    super().load_from_dict(data)
    files = data.get('files')
    if files is not None and isinstance(files, dict):
      self.restore(files)

  def to_dict(self) -> dict[str, Any]:
    """Serialize ``source``, ``pattern``, and base parameter fields.

    ``working_root`` is ephemeral and never serialized.

    Returns:
      Dict suitable for :meth:`from_dict`.
    """
    payload = super().to_dict()
    payload['source'] = self.source
    payload['pattern'] = self.pattern
    payload['files'] = self.snapshot()
    return payload

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'PathParameter':
    """Deserialize a :class:`PathParameter` from :meth:`to_dict` output.

    Args:
      data: Mapping produced by :meth:`to_dict`.

    Returns:
      Restored :class:`PathParameter` with optional id override.
    """
    data = dict(data)
    stored_id = data.pop('id', None)
    data.pop('type', None)
    source = data.pop('source')
    pattern = data.pop('pattern', '**/*')
    requires_grad = data.pop('requires_grad', True)
    items_raw = data.pop('items', [])
    data.pop('files', None)
    items = [Datum.from_dict(item) for item in items_raw]
    param = cls(source=source, pattern=pattern, items=items)
    param.requires_grad = requires_grad
    if stored_id:
      param._id = stored_id
    return param
