"""FileStore: content-addressed file versioning with worktree support.

Public API for parameter snapshots, checkout, diff, merge (three-step:
analysis -> preview -> apply), log, status, branch management, and worktree
isolation. Low-level object storage, hash computation, and file locking are
delegated to StorageBackend in autopilot.ai.store_lock.

Config-driven implementation using SHA-256, 2-char prefix sharding, JSON
snapshot manifests, and atomic writes. All paths come from config: store root,
objects, snapshots, worktrees, forest.json, refs.json.

Concurrency invariants (lock usage is implemented across ``file_store_core``,
``snapshot``, ``refs``, ``merge``, and related helpers):

- A single store-wide advisory lock serializes writers; loads that only read
  manifests or refs depend on atomic write/rename behavior, not holding the lock.
- Mutating paths re-read ``refs.json`` after acquiring the lock so logic never
  commits based on pre-lock state another process may have superseded (TOCTOU).
- ``StoreTransaction`` buffers multi-file updates and persists them together on
  successful exit so refs, manifests, and reflog never expose a half-applied op.

Constructor: FileStore(config: AutoPilotConfig). Parameters registered
separately via register_parameters(dict). Idempotent: creates .autopilot/
layout if not present, loads refs.json if exists. Manages multiple
experiments -- operations take experiment_id.

Named parameter keys: snapshot entries use 'name/state_key' format
(e.g. 'prompts/system.txt'). Names align with module attribute names via
register_parameters(). ParameterSchema is embedded in each manifest.

Directory layout::

  .autopilot/
    objects/          -- content-addressed parameter blobs (SHA-256)
    snapshots/        -- parameter snapshots per experiment/epoch
    stash/            -- numbered WIP stash manifests (0000.json, ...)
    experiments/      -- per-experiment artifact data
    worktrees/        -- parallel working directories
    forest.json       -- tree/node structure
    refs.json         -- heads, branches, tags, worktree registry
    reflog.jsonl      -- append-only audit trail of mutating operations

refs.json shape::

  {
    'HEAD': 'experiment-id-string',
    'branches': {
      'exp-id': {
        'latest_epoch': 5,
        'parent_id': 'parent-exp-id',
        'parent_epoch': 3,
        'merge_parents': [{'experiment_id': 'other-exp', 'epoch': 2}],
      }
    },
    'tags': {
      'v1.0': {
        'experiment_id': 'exp-id',
        'epoch': 5,
        'context': 'release candidate',
        'timestamp': '2026-01-01T00:00:00+00:00',
      }
    },
    'worktrees': {'exp-id': '/absolute/path/to/worktree'},
    'version': 2,
  }

HEAD field: active branch id persisted with refs. Written when checkout restores
files, snapshot records a new tip, materialize rewinds/stable-writes epoch 0, etc.

merge_parents: list of ``{experiment_id, epoch}`` entries appended by
``merge_apply`` to track merge lineage. Absent until the first merge; default
``[]``. ``version`` is bumped to ``2`` on the first merge_apply.

Merge (three-step API):

- ``merge_analysis``: cheap classification (fast-forward, clean, conflict,
  up-to-date) from refs + manifest key overlap since LCA; no blob reads.
- ``merge_preview``: materializes ``MergeIndex`` with ``ConflictEntry`` triples
  and auto-resolved entries per ``MergeStrategy``; sets cryptographic
  ``preview_token`` = SHA-256(target_exp, target_epoch, source_exp, source_epoch,
  strategy, sorted_keys, ancestor_tip).
- ``merge_apply``: validates token freshness (recompute from live refs),
  persists resolved ``MergeIndex`` as new epoch on target, appends
  ``merge_parents`` to branch entry.
- ``merge_and_apply``: convenience that runs analysis -> preview -> apply;
  raises ``StoreError`` on unresolved conflicts.

LCA algorithm: BFS on the refs DAG via ``parent_id``/``parent_epoch`` and
``merge_parents``; uses a visited set keyed by ``(experiment_id, epoch)`` pairs
to handle diamond histories. Fork boundaries are respected via ``parent_epoch``
(never implicit epoch 0) per BUG-023/BUG-026 fixes.

Fully decoupled from concrete parameter types: operates exclusively
through param.snapshot() / param.restore().
"""

from autopilot.ai.store.file_store_core import FileStoreCore
from autopilot.ai.store.file_store_doctor_mixin import FileStoreDoctorMixin
from autopilot.ai.store.file_store_merge_mixin import FileStoreMergeMixin


class FileStore(FileStoreDoctorMixin, FileStoreMergeMixin, FileStoreCore):
  """Content-addressed file store with worktree support.

    Uses AutopilotFileLock (filelock library, flock on POSIX) for store-wide
    locking. Lock is fail-fast by default: contention raises
    ConcurrentMutationError immediately. Set lock_timeout_s for timed or
    infinite waiting. Crash-safe: OS releases advisory lock on process death.
    All refs-mutating operations reload refs inside the lock to prevent
    TOCTOU races (plan 04).

    merge_apply and similar flows use ``StoreTransaction`` so callers either
    observe the pre-state or the fully committed post-state, never an intermediate
    mix of refs vs snapshot files.

  Constructor: FileStore(config: AutoPilotConfig).
  All paths come from config. Parameters registered separately via
  register_parameters(). Manages multiple experiments -- operations
  take experiment_id as parameter. experiment.id is the branch name by
  convention.

    Idempotent: creates .autopilot/ layout if not present, loads refs.json
    if exists. No auto-snapshot on construction -- call snapshot(id, 0) to
    create the first branch.

    Composite keys: snapshot entries use 'name/state_key' format
    (e.g. 'prompts/system.txt'). Named keys align with module attribute names.

  Example:
    >>> from pathlib import Path  # doctest: +SKIP
    >>> from autopilot.core.config import AutoPilotConfig  # doctest: +SKIP
    >>> from autopilot.ai.store.file_store import FileStore  # doctest: +SKIP
    >>>
    >>> config = AutoPilotConfig(workspace=Path('/my/project'))  # doctest: +SKIP
    >>> store = FileStore(config)  # doctest: +SKIP
    >>> store.register_parameters({'prompts': param})  # doctest: +SKIP
    >>> store.snapshot('exp-001', epoch=0)  # doctest: +SKIP
  """
