"""Base experiment entity with lifecycle state transitions.

Experiment is the data entity in the system. It holds identity, hypothesis,
status (internal Status enum), metrics, notes, timestamps, and epoch.

Inherits :class:`~autopilot.core.traceable.Traceable`, so every experiment
carries an append-only :class:`~autopilot.core.context.ContextLog` for
decision traceability. The context log is serialized in ``state_dict()``
and required by ``load_state_dict()`` (clean break: ``KeyError`` if the
``context_log`` key is missing from persisted state).

Two-tier design mirroring Module/AutoPilotModule:
- Experiment (this file): data + manual state transitions + no-op hooks.
- AutoPilotExperiment (ai/experiment.py): adds lifecycle hooks for
  Trainer integration. Subclass for Lightning-style automation.

Experiment supports context manager usage (``with experiment:``) for
automatic lifecycle management. ``__enter__`` calls ``start()`` when status
is ``pending``, or allows re-entry without calling ``start()`` when status
is already ``running`` (checkpoint resume: ``Trainer.fit(ckpt_path=...)``
restores ``Status.running`` before entering the context). ``__exit__``
calls ``complete()`` or ``fail()`` depending on whether an exception
occurred. ``__exit__`` never suppresses exceptions. A status guard in
``__exit__`` prevents double-completion when code inside the block
(e.g. ``Trainer.fit()``) already finalized the experiment.

Nested ``with experiment:`` blocks are supported via an internal depth
counter (``_context_depth``). Only the outermost ``__exit__`` finalizes;
inner exits decrement depth and pass exceptions through unchanged. The
depth counter is transient and not included in ``state_dict()``.

For non-``with`` usage, call ``start()`` / ``complete()`` / ``fail()`` directly.

Status is an internal enum (Lightning pattern). Prefer lifecycle methods
(start/complete/fail/cancel) over direct status assignment.

Hooks called by EpochLoop: on_epoch_complete(epoch, metrics, **kwargs) and
on_validation_complete(epoch, metrics, **kwargs). Both are no-ops on the base
class. Override in subclasses to react to epoch/validation completion.

Store uses experiment.id as the branch name by convention.
Path resolution is Config's responsibility, not Experiment's.

Epoch counters (BUG-047 documentation):

- ``store`` refs ``latest_epoch``: persisted tip for snapshots/materialize/checkouts.
  Managed by ``FileStore`` on ref mutations.
- ``Trainer.current_epoch``: loop cursor during ``fit``. Incremented by the epoch loop
  and reset on each ``fit`` call.
- ``Experiment.epoch``: logical experiment cursor. Must be explicitly aligned on
  rollback/checkout paths. ``rollback(epoch)`` sets ``experiment.epoch = epoch`` after
  ``store.checkout()``.

Rollback (BUG-038): ``rollback(epoch)`` checks out the store at the given epoch and
aligns ``experiment.epoch``. When ``store`` is None, rollback is a no-op. When epoch
is None, rollback is a no-op. A missing epoch propagates ``StoreError`` from checkout.

Post-complete snapshot policy (BUG-046): ``strict_snapshot_after_complete`` controls
whether ``FileStore.snapshot()`` raises ``ExperimentError`` (strict=True) or
proceeds silently (strict=False, default) when the experiment is completed.
"""

from autopilot.core.context import ContextLog
from autopilot.core.decision import DecisionEntry
from autopilot.core.enums import Status
from autopilot.core.errors import ExperimentError
from autopilot.core.traceable import Traceable
from autopilot.tracking.io import utc_now_iso
from collections import deque
from collections.abc import Callable
from typing import Any


def _require_status(
  experiment: 'Experiment',
  allowed: Status | tuple[Status, ...],
  action: str,
) -> None:
  """Raise ExperimentError unless experiment.status is in allowed set.

  Centralizes the lifecycle guard check used by ``start``, ``complete``,
  ``fail``, and ``advance_epoch``. Preserves the established error message
  format so existing tests and operators that match substrings continue to
  pass.

  Args:
    experiment: The experiment to check.
    allowed: A single ``Status`` value or tuple of allowed ``Status`` values
      that are valid for the action.
    action: Human-readable action verb for the error message (e.g. ``'start'``,
      ``'advance epoch'``).

  Raises:
    ExperimentError: When ``experiment.status`` is not in ``allowed``.
  """
  if isinstance(allowed, Status):
    allowed = (allowed,)
  if experiment.status not in allowed:
    expected = ' or '.join(s.value for s in allowed)
    msg = (
      f'cannot {action}: experiment id={experiment.id!r} '
      f'status={experiment.status.value!r}, expected {expected}'
    )
    raise ExperimentError(msg)


def validate_dependency_ids(
  candidate_deps: list[str],
  *,
  self_id: str,
  resolve: Callable[[str], Any | None],
  all_dependencies: Callable[[], dict[str, list[str]]],
) -> list[str]:
  """Validate and normalize dependency ids for an experiment.

  Checks that every id in *candidate_deps* resolves to an existing experiment
  (forest-wide) and that adding the edges ``self_id -> dep`` would not create
  a cycle in the directed dependency graph.

  Args:
    candidate_deps: Raw dependency id strings from CLI ``--depends-on``.
    self_id: The experiment id that will carry these dependencies.
    resolve: Callable that resolves an experiment id to a node (or ``None``
      if not found). Used for existence checking only.
    all_dependencies: Callable returning a dict mapping every experiment id
      in the forest to its current ``dependencies`` list. Used for cycle
      detection across the full graph.

  Returns:
    Sorted, de-duplicated list of validated dependency ids.

  Raises:
    ExperimentError: When a dependency id does not resolve or when adding
      the edges would create a cycle.
  """
  normalized = sorted(set(candidate_deps))
  for dep_id in normalized:
    if resolve(dep_id) is None:
      msg = (
        f'dependency {dep_id!r} not found in any tree; '
        'verify the experiment id exists before adding it as a dependency'
      )
      raise ExperimentError(msg)

  graph = all_dependencies()
  graph[self_id] = normalized

  if _has_cycle(graph, self_id):
    msg = (
      f'adding dependencies {normalized!r} to experiment {self_id!r} '
      'would create a cycle in the dependency graph'
    )
    raise ExperimentError(msg)

  return normalized


def _has_cycle(graph: dict[str, list[str]], start: str) -> bool:
  """Check if *start* can reach itself via BFS in the dependency graph.

  Args:
    graph: Adjacency list mapping experiment id to its dependency ids.
    start: Node to check for a cycle path back to itself.

  Returns:
    True when a cycle exists that includes *start*.
  """
  visited: set[str] = set()
  queue: deque[str] = deque(graph.get(start, []))
  while queue:
    node = queue.popleft()
    if node == start:
      return True
    if node in visited:
      continue
    visited.add(node)
    queue.extend(graph.get(node, []))
  return False


class Experiment(Traceable):
  """Base experiment. Subclass and override hooks as needed.

  Identity: public field ``id``, constructor kwonly ``experiment_id``. Accessing
  ``.experiment_id`` raises ``AttributeError`` with guidance to use ``.id``.

  Inherits traceability from :class:`~autopilot.core.traceable.Traceable`
  (context log via ``context_log`` property, ``add_context()`` method).
  ``state_dict()`` includes ``context_log`` and ``load_state_dict()`` requires
  it (clean break: ``KeyError`` if the key is missing from persisted state).

  Supports ``with experiment:`` context manager usage: ``__enter__`` calls
  ``start()`` when status is ``pending``, or allows re-entry when status is
  already ``running`` (checkpoint resume: ``Trainer.fit(ckpt_path=...)``
  restores ``Status.running`` before entering the context); raises
  ``ExperimentError`` for any other status. ``__exit__`` calls ``complete()``
  on normal exit or ``fail(error_msg)`` on exception, but only when status is
  still ``running`` (status guard prevents double-finalization). Exceptions
  are never suppressed (``__exit__`` always returns ``False``).

  Nested ``with experiment:`` blocks are safe: an internal ``_context_depth``
  counter ensures only the outermost ``__exit__`` finalizes. Inner exits
  decrement depth and pass exceptions through to the outer block.

  For non-``with`` usage, call ``start()`` / ``complete()`` / ``fail()`` directly.

  Properties: store (Store | None), last_accepted_epoch (int | None), epoch (int),
  status (Status), dataset_meta (dict[str, Any]), dependencies (list[str]),
  spec_version (str | None).

  Methods: start(), complete(metrics), fail(error, metrics), cancel(),
  invalidate(reason), rollback(epoch), advance_epoch(metrics),
  on_epoch_complete(epoch, metrics), on_validation_complete(epoch, metrics).

  ``complete()`` and ``fail()`` accept both ``pending`` and ``running`` status
  so CLI-only workflows that never call ``start()`` can finalize directly.
  ``cancel()`` accepts any non-terminal status.

  Two-tier design: Experiment is the base (data + manual state transitions).
  AutoPilotExperiment adds lifecycle hooks for Trainer integration.
  Trainer.fit uses ``with self._experiment:`` when an experiment is configured.

  Example:
    >>> from autopilot.core.experiment import Experiment
    >>>
    >>> experiment = Experiment('exp-demo')
    >>> with experiment:
    ...   experiment.complete({'accuracy': 1.0})
    >>>
    >>> experiment.status.value
    'completed'

  See ``Node``, ``Tree``, and ``Forest`` for experiment graph metadata in the
  workspace runtime.
  """

  should_rollback: bool = False

  def __init__(
    self,
    experiment_id: str,
    hypothesis: str | None = None,
    strict_snapshot_after_complete: bool = False,
  ) -> None:
    """Create an experiment in ``pending`` status with empty metrics.

    Calls ``super().__init__()`` first so ``Traceable`` initializes the
    ``context_log`` before any other fields are set.

    ``dataset_meta`` is an opaque ``dict[str, Any]`` for per-experiment dataset
    binding. Populate via ``compute_fingerprint(paths).to_dict()`` for
    interoperable fingerprint metadata. Included in ``state_dict`` /
    ``load_state_dict`` round-trips. Default empty dict -- no breaking change
    for callers who omit it.

    Args:
      experiment_id: Stable id (often matches store branch name).
      hypothesis: Optional human-readable hypothesis text.
      strict_snapshot_after_complete: When True, ``FileStore.snapshot()`` raises
        ``ExperimentError`` when the experiment is completed. When False
        (default), the snapshot proceeds silently (BUG-046).
    """
    super().__init__()
    self.id = experiment_id
    self.hypothesis = hypothesis
    self.strict_snapshot_after_complete = strict_snapshot_after_complete
    self.status: Status = Status.pending
    self.metrics: dict[str, Any] = {}
    self.dependencies: list[str] = []
    self.dataset_meta: dict[str, Any] = {}
    self.notes: str | None = None
    self.created_at: str = utc_now_iso()
    self.started_at: str | None = None
    self.completed_at: str | None = None
    self.failed_at: str | None = None
    self.cancelled_at: str | None = None
    self.invalidated_at: str | None = None
    self.epoch: int = -1
    self.error: str | None = None
    self.spec_version: str | None = None
    self._store: Any = None
    self._last_accepted_epoch: int | None = None
    self._context_depth: int = 0

  @property
  def is_terminal(self) -> bool:
    """True when experiment is completed, failed, cancelled, or invalidated."""
    return self.status.is_terminal

  def start(self) -> None:
    """Transition pending -> running. Sets started_at timestamp."""
    _require_status(self, Status.pending, 'start')
    self.status = Status.running
    self.started_at = utc_now_iso()

  def complete(self, metrics: dict[str, Any] | None = None) -> None:
    """Transition pending/running -> completed. Sets completed_at timestamp.

    Accepts both ``pending`` and ``running`` so CLI-only workflows that
    never call ``start()`` can finalize directly.

    Args:
      metrics: Optional final metrics dict. Replaces ``self.metrics`` when
        not None.
    """
    _require_status(self, (Status.pending, Status.running), 'complete')
    if metrics is not None:
      self.metrics = metrics
    self.status = Status.completed
    self.completed_at = utc_now_iso()

  def fail(self, error: str | None = None, metrics: dict[str, Any] | None = None) -> None:
    """Transition pending/running -> failed. Sets failed_at timestamp and error.

    Accepts both ``pending`` and ``running`` so CLI-only workflows that
    never call ``start()`` can mark experiments as failed directly.

    When ``metrics`` is provided, replaces ``self.metrics`` with the given
    dict before the status flip. This allows recording diagnostic metrics
    on failure (e.g. partial accuracy, error counts).

    Gate-reject path: when a policy gate rejects all epochs,
    ``Trainer._fit_success_path`` calls ``fail('policy gate rejected all
    epochs')`` instead of ``_complete_experiment_success``.

    Args:
      error: Optional failure reason string.
      metrics: Optional metrics dict to replace current metrics on failure.
    """
    _require_status(self, (Status.pending, Status.running), 'fail')
    if metrics is not None:
      self.metrics = metrics
    if error is not None:
      self.error = error
    self.status = Status.failed
    self.failed_at = utc_now_iso()

  def cancel(self) -> None:
    """Transition pending/running -> cancelled. Sets cancelled_at timestamp.

    Allowed from pending and running only.

    Raises:
      ExperimentError: When the experiment is already in a terminal state.
    """
    if self.status.is_terminal:
      msg = f'cannot cancel: experiment id={self.id!r} is terminal (status={self.status.value!r})'
      raise ExperimentError(
        msg,
      )
    self.status = Status.cancelled
    self.cancelled_at = utc_now_iso()

  def invalidate(self, reason: str) -> None:
    """Transition completed -> invalidated. Sets invalidated_at timestamp.

    Only experiments in ``completed`` status can be invalidated. This marks
    an experiment as historically bad without deleting it from the tree.
    Records the reason via ``add_context`` for audit traceability.

    Args:
      reason: Human-readable explanation for why this experiment is invalid.
    """
    _require_status(self, Status.completed, 'invalidate')
    self.status = Status.invalidated
    self.invalidated_at = utc_now_iso()
    self.add_context(
      reason,
      source='user',
      metadata={'action': 'invalidate'},
    )

  @property
  def store(self) -> Any:
    """The store backing this experiment. None if no store is wired."""
    return self._store

  @store.setter
  def store(self, value: Any) -> None:
    self._store = value

  @property
  def last_accepted_epoch(self) -> int | None:
    """Last training epoch where policy accepted (passed gate). none if none yet."""
    return self._last_accepted_epoch

  @last_accepted_epoch.setter
  def last_accepted_epoch(self, value: int | None) -> None:
    self._last_accepted_epoch = value

  def rollback(self, epoch: int | None) -> None:
    """Rollback to a previous epoch's state via Store checkout.

    Rewinds checked-out parameter files and aligns ``experiment.epoch``
    with the store tip for that epoch. When ``store`` is None or ``epoch``
    is None, this is a no-op (no store to checkout from).

    After the rollback is effective, appends a ``ContextEntry`` via
    ``add_context`` (DRY-07: no ``emit_context`` since no Trainer is
    in scope). Metadata is built via ``DecisionEntry.rollback()`` for
    typed filtering by ``_type`` discriminator.

    A missing epoch propagates ``StoreError`` from the store's checkout.

    Args:
      epoch: Target epoch to restore, or None for no-op.
    """
    if self.store is None or epoch is None:
      return
    reason_text = f'rolled back to epoch {epoch}'
    self.store.checkout(self.id, epoch, context=reason_text)
    self.epoch = epoch
    metadata = DecisionEntry.rollback(
      target_epoch=epoch,
      reason=reason_text,
    )
    self.add_context(
      reason_text,
      source='trainer',
      epoch=self.epoch,
      metadata=metadata,
    )

  def on_epoch_complete(self, epoch: int, metrics: dict[str, Any], **kwargs: Any) -> None:
    """No-op hook called by EpochLoop after each training epoch.

    Args:
      epoch: 0-based epoch index.
      metrics: Aggregated training metrics for the epoch.
      **kwargs: Reserved for forward compatibility.
    """

  def on_validation_complete(self, epoch: int, metrics: dict[str, Any], **kwargs: Any) -> None:
    """No-op hook called by EpochLoop after validation.

    Args:
      epoch: 0-based epoch index.
      metrics: Aggregated validation metrics for the epoch.
      **kwargs: Reserved for forward compatibility.
    """

  def advance_epoch(self, metrics: dict[str, Any] | None = None) -> None:
    """Increment epoch counter. Must be running.

    Called by EpochLoop after on_epoch_complete and on_validation_complete.
    Only increments the counter and optionally stores metrics.

    Args:
      metrics: Optional metrics dict. Replaces ``self.metrics`` when not None.
    """
    _require_status(self, Status.running, 'advance epoch')
    self.epoch += 1
    if metrics is not None:
      self.metrics = metrics

  def state_dict(self) -> dict[str, Any]:
    """Serialize all fields. Status as .value string.

    Includes ``context_log`` as a list of dicts (via ``ContextLog.to_list()``).

    Returns:
      Plain dict suitable for JSON or checkpoint storage.
    """
    return {
      'id': self.id,
      'hypothesis': self.hypothesis,
      'status': self.status.value,
      'metrics': dict(self.metrics),
      'dependencies': list(self.dependencies),
      'dataset_meta': dict(self.dataset_meta),
      'notes': self.notes,
      'created_at': self.created_at,
      'started_at': self.started_at,
      'completed_at': self.completed_at,
      'failed_at': self.failed_at,
      'cancelled_at': self.cancelled_at,
      'invalidated_at': self.invalidated_at,
      'epoch': self.epoch,
      'error': self.error,
      'spec_version': self.spec_version,
      'last_accepted_epoch': self._last_accepted_epoch,
      'strict_snapshot_after_complete': self.strict_snapshot_after_complete,
      'context_log': self._context_log.to_list(),
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    """Restore all fields from state dict. Resolves Status enum.

    Args:
      state: Dict previously returned by :meth:`state_dict`. Must contain
        ``context_log`` key (clean break: ``KeyError`` if missing).
    """
    self.id = state['id']
    self.hypothesis = state['hypothesis']
    self.status = Status(state['status'])
    self.metrics = dict(state['metrics'])
    self.dependencies = list(state.get('dependencies', []))
    self.dataset_meta = dict(state.get('dataset_meta', {}))
    self.notes = state['notes']
    self.created_at = state['created_at']
    self.started_at = state['started_at']
    self.completed_at = state['completed_at']
    self.failed_at = state['failed_at']
    self.cancelled_at = state['cancelled_at']
    self.invalidated_at = state.get('invalidated_at')
    self.epoch = state['epoch']
    self.error = state['error']
    self.spec_version = state.get('spec_version')
    self._last_accepted_epoch = state['last_accepted_epoch']
    self.strict_snapshot_after_complete = state.get('strict_snapshot_after_complete', False)
    self._context_log = ContextLog.from_list(state['context_log'])

  def __enter__(self) -> 'Experiment':
    """Start the experiment (or allow resume if already running).

    Re-entrant: nested ``with experiment:`` blocks increment an internal depth
    counter. The status check (pending -> start, running -> allow) only runs on
    the outermost entry (depth 0). Inner entries increment depth and return self
    without touching status.

    Returns:
      This experiment instance in running status.

    Raises:
      ExperimentError: When status is neither ``pending`` nor ``running``
        (checked only on outermost entry).
    """
    if self._context_depth == 0:
      if self.status == Status.pending:
        self.start()
      elif self.status != Status.running:
        msg = (
          f'cannot enter context: experiment id={self.id!r} status={self.status.value!r}; '
          'expected pending or running (running is allowed when resuming from checkpoint)'
        )
        raise ExperimentError(msg)
    self._context_depth += 1
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: Any,
  ) -> bool:
    """Complete or fail the experiment based on exception state.

    Re-entrant: decrements the internal depth counter. Only the outermost exit
    (depth reaching 0) finalizes the experiment. Inner exits return ``False``
    without touching status, allowing exceptions to propagate naturally to the
    outer block.

    If status is no longer ``running`` at the outermost exit (e.g.
    ``Trainer.fit()`` already called ``complete()`` or ``fail()``), this is a
    no-op -- prevents double-finalization that would raise ``ExperimentError``.

    Args:
      exc_type: Exception type, or None on normal exit.
      exc_val: Exception instance, or None. When exc_type is set but exc_val
        is None, the error message falls back to ``exc_type.__name__``.
      exc_tb: Traceback, or None.

    Returns:
      False -- never suppresses exceptions.
    """
    self._context_depth -= 1
    if self._context_depth > 0:
      return False
    if self.status != Status.running:
      return False
    if exc_type is None:
      self.complete()
    else:
      error_msg = (
        str(exc_val)
        if exc_val is not None
        else (exc_type.__name__ if exc_type is not None else 'unknown error')
      )
      self.fail(error_msg)
    return False

  def __getattr__(self, name: str) -> Any:
    """Provide guidance for common attribute access mistakes.

    Intercepts ``experiment_id`` to surface that the correct attribute is
    ``.id`` (constructor kwarg is ``experiment_id=...``, stored as ``id``).
    All other missing attributes raise the standard ``AttributeError``.

    Raises:
      AttributeError: Always (either with guidance for ``experiment_id``
        or the standard missing-attribute message).
    """
    if name == 'experiment_id':
      msg = (
        "Experiment has no attribute 'experiment_id'; use '.id' "
        '(constructor argument is experiment_id=..., stored as id).'
      )
      raise AttributeError(msg)
    msg = f'{type(self).__name__!r} object has no attribute {name!r}'
    raise AttributeError(msg)

  def __repr__(self) -> str:
    """Return a concise debug representation with id and status."""
    return f'{type(self).__name__}(id={self.id!r}, status={self.status.value})'
