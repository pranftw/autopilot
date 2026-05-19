"""Execution isolation abstraction for experiments.

Environment answers "where and how does experiment code execute?" --
worktree isolation, dependency setup, process sandboxing. It is a
project-level concern (set on Config), not an experiment-level concern.

``LocalEnvironment`` is the zero-config builtin (no isolation, current
working directory). Subclass ``Environment`` and override ``setup`` /
``teardown`` for custom isolation strategies (e.g. worktree-based
isolation in ``autopilot.ai.environment``).

The ``activate`` context manager on the base class composes setup/teardown
into a single ``with`` block. It works for all subclasses without override.
"""

from abc import ABC, abstractmethod
from autopilot.core.experiment import Experiment
from autopilot.core.module.module import Module
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol


class WorktreeStore(Protocol):
  """Structural type compatible with ``autopilot.core.store.base.Store``.

  Defined here to avoid a config->environment->store->config import cycle.
  Callers pass the real ``Store`` from ``core.store`` at runtime.
  Declares the subset of Store API used by Environment subclasses.
  """

  def create_worktree(self, experiment_id: str) -> Path:
    """Create an isolated working directory. Returns worktree path."""
    ...

  def remove_worktree(self, experiment_id: str) -> None:
    """Remove a worktree and clean up its directory."""
    ...

  def load_refs(self) -> dict[str, Any]:
    """Load the refs structure (branches, HEAD, etc.)."""
    ...

  def load_snapshot(self, experiment_id: str, epoch: int) -> Any:
    """Load a snapshot manifest for the given experiment and epoch."""
    ...

  def read_object(self, content_hash: str) -> bytes:
    """Read raw object content by content hash."""
    ...


class Environment(ABC):
  """Where and how experiment code executes in isolation.

  Subclass and override ``setup`` for custom isolation strategies.
  ``LocalEnvironment`` is the zero-config builtin (no isolation).
  ``teardown`` has an empty default and only needs overriding when
  cleanup is required.
  """

  @abstractmethod
  def setup(self, experiment: Experiment, store: WorktreeStore, module: Module) -> Path:
    """Prepare isolated workspace. Returns working directory path.

    Args:
      experiment: Experiment to isolate.
      store: Store for snapshot/worktree access.
      module: Module whose parameters drive copy-vs-symlink decisions.

    Returns:
      Path to the working directory for this experiment.
    """

  def teardown(self, experiment: Experiment) -> None:
    """Clean up isolation resources.

    Args:
      experiment: Experiment whose environment is being torn down.
    """

  @contextmanager
  def activate(
    self,
    experiment: Experiment,
    store: WorktreeStore,
    module: Module,
  ) -> Generator[Path, None, None]:
    """Context manager: setup on enter, teardown on exit.

    Args:
      experiment: Experiment to isolate.
      store: Store for snapshot/worktree access.
      module: Module whose parameters drive copy-vs-symlink decisions.

    Yields:
      Working directory path.
    """
    try:
      workspace = self.setup(experiment, store, module)
      yield workspace
    finally:
      self.teardown(experiment)


class LocalEnvironment(Environment):
  """Zero-isolation default. Runs in the current working directory."""

  def setup(self, experiment: Experiment, store: WorktreeStore, module: Module) -> Path:
    """Return current working directory without any isolation.

    Returns:
      Path.cwd() -- no worktree, no copies, no symlinks.
    """
    return Path.cwd()

  def teardown(self, experiment: Experiment) -> None:
    """No-op. Nothing to clean up for local execution."""
