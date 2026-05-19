"""Shared checkpoint parallel-run helpers for eval generator and judge agents."""

from autopilot.ai.evaluation.checkpoints import CheckpointManager
from autopilot.ai.evaluation.protocols import EvaluationOutputProtocol
from autopilot.ai.runtime import ParallelRunner, SlidingWindowLimiter
from autopilot.core.errors import AIError, TrackingError
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

CONFIG_HASH_HEX_LEN = 16


def hash_eval_config(config_data: dict[str, Any]) -> str:
  """SHA-256 hex prefix of serialized config for checkpoint identification.

  Produces a short deterministic digest suitable for tagging checkpoint files.
  Callers should pass ``config.model_dump(mode='json')`` to get a dict with
  JSON-compatible types that round-trips identically to Pydantic's
  ``model_dump_json()`` byte stream.

  Args:
    config_data: JSON-compatible dict (string keys, JSON-friendly values).

  Returns:
    First 16 hex characters of the SHA-256 digest.
  """
  raw = json.dumps(config_data, separators=(',', ':'))
  return hashlib.sha256(raw.encode()).hexdigest()[:CONFIG_HASH_HEX_LEN]


class EvalRunContext:
  """Shared evaluation run lifecycle: checkpoint, progress, resume.

  Encapsulates the repeated boilerplate in GeneratorAgent and JudgeAgent
  for managing output directories, checkpoint files, and resume state.

  Does NOT own or hold a ``CheckpointManager`` instance; each method that
  needs checkpoint access opens the checkpoint file read-only, matching the
  pre-refactor behavior.

  Args:
    output_dir: Directory containing checkpoint and output files.
    config_hash: Short hex digest identifying the config (from
      :func:`hash_eval_config`).
    total_items: Planned population size used for resume banners, not
      necessarily the number of lines in the checkpoint.
  """

  def __init__(
    self,
    output_dir: Path,
    config_hash: str,
    total_items: int,
  ) -> None:
    """Initialize an evaluation run context.

    Args:
      output_dir: Directory containing checkpoint and output files.
      config_hash: Short hex digest from :func:`hash_eval_config`.
      total_items: Planned population size for resume banners.
    """
    self._output_dir = output_dir
    self._config_hash = config_hash
    self._total_items = total_items

  @property
  def output_dir(self) -> Path:
    """The output directory for this evaluation run."""
    return self._output_dir

  @property
  def config_hash(self) -> str:
    """Short hex digest identifying the evaluation config."""
    return self._config_hash

  @property
  def total_items(self) -> int:
    """Planned population size for the run."""
    return self._total_items

  def checkpoint_path(self) -> Path:
    """Return the canonical checkpoint file path.

    Returns:
      ``output_dir / 'checkpoint.jsonl'``.
    """
    return self._output_dir / 'checkpoint.jsonl'

  def _completed_ids(self) -> set[str]:
    """Read completed item ids from checkpoint without caching.

    Returns:
      Set of item id strings that have ``result`` events in the checkpoint.
    """
    ckpt_path = self.checkpoint_path()
    if not ckpt_path.exists():
      return set()
    try:
      mgr = CheckpointManager(ckpt_path)
    except (AIError, TrackingError, OSError, ValueError):
      return set()
    return mgr.completed_ids()

  def load_completed_count(self) -> int:
    """Count completed items by reading the checkpoint file.

    Opens the checkpoint read-only on each call (no cached manager).

    Returns:
      Number of items with ``result`` events, or ``0`` when the file is
      missing or empty.
    """
    return len(self._completed_ids())

  def should_skip(self, item_id: str) -> bool:
    """Check whether an item was already completed in the checkpoint.

    Opens the checkpoint read-only on each call (no cached manager).

    Args:
      item_id: Stable string identifier matching ``CheckpointManager`` ids.

    Returns:
      ``True`` when ``item_id`` has a prior ``result`` event.
    """
    return item_id in self._completed_ids()


async def run_parallel_items(
  items: list[T],
  process_fn: Callable[[T], Awaitable[R]],
  limiter: SlidingWindowLimiter | None,
  max_concurrent: int,
  output: EvaluationOutputProtocol,
  *,
  on_complete: Callable[[R], None] | None = None,
) -> list[R]:
  """Run items through process_fn with concurrency and rate limiting.

  Returns:
    List of per-item results from ``process_fn``, in input order.
  """
  _ = output
  runner = ParallelRunner(max_concurrent, limiter=limiter)
  return await runner.run(items, process_fn, on_complete=on_complete)


def resume_from_checkpoint(
  checkpoint_mgr: CheckpointManager,
  output: EvaluationOutputProtocol,
  total_planned: int,
) -> tuple[int, int]:
  """Print resume banner and return (n_done, n_remaining).

  Returns:
    Tuple ``(completed_count, remaining_count)`` based on the checkpoint.
  """
  n_done = len(checkpoint_mgr.completed_ids())
  n_remaining = total_planned - n_done
  output.info(f'Resuming: {n_done} done, {n_remaining} remaining')
  return n_done, n_remaining


def write_checkpoint_header(
  output_dir: Path,
  config_hash: str,
  subsystem: str,
  metadata: dict,
) -> CheckpointManager:
  """Create output_dir, write checkpoint header, return the manager.

  Returns:
    Initialized :class:`CheckpointManager` for the run's JSONL path.
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  ckpt_path = output_dir / 'checkpoint.jsonl'
  ckpt = CheckpointManager(ckpt_path)
  ckpt.save_header(config_hash=config_hash, subsystem=subsystem, args=dict(metadata))
  return ckpt


def log_item_failure(
  item_id: str,
  exc: BaseException,
  output: EvaluationOutputProtocol,
  *,
  unexpected: bool = False,
) -> None:
  """Log a per-item failure via logger and Output.warn."""
  if unexpected:
    logger.warning('%s failed (unexpected): %s', item_id, exc, exc_info=exc)
  else:
    logger.warning('%s failed: %s', item_id, exc)
  output.warn(f'{item_id} failed ({type(exc).__name__}): {exc}')
