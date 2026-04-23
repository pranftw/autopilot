"""Store-related Trainer callbacks: snapshot and promote."""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.models import Result
from typing import Any, Callable


def _epoch_for_store(epoch: int, result: Any) -> int:
  if isinstance(result, dict) and 'epoch' in result:
    return int(result['epoch'])
  return epoch


class StoreCheckpointCallback(Callback):
  """Snapshots the store at each epoch end. Like ModelCheckpoint.

  Gets store from trainer.experiment.store -- single source of truth.
  """

  def __init__(self) -> None:
    self._last_epoch: int | None = None

  def on_epoch_end(
    self,
    trainer: Any,
    epoch: int,
    result: Result | None = None,
  ) -> None:
    experiment = trainer.experiment
    if not experiment or not experiment.store:
      return
    snap_epoch = _epoch_for_store(epoch, result)
    experiment.store.snapshot(snap_epoch)
    self._last_epoch = snap_epoch

  def state_dict(self) -> dict[str, Any]:
    return {'last_epoch': self._last_epoch}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    self._last_epoch = state_dict.get('last_epoch')


class StorePromoterCallback(Callback):
  """Promotes a store epoch when a predicate returns True.

  Gets store from trainer.experiment.store -- single source of truth.
  """

  def __init__(self, promote_on: Callable[[int, Any], bool]) -> None:
    self._promote_on = promote_on

  def on_epoch_end(
    self,
    trainer: Any,
    epoch: int,
    result: Result | None = None,
  ) -> None:
    experiment = trainer.experiment
    if not experiment or not experiment.store:
      return
    eff_epoch = _epoch_for_store(epoch, result)
    if self._promote_on(eff_epoch, result):
      experiment.store.promote(eff_epoch)
