from autopilot.core.callbacks import Callback
from autopilot.core.models import Result
from autopilot.core.store import Store
from typing import Any, Callable


def _epoch_for_store(epoch: int, result: Any) -> int:
  if isinstance(result, dict) and 'epoch' in result:
    return int(result['epoch'])
  return epoch


class StoreCheckpoint(Callback):
  """Snapshots the store at each epoch end. Like ModelCheckpoint."""

  def __init__(self, store: Store) -> None:
    self._store = store
    self._last_epoch: int | None = None

  def on_epoch_end(
    self,
    trainer: Any,
    epoch: int,
    result: Result | None = None,
  ) -> None:
    snap_epoch = _epoch_for_store(epoch, result)
    self._store.snapshot(snap_epoch)
    self._last_epoch = snap_epoch

  def state_dict(self) -> dict[str, Any]:
    return {'last_epoch': self._last_epoch}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    self._last_epoch = state_dict.get('last_epoch')


class StorePromoter(Callback):
  """Promotes a store epoch when a predicate returns True."""

  def __init__(self, store: Store, promote_on: Callable[[int, Any], bool]) -> None:
    self._store = store
    self._promote_on = promote_on

  def on_epoch_end(
    self,
    trainer: Any,
    epoch: int,
    result: Result | None = None,
  ) -> None:
    eff_epoch = _epoch_for_store(epoch, result)
    if self._promote_on(eff_epoch, result):
      self._store.promote(eff_epoch)
