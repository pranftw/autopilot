"""Tests for store-related Trainer callbacks."""

from autopilot.core.store import SnapshotManifest
from autopilot.core.store_callbacks import StoreCheckpoint, StorePromoter


class RecordingStore:
  """Minimal stand-in for Store snapshot/promote hooks."""

  def __init__(self) -> None:
    self.snapshot_epochs: list[int] = []
    self.promote_epochs: list[int] = []

  def snapshot(self, epoch: int) -> SnapshotManifest:
    self.snapshot_epochs.append(epoch)
    return SnapshotManifest(epoch=epoch, timestamp='', entries={})

  def promote(self, epoch: int) -> None:
    self.promote_epochs.append(epoch)


class TestStoreCheckpoint:
  def test_snapshot_per_epoch(self) -> None:
    store = RecordingStore()
    cb = StoreCheckpoint(store)
    for epoch in (1, 2, 3):
      cb.on_epoch_end(trainer=None, epoch=epoch, result={'epoch': epoch})
    assert store.snapshot_epochs == [1, 2, 3]

  def test_epoch_from_result_dict(self) -> None:
    store = RecordingStore()
    cb = StoreCheckpoint(store)
    cb.on_epoch_end(trainer=None, epoch=1, result={'epoch': 42})
    assert store.snapshot_epochs == [42]

  def test_missing_epoch_in_result_uses_hook_epoch(self) -> None:
    store = RecordingStore()
    cb = StoreCheckpoint(store)
    cb.on_epoch_end(trainer=None, epoch=7, result={})
    cb.on_epoch_end(trainer=None, epoch=8, result=None)
    assert store.snapshot_epochs == [7, 8]

  def test_state_dict_returns_last_epoch(self) -> None:
    store = RecordingStore()
    cb = StoreCheckpoint(store)
    cb.on_epoch_end(trainer=None, epoch=1, result=None)
    assert cb.state_dict() == {'last_epoch': 1}

  def test_load_state_dict_restores_last_epoch(self) -> None:
    store = RecordingStore()
    cb = StoreCheckpoint(store)
    cb.load_state_dict({'last_epoch': 9})
    assert cb._last_epoch == 9

  def test_state_dict_fresh_is_none(self) -> None:
    store = RecordingStore()
    cb = StoreCheckpoint(store)
    assert cb.state_dict() == {'last_epoch': None}


class TestStorePromoter:
  def test_promotes_when_true(self) -> None:
    store = RecordingStore()
    cb = StorePromoter(store, lambda e, r: True)
    cb.on_epoch_end(trainer=None, epoch=2, result=None)
    assert store.promote_epochs == [2]

  def test_no_promote_when_false(self) -> None:
    store = RecordingStore()
    cb = StorePromoter(store, lambda e, r: False)
    cb.on_epoch_end(trainer=None, epoch=2, result=None)
    assert store.promote_epochs == []

  def test_predicate_receives_epoch_and_result(self) -> None:
    store = RecordingStore()
    seen: list[tuple[int, object]] = []

    def promote_on(e: int, r: object) -> bool:
      seen.append((e, r))
      return False

    cb = StorePromoter(store, promote_on)
    payload = {'epoch': 5, 'loss': 0.1}
    cb.on_epoch_end(trainer=None, epoch=1, result=payload)
    assert seen == [(5, payload)]

  def test_multiple_epochs_sequential(self) -> None:
    store = RecordingStore()
    cb = StoreCheckpoint(store)
    for n in range(1, 5):
      cb.on_epoch_end(trainer=None, epoch=n, result=None)
    assert store.snapshot_epochs == [1, 2, 3, 4]


class TestStorePromoterEvenEpochs:
  def test_promote_only_on_predicate_epochs(self) -> None:
    store = RecordingStore()
    cb = StorePromoter(store, lambda e, _: e % 2 == 0)
    for n in (1, 2, 3, 4):
      cb.on_epoch_end(trainer=None, epoch=n, result=None)
    assert store.promote_epochs == [2, 4]
