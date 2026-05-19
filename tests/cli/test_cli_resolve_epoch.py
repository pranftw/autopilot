"""Tests for resolve_epoch and CLIError in autopilot.cli.helpers."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.helpers import CLIError, resolve_epoch
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import AutoPilotError
from pathlib import Path
import pytest


def _make_store_with_epochs(
  tmp_path: Path,
  experiment_id: str,
  epoch_count: int,
) -> FileStore:
  """Create a FileStore with ``epoch_count`` snapshots for ``experiment_id``."""
  src = tmp_path / 'src'
  src.mkdir(parents=True, exist_ok=True)
  (src / 'file.txt').write_text('v0')
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path = tmp_path / '.autopilot'
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  for epoch in range(epoch_count):
    (src / 'file.txt').write_text(f'v{epoch}')
    store.snapshot(experiment_id, epoch)
  return store


class TestCLIError:
  def test_is_autopilot_error_subclass(self) -> None:
    assert issubclass(CLIError, AutoPilotError)

  def test_message_round_trip(self) -> None:
    exc = CLIError('something went wrong')
    assert str(exc) == 'something went wrong'


class TestResolveEpochInt:
  def test_valid_int(self, tmp_path: Path) -> None:
    store = _make_store_with_epochs(tmp_path, 'exp', 3)
    assert resolve_epoch(0, store, 'exp') == 0
    assert resolve_epoch(1, store, 'exp') == 1
    assert resolve_epoch(2, store, 'exp') == 2

  def test_negative_int_raises(self, tmp_path: Path) -> None:
    store = _make_store_with_epochs(tmp_path, 'exp', 3)
    with pytest.raises(CLIError, match='invalid epoch'):
      resolve_epoch(-1, store, 'exp')

  def test_greater_than_latest_raises(self, tmp_path: Path) -> None:
    store = _make_store_with_epochs(tmp_path, 'exp', 3)
    with pytest.raises(CLIError, match='not found'):
      resolve_epoch(10, store, 'exp')


class TestResolveEpochLatest:
  def test_latest_resolves_to_tip(self, tmp_path: Path) -> None:
    store = _make_store_with_epochs(tmp_path, 'exp', 5)
    assert resolve_epoch('latest', store, 'exp') == 4

  def test_latest_case_insensitive(self, tmp_path: Path) -> None:
    store = _make_store_with_epochs(tmp_path, 'exp', 2)
    assert resolve_epoch('Latest', store, 'exp') == 1
    assert resolve_epoch('LATEST', store, 'exp') == 1

  def test_latest_empty_store_raises(self, tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir(parents=True, exist_ok=True)
    (src / 'file.txt').write_text('v0')
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path = tmp_path / '.autopilot'
    param = PathParameter(source=str(src), pattern='*')
    store = FileStore(config)
    store.register_parameters({'source': param})
    with pytest.raises(CLIError):
      resolve_epoch('latest', store, 'exp')


class TestResolveEpochNumericString:
  def test_numeric_string(self, tmp_path: Path) -> None:
    store = _make_store_with_epochs(tmp_path, 'exp', 3)
    assert resolve_epoch('2', store, 'exp') == 2

  def test_numeric_string_invalid_raises(self, tmp_path: Path) -> None:
    store = _make_store_with_epochs(tmp_path, 'exp', 3)
    with pytest.raises(CLIError, match='not found'):
      resolve_epoch('99', store, 'exp')


class TestResolveEpochInvalidInput:
  def test_non_numeric_string_raises(self, tmp_path: Path) -> None:
    store = _make_store_with_epochs(tmp_path, 'exp', 3)
    with pytest.raises(CLIError, match='invalid epoch'):
      resolve_epoch('abc', store, 'exp')

  def test_whitespace_string_raises(self, tmp_path: Path) -> None:
    store = _make_store_with_epochs(tmp_path, 'exp', 3)
    with pytest.raises(CLIError, match='invalid epoch'):
      resolve_epoch('  ', store, 'exp')

  def test_negative_numeric_string_raises(self, tmp_path: Path) -> None:
    store = _make_store_with_epochs(tmp_path, 'exp', 3)
    with pytest.raises(CLIError, match='invalid epoch'):
      resolve_epoch('-1', store, 'exp')
