"""Tests for `_latest_snapshot_file` and `AutoPilotConfig.stabilize` snapshot selection."""

from autopilot.core.config import AutoPilotConfig, _latest_snapshot_file
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
import hashlib


def test_latest_snapshot_prefers_epoch_11_over_epoch_9(tmp_path: Path) -> None:
  """Numeric epoch ordering must beat lexicographic filename ordering."""
  snapshots = tmp_path / 'snapshots'
  snapshots.mkdir()
  for epoch in range(12):
    (snapshots / f'epoch_{epoch}.json').write_text('{}', encoding='utf-8')
  result = _latest_snapshot_file(snapshots)
  assert result is not None
  assert result.name == 'epoch_11.json'


def test_latest_snapshot_prefers_epoch_10_over_epoch_9(tmp_path: Path) -> None:
  """Two-digit epoch suffix must rank above single-digit when 10 > 9 numerically."""
  snapshots = tmp_path / 'snapshots'
  snapshots.mkdir()
  (snapshots / 'epoch_9.json').write_text('{}', encoding='utf-8')
  (snapshots / 'epoch_10.json').write_text('{}', encoding='utf-8')
  result = _latest_snapshot_file(snapshots)
  assert result is not None
  assert result.name == 'epoch_10.json'


def test_latest_snapshot_single_epoch_zero(tmp_path: Path) -> None:
  """A sole epoch_0.json file is selected."""
  snapshots = tmp_path / 'snapshots'
  snapshots.mkdir()
  sole = snapshots / 'epoch_0.json'
  sole.write_text('{}', encoding='utf-8')
  result = _latest_snapshot_file(snapshots)
  assert result == sole


def test_latest_snapshot_two_epochs_picks_max(tmp_path: Path) -> None:
  """Among multiple epoch files, the greatest epoch wins."""
  snapshots = tmp_path / 'snapshots'
  snapshots.mkdir()
  (snapshots / 'epoch_1.json').write_text('{}', encoding='utf-8')
  expected = snapshots / 'epoch_4.json'
  expected.write_text('{}', encoding='utf-8')
  result = _latest_snapshot_file(snapshots)
  assert result == expected


def test_latest_snapshot_empty_returns_none(tmp_path: Path) -> None:
  """An existing but empty directory yields no snapshot."""
  snapshots = tmp_path / 'snapshots'
  snapshots.mkdir()
  assert _latest_snapshot_file(snapshots) is None


def test_latest_snapshot_ignores_non_epoch_json(tmp_path: Path) -> None:
  """Only filenames matching epoch_<int>.json participate."""
  snapshots = tmp_path / 'snapshots'
  snapshots.mkdir()
  (snapshots / 'notes.json').write_text('{}', encoding='utf-8')
  expected = snapshots / 'epoch_3.json'
  expected.write_text('{}', encoding='utf-8')
  result = _latest_snapshot_file(snapshots)
  assert result == expected


def test_latest_snapshot_ignores_epoch_with_extra_suffix(tmp_path: Path) -> None:
  """Names like epoch_001.backup.json do not match the epoch snapshot pattern."""
  snapshots = tmp_path / 'snapshots'
  snapshots.mkdir()
  (snapshots / 'epoch_001.backup.json').write_text('{}', encoding='utf-8')
  expected = snapshots / 'epoch_2.json'
  expected.write_text('{}', encoding='utf-8')
  result = _latest_snapshot_file(snapshots)
  assert result == expected


def test_latest_snapshot_skips_directories(tmp_path: Path) -> None:
  """A directory named like an epoch file is ignored."""
  snapshots = tmp_path / 'snapshots'
  snapshots.mkdir()
  (snapshots / 'epoch_5.json').mkdir()
  expected = snapshots / 'epoch_1.json'
  expected.write_text('{}', encoding='utf-8')
  result = _latest_snapshot_file(snapshots)
  assert result == expected


def test_latest_snapshot_nonexistent_dir_returns_none(tmp_path: Path) -> None:
  """Missing snapshots directory returns None."""
  missing = tmp_path / 'no_such_snapshots'
  assert _latest_snapshot_file(missing) is None


def test_autopilot_config_stabilize_uses_numeric_latest(tmp_path: Path) -> None:
  """`stabilize` loads the highest numeric epoch, not lexicographic order."""
  config = AutoPilotConfig(workspace=tmp_path)
  snapshots_dir = config.snapshots_path / 'exp-001'
  snapshots_dir.mkdir(parents=True, exist_ok=True)
  objects_dir = config.objects_path
  objects_dir.mkdir(parents=True, exist_ok=True)

  rel_path = 'stabilized/out.txt'
  content_nine = b'content from epoch 9'
  hash_nine = hashlib.sha256(content_nine).hexdigest()
  nine_blob = objects_dir / hash_nine[:2] / hash_nine[2:]
  nine_blob.parent.mkdir(parents=True, exist_ok=True)
  nine_blob.write_bytes(content_nine)

  manifest_nine = {
    'epoch': 9,
    'timestamp': '2026-05-01T12:00:00+00:00',
    'entries': {
      'param': {
        'digest': hash_nine,
        'size': len(content_nine),
        'mtime': 0.0,
        'original_path': rel_path,
      },
    },
  }
  atomic_write_json(snapshots_dir / 'epoch_9.json', manifest_nine)

  content_eleven = b'content from epoch 11'
  hash_eleven = hashlib.sha256(content_eleven).hexdigest()
  eleven_blob = objects_dir / hash_eleven[:2] / hash_eleven[2:]
  eleven_blob.parent.mkdir(parents=True, exist_ok=True)
  eleven_blob.write_bytes(content_eleven)

  manifest_eleven = {
    'epoch': 11,
    'timestamp': '2026-05-02T12:00:00+00:00',
    'entries': {
      'param': {
        'digest': hash_eleven,
        'size': len(content_eleven),
        'mtime': 0.0,
        'original_path': rel_path,
      },
    },
  }
  atomic_write_json(snapshots_dir / 'epoch_11.json', manifest_eleven)

  copied = config.stabilize('exp-001')
  dst = tmp_path / rel_path
  assert dst.read_bytes() == content_eleven
  assert copied == [dst]
