"""Dataset fingerprinting utilities for reproducibility and drift detection.

Provides cryptographic hashing of files, directories, and JSONL data sources.
Use ``fingerprint_file`` for individual files, ``fingerprint_directory`` for
directory trees, and ``fingerprint_jsonl`` for JSONL data files. All hashing
is SHA-256 on raw bytes (no text normalization).

``compute_fingerprint`` bundles multiple paths into a single
``DatasetFingerprint`` with per-path hashes and a combined ``bundle_hash``.
``detect_drift`` compares two fingerprints by hash content; timestamp
differences alone do not constitute drift.

Streaming reads (8 KiB chunks) prevent memory exhaustion on large files.

Placement rationale: fingerprinting is agent-oriented reproducibility tooling
(``ai/`` layer), not data sampling (``data/``).
"""

from autopilot.core.serialization import DictMixin
from autopilot.tracking.io import utc_now_iso
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

_CHUNK_SIZE = 8192


def _hash_file_streaming(path: Path) -> str:
  """SHA-256 hex digest of file bytes using streaming reads.

  Args:
    path: File to hash.

  Returns:
    64-character lowercase hex digest.

  Raises:
    FileNotFoundError: When ``path`` does not exist or is not a file.
  """
  if not path.is_file():
    msg = f'file not found: {path}'
    raise FileNotFoundError(msg)
  hasher = hashlib.sha256()
  with path.open('rb') as fh:
    while True:
      chunk = fh.read(_CHUNK_SIZE)
      if not chunk:
        break
      hasher.update(chunk)
  return hasher.hexdigest()


def fingerprint_file(path: Path) -> str:
  """SHA-256 hex digest of raw file bytes (no normalization).

  Uses streaming reads to avoid loading entire files into memory.

  Args:
    path: Path to the file to hash.

  Returns:
    64-character lowercase hex digest.
  """
  return _hash_file_streaming(path)


def fingerprint_directory(path: Path, pattern: str = '**/*') -> str:
  r"""SHA-256 of sorted ``(relative_path, file_hash)`` pairs.

  Files matching ``pattern`` under ``path`` are discovered, sorted
  lexicographically by their path relative to ``path``, individually
  hashed via ``fingerprint_file``, then combined into one digest.

  Concatenation rule: for each file, ``relative_path + '\0' + file_hash``
  is fed to the hasher. Files are processed in sorted relative-path order.
  An empty directory yields the digest of an empty byte string.

  Args:
    path: Root directory to fingerprint.
    pattern: Glob pattern for file discovery (default ``'**/*'``).

  Returns:
    64-character lowercase hex digest covering the full directory tree.
  """
  hasher = hashlib.sha256()
  matched = sorted((p.relative_to(path), p) for p in path.glob(pattern) if p.is_file())
  for rel_path, abs_path in matched:
    file_hash = _hash_file_streaming(abs_path)
    hasher.update(f'{rel_path}\0{file_hash}'.encode())
  return hasher.hexdigest()


def fingerprint_jsonl(path: Path) -> str:
  """SHA-256 hex digest of raw JSONL file bytes (no normalization).

  JSONL is treated as opaque bytes -- the hash reflects exact on-disk
  contents including line ordering, duplicates, and trailing newlines.
  Order-sensitive for multi-file bundles.

  Args:
    path: Path to the JSONL file.

  Returns:
    64-character lowercase hex digest.
  """
  return _hash_file_streaming(path)


@dataclass
class DatasetFingerprint(DictMixin):
  """Fingerprint metadata for a dataset.

  ``paths`` and ``hashes`` are parallel lists: ``hashes[i]`` is the hash
  for ``paths[i]``. ``bundle_hash`` covers the full set. ``row_count``
  is optional and only populated when derivable without extra I/O.
  ``timestamp`` records when the fingerprint was computed.

  Round-trip via ``DictMixin``: ``to_dict()`` / ``from_dict()``.
  """

  paths: list[str] = field(default_factory=list)
  hashes: list[str] = field(default_factory=list)
  row_count: int | None = None
  bundle_hash: str | None = None
  timestamp: str | None = None


def compute_fingerprint(
  paths: list[Path],
  pattern: str = '**/*',
) -> DatasetFingerprint:
  """Compute bundle fingerprint for multiple dataset paths.

  Each path is individually hashed: files via ``fingerprint_file``,
  directories via ``fingerprint_directory``. Results are collected into
  parallel ``paths`` / ``hashes`` lists. The ``bundle_hash`` is a SHA-256
  over sorted ``(str(path), hash)`` pairs concatenated with null separators.

  An empty ``paths`` list yields an empty fingerprint with ``bundle_hash``
  equal to the SHA-256 of an empty byte string.

  Args:
    paths: Dataset file/directory roots to fingerprint.
    pattern: Glob pattern for directory fingerprinting (default ``'**/*'``).

  Returns:
    ``DatasetFingerprint`` with per-path hashes and a combined bundle hash.
  """
  path_strs: list[str] = []
  hash_list: list[str] = []
  for p in paths:
    h = fingerprint_directory(p, pattern) if p.is_dir() else fingerprint_file(p)
    path_strs.append(str(p))
    hash_list.append(h)

  bundle_hasher = hashlib.sha256()
  for ps, h in sorted(zip(path_strs, hash_list, strict=True)):
    bundle_hasher.update(f'{ps}\0{h}'.encode())
  bundle_hash = bundle_hasher.hexdigest()

  return DatasetFingerprint(
    paths=path_strs,
    hashes=hash_list,
    bundle_hash=bundle_hash,
    timestamp=utc_now_iso(),
  )


def detect_drift(
  before: DatasetFingerprint,
  after: DatasetFingerprint,
) -> bool:
  """True if hashes changed or structural mismatch between fingerprints.

  Compares ``paths`` and ``hashes`` lists element-wise. Differing list
  lengths are treated as drift. Timestamp differences alone do NOT
  constitute drift -- only hash and path content matter.

  Args:
    before: Baseline fingerprint.
    after: Current fingerprint.

  Returns:
    ``True`` when any hash or structural difference is detected.
  """
  if len(before.paths) != len(after.paths):
    return True
  if len(before.hashes) != len(after.hashes):
    return True
  if before.paths != after.paths:
    return True
  if before.hashes != after.hashes:
    return True
  return before.bundle_hash != after.bundle_hash
