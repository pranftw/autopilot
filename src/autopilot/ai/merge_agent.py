"""Agent-driven conflict resolution for store merges.

MergeAgent wraps an inner Agent to resolve merge conflicts through structured
prompts. It builds BASE/OURS/THEIRS context per conflict key, sends the prompt
to the agent, parses the delimited response, writes resolved blobs to the store,
and mutates the MergeIndex staging area.

Delimiter grammar expected from the agent::

  RESOLVED: <key>
  ```
  <resolved file content>
  ```

Each conflict key must appear exactly once as a ``RESOLVED: <key>`` header
followed by a triple-backtick fenced code block containing the resolved content.
Parse failures raise ``ValueError``.

Binary / non-UTF-8 safety: when a side's blob cannot be decoded as UTF-8,
``build_resolution_prompt`` emits a sentinel instead of crashing. The sentinel
format is ``(binary blob, <digest> <size> bytes -- cannot display)``.
"""

from autopilot.ai.agents.agent import Agent
from autopilot.ai.store_lock import hash_content
from autopilot.core.errors import StoreError
from autopilot.core.snapshot import FileEntry, SnapshotManifest
from autopilot.core.store.base import Store
from autopilot.core.store.types import MergeIndex
import re

_RESOLVED_RE = re.compile(r'^RESOLVED:\s*(.+)$', re.MULTILINE)


class MergeAgent:
  """Agent-driven conflict resolution for store merges.

  Wraps an ``Agent`` and a ``Store`` to provide structured conflict resolution.
  The agent receives a prompt with BASE/OURS/THEIRS sections per conflict key
  and replies with ``RESOLVED: <key>`` headers followed by fenced code blocks.

  Attributes:
    _agent: Inner agent implementing the ``run(prompt) -> str`` protocol.
    _store: Store instance for blob reads and merge_apply.
  """

  def __init__(self, agent: Agent, store: Store) -> None:
    """Create a MergeAgent with an inner agent and store.

    Args:
      agent: Agent implementing ``run(prompt: str) -> str``.
      store: Store for reading conflict blobs and applying merges.
    """
    self._agent = agent
    self._store = store

  def build_resolution_prompt(self, merge_index: MergeIndex) -> str:
    """Build structured prompt with BASE/OURS/THEIRS per conflict.

    For each conflict key in ``merge_index.conflicts``, loads side contents
    from the store using each ``FileEntry`` digest when present.

    Keys with no conflicting entries are omitted. Empty files are represented
    as an explicit empty section body to keep parsing deterministic.

    Binary / non-UTF-8 blobs emit a sentinel instead of crashing:
    ``(binary blob, <digest> <size> bytes -- cannot display)``.

    Args:
      merge_index: MergeIndex with unresolved conflicts.

    Returns:
      Formatted prompt string with conflict sections in sorted key order.
    """
    sections: list[str] = []
    for key in sorted(merge_index.conflicts):
      conflict = merge_index.conflicts[key]
      parts = [f'CONFLICT: {key}']
      parts.extend(
        [
          '--- BASE ---',
          self._load_side_text(conflict.ancestor),
          '--- OURS ---',
          self._load_side_text(conflict.ours),
          '--- THEIRS ---',
          self._load_side_text(conflict.theirs),
        ]
      )
      sections.append('\n'.join(parts))
    return '\n\n'.join(sections)

  def resolve_conflicts(self, merge_index: MergeIndex) -> MergeIndex:
    """Resolve all conflicts via the inner agent.

    Flow:
      1. Build prompt from ``merge_index``
      2. Call ``self._agent.run(prompt)``
      3. Parse response into per-key resolved text bodies
      4. For each resolved key, encode UTF-8, hash, store blob, resolve in index
      5. Return the mutated ``merge_index``

    Parsing contract:
      Agent must reply with machine-delimited blocks: one ``RESOLVED: <key>``
      header followed by a triple-backtick fenced code block per key. Parse
      failures raise ``ValueError``.

    Args:
      merge_index: MergeIndex with unresolved conflicts.

    Returns:
      The mutated merge_index (callers verify ``is_resolved()`` before apply).

    Raises:
      ValueError: If conflicts contain binary/non-UTF-8 sides, or if the agent
        response has malformed delimiter grammar or missing keys.
    """
    if not merge_index.conflicts:
      return merge_index

    binary_keys = self._detect_binary_keys(merge_index)
    if binary_keys:
      msg = (
        f'cannot agent-resolve binary conflict keys: {sorted(binary_keys)}; '
        f'use merge-resolve --content to resolve manually'
      )
      raise ValueError(msg)

    prompt = self.build_resolution_prompt(merge_index)
    response = self._agent.run(prompt)
    resolutions = self._parse_response(response, set(merge_index.conflicts))

    for key, content in resolutions.items():
      self._apply_single_resolution(merge_index, key, content)

    return merge_index

  def apply_resolution(
    self,
    merge_index: MergeIndex,
    resolutions: dict[str, str],
  ) -> SnapshotManifest:
    """Write explicit resolved text bodies, then persist via ``Store.merge_apply``.

    For each ``(key, content)`` in ``resolutions``, encodes UTF-8, hashes,
    stores the blob, and resolves the key in the merge index.

    The next snapshot epoch is chosen inside ``merge_apply`` from refs; this
    method does not take an ``epoch`` argument.

    Args:
      merge_index: Staged MergeIndex for the merge being completed. Token
        validation happens inside ``merge_apply``.
      resolutions: Mapping from conflict key to resolved text content.

    Returns:
      The ``SnapshotManifest`` returned by ``self._store.merge_apply(merge_index)``.
    """
    for key, content in resolutions.items():
      self._apply_single_resolution(merge_index, key, content)

    return self._store.merge_apply(merge_index)

  def _apply_single_resolution(
    self,
    merge_index: MergeIndex,
    key: str,
    content: str,
  ) -> None:
    """Encode, hash, store, and resolve a single conflict key.

    Shared per-item application logic used by both ``resolve_conflicts``
    (agent-driven) and ``apply_resolution`` (explicit text bodies).

    Args:
      merge_index: MergeIndex to mutate.
      key: Conflict key to resolve.
      content: Resolved text content for this key.
    """
    content_bytes = content.encode('utf-8')
    digest = hash_content(content)
    self._store_blob(digest, content_bytes)
    entry = FileEntry(digest=digest, size=len(content_bytes), mtime=0.0)
    merge_index.resolve(key, entry)

  def _load_side_text(self, entry: FileEntry | None) -> str:
    """Load and decode a conflict side, with fallbacks for missing/binary.

    Args:
      entry: FileEntry for the side, or None if deleted/absent.

    Returns:
      Decoded text, ``(deleted)`` sentinel, missing-object sentinel, or
      binary sentinel.
    """
    if entry is None:
      return '(deleted)'
    try:
      data = self._store.read_object(entry.digest)
    except StoreError:
      return f'(missing object, {entry.digest} -- not in store)'
    try:
      return data.decode('utf-8')
    except UnicodeDecodeError:
      return f'(binary blob, {entry.digest} {entry.size} bytes -- cannot display)'

  def _detect_binary_keys(self, merge_index: MergeIndex) -> set[str]:
    """Identify conflict keys where any side is binary / non-UTF-8.

    Args:
      merge_index: MergeIndex with unresolved conflicts.

    Returns:
      Set of conflict keys that contain non-UTF-8 sides.
    """
    binary_keys: set[str] = set()
    for key, conflict in merge_index.conflicts.items():
      for side in (conflict.ancestor, conflict.ours, conflict.theirs):
        if side is None:
          continue
        text = self._load_side_text(side)
        if text.startswith(('(binary blob,', '(missing object,')):
          binary_keys.add(key)
          break
    return binary_keys

  def _store_blob(self, digest: str, data: bytes) -> None:
    """Write a blob to the store's object backend.

    Delegates to the store's public ``store_blob`` method, keeping all blob
    writes on the canonical path.

    Args:
      digest: SHA-256 hex digest of the data.
      data: Raw bytes to store.
    """
    self._store.store_blob(digest, data)

  def _parse_response(
    self,
    response: str,
    expected_keys: set[str],
  ) -> dict[str, str]:
    """Parse agent response into per-key resolved content.

    Expected format per key::

      RESOLVED: <key>
      ```
      <content>
      ```

    Args:
      response: Raw agent response text.
      expected_keys: Set of conflict keys that must be resolved.

    Returns:
      Mapping from key to resolved text content.

    Raises:
      ValueError: If delimiter grammar is malformed or keys are missing.
    """
    result: dict[str, str] = {}
    matches = list(_RESOLVED_RE.finditer(response))
    if not matches:
      msg = (
        f'agent response contains no RESOLVED: headers; '
        f'expected {len(expected_keys)} resolution(s) for keys: {sorted(expected_keys)}'
      )
      raise ValueError(msg)

    for i, match in enumerate(matches):
      key = match.group(1).strip()
      after = response[match.end() :]
      if i + 1 < len(matches):
        after = response[match.end() : matches[i + 1].start()]

      fence_start = after.find('```')
      if fence_start == -1:
        msg = f'no fenced code block found after RESOLVED: {key}'
        raise ValueError(msg)
      body_start = after.find('\n', fence_start)
      if body_start == -1:
        msg = f'malformed fenced code block after RESOLVED: {key}'
        raise ValueError(msg)
      body_start += 1
      remaining = after[body_start:]
      fence_end = remaining.find('```')
      if fence_end == -1:
        msg = f'unclosed fenced code block for RESOLVED: {key}'
        raise ValueError(msg)
      content = remaining[:fence_end]
      content = content.removesuffix('\n')
      result[key] = content

    missing = expected_keys - set(result)
    if missing:
      msg = f'agent did not resolve all conflicts; missing keys: {sorted(missing)}'
      raise ValueError(msg)

    return result
