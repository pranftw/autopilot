"""FileStore merge integration tests: LCA, strategies, apply, token, refs."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.errors import StoreError
from autopilot.core.store.types import MergeStrategy
from pathlib import Path
from tests.ai.conftest import make_source_dir, make_store_config
import pytest


def _make_store(
  tmp_path: Path,
  files: dict[str, str] | None = None,
) -> tuple[FileStore, Path, PathParameter]:
  if files is None:
    files = {'main.py': 'print("hello")\n'}
  src = make_source_dir(tmp_path, files=files)
  config = make_store_config(tmp_path)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('root', 0)
  return store, src, param


def _setup_diverged(
  tmp_path: Path,
  base_files: dict[str, str],
  ours_edits: dict[str, str],
  theirs_edits: dict[str, str],
) -> tuple[FileStore, Path]:
  """Set up root -> exp-a (ours) and exp-b (theirs) with diverged edits."""
  store, src, _ = _make_store(tmp_path, files=base_files)
  store.branch('exp-a')
  store.branch('exp-b')

  store.checkout('exp-a', 0)
  for name, content in ours_edits.items():
    (src / name).write_text(content)
  store.snapshot('exp-a', 1)

  store.checkout('exp-b', 0)
  for name, content in theirs_edits.items():
    (src / name).write_text(content)
  store.snapshot('exp-b', 1)

  return store, src


# -- merge_analysis tests --


class TestMergeAnalysisFastForward:
  def test_ancestor_equals_ours(self, tmp_path: Path) -> None:
    """Ancestor equals ours; theirs ahead -> fast_forward."""
    store, src, _ = _make_store(tmp_path, files={'f.txt': 'base'})
    store.branch('feature')
    store.checkout('feature', 0)
    (src / 'f.txt').write_text('advanced')
    store.snapshot('feature', 1)
    result = store.merge_analysis('root', 'feature')
    assert result.classification == 'fast_forward'
    assert result.can_fast_forward is True


class TestMergeAnalysisUpToDate:
  def test_tips_equal(self, tmp_path: Path) -> None:
    """Source is ancestor of target -> up_to_date."""
    store, src, _ = _make_store(tmp_path, files={'f.txt': 'base'})
    store.branch('feature')
    store.checkout('root', 0)
    (src / 'f.txt').write_text('advanced root')
    store.snapshot('root', 1)
    result = store.merge_analysis('root', 'feature')
    assert result.classification == 'up_to_date'
    assert result.conflict_count == 0


class TestMergeAnalysisCleanThreeWay:
  def test_independent_files_edited(self, tmp_path: Path) -> None:
    """Independent file edits -> clean classification."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'a.txt': 'base a', 'b.txt': 'base b'},
      ours_edits={'a.txt': 'ours a'},
      theirs_edits={'b.txt': 'theirs b'},
    )
    result = store.merge_analysis('exp-a', 'exp-b')
    assert result.classification == 'clean'
    assert result.has_conflicts is False


class TestMergeAnalysisConflictPrediction:
  def test_same_key_touched_both_sides(self, tmp_path: Path) -> None:
    """Both sides touched the same manifest key -> conflict prediction."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'shared.txt': 'base'},
      ours_edits={'shared.txt': 'ours'},
      theirs_edits={'shared.txt': 'theirs'},
    )
    result = store.merge_analysis('exp-a', 'exp-b')
    assert result.has_conflicts is True
    assert result.classification == 'conflict'


# -- merge_preview tests --


class TestMergePreviewNonOverlappingEdits:
  def test_distinct_keys_no_conflicts(self, tmp_path: Path) -> None:
    """Distinct keys touched -> empty conflicts dict."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'a.txt': 'base a', 'b.txt': 'base b'},
      ours_edits={'a.txt': 'ours a'},
      theirs_edits={'b.txt': 'theirs b'},
    )
    idx = store.merge_preview('exp-a', 'exp-b')
    assert idx.conflicts == {}
    assert idx.is_resolved()


class TestMergePreviewOverlappingEditsConflictTriples:
  def test_conflict_has_all_three_sides(self, tmp_path: Path) -> None:
    """Conflict includes ancestor/ours/theirs entries with distinct digests."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'shared.txt': 'base content'},
      ours_edits={'shared.txt': 'ours content'},
      theirs_edits={'shared.txt': 'theirs content'},
    )
    idx = store.merge_preview('exp-a', 'exp-b')
    assert len(idx.conflicts) > 0
    key = next(k for k in idx.conflicts if 'shared.txt' in k)
    conflict = idx.conflicts[key]
    assert conflict.ancestor is not None
    assert conflict.ours is not None
    assert conflict.theirs is not None
    assert conflict.ours.digest != conflict.theirs.digest


class TestMergePreviewOneSideAdd:
  def test_key_absent_ancestor_present_one_side(self, tmp_path: Path) -> None:
    """Key absent in ancestor, present on one side -> lands in resolved."""
    store, src, _ = _make_store(tmp_path, files={'existing.txt': 'base'})
    store.branch('feature')
    store.checkout('feature', 0)
    (src / 'new_file.txt').write_text('new content')
    store.snapshot('feature', 1)
    idx = store.merge_preview('root', 'feature')
    assert idx.is_resolved()
    has_new = any('new_file.txt' in k for k in idx.resolved)
    assert has_new


class TestMergePreviewDeleteVsModifyConflict:
  def test_delete_vs_modify(self, tmp_path: Path) -> None:
    """Deletion vs mutation -> ConflictEntry with appropriate None sides."""
    store, src, _ = _make_store(tmp_path, files={'target.txt': 'base', 'keep.txt': 'stable'})
    store.branch('exp-a')
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (src / 'target.txt').unlink()
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (src / 'target.txt').write_text('modified content')
    store.snapshot('exp-b', 1)

    idx = store.merge_preview('exp-a', 'exp-b')
    key = next(k for k in idx.conflicts if 'target.txt' in k)
    conflict = idx.conflicts[key]
    assert conflict.ours is None
    assert conflict.theirs is not None


class TestMergePreviewBothAddIdentical:
  def test_same_blob_resolves(self, tmp_path: Path) -> None:
    """Both sides add same key with identical content -> resolved single entry."""
    store, src, _ = _make_store(tmp_path, files={'base.txt': 'base'})
    store.branch('exp-a')
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (src / 'new.txt').write_text('identical')
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (src / 'new.txt').write_text('identical')
    store.snapshot('exp-b', 1)

    idx = store.merge_preview('exp-a', 'exp-b')
    has_new = any('new.txt' in k for k in idx.resolved)
    assert has_new
    assert not any('new.txt' in k for k in idx.conflicts)


class TestMergePreviewBothAddDifferentConflict:
  def test_divergent_new_blobs_conflict(self, tmp_path: Path) -> None:
    """Both sides add same key with different content -> conflict."""
    store, src, _ = _make_store(tmp_path, files={'base.txt': 'base'})
    store.branch('exp-a')
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (src / 'new.txt').write_text('content a')
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (src / 'new.txt').write_text('content b')
    store.snapshot('exp-b', 1)

    idx = store.merge_preview('exp-a', 'exp-b')
    has_conflict = any('new.txt' in k for k in idx.conflicts)
    assert has_conflict


# -- merge_apply tests --


class TestMergeApplyPersistsNewEpoch:
  def test_refs_latest_epoch_increments(self, tmp_path: Path) -> None:
    """After apply, refs latest_epoch increments; manifest is readable."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'a.txt': 'base a', 'b.txt': 'base b'},
      ours_edits={'a.txt': 'ours a'},
      theirs_edits={'b.txt': 'theirs b'},
    )
    idx = store.merge_preview('exp-a', 'exp-b')
    manifest = store.merge_apply(idx)
    assert manifest.epoch == 2
    refs = store.load_refs()
    assert refs['branches']['exp-a']['latest_epoch'] == 2
    reloaded = store.load_snapshot('exp-a', 2)
    assert len(reloaded.entries) > 0


class TestMergeApplyUnresolvedRaises:
  def test_normal_strategy_with_conflicts_raises(self, tmp_path: Path) -> None:
    """Normal strategy with open conflicts -> StoreError."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'shared.txt': 'base'},
      ours_edits={'shared.txt': 'ours'},
      theirs_edits={'shared.txt': 'theirs'},
    )
    idx = store.merge_preview('exp-a', 'exp-b')
    assert not idx.is_resolved()
    with pytest.raises(StoreError, match='unresolved'):
      store.merge_apply(idx)


# -- merge_and_apply tests --


class TestMergeAndApplyClean:
  def test_end_to_end_success(self, tmp_path: Path) -> None:
    """Clean merge returns manifest with merged content."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'a.txt': 'base a', 'b.txt': 'base b'},
      ours_edits={'a.txt': 'ours a'},
      theirs_edits={'b.txt': 'theirs b'},
    )
    manifest = store.merge_and_apply('exp-a', 'exp-b')
    assert manifest.epoch == 2
    has_a = any('a.txt' in k for k in manifest.entries)
    has_b = any('b.txt' in k for k in manifest.entries)
    assert has_a
    assert has_b


class TestMergeAndApplyConflictsRaise:
  def test_forced_conflict_raises_store_error(self, tmp_path: Path) -> None:
    """Conflict with normal strategy -> StoreError."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'shared.txt': 'base'},
      ours_edits={'shared.txt': 'ours'},
      theirs_edits={'shared.txt': 'theirs'},
    )
    with pytest.raises(StoreError, match='unresolved'):
      store.merge_and_apply('exp-a', 'exp-b')


# -- LCA tests --


class TestLcaSimpleChain:
  def test_linear_history_ancestor(self, tmp_path: Path) -> None:
    """Linear history -> ancestor is the fork point."""
    store, src, _ = _make_store(tmp_path, files={'f.txt': 'base'})
    store.branch('feature')
    store.checkout('root', 0)
    (src / 'f.txt').write_text('root v1')
    store.snapshot('root', 1)
    store.checkout('feature', 0)
    (src / 'f.txt').write_text('feature v1')
    store.snapshot('feature', 1)

    refs = store.load_refs()
    lca_exp, lca_epoch = store._find_lca('root', 'feature', refs)
    assert lca_exp == 'root'
    assert lca_epoch == 0


class TestLcaDiamond:
  def test_diamond_merge_base(self, tmp_path: Path) -> None:
    """Two branches from same root -> LCA is root tip."""
    store, src, _ = _make_store(tmp_path, files={'f.txt': 'base'})
    store.branch('branch-a')
    store.branch('branch-b')

    store.checkout('branch-a', 0)
    (src / 'f.txt').write_text('a changes')
    store.snapshot('branch-a', 1)

    store.checkout('branch-b', 0)
    (src / 'f.txt').write_text('b changes')
    store.snapshot('branch-b', 1)

    refs = store.load_refs()
    lca_exp, lca_epoch = store._find_lca('branch-a', 'branch-b', refs)
    assert lca_exp == 'root'
    assert lca_epoch == 0


class TestLcaDiamondRemerge:
  def test_diamond_remerge_topology(self, tmp_path: Path) -> None:
    """Diamond re-merge: A and B diverge from root, B merges A, then A merges B.

    Root -> branch-a (diverge) -> branch-b (diverge) -> merge A into B
    -> then merge B into A.  LCA for second merge should be branch-a's
    tip at the time of the first merge.
    """
    store, src, _ = _make_store(tmp_path, files={'a.txt': 'base-a', 'b.txt': 'base-b'})
    store.branch('branch-a')
    store.branch('branch-b')

    store.checkout('branch-a', 0)
    (src / 'a.txt').write_text('a-only-change')
    store.snapshot('branch-a', 1)

    store.checkout('branch-b', 0)
    (src / 'b.txt').write_text('b-only-change')
    store.snapshot('branch-b', 1)

    idx = store.merge_preview('branch-b', 'branch-a')
    assert idx.is_resolved()
    store.merge_apply(idx)

    refs = store.load_refs()
    b_info = refs['branches']['branch-b']
    assert len(b_info.get('merge_parents', [])) == 1
    assert b_info['merge_parents'][0]['experiment_id'] == 'branch-a'

    store.checkout('branch-a', 1)
    (src / 'a.txt').write_text('a-second-change')
    store.snapshot('branch-a', 2)

    lca_exp, lca_epoch = store._find_lca('branch-a', 'branch-b', refs)
    assert lca_exp is not None
    assert lca_epoch is not None


class TestLcaDeepChainForkEpochs:
  def test_nested_forks_ancestor_epoch_respects_fork(self, tmp_path: Path) -> None:
    """Nested forks: ancestor epoch respects fork metadata."""
    store, src, _ = _make_store(tmp_path, files={'f.txt': 'base'})
    (src / 'f.txt').write_text('root v1')
    store.snapshot('root', 1)
    (src / 'f.txt').write_text('root v2')
    store.snapshot('root', 2)

    store.branch('child')
    store.checkout('child', 0)
    (src / 'f.txt').write_text('child v1')
    store.snapshot('child', 1)

    refs = store.load_refs()
    lca_exp, lca_epoch = store._find_lca('root', 'child', refs)
    assert lca_exp == 'root'
    assert lca_epoch == 2


# -- BUG-025 regression --


class TestMergeInsertsSameRegionProducesConflict:
  def test_concurrent_inserts_same_hunk_conflict(self, tmp_path: Path) -> None:
    """Concurrent inserts at same line -> conflict, ours NOT discarded."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'f.txt': 'line1\nline2\nline3\n'},
      ours_edits={'f.txt': 'line1\nours_insert\nline2\nline3\n'},
      theirs_edits={'f.txt': 'line1\ntheirs_insert\nline2\nline3\n'},
    )
    idx = store.merge_preview('exp-a', 'exp-b')
    key = next(k for k in idx.conflicts if 'f.txt' in k)
    assert key in idx.conflicts, 'BUG-025 regression: same-hunk insert must be a conflict'
    assert key not in idx.resolved, 'BUG-025 regression: same-hunk insert must not auto-resolve'
    conflict = idx.conflicts[key]
    assert conflict.ours is not None, 'ours side must not be discarded'
    assert conflict.theirs is not None, 'theirs side must be present'


# -- strategy auto-resolve tests --


class TestStrategyOursAutoresolves:
  def test_all_conflicts_resolved_to_ours(self, tmp_path: Path) -> None:
    """Strategy ours auto-resolves all conflicts; apply succeeds."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'shared.txt': 'base'},
      ours_edits={'shared.txt': 'ours'},
      theirs_edits={'shared.txt': 'theirs'},
    )
    idx = store.merge_preview('exp-a', 'exp-b', strategy=MergeStrategy.ours)
    assert idx.is_resolved()
    manifest = store.merge_apply(idx)
    assert manifest.epoch == 2


class TestStrategyTheirsAutoresolves:
  def test_all_conflicts_resolved_to_theirs(self, tmp_path: Path) -> None:
    """Strategy theirs auto-resolves all conflicts; apply succeeds."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'shared.txt': 'base'},
      ours_edits={'shared.txt': 'ours'},
      theirs_edits={'shared.txt': 'theirs'},
    )
    idx = store.merge_preview('exp-a', 'exp-b', strategy=MergeStrategy.theirs)
    assert idx.is_resolved()
    manifest = store.merge_apply(idx)
    key = next(k for k in manifest.entries if 'shared.txt' in k)
    content = store.read_object(manifest.entries[key].digest).decode('utf-8')
    assert content == 'theirs'


# -- preview token tests --


class TestPreviewTokenPreventsStaleApply:
  def test_mutated_refs_between_preview_and_apply(self, tmp_path: Path) -> None:
    """Mutating refs between preview and apply -> StoreError on apply."""
    store, src = _setup_diverged(
      tmp_path,
      base_files={'a.txt': 'base a', 'b.txt': 'base b'},
      ours_edits={'a.txt': 'ours a'},
      theirs_edits={'b.txt': 'theirs b'},
    )
    idx = store.merge_preview('exp-a', 'exp-b')

    store.checkout('exp-a', 1)
    (src / 'a.txt').write_text('ours a v2')
    store.snapshot('exp-a', 2)

    with pytest.raises(StoreError, match='stale preview token'):
      store.merge_apply(idx)


# -- refs merge_parents tests --


class TestRefsRecordsMergeParents:
  def test_merge_parents_entry_after_apply(self, tmp_path: Path) -> None:
    """After apply, branch has merge_parents entry with source id/epoch."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'a.txt': 'base a', 'b.txt': 'base b'},
      ours_edits={'a.txt': 'ours a'},
      theirs_edits={'b.txt': 'theirs b'},
    )
    store.merge_and_apply('exp-a', 'exp-b')
    refs = store.load_refs()
    branch = refs['branches']['exp-a']
    assert 'merge_parents' in branch
    assert len(branch['merge_parents']) == 1
    mp = branch['merge_parents'][0]
    assert mp['experiment_id'] == 'exp-b'
    assert mp['epoch'] == 1


# -- empty manifest merge --


class TestEmptyManifestMerge:
  def test_empty_manifests_merge_without_error(self, tmp_path: Path) -> None:
    """Empty manifests merge cleanly without raising."""
    config = make_store_config(tmp_path)
    src = tmp_path / 'src'
    src.mkdir(parents=True, exist_ok=True)
    param = PathParameter(source=str(src), pattern='*')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('root', 0)
    store.branch('feature')
    idx = store.merge_preview('root', 'feature')
    assert idx.is_resolved()


# -- union strategy --


class TestUnionStrategyNonOverlapping:
  def test_union_concatenation(self, tmp_path: Path) -> None:
    """Union concatenation for text-like blobs where both sides add content."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'shared.txt': 'base'},
      ours_edits={'shared.txt': 'ours'},
      theirs_edits={'shared.txt': 'theirs'},
    )
    idx = store.merge_preview('exp-a', 'exp-b', strategy=MergeStrategy.union)
    assert idx.is_resolved()
    key = next(k for k in idx.resolved if 'shared.txt' in k)
    content = store.read_object(idx.resolved[key].digest).decode('utf-8')
    assert 'ours' in content
    assert 'theirs' in content


# -- MergeIndex default strategy test --


class TestMergeIndexMergeStrategyNormalDefault:
  def test_default_strategy_surfaces_conflicts(self, tmp_path: Path) -> None:
    """Default MergeStrategy.normal surfaces conflicts on overlap."""
    store, _src = _setup_diverged(
      tmp_path,
      base_files={'shared.txt': 'base'},
      ours_edits={'shared.txt': 'ours'},
      theirs_edits={'shared.txt': 'theirs'},
    )
    idx = store.merge_preview('exp-a', 'exp-b')
    assert idx.strategy == MergeStrategy.normal
    assert len(idx.conflicts) > 0
