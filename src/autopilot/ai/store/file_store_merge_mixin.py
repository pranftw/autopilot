"""Internal merge helper delegations mixed into FileStore."""

from autopilot.ai.store import merge as merge_mod
from autopilot.ai.store import merge_ancestry as ancestry_mod
from autopilot.ai.store import text_merge as text_merge_mod


class FileStoreMergeMixin:
  """Mixin wiring FileStore merge internals to merge_mod."""

  def _merge_key_three_way(self, key, ancestor_snap, ours_snap, theirs_snap, conflicts, resolved):
    merge_mod.merge_key_three_way(
      self, key, ancestor_snap, ours_snap, theirs_snap, conflicts, resolved
    )

  def _try_text_merge(self, key, ancestor_entry, ours_entry, theirs_entry):
    return text_merge_mod.try_text_merge(self, key, ancestor_entry, ours_entry, theirs_entry)

  def _three_way_merge_text(self, base, ours, theirs):
    return text_merge_mod.three_way_merge_text(base, ours, theirs)

  def _apply_non_overlapping_changes(self, base_lines, ours_lines, theirs_lines):
    return text_merge_mod.apply_non_overlapping_changes(base_lines, ours_lines, theirs_lines)

  def _collect_edits(self, base_lines, modified_lines):
    return text_merge_mod.collect_edits(base_lines, modified_lines)

  def _extract_changed_line_numbers(self, base_lines, modified_lines):
    return text_merge_mod.extract_changed_line_numbers(base_lines, modified_lines)

  def _auto_resolve_strategy(self, conflicts, resolved, strategy):
    merge_mod.auto_resolve_strategy(conflicts, resolved, strategy)

  def _auto_resolve_union(self, conflicts, resolved, ancestor_snap):
    merge_mod.auto_resolve_union(self, conflicts, resolved, ancestor_snap)

  def _compute_preview_token(
    self,
    target_exp,
    target_epoch,
    source_exp,
    source_epoch,
    strategy,
    sorted_keys,
    ancestor_exp,
    ancestor_epoch,
  ):
    return merge_mod.compute_preview_token(
      target_exp,
      target_epoch,
      source_exp,
      source_epoch,
      strategy,
      sorted_keys,
      ancestor_exp,
      ancestor_epoch,
    )

  def _recompute_preview_token(self, experiment_id, source_id, strategy, refs):
    return merge_mod.recompute_preview_token(self, experiment_id, source_id, strategy, refs)

  def _find_lca(self, exp_a, exp_b, refs):
    return merge_mod.find_lca(exp_a, exp_b, refs)

  def _ancestor_set(self, experiment_id, branches):
    return ancestry_mod.ancestor_set(experiment_id, branches)

  def _enqueue_parent(self, branch_info, branches, visited, queue):
    ancestry_mod.enqueue_parent(branch_info, branches, visited, queue)

  def _enqueue_merge_parents(self, branch_info, branches, visited, queue):
    ancestry_mod.enqueue_merge_parents(branch_info, branches, visited, queue)

  def _ancestor_chain_ordered(self, experiment_id, branches):
    return ancestry_mod.ancestor_chain_ordered(experiment_id, branches)

  def _load_ancestor_snapshot(self, ancestor_exp, ancestor_epoch):
    return merge_mod.load_ancestor_snapshot(self, ancestor_exp, ancestor_epoch)

  def _require_both_branches(self, experiment_id, from_experiment_id, branches):
    merge_mod.require_both_branches(experiment_id, from_experiment_id, branches)

  def _changed_keys(self, ancestor_snap, current_snap):
    return merge_mod.changed_keys(ancestor_snap, current_snap)

  def _divergent_keys(self, overlap, ours_snap, theirs_snap):
    return merge_mod.divergent_keys(overlap, ours_snap, theirs_snap)
