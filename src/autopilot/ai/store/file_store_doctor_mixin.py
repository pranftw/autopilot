"""Internal doctor helper delegations mixed into FileStore."""

from autopilot.ai.store import doctor as doctor_mod
from autopilot.ai.store import repair as repair_mod


class FileStoreDoctorMixin:
  """Mixin wiring FileStore doctor internals to doctor_mod."""

  def _repair_orphan_blob(self, entry):
    repair_mod.repair_orphan_blob(self, entry)

  def _repair_stale_lock(self, entry):
    repair_mod.repair_stale_lock(entry)

  def _repair_broken_ref(self, entry):
    repair_mod.repair_broken_ref(self, entry)

  def _repair_reflog_gap(self, entry, context):
    repair_mod.repair_reflog_gap(self, entry, context)

  def _detect_stale_locks(self, diagnostics):
    doctor_mod.detect_stale_locks(self, diagnostics)

  def _is_stale_lock(self, lock_path):
    return doctor_mod.is_stale_lock(lock_path)

  def _detect_reflog_gaps(self, diagnostics):
    doctor_mod.detect_reflog_gaps(self, diagnostics)

  def _detect_ghost_epochs(self, diagnostics):
    doctor_mod.detect_ghost_epochs(self, diagnostics)

  def _walk_manifests(self):
    return doctor_mod.walk_manifests(self)

  def _walk_single_manifest(self, path, reachable, errors):
    doctor_mod.walk_single_manifest(path, reachable, errors)

  def _collect_on_disk_blobs(self):
    return doctor_mod.collect_on_disk_blobs(self)

  def _check_refs_consistency(self, issues):
    doctor_mod.check_refs_consistency(self, issues)

  def _check_forest_health(self):
    return doctor_mod.check_forest_health(self)

  def _validate_forest_tree(self, idx, tree_state, errors):
    doctor_mod.validate_forest_tree(idx, tree_state, errors)

  def _collect_reachable_digests(self):
    return doctor_mod.collect_reachable_digests(self)

  def _remove_unreachable_blobs(self, reachable):
    return doctor_mod.remove_unreachable_blobs(self, reachable)
