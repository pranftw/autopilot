"""Shared CLI user-facing string constants.

Single source for error and guard messages used across multiple command
handlers. Constants use a ``MSG_`` prefix and contain actionable guidance
(which flag to pass, which command to run first). Dynamic fragments stay
outside constants -- build final text in the handler via ``format`` or
f-strings after importing the static template.

Add a constant here when the same message appears in two or more command
handlers. One-off messages that are unlikely to drift can stay inline.
"""

# -- experiment / workspace --------------------------------------------------

MSG_EXPERIMENT_SLUG_REQUIRED = (
  'experiment slug required (--experiment); pass --experiment <id> or set active experiment'
)

MSG_NO_ACTIVE_TREE = 'no active tree; create or switch to a tree first'

MSG_NO_MODULE_CONFIGURED = 'no module configured; ensure the project passes a Module to run()'

MSG_NO_TRAINER_CONFIGURED = 'no trainer configured; ensure the project passes a Module to run()'

# -- epoch --------------------------------------------------------------------

MSG_EPOCH_REQUIRED = '--epoch is required'

MSG_EPOCH_INVALID = "invalid epoch {value!r}: expected a non-negative integer or 'latest'"

MSG_EPOCH_NOT_FOUND = (
  'epoch {epoch} not found for experiment {experiment_id!r}; valid range is 0..{latest}'
)

MSG_EPOCH_EMPTY_STORE = (
  "cannot resolve 'latest' epoch for experiment {experiment_id!r}: store has no snapshots"
)

# -- forest-only store errors -------------------------------------------------

MSG_FOREST_ONLY_STORE = (
  'experiment {experiment_id!r} is forest-only (no store branch). '
  "Run `store create --context 'reason'` to enable versioning."
)
