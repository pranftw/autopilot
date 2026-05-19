"""Structured doctor diagnostics with repair metadata.

DiagnosticEntry is the machine-readable output from detect-diagnose-repair
pipelines on FileStore and workspace health checks. Each entry carries a
diagnostic code, severity, optional path, human message, and repair metadata
(repairable flag + repair_action).

Three-phase naming convention used throughout docstrings:
  - **detect**: scan filesystem / refs for anomalies.
  - **diagnose**: classify anomalies into DiagnosticEntry records.
  - **repair**: act only when repairable=True and CLI confirms.

Diagnostic code -> repair_action mapping:
  orphan_blob    -> 'delete'   (remove unreferenced blob)
  stale_lock     -> 'delete'   (remove stale lock file)
  broken_ref     -> 'reset'    (reset branch ref to last valid epoch)
  missing_blob   -> None       (not auto-repairable)
  manifest_error -> None       (not auto-repairable)
  reflog_gap     -> 'backfill' (append synthetic reflog entry)
  ghost_epoch    -> 'delete'   (remove manifest beyond latest_epoch)
  forest_missing -> None       (informational, not repairable)
  forest_corrupt -> None       (not auto-repairable)
"""

from autopilot.core.serialization import DictMixin
import dataclasses

VALID_DIAGNOSTIC_CODES = frozenset(
  {
    'orphan_blob',
    'manifest_error',
    'stale_lock',
    'broken_ref',
    'missing_blob',
    'reflog_gap',
    'ghost_epoch',
    'forest_missing',
    'forest_corrupt',
  }
)

VALID_SEVERITY_LEVELS = frozenset({'error', 'warning', 'info'})

VALID_REPAIR_ACTIONS = frozenset(
  {
    'delete',
    'reset',
    'backfill',
  }
)


@dataclasses.dataclass
class DiagnosticEntry(DictMixin):
  """Structured store/workspace doctor finding with repair metadata.

  Attributes:
    code: Diagnostic code from VALID_DIAGNOSTIC_CODES.
    severity: One of 'error', 'warning', 'info'.
    path: Optional filesystem path related to the finding.
    message: Human-readable description of the issue.
    repairable: Whether this finding can be auto-repaired.
    repair_action: Repair action from VALID_REPAIR_ACTIONS, or None.
  """

  code: str
  severity: str
  path: str | None
  message: str
  repairable: bool
  repair_action: str | None = None

  def __post_init__(self) -> None:
    """Validate code, severity, repair_action membership and message content.

    Raises:
      ValueError: When code, severity, or repair_action is invalid, message
        is empty, or repairable is True without a repair_action.
    """
    if self.code not in VALID_DIAGNOSTIC_CODES:
      msg = (
        f'invalid diagnostic code {self.code!r}; '
        f'must be one of: {", ".join(sorted(VALID_DIAGNOSTIC_CODES))}'
      )
      raise ValueError(msg)
    if self.severity not in VALID_SEVERITY_LEVELS:
      msg = (
        f'invalid severity {self.severity!r}; '
        f'must be one of: {", ".join(sorted(VALID_SEVERITY_LEVELS))}'
      )
      raise ValueError(msg)
    if self.repair_action is not None and self.repair_action not in VALID_REPAIR_ACTIONS:
      msg = (
        f'invalid repair_action {self.repair_action!r}; '
        f'must be one of: {", ".join(sorted(VALID_REPAIR_ACTIONS))}'
      )
      raise ValueError(msg)
    if not self.message:
      msg = (
        'message must not be empty; provide a human-readable description of the diagnostic issue'
      )
      raise ValueError(msg)
    if self.repairable and self.repair_action is None:
      msg = (
        'repairable entry must specify repair_action; '
        'set repair_action to the operation that fixes this issue '
        "(e.g. 'delete', 'reset', 'backfill')"
      )
      raise ValueError(msg)
