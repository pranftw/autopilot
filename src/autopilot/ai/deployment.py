"""Deployment event system: append-only JSONL log for deploy/undeploy/replace.

DeploymentEvent records a single deployment lifecycle action. DeploymentLog
provides append and query over the workspace-scoped JSONL file at
``{workspace}/.autopilot/deployment_events.jsonl``.

Classes:
  DeploymentEvent -- one deployment lifecycle event (DictMixin dataclass)
  DeploymentLog -- append-only JSONL deployment history reader/writer
"""

from autopilot.core.serialization import DictMixin
from autopilot.tracking.io import append_jsonl, iter_jsonl_lines, utc_now_iso
from pathlib import Path
from typing import Any
import dataclasses
import json

VALID_DEPLOYMENT_ACTIONS = frozenset({'deploy', 'undeploy', 'replace'})


@dataclasses.dataclass
class DeploymentEvent(DictMixin):
  """One deployment lifecycle event.

  Records a deploy, undeploy, or replace action for audit trail purposes.
  Validated at construction time via ``__post_init__``.

  Attributes:
    label: Deployment target name (e.g. 'production', 'staging').
    experiment_id: Experiment that is being deployed or undeployed.
    action: One of 'deploy', 'undeploy', 'replace'.
    previous_experiment_id: Prior occupant of the label (for replace/deploy).
    timestamp: ISO 8601 timestamp string.
    context: Optional reason/provenance string.
  """

  label: str
  experiment_id: str
  action: str
  previous_experiment_id: str | None
  timestamp: str
  context: str | None

  def __post_init__(self) -> None:
    """Validate action and label constraints.

    Raises:
      ValueError: When action is not in VALID_DEPLOYMENT_ACTIONS or label is empty.
    """
    if self.action not in VALID_DEPLOYMENT_ACTIONS:
      msg = (
        f'Invalid deployment action {self.action!r}. '
        f'Must be one of: {", ".join(sorted(VALID_DEPLOYMENT_ACTIONS))}'
      )
      raise ValueError(msg)
    if not self.label:
      msg = 'DeploymentEvent label must not be empty.'
      raise ValueError(msg)


class DeploymentLog:
  """Append-only JSONL deployment history.

  Reads and writes deployment events to a workspace-scoped JSONL file.
  Malformed lines are silently skipped on read (no warnings module usage).

  Args:
    path: Path to the deployment_events.jsonl file.
  """

  def __init__(self, path: Path) -> None:
    """Initialize with the JSONL file path.

    Args:
      path: Filesystem path for the deployment log JSONL file.
    """
    self._path = path

  def append(self, event: DeploymentEvent) -> None:
    """Append a deployment event to the log.

    Args:
      event: Validated DeploymentEvent to persist.
    """
    append_jsonl(self._path, event.to_dict())

  def query(
    self,
    *,
    label: str | None = None,
    experiment_id: str | None = None,
  ) -> list[DeploymentEvent]:
    """Return events matching optional filters, chronological order.

    Malformed JSONL lines are silently skipped. Filters are AND-combined
    when both are provided.

    Args:
      label: Filter to events with this deployment label.
      experiment_id: Filter to events involving this experiment.

    Returns:
      List of matching DeploymentEvent instances in append order (oldest first).
    """
    events: list[DeploymentEvent] = []
    for line in iter_jsonl_lines(self._path):
      record = _try_parse_event(line)
      if record is None:
        continue
      if label is not None and record.label != label:
        continue
      if experiment_id is not None and record.experiment_id != experiment_id:
        continue
      events.append(record)
    return events

  def latest_for(self, label: str) -> DeploymentEvent | None:
    """Return the most recent event for a label, or None.

    Args:
      label: Deployment label to look up.

    Returns:
      The last appended event matching the label, or None if no match.
    """
    matching = self.query(label=label)
    if not matching:
      return None
    return matching[-1]


def _try_parse_event(line: str) -> DeploymentEvent | None:
  """Attempt to parse a JSONL line into a DeploymentEvent.

  Returns None on any parse or validation failure (malformed lines skipped).

  Args:
    line: Stripped non-empty JSONL line.

  Returns:
    DeploymentEvent on success, None on failure.
  """
  try:
    data: dict[str, Any] = json.loads(line)
  except (json.JSONDecodeError, ValueError):
    return None
  if not isinstance(data, dict):
    return None
  try:
    return DeploymentEvent.from_dict(data)
  except (TypeError, ValueError, KeyError):
    return None


def emit_deployment_event(
  log: DeploymentLog,
  label: str,
  experiment_id: str,
  action: str,
  previous_experiment_id: str | None = None,
  context: str | None = None,
) -> DeploymentEvent:
  """Create and append a deployment event to the log.

  Convenience function for Forest/CLI deploy paths. Constructs a
  DeploymentEvent with the current UTC timestamp and appends it.

  Args:
    log: DeploymentLog to append to.
    label: Deployment target name.
    experiment_id: Experiment being deployed/undeployed.
    action: One of 'deploy', 'undeploy', 'replace'.
    previous_experiment_id: Prior occupant of the label, if any.
    context: Optional reason string from --context.

  Returns:
    The constructed and appended DeploymentEvent.
  """
  event = DeploymentEvent(
    label=label,
    experiment_id=experiment_id,
    action=action,
    previous_experiment_id=previous_experiment_id,
    timestamp=utc_now_iso(),
    context=context,
  )
  log.append(event)
  return event


def deployment_log_for_workspace(workspace: Path) -> DeploymentLog:
  """Return the DeploymentLog for a workspace root.

  Resolves to ``{workspace}/.autopilot/deployment_events.jsonl``.

  Args:
    workspace: Workspace root directory.

  Returns:
    DeploymentLog instance for the canonical deployment events file.
  """
  return DeploymentLog(workspace / '.autopilot' / 'deployment_events.jsonl')
