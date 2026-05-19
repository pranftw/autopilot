"""Unit tests for DeploymentEvent and DeploymentLog."""

from autopilot.ai.deployment import (
  DeploymentEvent,
  DeploymentLog,
  emit_deployment_event,
)
from autopilot.tracking.io import utc_now_iso
from pathlib import Path
import json
import pytest


class TestDeploymentEvent:
  """Tests for DeploymentEvent dataclass."""

  def test_deployment_event_roundtrip(self) -> None:
    """DictMixin to_dict / from_dict equality for a valid event."""
    event = DeploymentEvent(
      label='production',
      experiment_id='exp-001',
      action='deploy',
      previous_experiment_id=None,
      timestamp='2025-01-01T00:00:00+00:00',
      context='initial deploy',
    )
    data = event.to_dict()
    restored = DeploymentEvent.from_dict(data)
    assert restored.label == event.label
    assert restored.experiment_id == event.experiment_id
    assert restored.action == event.action
    assert restored.previous_experiment_id == event.previous_experiment_id
    assert restored.timestamp == event.timestamp
    assert restored.context == event.context

  def test_deployment_event_constructor_invalid_action(self) -> None:
    """ValueError for action not in VALID_DEPLOYMENT_ACTIONS."""
    with pytest.raises(ValueError, match='Invalid deployment action'):
      DeploymentEvent(
        label='staging',
        experiment_id='exp-002',
        action='promote',
        previous_experiment_id=None,
        timestamp='2025-01-01T00:00:00+00:00',
        context=None,
      )

  def test_deployment_event_constructor_empty_label(self) -> None:
    """ValueError for empty label."""
    with pytest.raises(ValueError, match='label must not be empty'):
      DeploymentEvent(
        label='',
        experiment_id='exp-003',
        action='deploy',
        previous_experiment_id=None,
        timestamp='2025-01-01T00:00:00+00:00',
        context=None,
      )

  def test_deployment_event_timestamp(self) -> None:
    """Timestamp field preserved in round-trip."""
    ts = '2025-06-15T12:30:00+00:00'
    event = DeploymentEvent(
      label='canary',
      experiment_id='exp-004',
      action='deploy',
      previous_experiment_id=None,
      timestamp=ts,
      context=None,
    )
    data = event.to_dict()
    assert data['timestamp'] == ts
    restored = DeploymentEvent.from_dict(data)
    assert restored.timestamp == ts

  def test_deployment_event_includes_previous(self) -> None:
    """previous_experiment_id round-trip with non-None value."""
    event = DeploymentEvent(
      label='production',
      experiment_id='exp-new',
      action='replace',
      previous_experiment_id='exp-old',
      timestamp='2025-01-01T00:00:00+00:00',
      context=None,
    )
    data = event.to_dict()
    assert data['previous_experiment_id'] == 'exp-old'
    restored = DeploymentEvent.from_dict(data)
    assert restored.previous_experiment_id == 'exp-old'

  def test_deployment_event_context_entry(self) -> None:
    """context str | None round-trip."""
    for ctx_val in ['deploy reason', None]:
      event = DeploymentEvent(
        label='staging',
        experiment_id='exp-005',
        action='deploy',
        previous_experiment_id=None,
        timestamp='2025-01-01T00:00:00+00:00',
        context=ctx_val,
      )
      data = event.to_dict()
      assert data['context'] == ctx_val
      restored = DeploymentEvent.from_dict(data)
      assert restored.context == ctx_val


class TestDeploymentLog:
  """Tests for DeploymentLog read/write/query operations."""

  def test_deployment_log_query_by_experiment(self, tmp_path: Path) -> None:
    """query(experiment_id=...) returns expected rows."""
    log = DeploymentLog(tmp_path / 'events.jsonl')
    log.append(
      DeploymentEvent(
        label='prod',
        experiment_id='exp-a',
        action='deploy',
        previous_experiment_id=None,
        timestamp=utc_now_iso(),
        context=None,
      )
    )
    log.append(
      DeploymentEvent(
        label='staging',
        experiment_id='exp-b',
        action='deploy',
        previous_experiment_id=None,
        timestamp=utc_now_iso(),
        context=None,
      )
    )
    log.append(
      DeploymentEvent(
        label='canary',
        experiment_id='exp-a',
        action='deploy',
        previous_experiment_id=None,
        timestamp=utc_now_iso(),
        context=None,
      )
    )

    results = log.query(experiment_id='exp-a')
    assert len(results) == 2
    assert all(r.experiment_id == 'exp-a' for r in results)

  def test_deployment_log_latest_for(self, tmp_path: Path) -> None:
    """latest_for returns last appended matching label."""
    log = DeploymentLog(tmp_path / 'events.jsonl')
    log.append(
      DeploymentEvent(
        label='prod',
        experiment_id='exp-1',
        action='deploy',
        previous_experiment_id=None,
        timestamp='2025-01-01T00:00:00+00:00',
        context=None,
      )
    )
    log.append(
      DeploymentEvent(
        label='prod',
        experiment_id='exp-2',
        action='replace',
        previous_experiment_id='exp-1',
        timestamp='2025-01-02T00:00:00+00:00',
        context=None,
      )
    )
    log.append(
      DeploymentEvent(
        label='staging',
        experiment_id='exp-3',
        action='deploy',
        previous_experiment_id=None,
        timestamp='2025-01-03T00:00:00+00:00',
        context=None,
      )
    )

    latest = log.latest_for('prod')
    assert latest is not None
    assert latest.experiment_id == 'exp-2'
    assert latest.action == 'replace'

  def test_deploy_log_corrupt_jsonl_line(self, tmp_path: Path) -> None:
    """Garbage line skipped; valid lines still returned."""
    log_path = tmp_path / 'events.jsonl'
    valid_event = DeploymentEvent(
      label='prod',
      experiment_id='exp-ok',
      action='deploy',
      previous_experiment_id=None,
      timestamp=utc_now_iso(),
      context=None,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
      json.dumps(valid_event.to_dict())
      + '\n'
      + 'this is not valid json\n'
      + json.dumps(valid_event.to_dict())
      + '\n',
      encoding='utf-8',
    )

    log = DeploymentLog(log_path)
    results = log.query()
    assert len(results) == 2
    assert all(r.experiment_id == 'exp-ok' for r in results)

  def test_deploy_log_query_combined_filters(self, tmp_path: Path) -> None:
    """query(label=..., experiment_id=...) returns intersection."""
    log = DeploymentLog(tmp_path / 'events.jsonl')
    log.append(
      DeploymentEvent(
        label='prod',
        experiment_id='exp-a',
        action='deploy',
        previous_experiment_id=None,
        timestamp=utc_now_iso(),
        context=None,
      )
    )
    log.append(
      DeploymentEvent(
        label='staging',
        experiment_id='exp-a',
        action='deploy',
        previous_experiment_id=None,
        timestamp=utc_now_iso(),
        context=None,
      )
    )
    log.append(
      DeploymentEvent(
        label='prod',
        experiment_id='exp-b',
        action='deploy',
        previous_experiment_id=None,
        timestamp=utc_now_iso(),
        context=None,
      )
    )

    results = log.query(label='prod', experiment_id='exp-a')
    assert len(results) == 1
    assert results[0].experiment_id == 'exp-a'
    assert results[0].label == 'prod'

  def test_deploy_log_latest_for_nonexistent_label(self, tmp_path: Path) -> None:
    """latest_for('nonexistent') returns None."""
    log = DeploymentLog(tmp_path / 'events.jsonl')
    assert log.latest_for('nonexistent') is None

  def test_deploy_appends_event(self, tmp_path: Path) -> None:
    """After deploy, log contains one deploy event with correct ids."""
    log = DeploymentLog(tmp_path / 'events.jsonl')
    event = emit_deployment_event(
      log,
      label='prod',
      experiment_id='exp-x',
      action='deploy',
    )
    assert event.action == 'deploy'
    assert event.label == 'prod'
    results = log.query()
    assert len(results) == 1
    assert results[0].experiment_id == 'exp-x'

  def test_undeploy_appends_event(self, tmp_path: Path) -> None:
    """After undeploy, log contains undeploy event."""
    log = DeploymentLog(tmp_path / 'events.jsonl')
    emit_deployment_event(
      log,
      label='prod',
      experiment_id='exp-y',
      action='undeploy',
    )
    results = log.query()
    assert len(results) == 1
    assert results[0].action == 'undeploy'

  def test_replace_appends_single_replace_event(self, tmp_path: Path) -> None:
    """Replace produces one event with action='replace' and previous_experiment_id."""
    log = DeploymentLog(tmp_path / 'events.jsonl')
    emit_deployment_event(
      log,
      label='prod',
      experiment_id='exp-new',
      action='replace',
      previous_experiment_id='exp-old',
    )
    results = log.query()
    assert len(results) == 1
    assert results[0].action == 'replace'
    assert results[0].previous_experiment_id == 'exp-old'
    assert results[0].experiment_id == 'exp-new'

  def test_deploy_log_across_trees(self, tmp_path: Path) -> None:
    """Events from operations on different trees appear in one file."""
    log = DeploymentLog(tmp_path / 'events.jsonl')
    emit_deployment_event(
      log,
      label='prod',
      experiment_id='tree-a-exp',
      action='deploy',
    )
    emit_deployment_event(
      log,
      label='staging',
      experiment_id='tree-b-exp',
      action='deploy',
    )
    results = log.query()
    assert len(results) == 2
    ids = {r.experiment_id for r in results}
    assert ids == {'tree-a-exp', 'tree-b-exp'}

  def test_deploy_emits_context_entry(self, tmp_path: Path) -> None:
    """After deploy, experiment context_log contains deployment source entry."""
    from autopilot.core.experiment import Experiment

    log = DeploymentLog(tmp_path / 'events.jsonl')
    exp = Experiment('exp-ctx')
    event = emit_deployment_event(
      log,
      label='prod',
      experiment_id='exp-ctx',
      action='deploy',
    )
    exp.add_context(
      f'deployment: {event.action} as {event.label}',
      source='deployment',
      metadata=event.to_dict(),
    )
    entries = exp.context_log.entries
    assert len(entries) == 1
    assert entries[0].source == 'deployment'
    assert entries[0].metadata['action'] == 'deploy'
    assert entries[0].metadata['label'] == 'prod'
