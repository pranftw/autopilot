"""Factory for typed ContextEntry metadata dicts.

DecisionEntry produces structured ``dict[str, Any]`` payloads with a ``_type``
discriminator key, designed for use with ``Experiment.add_context(..., metadata=...)``
or ``Trainer.emit_context(..., metadata=...)``.

The ``_type`` field enables machine filtering of context log entries by decision
kind (deployment, rollback, comparison, policy gate, plateau stop) without
introducing registries or string-key command dispatch.

Usage::

  metadata = DecisionEntry.deployment(
    label='production',
    experiment_id='abc123',
    previous_id='def456',
  )
  experiment.add_context(
    'deployed to production',
    source='deployment',
    metadata=metadata,
  )

  # later, filter entries by type:
  deployments = [
    e
    for e in experiment.context_log
    if e.metadata.get('_type') == DecisionEntry.DEPLOYMENT_TYPE
  ]

  # filter plateau stops:
  plateau_stops = [
    e
    for e in experiment.context_log
    if e.metadata.get('_type') == DecisionEntry.PLATEAU_STOP_TYPE
  ]
"""

from typing import Any


class DecisionEntry:
  """Factory for typed ContextEntry metadata dicts.

  Each class method returns a ``dict[str, Any]`` containing a ``_type``
  discriminator and the relevant fields for that decision kind. All methods
  validate required string arguments are non-empty and raise ``ValueError``
  with actionable guidance on failure.

  Class-level constants:
    DEPLOYMENT_TYPE: discriminator for deployment decisions.
    ROLLBACK_TYPE: discriminator for rollback decisions.
    COMPARISON_TYPE: discriminator for experiment comparisons.
    POLICY_GATE_TYPE: discriminator for policy gate outcomes.
    PLATEAU_STOP_TYPE: discriminator for plateau stop decisions.
  """

  DEPLOYMENT_TYPE = 'deployment'
  ROLLBACK_TYPE = 'rollback'
  COMPARISON_TYPE = 'comparison'
  POLICY_GATE_TYPE = 'policy_gate'
  PLATEAU_STOP_TYPE = 'plateau_stop'
  OPTIMIZER_STEP_TYPE = 'optimizer_step'

  @classmethod
  def deployment(
    cls,
    label: str,
    experiment_id: str,
    previous_id: str | None = None,
    evidence: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Build a deployment decision metadata dict.

    Args:
      label: Deployment target label (e.g. 'production', 'staging').
      experiment_id: ID of the experiment being deployed.
      previous_id: ID of the previously deployed experiment, if any.
      evidence: Supporting evidence dict (e.g. metric deltas).

    Returns:
      Dict with ``_type='deployment'`` and deployment fields.

    Raises:
      ValueError: When label or experiment_id is empty or whitespace-only.
    """
    if not label or not label.strip():
      msg = (
        'label must not be empty or whitespace-only. '
        "Provide a deployment target name (e.g. 'production')."
      )
      raise ValueError(msg)
    if not experiment_id or not experiment_id.strip():
      msg = (
        'experiment_id must not be empty or whitespace-only. '
        'Provide the ID of the experiment being deployed.'
      )
      raise ValueError(msg)
    result: dict[str, Any] = {
      '_type': cls.DEPLOYMENT_TYPE,
      'label': label,
      'experiment_id': experiment_id,
    }
    if previous_id is not None:
      result['previous_id'] = previous_id
    if evidence is not None:
      result['evidence'] = evidence
    return result

  @classmethod
  def rollback(
    cls,
    target_epoch: int,
    reason: str,
    metrics_before: dict[str, float] | None = None,
  ) -> dict[str, Any]:
    """Build a rollback decision metadata dict.

    Args:
      target_epoch: Epoch to roll back to.
      reason: Why the rollback is being performed.
      metrics_before: Metrics at the time of rollback decision.

    Returns:
      Dict with ``_type='rollback'`` and rollback fields.

    Raises:
      ValueError: When reason is empty or whitespace-only.
    """
    if not reason or not reason.strip():
      msg = (
        'reason must not be empty or whitespace-only. '
        'Provide an explanation for why the rollback is needed.'
      )
      raise ValueError(msg)
    result: dict[str, Any] = {
      '_type': cls.ROLLBACK_TYPE,
      'target_epoch': target_epoch,
      'reason': reason,
    }
    if metrics_before is not None:
      result['metrics_before'] = metrics_before
    return result

  @classmethod
  def comparison(
    cls,
    baseline_id: str,
    candidate_id: str,
    verdict: str,
    deltas: list[dict[str, Any]] | None = None,
    confidence: str | None = None,
    *,
    proposal_id: str | None = None,
    baseline_epoch: int | None = None,
    candidate_epoch: int | None = None,
  ) -> dict[str, Any]:
    """Build a comparison decision metadata dict.

    Args:
      baseline_id: ID of the baseline experiment.
      candidate_id: ID of the candidate experiment.
      verdict: Comparison outcome (e.g. 'improved', 'regressed', 'inconclusive').
      deltas: Per-metric delta records.
      confidence: Confidence level of the verdict.
      proposal_id: Optional proposal identifier that triggered the comparison.
      baseline_epoch: Optional baseline epoch index.
      candidate_epoch: Optional candidate epoch index.

    Returns:
      Dict with ``_type='comparison'`` and comparison fields.

    Raises:
      ValueError: When baseline_id, candidate_id, or verdict is empty or whitespace-only.
    """
    if not baseline_id or not baseline_id.strip():
      msg = (
        'baseline_id must not be empty or whitespace-only. '
        'Provide the ID of the baseline experiment.'
      )
      raise ValueError(msg)
    if not candidate_id or not candidate_id.strip():
      msg = (
        'candidate_id must not be empty or whitespace-only. '
        'Provide the ID of the candidate experiment.'
      )
      raise ValueError(msg)
    if not verdict or not verdict.strip():
      msg = (
        'verdict must not be empty or whitespace-only. '
        "Provide a comparison outcome (e.g. 'improved', 'regressed')."
      )
      raise ValueError(msg)
    result: dict[str, Any] = {
      '_type': cls.COMPARISON_TYPE,
      'baseline_id': baseline_id,
      'candidate_id': candidate_id,
      'verdict': verdict,
    }
    if deltas is not None:
      result['deltas'] = deltas
    if confidence is not None:
      result['confidence'] = confidence
    if proposal_id is not None:
      result['proposal_id'] = proposal_id
    if baseline_epoch is not None:
      result['baseline_epoch'] = baseline_epoch
    if candidate_epoch is not None:
      result['candidate_epoch'] = candidate_epoch
    return result

  @classmethod
  def policy_gate(
    cls,
    gate_name: str,
    passed: bool,
    value: float | None = None,
    threshold: str | None = None,
  ) -> dict[str, Any]:
    """Build a policy gate decision metadata dict.

    Args:
      gate_name: Name of the gate that produced this result.
      passed: Whether the gate passed.
      value: Metric value evaluated by the gate.
      threshold: Human-readable threshold description.

    Returns:
      Dict with ``_type='policy_gate'`` and gate fields.

    Raises:
      ValueError: When gate_name is empty or whitespace-only.
    """
    if not gate_name or not gate_name.strip():
      msg = 'gate_name must not be empty or whitespace-only. Provide the name of the policy gate.'
      raise ValueError(msg)
    result: dict[str, Any] = {
      '_type': cls.POLICY_GATE_TYPE,
      'gate_name': gate_name,
      'passed': passed,
    }
    if value is not None:
      result['value'] = value
    if threshold is not None:
      result['threshold'] = threshold
    return result

  @classmethod
  def plateau_stop(
    cls,
    monitor: str,
    epoch: int,
    *,
    plateau_window: int,
    plateau_threshold: float,
    values: list[float],
  ) -> dict[str, Any]:
    """Build a plateau-stop decision metadata dict.

    Args:
      monitor: Metric key that plateaued (e.g. 'accuracy', 'val_accuracy').
      epoch: 0-based epoch index when plateau was detected (last completed epoch).
      plateau_window: Number of consecutive epochs in the detection window.
      plateau_threshold: Relative range threshold used for detection.
      values: Monitored metric values over the window (oldest to newest).

    Returns:
      Dict with ``_type='plateau_stop'`` and plateau fields.

    Raises:
      ValueError: When monitor is empty or whitespace-only.
    """
    if not monitor or not monitor.strip():
      msg = (
        'monitor must not be empty or whitespace-only. '
        "Provide the metric name (e.g. 'val_accuracy')."
      )
      raise ValueError(msg)
    return {
      '_type': cls.PLATEAU_STOP_TYPE,
      'monitor': monitor,
      'epoch': epoch,
      'plateau_window': plateau_window,
      'plateau_threshold': plateau_threshold,
      'values': values,
    }

  @classmethod
  def optimizer_step(
    cls,
    *,
    epoch: int,
    param_summaries: list[dict[str, str]],
  ) -> dict[str, Any]:
    """Build an optimizer-step evidence metadata dict.

    Emitted once per accepted epoch (non-agentic) after the gate accepts,
    capturing post-step parameter state summaries.

    Args:
      epoch: 0-based epoch index when the optimizer step ran.
      param_summaries: Per-parameter state summaries (param_name, param_type,
        value_preview).

    Returns:
      Dict with ``_type='optimizer_step'`` and evidence fields.
    """
    return {
      '_type': cls.OPTIMIZER_STEP_TYPE,
      'epoch': epoch,
      'param_summaries': param_summaries,
    }
