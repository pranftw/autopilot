"""Shared experiment factory helpers for tree/forest tests."""

from autopilot.core.experiment import Experiment


def make_experiment(
  id_: str,
  *,
  status: str = 'pending',
  hypothesis: str | None = None,
  metrics: dict | None = None,
  auto_hypothesis: bool = True,
) -> Experiment:
  """Build an Experiment with the given status and optional metrics.

  Args:
    id_: Experiment identifier.
    status: Desired lifecycle state ('pending', 'running', 'completed',
      'failed', 'cancelled').
    hypothesis: Explicit hypothesis string.  When ``None`` and
      *auto_hypothesis* is ``True``, defaults to ``'{id_} hypothesis'``.
    metrics: Metric dict attached on completion (or forced onto the
      experiment when *status* is not ``'completed'``).
    auto_hypothesis: When ``True`` (default) and *hypothesis* is ``None``,
      generates ``'{id_} hypothesis'``.  Set to ``False`` to leave
      hypothesis unset.

  Returns:
    Experiment instance transitioned to the requested status.
  """
  h = hypothesis if hypothesis is not None else (f'{id_} hypothesis' if auto_hypothesis else None)
  exp = Experiment(experiment_id=id_, hypothesis=h)
  if status == 'running':
    exp.start()
  elif status == 'completed':
    exp.start()
    exp.complete(metrics=metrics)
  elif status == 'failed':
    exp.start()
    exp.fail(error='test error')
  elif status == 'cancelled':
    exp.cancel()
  if metrics and status != 'completed':
    exp.metrics = metrics
  return exp


def completed_exp(id_: str, metrics: dict | None = None, **kw) -> Experiment:
  """Shortcut for a completed experiment."""
  return make_experiment(id_, status='completed', metrics=metrics, **kw)


def pending_exp(id_: str, **kw) -> Experiment:
  """Shortcut for a pending experiment."""
  return make_experiment(id_, status='pending', **kw)


def running_exp(id_: str, **kw) -> Experiment:
  """Shortcut for a running experiment."""
  return make_experiment(id_, status='running', **kw)


def failed_exp(id_: str, **kw) -> Experiment:
  """Shortcut for a failed experiment."""
  return make_experiment(id_, status='failed', **kw)


def cancelled_exp(id_: str, **kw) -> Experiment:
  """Shortcut for a cancelled experiment."""
  return make_experiment(id_, status='cancelled', **kw)
