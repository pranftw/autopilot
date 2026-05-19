"""Metric direction inference and prefix stripping utilities.

Shared by Trainer (train/val metric merging) and CLI comparison/query
commands. Consolidates three prior copies of prefix-strip logic and
the ``infer_direction`` heuristic into one canonical location.
"""

LOWER_IS_BETTER_PATTERNS = (
  'loss',
  'error',
  'latency',
  'cost',
  'perplexity',
)

LOWER_IS_BETTER_SEGMENT_PATTERNS = (
  'cer',
  'wer',
)


def infer_direction(metric_name: str) -> bool:
  """Return True if higher values are better for this metric name.

  Uses substring matching on the lowercased name against
  ``LOWER_IS_BETTER_PATTERNS`` for long tokens (loss, error, etc.).
  Uses segment matching (split on ``_``) for short tokens (cer, wer)
  to avoid false positives like ``answer`` matching ``wer``.

  Args:
    metric_name: Raw metric key (e.g. ``'val_loss'``, ``'accuracy'``).

  Returns:
    ``True`` for higher-is-better, ``False`` for lower-is-better.
  """
  lower = metric_name.lower()
  if any(pattern in lower for pattern in LOWER_IS_BETTER_PATTERNS):
    return False
  segments = lower.split('_')
  return all(pattern not in segments for pattern in LOWER_IS_BETTER_SEGMENT_PATTERNS)


def strip_metric_prefix(key: str) -> tuple[str, str]:
  """Strip train_/val_ prefix from a metric key.

  Single-pass strip: recognizes only ``train_`` and ``val_`` prefixes.
  Does not iteratively strip (e.g. ``train_train_loss`` becomes
  ``train_loss`` after one strip, not ``loss``).

  Args:
    key: Raw metric key (e.g. ``'train_loss'``, ``'val_accuracy'``).

  Returns:
    Tuple of (base_key, prefix). prefix is ``''`` if no recognized prefix.
  """
  if key.startswith('train_'):
    return key[6:], 'train_'
  if key.startswith('val_'):
    return key[4:], 'val_'
  return key, ''


def metric_base_name(key: str) -> str:
  """Return the base name of a metric key after stripping train_/val_ prefix.

  Convenience wrapper around ``strip_metric_prefix`` for callers that
  only need the base name without the prefix string.

  Args:
    key: Raw metric key (e.g. ``'val_accuracy'``).

  Returns:
    Base name without prefix (e.g. ``'accuracy'``).
  """
  base, _ = strip_metric_prefix(key)
  return base
