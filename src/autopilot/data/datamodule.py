"""DataModule with Stage enum and checkpoint hooks.

Mirrors Lightning's LightningDataModule with stricter typing: ``setup`` and
``teardown`` take a ``Stage`` enum member (not a raw string).  Passing a
non-``Stage`` value raises ``TypeError`` via the ``ensure_stage`` helper.

Checkpoint integration: ``state_dict`` / ``load_state_dict`` allow the Trainer
to persist and restore data-module state across runs.

Dataset fingerprint: concrete ``DataModule`` subclasses may set
``self.dataset_fingerprint`` to a ``DatasetFingerprint`` instance (from
``autopilot.ai.fingerprint``). When set, the base ``state_dict()`` includes
the key ``'dataset_fingerprint'`` (serialized via ``DictMixin.to_dict()``),
and ``load_state_dict()`` restores it. This embeds dataset identity into
Trainer checkpoints for reproducibility.
"""

from autopilot.ai.fingerprint import DatasetFingerprint
from autopilot.data.dataloader import DataLoader
from typing import Any
import enum


class Stage(enum.Enum):
  """Trainer/data lifecycle stage (Lightning-aligned)."""

  fit = 'fit'
  validate = 'validate'
  test = 'test'
  predict = 'predict'


def ensure_stage(stage: Any) -> Stage:
  """Return ``stage`` if it is a ``Stage`` enum member; else raise ``TypeError``.

  Args:
    stage: Expected ``Stage`` instance.

  Returns:
    The same ``Stage`` instance.

  Raises:
    TypeError: When ``stage`` is not a ``Stage`` enum member.
  """
  if not isinstance(stage, Stage):
    msg = (
      f'expected a Stage enum member, got {type(stage).__name__}={stage!r}. '
      f'Use Stage.fit, Stage.validate, Stage.test, or Stage.predict.'
    )
    raise TypeError(msg)
  return stage


class DataModule:
  """Lifecycle for data with Stage-typed hooks and checkpoint support.

  Mirrors LightningDataModule.  ``setup`` and ``teardown`` accept a ``Stage``
  enum member -- raw strings are rejected at runtime via ``ensure_stage``.

  Checkpoint hooks (``state_dict`` / ``load_state_dict``) default to empty-dict
  round-trips.  Subclasses override to persist custom state (e.g. split indices,
  counters, preprocessing artifacts).

  Dataset fingerprint: set ``self.dataset_fingerprint`` to a
  ``DatasetFingerprint`` instance when fingerprint data is available.
  The base ``state_dict()`` / ``load_state_dict()`` automatically include
  and restore this field in Trainer checkpoints.

  Works with ``Dataset`` sources, ``DataLoader`` batching, and ``Stage``-scoped
  hooks across Trainer entry points.

  Example:
    >>> from autopilot.data.datamodule import DataModule, Stage
    >>> from autopilot.data.dataset import Dataset
    >>> from autopilot.data.dataloader import DataLoader
    >>>
    >>> class FixedRows(Dataset[dict]):
    ...   def __getitem__(self, index):
    ...     return {'success': True}
    ...
    ...   def __len__(self):
    ...     return 1
    >>>
    >>> class TinyDataModule(DataModule):
    ...   def setup(self, stage):
    ...     self._train = FixedRows()
    ...
    ...   def train_dataloader(self):
    ...     return DataLoader(self._train, batch_size=1)
    ...
    ...   def val_dataloader(self):
    ...     return DataLoader(self._train, batch_size=1)
    >>>
    >>> dm = TinyDataModule()
    >>> dm.setup(Stage.fit)
    >>> isinstance(dm.train_dataloader(), DataLoader)
    True
  """

  dataset_fingerprint: DatasetFingerprint | None = None

  def prepare_data(self) -> None:
    """Download or prepare static assets (optional hook)."""

  def setup(self, stage: Stage) -> None:
    """Split or assign datasets for a pipeline stage (optional hook).

    Args:
      stage: ``Stage`` enum member identifying the current lifecycle phase.
    """

  def teardown(self, stage: Stage) -> None:
    """Release resources after a stage completes (optional hook).

    Args:
      stage: ``Stage`` enum member identifying the lifecycle phase being torn down.
    """

  def state_dict(self) -> dict[str, Any]:
    """Serializable data-module state for checkpoints.

    Includes ``'dataset_fingerprint'`` when ``self.dataset_fingerprint``
    is not None (serialized via ``DatasetFingerprint.to_dict()``).
    Subclasses should call ``super().state_dict()`` and merge their own
    keys to preserve fingerprint state.

    Returns:
      Dict of checkpoint-safe state.
    """
    result: dict[str, Any] = {}
    if self.dataset_fingerprint is not None:
      result['dataset_fingerprint'] = self.dataset_fingerprint.to_dict()
    return result

  def load_state_dict(self, state: dict[str, Any]) -> None:
    """Restore state from ``state_dict()`` output.

    Restores ``dataset_fingerprint`` from the ``'dataset_fingerprint'``
    key when present. Subclasses should call ``super().load_state_dict(state)``
    to preserve fingerprint restoration.

    Args:
      state: Dict previously returned by ``state_dict()``.
    """
    fp_data = state.get('dataset_fingerprint')
    if fp_data is not None:
      self.dataset_fingerprint = DatasetFingerprint.from_dict(fp_data)
    else:
      self.dataset_fingerprint = None

  def train_dataloader(self) -> DataLoader:
    """Build the training dataloader; subclasses must implement.

    Raises:
      NotImplementedError: On the base class.
    """
    raise NotImplementedError

  def val_dataloader(self) -> DataLoader:
    """Build the validation dataloader; subclasses must implement.

    Raises:
      NotImplementedError: On the base class.
    """
    raise NotImplementedError

  def test_dataloader(self) -> DataLoader:
    """Build the test dataloader; subclasses must implement.

    Raises:
      NotImplementedError: On the base class.
    """
    raise NotImplementedError

  def predict_dataloader(self) -> DataLoader:
    """Build the predict dataloader; subclasses must implement.

    Raises:
      NotImplementedError: On the base class.
    """
    msg = (
      f'{type(self).__name__} does not implement predict_dataloader(). '
      'Override it to return a DataLoader for Trainer.predict().'
    )
    raise NotImplementedError(msg)
