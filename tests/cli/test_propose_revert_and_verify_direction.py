"""Tests for propose command fixes (Plan 08).

BUG-PROPOSE-REVERT-E0: revert wrongly rejects restore_epoch == 0.
FR-008: per-metric direction for propose verify via CLI flags.
"""

from autopilot.ai.proposal import ChangeProposal, record_proposal
from autopilot.cli.commands.propose import ProposeCommand
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from tests.cli.conftest import make_mock_cli_context
from unittest.mock import MagicMock, patch
import inspect
import json
import pytest


def _make_ctx(
  tmp_path: Path,
  experiment: str = 'test-exp',
  epoch: int | None = 1,
) -> MagicMock:
  """Build a mock CLIContext for propose tests."""
  return make_mock_cli_context(tmp_path, experiment=experiment, epoch=epoch, project=None)


def _seed_proposal(
  exp_dir: Path,
  proposal_id: str = 'abc123',
  epoch: int = 1,
) -> None:
  """Record a proposal for testing."""
  record_proposal(
    exp_dir,
    ChangeProposal(
      proposal_id=proposal_id,
      hypothesis='test hypothesis',
      target_node='accuracy',
      change_type='rule_change',
      epoch=epoch,
      status='proposed',
    ),
  )


def _seed_epoch_metrics(
  exp_dir: Path,
  epoch: int,
  metrics: dict[str, float],
) -> None:
  """Write epoch metrics file for testing."""
  atomic_write_json(exp_dir / f'epoch_{epoch}_metrics.json', metrics)


class TestProposeRevertEpochZero:
  """BUG-PROPOSE-REVERT-E0: revert wrongly rejects restore_epoch == 0."""

  def test_propose_revert_epoch_zero_succeeds(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
  ) -> None:
    """When restore_epoch resolves to 0, the guard must not fire.

    Regression test for BUG-PROPOSE-REVERT-E0: old code used ``<= 0``
    which rejected the valid epoch 0.
    """
    ctx = _make_ctx(tmp_path, epoch=0)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)

    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    (source_dir / 'data.txt').write_text('hello')

    cmd = ProposeCommand()
    args = MagicMock(
      proposal_id='abc123',
      source=str(source_dir),
      store=str(tmp_path / '.store'),
      pattern='**/*',
    )

    with patch('autopilot.cli.commands.propose.FileStore') as mock_store_cls:
      mock_instance = MagicMock()
      mock_store_cls.return_value = mock_instance
      cmd.revert(ctx, args)
      mock_instance.checkout.assert_called_once_with('test-exp', 0, context=None)

    captured = capsys.readouterr()
    result = json.loads(captured.out)['result']
    assert result['status'] == 'reverted'
    assert result['restored_epoch'] == 0

  def test_revert_epoch_zero_via_proposal_epoch_1(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
  ) -> None:
    """Proposal at epoch 1 with no explicit --epoch resolves to max(1-1, 0) = 0."""
    ctx = _make_ctx(tmp_path, epoch=None)
    ctx.epoch = None
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)

    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    (source_dir / 'data.txt').write_text('hello')

    cmd = ProposeCommand()
    args = MagicMock(
      proposal_id='abc123',
      source=str(source_dir),
      store=str(tmp_path / '.store'),
      pattern='**/*',
    )

    with patch('autopilot.cli.commands.propose.FileStore') as mock_store_cls:
      mock_instance = MagicMock()
      mock_store_cls.return_value = mock_instance
      cmd.revert(ctx, args)
      mock_instance.checkout.assert_called_once_with('test-exp', 0, context=None)

    captured = capsys.readouterr()
    result = json.loads(captured.out)['result']
    assert result['restored_epoch'] == 0

  def test_revert_negative_epoch_rejected(self, tmp_path: Path) -> None:
    """Explicit negative epoch is rejected with typed error message."""
    ctx = _make_ctx(tmp_path, epoch=-1)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)

    cmd = ProposeCommand()
    args = MagicMock(
      proposal_id='abc123',
      source=str(tmp_path / 'source'),
      store=str(tmp_path / '.store'),
      pattern='**/*',
    )

    with pytest.raises(SystemExit) as exc_info:
      cmd.revert(ctx, args)
    assert exc_info.value.code == 1


class TestProposeVerifyDirectionFlags:
  """FR-008: per-metric direction for propose verify."""

  def test_propose_verify_lower_is_better_flag(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
  ) -> None:
    """Loss metric with --lower-is-better should show 'improved' when it decreases."""
    ctx = _make_ctx(tmp_path, epoch=1)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)

    _seed_epoch_metrics(exp_dir, 0, {'loss': 0.8})
    _seed_epoch_metrics(exp_dir, 1, {'loss': 0.3})

    cmd = ProposeCommand()
    args = MagicMock(
      proposal_id='abc123',
      higher_is_better=None,
      lower_is_better=['loss'],
    )
    cmd.verify(ctx, args)

    captured = capsys.readouterr()
    result = json.loads(captured.out)['result']
    assert result['verdict'] == 'improved'
    assert len(result['deltas']) == 1
    delta = result['deltas'][0]
    assert delta['higher_is_better'] is False
    assert delta['delta'] < 0

  def test_propose_verify_default_higher_is_better(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
  ) -> None:
    """Without direction flags, metrics default to higher_is_better=True."""
    ctx = _make_ctx(tmp_path, epoch=1)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)

    _seed_epoch_metrics(exp_dir, 0, {'accuracy': 0.7})
    _seed_epoch_metrics(exp_dir, 1, {'accuracy': 0.9})

    cmd = ProposeCommand()
    args = MagicMock(
      proposal_id='abc123',
      higher_is_better=None,
      lower_is_better=None,
    )
    cmd.verify(ctx, args)

    captured = capsys.readouterr()
    result = json.loads(captured.out)['result']
    assert result['verdict'] == 'improved'
    assert result['deltas'][0]['higher_is_better'] is True

  def test_propose_verify_mixed_metric_directions(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
  ) -> None:
    """Two metrics: accuracy (default higher) and loss (flagged lower)."""
    ctx = _make_ctx(tmp_path, epoch=1)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)

    _seed_epoch_metrics(exp_dir, 0, {'accuracy': 0.7, 'loss': 0.8})
    _seed_epoch_metrics(exp_dir, 1, {'accuracy': 0.9, 'loss': 0.3})

    cmd = ProposeCommand()
    args = MagicMock(
      proposal_id='abc123',
      higher_is_better=None,
      lower_is_better=['loss'],
    )
    cmd.verify(ctx, args)

    captured = capsys.readouterr()
    result = json.loads(captured.out)['result']
    assert result['verdict'] == 'improved'

    deltas_by_name = {d['metric']: d for d in result['deltas']}
    assert deltas_by_name['accuracy']['higher_is_better'] is True
    assert deltas_by_name['loss']['higher_is_better'] is False

  def test_propose_verify_direction_conflict_raises(self, tmp_path: Path) -> None:
    """Same metric in both --higher-is-better and --lower-is-better is rejected."""
    ctx = _make_ctx(tmp_path, epoch=1)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)

    cmd = ProposeCommand()
    args = MagicMock(
      proposal_id='abc123',
      higher_is_better=['accuracy'],
      lower_is_better=['accuracy'],
    )

    with pytest.raises(SystemExit) as exc_info:
      cmd.verify(ctx, args)
    assert exc_info.value.code == 1

  def test_propose_verify_direction_conflict_message(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
  ) -> None:
    """Conflict error message includes the conflicting metric name."""
    ctx = _make_ctx(tmp_path, epoch=1)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)

    cmd = ProposeCommand()
    args = MagicMock(
      proposal_id='abc123',
      higher_is_better=['f1', 'accuracy'],
      lower_is_better=['accuracy'],
    )

    with pytest.raises(SystemExit):
      cmd.verify(ctx, args)

    captured = capsys.readouterr()
    assert 'accuracy' in captured.err or 'accuracy' in captured.out

  def test_propose_verify_lower_is_better_regressed(
    self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
  ) -> None:
    """Loss increasing with lower-is-better should show 'regressed'."""
    ctx = _make_ctx(tmp_path, epoch=1)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)

    _seed_epoch_metrics(exp_dir, 0, {'loss': 0.3})
    _seed_epoch_metrics(exp_dir, 1, {'loss': 0.8})

    cmd = ProposeCommand()
    args = MagicMock(
      proposal_id='abc123',
      higher_is_better=None,
      lower_is_better=['loss'],
    )
    cmd.verify(ctx, args)

    captured = capsys.readouterr()
    result = json.loads(captured.out)['result']
    assert result['verdict'] == 'regressed'


class TestProposeVerifyDocstring:
  """Verify docstring documents default metric direction."""

  def test_propose_verify_docstring_mentions_default_direction(self) -> None:
    """Module or method docstring documents the higher_is_better default."""
    method_doc = inspect.getdoc(ProposeCommand.verify) or ''
    assert 'higher_is_better' in method_doc

  def test_propose_module_docstring_mentions_direction(self) -> None:
    """Module docstring documents the metric direction flags."""
    from autopilot.cli.commands import propose

    module_doc = inspect.getdoc(propose) or propose.__doc__ or ''
    assert 'higher_is_better' in module_doc
