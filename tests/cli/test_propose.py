"""Tests for propose CLI command."""

from autopilot.ai.proposal import ChangeProposal, read_proposals, record_proposal
from autopilot.cli.commands.propose import ProposeCommand
from autopilot.core.artifacts.epoch import MetricComparisonArtifact
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from tests.cli.conftest import make_mock_cli_context
from unittest.mock import MagicMock, patch
import argparse
import json
import pytest

_mc = MetricComparisonArtifact()


def _verify_args(proposal_id: str = 'abc123') -> argparse.Namespace:
  """Build verify args with required attributes."""
  return argparse.Namespace(
    proposal_id=proposal_id,
    higher_is_better=None,
    lower_is_better=None,
  )


def _make_ctx(tmp_path: Path, experiment: str = 'test-exp') -> MagicMock:
  return make_mock_cli_context(tmp_path, experiment=experiment, epoch=1, project=None)


def _seed_proposal(exp_dir: Path, proposal_id: str = 'abc123', epoch: int = 1) -> None:
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


class TestProposeCommand:
  def test_instantiates(self):
    cmd = ProposeCommand()
    assert cmd.name == 'propose'

  def test_create(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    cmd = ProposeCommand()
    args = MagicMock(target='node_a', hypothesis='will improve accuracy', category='rule_change')
    cmd.create(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['status'] == 'created'
    assert 'proposal_id' in envelope['result']

    exp_dir = tmp_path / 'test-exp'
    proposals = read_proposals(exp_dir)
    assert len(proposals) == 1
    assert proposals[0].hypothesis == 'will improve accuracy'

  def test_verify_no_id(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    cmd = ProposeCommand()
    args = _verify_args(proposal_id='')
    with pytest.raises(SystemExit) as exc_info:
      cmd.verify(ctx, args)
    assert exc_info.value.code == 1

  def test_verify_returns_inconclusive(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)
    _mc.write(
      {
        'regression_detected': True,
        'regressions': [{'metric': 'accuracy', 'delta': -0.3}],
      },
      exp_dir,
      epoch=1,
    )

    cmd = ProposeCommand()
    args = _verify_args()
    cmd.verify(ctx, args)
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['verdict'] == 'inconclusive'

  def test_verify_inconclusive_no_comparison(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)
    atomic_write_json(exp_dir / 'best_baseline.json', {'accuracy': 0.8})

    cmd = ProposeCommand()
    args = _verify_args()
    cmd.verify(ctx, args)
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['verdict'] == 'inconclusive'

  def test_verify_inconclusive_no_improvement(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)
    atomic_write_json(exp_dir / 'best_baseline.json', {'accuracy': 0.8})
    _mc.write(
      {
        'per_metric_deltas': {'accuracy': 0.0},
        'regressions': [],
        'improvements': [],
      },
      exp_dir,
      epoch=1,
    )

    cmd = ProposeCommand()
    args = _verify_args()
    cmd.verify(ctx, args)
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['verdict'] == 'inconclusive'

  def test_revert_no_id(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    cmd = ProposeCommand()
    args = MagicMock(proposal_id='', source='', store='', pattern='**/*')
    with pytest.raises(SystemExit) as exc_info:
      cmd.revert(ctx, args)
    assert exc_info.value.code == 1

  def test_revert_no_source(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=2)

    cmd = ProposeCommand()
    args = MagicMock(proposal_id='abc123', source='', store='', pattern='**/*')
    with pytest.raises(SystemExit) as exc_info:
      cmd.revert(ctx, args)
    assert exc_info.value.code == 1

  def test_revert_calls_store_checkout(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    ctx.epoch = 2
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=3)

    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    (source_dir / 'rules.json').write_text('[]')

    cmd = ProposeCommand()
    args = MagicMock(
      proposal_id='abc123',
      source=str(source_dir),
      store=str(tmp_path / '.store'),
      pattern='**/*',
    )

    with patch('autopilot.cli.commands.propose.FileStore') as mock_store:
      mock_instance = MagicMock()
      mock_store.return_value = mock_instance
      cmd.revert(ctx, args)
      mock_instance.checkout.assert_called_once_with('test-exp', 2, context=None)

    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['status'] == 'reverted'
    assert r['restored_epoch'] == 2

  def test_list_empty(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    cmd = ProposeCommand()
    args = MagicMock()
    cmd.list_proposals(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['count'] == 0
    assert envelope['result']['proposals'] == []

  def test_list_after_create(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    cmd = ProposeCommand()
    args_create = MagicMock(target='x', hypothesis='y', category='z')
    cmd.create(ctx, args_create)
    capsys.readouterr()

    args_list = MagicMock()
    cmd.list_proposals(ctx, args_list)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['count'] == 1
