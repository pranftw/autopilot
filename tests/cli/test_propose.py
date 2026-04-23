"""Tests for propose CLI command."""

from autopilot.cli.commands.propose import ProposeCommand
from autopilot.cli.output import Output
from autopilot.core.artifacts.epoch import MetricComparisonArtifact
from autopilot.core.proposal import ChangeProposal, read_proposals, record_proposal
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

_mc = MetricComparisonArtifact()


def _make_ctx(tmp_path: Path, experiment: str = 'test-exp') -> MagicMock:
  ctx = MagicMock()
  ctx.experiment = experiment
  ctx.epoch = 1
  ctx.workspace = tmp_path
  ctx.project = None
  ctx.output = Output(use_json=True)
  exp_dir = tmp_path / experiment
  exp_dir.mkdir(parents=True, exist_ok=True)
  ctx.experiment_dir.return_value = exp_dir
  return ctx


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
    ctx.output = MagicMock()
    cmd = ProposeCommand()
    args = MagicMock(proposal_id='')
    cmd.verify(ctx, args)
    ctx.output.error.assert_called_once()
    ctx.output.result.assert_not_called()

  def test_verify_regression_detected(self, tmp_path, capsys):
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
    args = MagicMock(proposal_id='abc123')
    cmd.verify(ctx, args)
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['verdict'] == 'regression_after_change'
    assert 'accuracy' in r['regressed_metrics']

  def test_verify_fix_confirmed(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)
    atomic_write_json(exp_dir / 'best_baseline.json', {'accuracy': 0.5})
    _mc.write(
      {
        'per_metric_deltas': {'accuracy': 0.3},
        'regressions': [],
        'improvements': [{'metric': 'accuracy', 'delta': 0.3, 'baseline': 0.5, 'candidate': 0.8}],
      },
      exp_dir,
      epoch=1,
    )

    cmd = ProposeCommand()
    args = MagicMock(proposal_id='abc123')
    cmd.verify(ctx, args)
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['verdict'] == 'fix_confirmed'

  def test_verify_inconclusive_no_comparison(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=1)
    atomic_write_json(exp_dir / 'best_baseline.json', {'accuracy': 0.8})

    cmd = ProposeCommand()
    args = MagicMock(proposal_id='abc123')
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
    args = MagicMock(proposal_id='abc123')
    cmd.verify(ctx, args)
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['verdict'] == 'inconclusive'

  def test_revert_no_id(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    ctx.output = MagicMock()
    cmd = ProposeCommand()
    args = MagicMock(proposal_id='', source='', store='', pattern='**/*')
    cmd.revert(ctx, args)
    ctx.output.error.assert_called_once()
    ctx.output.result.assert_not_called()

  def test_revert_no_source(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    ctx.output = MagicMock()
    exp_dir = tmp_path / 'test-exp'
    _seed_proposal(exp_dir, 'abc123', epoch=2)

    cmd = ProposeCommand()
    args = MagicMock(proposal_id='abc123', source='', store='', pattern='**/*')
    cmd.revert(ctx, args)
    ctx.output.error.assert_called()

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

    with patch('autopilot.cli.commands.propose.FileStore') as MockStore:
      mock_instance = MagicMock()
      MockStore.return_value = mock_instance
      cmd.revert(ctx, args)
      mock_instance.checkout.assert_called_once_with(2)

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
