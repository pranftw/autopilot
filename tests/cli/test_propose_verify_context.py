"""Tests for propose verify auto-emitting comparison context (sub-plan 10).

Covers:
  - test_propose_verify_emits_context_entry
  - test_propose_verify_context_contains_verdict
  - test_propose_verify_no_experiment_skips_context
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.proposal import ChangeProposal, record_proposal
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.propose import ProposeCommand
from autopilot.core.artifacts.epoch import DataArtifact
from autopilot.core.config import AutoPilotConfig
from autopilot.core.decision import DecisionEntry
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import make_mock_cli_context
import argparse
import json


def _seed_proposal(exp_dir: Path, proposal_id: str, epoch: int) -> ChangeProposal:
  """Write a proposal to the experiment directory."""
  proposal = ChangeProposal(
    proposal_id=proposal_id,
    hypothesis='test hypothesis',
    target_node=None,
    change_type='general',
    epoch=epoch,
    status='proposed',
  )
  record_proposal(exp_dir, proposal)
  return proposal


def _seed_epoch_metrics(exp_dir: Path, epoch: int, metrics: dict) -> None:
  """Write metrics for a given epoch."""
  metrics_file = exp_dir / f'epoch_{epoch}_metrics.json'
  metrics_file.write_text(json.dumps(metrics), encoding='utf-8')


def _seed_data_artifact(exp_dir: Path, epoch: int) -> None:
  """Write minimal data artifact for items_tested."""
  DataArtifact().write([{'input': 'x', 'output': 'y'}], exp_dir, epoch=epoch)


def _make_forest_with_experiment(
  workspace: Path,
  experiment_id: str,
  metrics: dict | None = None,
) -> FileForest:
  """Build a forest with one tree and one experiment."""
  config = AutoPilotConfig(workspace=workspace)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')
  exp = Experiment(experiment_id=experiment_id)
  exp.start()
  exp.complete(metrics=metrics)
  node = Node(experiment=exp)
  tree.add(node)
  forest.save()
  return forest


class TestProposeVerifyEmitsContext:
  """Verify that propose verify emits comparison context to the experiment."""

  def test_propose_verify_emits_context_entry(self, tmp_path: Path) -> None:
    """After verify, experiment context log has source='proposal' entry."""
    exp_id = 'exp-ctx-1'
    _make_forest_with_experiment(tmp_path, exp_id, metrics={'accuracy': 0.8})

    exp_dir = tmp_path / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    _seed_proposal(exp_dir, 'abc12345', epoch=0)
    _seed_epoch_metrics(exp_dir, 0, {'accuracy': 0.7})
    _seed_epoch_metrics(exp_dir, 1, {'accuracy': 0.9})
    _seed_data_artifact(exp_dir, 1)

    ctx = make_mock_cli_context(tmp_path, experiment=exp_id, epoch=1)
    args = argparse.Namespace(
      proposal_id='abc12345',
      higher_is_better=None,
      lower_is_better=None,
    )

    cmd = ProposeCommand()
    cmd.verify(ctx, args)

    reloaded = FileForest(FileStore(AutoPilotConfig(workspace=tmp_path)))
    found = reloaded.find_experiment(exp_id)
    assert found is not None
    experiment = found[0].experiment
    context_entries = experiment.context_log.filter_by_source('proposal')
    assert len(context_entries) == 1
    entry = context_entries[0]
    assert entry.metadata['_type'] == DecisionEntry.COMPARISON_TYPE

  def test_propose_verify_context_contains_verdict(self, tmp_path: Path) -> None:
    """Metadata includes verdict, deltas, proposal_id, epoch fields."""
    exp_id = 'exp-ctx-2'
    _make_forest_with_experiment(tmp_path, exp_id, metrics={'accuracy': 0.8})

    exp_dir = tmp_path / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    _seed_proposal(exp_dir, 'def67890', epoch=1)
    _seed_epoch_metrics(exp_dir, 0, {'accuracy': 0.7})
    _seed_epoch_metrics(exp_dir, 2, {'accuracy': 0.9})
    _seed_data_artifact(exp_dir, 2)

    ctx = make_mock_cli_context(tmp_path, experiment=exp_id, epoch=2)
    args = argparse.Namespace(
      proposal_id='def67890',
      higher_is_better=None,
      lower_is_better=None,
    )

    cmd = ProposeCommand()
    cmd.verify(ctx, args)

    reloaded = FileForest(FileStore(AutoPilotConfig(workspace=tmp_path)))
    found = reloaded.find_experiment(exp_id)
    assert found is not None
    experiment = found[0].experiment
    context_entries = experiment.context_log.filter_by_source('proposal')
    assert len(context_entries) == 1
    meta = context_entries[0].metadata
    assert meta['verdict'] == 'improved'
    assert isinstance(meta['deltas'], list)
    assert meta['proposal_id'] == 'def67890'
    assert meta['baseline_epoch'] == 0
    assert meta['candidate_epoch'] == 2

  def test_propose_verify_no_experiment_skips_context(self, tmp_path: Path) -> None:
    """Verify exits 0 without crash when forest has no matching experiment."""
    exp_id = 'exp-no-forest'
    exp_dir = tmp_path / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    _seed_proposal(exp_dir, 'aaa11111', epoch=0)
    _seed_epoch_metrics(exp_dir, 0, {'accuracy': 0.7})
    _seed_epoch_metrics(exp_dir, 1, {'accuracy': 0.9})
    _seed_data_artifact(exp_dir, 1)

    ctx = make_mock_cli_context(tmp_path, experiment=exp_id, epoch=1)
    args = argparse.Namespace(
      proposal_id='aaa11111',
      higher_is_better=None,
      lower_is_better=None,
    )

    cmd = ProposeCommand()
    cmd.verify(ctx, args)
