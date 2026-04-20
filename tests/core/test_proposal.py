"""Tests for proposal I/O helpers."""

from autopilot.core.proposal import read_proposals, read_verdict, record_proposal, record_verdict
from autopilot.core.stage_models import ChangeProposal, ProposalVerdict


class TestProposalIO:
  def test_record_proposal(self, tmp_path):
    p = ChangeProposal(proposal_id='p1', hypothesis='test hypo', epoch=1)
    record_proposal(tmp_path, p)
    assert (tmp_path / 'hypothesis_log.jsonl').exists()

  def test_read_proposals(self, tmp_path):
    for i in range(3):
      record_proposal(tmp_path, ChangeProposal(proposal_id=f'p{i}', epoch=i))
    proposals = read_proposals(tmp_path)
    assert len(proposals) == 3
    assert proposals[0].proposal_id == 'p0'
    assert proposals[2].proposal_id == 'p2'

  def test_read_proposals_empty(self, tmp_path):
    assert read_proposals(tmp_path) == []

  def test_record_verdict(self, tmp_path):
    v = ProposalVerdict(proposal_id='p1', verdict='fix_confirmed', items_tested=5)
    record_verdict(tmp_path, epoch=1, verdict=v)
    result = read_verdict(tmp_path, epoch=1)
    assert result is not None
    assert result.verdict == 'fix_confirmed'

  def test_read_verdict_missing(self, tmp_path):
    assert read_verdict(tmp_path, epoch=99) is None

  def test_proposal_id_in_output(self, tmp_path):
    p = ChangeProposal(proposal_id='unique-id', hypothesis='test')
    record_proposal(tmp_path, p)
    proposals = read_proposals(tmp_path)
    assert proposals[0].proposal_id == 'unique-id'

  def test_persistence_across_calls(self, tmp_path):
    record_proposal(tmp_path, ChangeProposal(proposal_id='p1'))
    proposals = read_proposals(tmp_path)
    assert len(proposals) == 1
    record_proposal(tmp_path, ChangeProposal(proposal_id='p2'))
    proposals = read_proposals(tmp_path)
    assert len(proposals) == 2
