"""Tests for Diagnostics base class."""

from autopilot.core.diagnostics import (
  DiagnosisEntry,
  DiagnosticResult,
  Diagnostics,
  NodeScore,
)


class TestDiagnosisEntryRoundTrip:
  def test_to_dict_from_dict(self):
    entry = DiagnosisEntry(category='syntax', count=3, sample_errors=['a', 'b', 'c'])
    d = entry.to_dict()
    restored = DiagnosisEntry.from_dict(d)
    assert restored.category == 'syntax'
    assert restored.count == 3
    assert restored.sample_errors == ['a', 'b', 'c']


class TestNodeScoreRoundTrip:
  def test_to_dict_from_dict(self):
    score = NodeScore(total=10, failed=3, error_rate=0.3)
    d = score.to_dict()
    restored = NodeScore.from_dict(d)
    assert restored.total == 10
    assert restored.failed == 3
    assert restored.error_rate == 0.3


class TestDiagnosticsAnalyze:
  def test_pure_computation_no_io(self, tmp_path):
    diag = Diagnostics(tmp_path)
    data = [
      {'id': 'n1', 'success': True, 'metadata': {'category': 'syntax'}},
      {
        'id': 'n2',
        'success': False,
        'error_message': 'parse error',
        'metadata': {'category': 'syntax'},
      },
      {
        'id': 'n3',
        'success': False,
        'error_message': 'timeout',
        'metadata': {'category': 'network'},
      },
      {'id': 'n1', 'success': True, 'metadata': {'category': 'syntax'}},
    ]
    result = diag.analyze(data, epoch=1)
    assert isinstance(result, DiagnosticResult)
    assert result.epoch == 1
    assert len(result.diagnoses) == 2
    categories = {d.category for d in result.diagnoses}
    assert 'syntax' in categories
    assert 'network' in categories
    assert result.heatmap['n1'].total == 2
    assert result.heatmap['n1'].failed == 0
    assert result.heatmap['n2'].failed == 1
    assert result.heatmap['n3'].error_rate == 1.0

  def test_empty_data(self, tmp_path):
    diag = Diagnostics(tmp_path)
    result = diag.analyze([], epoch=1)
    assert result.diagnoses == []
    assert result.heatmap == {}

  def test_all_successes(self, tmp_path):
    diag = Diagnostics(tmp_path)
    data = [
      {'id': 'a', 'success': True, 'metadata': {}},
      {'id': 'b', 'success': True, 'metadata': {}},
    ]
    result = diag.analyze(data, epoch=1)
    assert result.diagnoses == []
    assert result.heatmap['a'].failed == 0
    assert result.heatmap['b'].failed == 0

  def test_all_failures(self, tmp_path):
    diag = Diagnostics(tmp_path)
    data = [
      {'id': f'n{i}', 'success': False, 'error_message': f'err{i}', 'metadata': {'category': 'bug'}}
      for i in range(10)
    ]
    result = diag.analyze(data, epoch=1)
    assert len(result.diagnoses) == 1
    assert result.diagnoses[0].category == 'bug'
    assert result.diagnoses[0].count == 10
    assert len(result.diagnoses[0].sample_errors) == 5

  def test_samples_capped_at_limit(self, tmp_path):
    diag = Diagnostics(tmp_path)
    data = [
      {'id': 'n', 'success': False, 'error_message': f'e{i}', 'metadata': {'category': 'x'}}
      for i in range(20)
    ]
    result = diag.analyze(data, epoch=1)
    assert len(result.diagnoses[0].sample_errors) == 5


class TestDiagnosticsWriteRead:
  def test_write_read_diagnoses_round_trip(self, tmp_path):
    diag = Diagnostics(tmp_path)
    result = DiagnosticResult(
      epoch=1,
      diagnoses=[
        DiagnosisEntry(category='syntax', count=2, sample_errors=['a', 'b']),
        DiagnosisEntry(category='network', count=1, sample_errors=['c']),
      ],
      heatmap={
        'node1': NodeScore(total=5, failed=2, error_rate=0.4),
      },
    )
    diag.write(result)
    entries = diag.read_diagnoses(1)
    assert len(entries) == 2
    assert entries[0].category == 'syntax'
    assert entries[1].category == 'network'

  def test_write_read_heatmap_round_trip(self, tmp_path):
    diag = Diagnostics(tmp_path)
    result = DiagnosticResult(
      epoch=2,
      diagnoses=[],
      heatmap={
        'node1': NodeScore(total=10, failed=3, error_rate=0.3),
        'node2': NodeScore(total=5, failed=0, error_rate=0.0),
      },
    )
    diag.write(result)
    hm = diag.read_heatmap(2)
    assert hm['node1'].total == 10
    assert hm['node2'].failed == 0

  def test_read_diagnoses_missing_epoch(self, tmp_path):
    diag = Diagnostics(tmp_path)
    entries = diag.read_diagnoses(99)
    assert entries == []

  def test_read_heatmap_missing_epoch(self, tmp_path):
    diag = Diagnostics(tmp_path)
    hm = diag.read_heatmap(99)
    assert hm == {}


class TestDiagnosticsCustomSubclass:
  def test_custom_categorize(self, tmp_path):
    class CustomDiag(Diagnostics):
      def categorize(self, item):
        return 'custom_cat'

    diag = CustomDiag(tmp_path)
    data = [{'id': 'x', 'success': False, 'error_message': 'e', 'metadata': {}}]
    result = diag.analyze(data, epoch=1)
    assert result.diagnoses[0].category == 'custom_cat'

  def test_custom_is_failure(self, tmp_path):
    class ScoreDiag(Diagnostics):
      def is_failure(self, item):
        return item.get('score', 1.0) < 0.5

    diag = ScoreDiag(tmp_path)
    data = [
      {'id': 'a', 'score': 0.3, 'metadata': {'category': 'low'}},
      {'id': 'b', 'score': 0.8, 'metadata': {'category': 'high'}},
    ]
    result = diag.analyze(data, epoch=1)
    assert len(result.diagnoses) == 1
    assert result.heatmap['a'].failed == 1
    assert result.heatmap['b'].failed == 0

  def test_custom_resolve_node(self, tmp_path):
    class NodeDiag(Diagnostics):
      def resolve_node(self, item):
        return item.get('custom_node', 'unknown')

    diag = NodeDiag(tmp_path)
    data = [{'custom_node': 'my_node', 'success': True, 'metadata': {}}]
    result = diag.analyze(data, epoch=1)
    assert 'my_node' in result.heatmap

  def test_custom_score_node(self, tmp_path):
    class WeightedDiag(Diagnostics):
      def score_node(self, node, items):
        return NodeScore(total=len(items), failed=0, error_rate=0.0)

    diag = WeightedDiag(tmp_path)
    data = [{'id': 'x', 'success': False, 'error_message': 'e', 'metadata': {'category': 'a'}}]
    result = diag.analyze(data, epoch=1)
    assert result.heatmap['x'].failed == 0

  def test_custom_select_samples(self, tmp_path):
    class CustomSamples(Diagnostics):
      def select_samples(self, items, limit=5):
        return ['custom_sample']

    diag = CustomSamples(tmp_path)
    data = [{'id': 'x', 'success': False, 'error_message': 'e', 'metadata': {'category': 'a'}}]
    result = diag.analyze(data, epoch=1)
    assert result.diagnoses[0].sample_errors == ['custom_sample']


class TestDiagnosticsArtifactRegistration:
  def test_artifacts_registered(self, tmp_path):
    diag = Diagnostics(tmp_path)
    arts = diag.artifacts
    assert 'diagnoses_artifact' in arts
    assert 'heatmap_artifact' in arts
