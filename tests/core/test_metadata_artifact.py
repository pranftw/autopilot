"""Unit tests for MetadataArtifact and QueryBuilder.metadata_contains."""

from autopilot.core.experiment import Experiment
from autopilot.core.metadata import MetadataArtifact
from autopilot.core.node import Node
from autopilot.core.query import QueryBuilder
from pathlib import Path
import pytest


class TestMetadataArtifactRoundtrip:
  """MetadataArtifact set/get/show file round-trip via tmp_path."""

  def test_metadata_artifact_roundtrip(self, tmp_path: Path) -> None:
    """Set/show matches; file exists on disk."""
    artifact = MetadataArtifact()
    artifact.set('env', 'production', base_dir=tmp_path)
    artifact.set('version', '2.0', base_dir=tmp_path)

    result = artifact.show(base_dir=tmp_path)
    assert result == {'env': 'production', 'version': '2.0'}
    assert (tmp_path / 'metadata.json').exists()

  def test_metadata_set_invalid_key(self, tmp_path: Path) -> None:
    """Empty key raises ValueError with actionable message."""
    artifact = MetadataArtifact()
    with pytest.raises(ValueError, match='must not be empty'):
      artifact.set('', 'value', base_dir=tmp_path)

  def test_metadata_get_missing_key(self, tmp_path: Path) -> None:
    """Get returns None for absent key."""
    artifact = MetadataArtifact()
    assert artifact.get('nonexistent', base_dir=tmp_path) is None

  def test_metadata_show_missing_file(self, tmp_path: Path) -> None:
    """Show returns empty dict when file does not exist."""
    artifact = MetadataArtifact()
    assert artifact.show(base_dir=tmp_path) == {}

  def test_metadata_overwrite(self, tmp_path: Path) -> None:
    """Second set replaces prior value for same key."""
    artifact = MetadataArtifact()
    artifact.set('key', 'first', base_dir=tmp_path)
    artifact.set('key', 'second', base_dir=tmp_path)
    assert artifact.get('key', base_dir=tmp_path) == 'second'

  def test_metadata_show_returns_shallow_copy(self, tmp_path: Path) -> None:
    """Mutating returned dict does not affect stored state."""
    artifact = MetadataArtifact()
    artifact.set('x', 'y', base_dir=tmp_path)
    data = artifact.show(base_dir=tmp_path)
    data['x'] = 'mutated'
    assert artifact.get('x', base_dir=tmp_path) == 'y'

  def test_metadata_non_string_values(self, tmp_path: Path) -> None:
    """Non-string values (int, list, dict) round-trip correctly."""
    artifact = MetadataArtifact()
    artifact.set('count', 42, base_dir=tmp_path)
    artifact.set('tags', ['a', 'b'], base_dir=tmp_path)
    artifact.set('nested', {'k': 'v'}, base_dir=tmp_path)

    assert artifact.get('count', base_dir=tmp_path) == 42
    assert artifact.get('tags', base_dir=tmp_path) == ['a', 'b']
    assert artifact.get('nested', base_dir=tmp_path) == {'k': 'v'}


class TestQueryBuilderMetadataContains:
  """QueryBuilder.metadata_contains filter tests."""

  def _make_node(self, experiment_id: str) -> Node:
    """Create a minimal completed node."""
    exp = Experiment(experiment_id=experiment_id)
    exp.start()
    exp.complete(metrics={'accuracy': 0.9})
    return Node(experiment=exp)

  def test_metadata_query_filter(self, tmp_path: Path) -> None:
    """Builder excludes non-matching experiments."""
    experiments_path = tmp_path / 'experiments'
    experiments_path.mkdir()

    node_a = self._make_node('exp-a')
    node_b = self._make_node('exp-b')
    node_c = self._make_node('exp-c')

    exp_a_dir = experiments_path / 'exp-a'
    exp_a_dir.mkdir()
    exp_b_dir = experiments_path / 'exp-b'
    exp_b_dir.mkdir()
    exp_c_dir = experiments_path / 'exp-c'
    exp_c_dir.mkdir()

    artifact = MetadataArtifact()
    artifact.set('env', 'prod', base_dir=exp_a_dir)
    artifact.set('env', 'staging', base_dir=exp_b_dir)

    nodes = [node_a, node_b, node_c]
    resolver = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, resolver.get)
    qb = qb.metadata_contains('env', 'prod', experiments_path)

    results = qb.all()
    assert len(results) == 1
    assert results[0].experiment.id == 'exp-a'

  def test_metadata_contains_string_coercion(self, tmp_path: Path) -> None:
    """Non-string metadata values matched via str() coercion."""
    experiments_path = tmp_path / 'experiments'
    experiments_path.mkdir()

    node = self._make_node('exp-x')
    exp_dir = experiments_path / 'exp-x'
    exp_dir.mkdir()

    artifact = MetadataArtifact()
    artifact.set('count', 42, base_dir=exp_dir)

    nodes = [node]
    resolver = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, resolver.get)
    qb = qb.metadata_contains('count', '42', experiments_path)

    results = qb.all()
    assert len(results) == 1

  def test_metadata_contains_no_match(self, tmp_path: Path) -> None:
    """Query returns empty when no experiment matches."""
    experiments_path = tmp_path / 'experiments'
    experiments_path.mkdir()

    node = self._make_node('exp-y')
    (experiments_path / 'exp-y').mkdir()

    nodes = [node]
    resolver = {n.experiment.id: n for n in nodes}
    qb = QueryBuilder(nodes, resolver.get)
    qb = qb.metadata_contains('missing', 'val', experiments_path)

    assert qb.all() == []

  def test_metadata_contains_immutable_chain(self, tmp_path: Path) -> None:
    """metadata_contains returns a new builder (immutable chain)."""
    experiments_path = tmp_path / 'experiments'
    experiments_path.mkdir()

    node = self._make_node('exp-z')
    (experiments_path / 'exp-z').mkdir()

    nodes = [node]
    resolver = {n.experiment.id: n for n in nodes}
    original = QueryBuilder(nodes, resolver.get)
    filtered = original.metadata_contains('k', 'v', experiments_path)

    assert original is not filtered
    assert len(original.all()) == 1
    assert len(filtered.all()) == 0


class TestMetadataCorruptJsonHandling:
  """MetadataArtifact degrades gracefully on corrupt JSON."""

  def test_show_returns_empty_on_corrupt_json(self, tmp_path: Path) -> None:
    """show() returns {} when metadata.json contains invalid JSON."""
    artifact = MetadataArtifact()
    (tmp_path / 'metadata.json').write_text('{invalid json!!!', encoding='utf-8')
    result = artifact.show(tmp_path)
    assert result == {}

  def test_show_returns_empty_on_non_dict_json(self, tmp_path: Path) -> None:
    """show() returns {} when metadata.json contains a non-dict JSON value."""
    artifact = MetadataArtifact()
    (tmp_path / 'metadata.json').write_text('"just a string"', encoding='utf-8')
    result = artifact.show(tmp_path)
    assert result == {}

  def test_get_returns_none_on_corrupt_json(self, tmp_path: Path) -> None:
    """get() returns None for any key when metadata.json is corrupt."""
    artifact = MetadataArtifact()
    (tmp_path / 'metadata.json').write_text('{bad', encoding='utf-8')
    assert artifact.get('key', tmp_path) is None

  def test_set_after_corrupt_json_recovers(self, tmp_path: Path) -> None:
    """set() on corrupt file starts from empty dict and recovers."""
    artifact = MetadataArtifact()
    (tmp_path / 'metadata.json').write_text('{bad', encoding='utf-8')
    artifact.set('key', 'value', tmp_path)
    assert artifact.get('key', tmp_path) == 'value'

  def test_show_returns_empty_on_json_array(self, tmp_path: Path) -> None:
    """show() returns {} when metadata.json contains a JSON array."""
    artifact = MetadataArtifact()
    (tmp_path / 'metadata.json').write_text('[1, 2, 3]', encoding='utf-8')
    result = artifact.show(tmp_path)
    assert result == {}

  def test_show_returns_empty_on_empty_file(self, tmp_path: Path) -> None:
    """show() returns {} when metadata.json is an empty file."""
    artifact = MetadataArtifact()
    (tmp_path / 'metadata.json').write_text('', encoding='utf-8')
    result = artifact.show(tmp_path)
    assert result == {}

  def test_show_returns_empty_on_null_json(self, tmp_path: Path) -> None:
    """show() returns {} when metadata.json contains null."""
    artifact = MetadataArtifact()
    (tmp_path / 'metadata.json').write_text('null', encoding='utf-8')
    result = artifact.show(tmp_path)
    assert result == {}
