"""Tests for stage artifact I/O."""

from autopilot.core.stage_io import (
  append_epoch_artifact,
  append_experiment_artifact,
  read_epoch_artifact,
  read_epoch_artifact_lines,
  read_experiment_artifact,
  read_experiment_artifact_lines,
  write_epoch_artifact,
  write_experiment_artifact,
)


class TestEpochArtifacts:
  def test_write_and_read_round_trip(self, tmp_path):
    write_epoch_artifact(tmp_path, 1, 'metrics.json', {'accuracy': 0.8})
    result = read_epoch_artifact(tmp_path, 1, 'metrics.json')
    assert result == {'accuracy': 0.8}

  def test_append_and_read_lines(self, tmp_path):
    append_epoch_artifact(tmp_path, 1, 'data.jsonl', {'item': 'a'})
    append_epoch_artifact(tmp_path, 1, 'data.jsonl', {'item': 'b'})
    append_epoch_artifact(tmp_path, 1, 'data.jsonl', {'item': 'c'})
    lines = read_epoch_artifact_lines(tmp_path, 1, 'data.jsonl')
    assert len(lines) == 3

  def test_epoch_dir_auto_created(self, tmp_path):
    write_epoch_artifact(tmp_path, 5, 'test.json', {'ok': True})
    assert (tmp_path / 'epoch_5' / 'test.json').exists()

  def test_read_nonexistent_returns_none(self, tmp_path):
    assert read_epoch_artifact(tmp_path, 99, 'missing.json') is None

  def test_read_lines_nonexistent_returns_empty(self, tmp_path):
    assert read_epoch_artifact_lines(tmp_path, 99, 'missing.jsonl') == []


class TestExperimentArtifacts:
  def test_write_and_read(self, tmp_path):
    write_experiment_artifact(tmp_path, 'summary.json', {'total': 5})
    result = read_experiment_artifact(tmp_path, 'summary.json')
    assert result == {'total': 5}

  def test_append_and_read_lines(self, tmp_path):
    append_experiment_artifact(tmp_path, 'log.jsonl', {'epoch': 1})
    append_experiment_artifact(tmp_path, 'log.jsonl', {'epoch': 2})
    lines = read_experiment_artifact_lines(tmp_path, 'log.jsonl')
    assert len(lines) == 2

  def test_read_nonexistent(self, tmp_path):
    assert read_experiment_artifact(tmp_path, 'nope.json') is None

  def test_read_lines_nonexistent(self, tmp_path):
    assert read_experiment_artifact_lines(tmp_path, 'nope.jsonl') == []

  def test_empty_payload(self, tmp_path):
    write_experiment_artifact(tmp_path, 'empty.json', {})
    result = read_experiment_artifact(tmp_path, 'empty.json')
    assert result == {}
