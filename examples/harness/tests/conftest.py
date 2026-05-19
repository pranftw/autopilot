"""Shared fixtures for harness tests."""

from pathlib import Path
import json
import pytest


def _make_record(task_id: str, message: str, tool: str, comm: str, assertion: str) -> dict:
  """Build a single scenario record matching the JSONL schema."""
  return {
    'task_id': task_id,
    'initial_message': message,
    'user_instructions': {
      'reason_for_call': message,
      'known_info': {'sku': f'sku-{task_id}'},
      'task_instructions': f'do task {task_id}',
    },
    'evaluation_criteria': {
      'expected_actions': [{'tool': tool, 'args': {'q': task_id}}],
      'communicate_info': [comm],
      'nl_assertions': [assertion],
    },
  }


@pytest.fixture
def sample_records() -> list[dict]:
  """Three distinct scenario records for test use."""
  return [
    _make_record('t0', 'hello, I need help', 'find', 'say refund policy', 'agent was polite'),
    _make_record('t1', 'I want to return', 'lookup', 'confirm address', 'agent asked for order'),
    _make_record('t2', 'exchange request', 'search', 'explain options', 'agent offered alt'),
  ]


@pytest.fixture
def scenarios_dir(tmp_path: Path, sample_records: list[dict]) -> Path:
  """Create a temporary scenarios directory with train/val/test JSONL files."""
  root = tmp_path / 'scenarios'
  root.mkdir()
  for name, subset in (
    ('train.jsonl', sample_records[:2]),
    ('val.jsonl', sample_records[2:]),
    ('test.jsonl', sample_records[1:]),
  ):
    path = root / name
    lines = [json.dumps(r, ensure_ascii=False) for r in subset]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
  return root
