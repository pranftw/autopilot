"""Documentation completeness tests for dogfood V8 plan 13.

Verifies that key documentation sections added during the dogfood-v8
documentation wave (plan 13) are present in CLAUDE.md and AGENTS.md.
These are documentation drift guards that prevent accidental removal.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAUDE_MD = (REPO_ROOT / 'CLAUDE.md').read_text()
AGENTS_MD = (REPO_ROOT / 'AGENTS.md').read_text()


def test_graph_preservation_documented():
  """Graph preservation rule for Module.forward() must be present."""
  assert 'Graph preservation in `Module.forward()`' in CLAUDE_MD


def test_metrics_vs_metadata_documented():
  """Metrics vs Metadata boundary section must be present."""
  assert 'Metrics vs Metadata boundary' in CLAUDE_MD


def test_experiment_creation_rationale_documented():
  """Experiment-creation rationale usage pattern must be present."""
  assert 'Experiment-creation rationale' in CLAUDE_MD


def test_gradient_accumulation_detail_documented():
  """Gradient accumulation detail section must be present."""
  assert 'Gradient accumulation detail' in CLAUDE_MD


def test_agents_md_mirrors_claude_md():
  """All four key documentation phrases must also appear in AGENTS.md."""
  phrases = [
    'Graph preservation in `Module.forward()`',
    'Metrics vs Metadata boundary',
    'Experiment-creation rationale',
    'Gradient accumulation detail',
  ]
  for phrase in phrases:
    assert phrase in AGENTS_MD, f'{phrase!r} missing from AGENTS.md'
