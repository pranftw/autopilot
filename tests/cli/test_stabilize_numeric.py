"""Test that stabilize CLI picks epoch 11 over epoch 9 (BUG-034, section 4.1 #10)."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from pathlib import Path


def test_stabilize_cli_invokes_numeric_selection(tmp_path: Path) -> None:
  """End-to-end: config.stabilize picks epoch 11 content, not epoch 9."""
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'main.py').write_text('v0')
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path = tmp_path / '.autopilot'
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('exp-001', 0)

  for epoch in range(1, 12):
    (src / 'main.py').write_text(f'v{epoch}')
    store.snapshot('exp-001', epoch)

  copied = config.stabilize('exp-001')

  assert len(copied) > 0
  content = (src / 'main.py').read_text()
  assert content == 'v11'
