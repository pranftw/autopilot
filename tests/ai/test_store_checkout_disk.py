"""AP-2: Real FileStore.checkout tests with seeded snapshots.

Validates that ``FileStore.checkout`` restores file contents on disk after
mutation, using real ``PathParameter`` registration and ``snapshot()`` calls.
No mocks on checkout paths.

See plan 22, subplan 2.2.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from pathlib import Path


def _setup_store_with_param(
  tmp_path: Path,
) -> tuple[FileStore, PathParameter, Path]:
  """Create a store, register a PathParameter, and return (store, param, source_dir)."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  source_dir = ws / 'params'
  source_dir.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  param = PathParameter(source=str(source_dir), pattern='**/*.txt')
  store.register_parameters({'prompts': param})
  return store, param, source_dir


def test_store_checkout_restores_file_contents_epoch_0(tmp_path: Path) -> None:
  """Checkout epoch 0 restores the original file bytes after mutation.

  AP-2: seed a PathParameter file, snapshot at epoch 0, mutate, checkout,
  assert original content is restored.
  """
  store, _param, source_dir = _setup_store_with_param(tmp_path)
  experiment_id = 'exp-checkout'

  original_content = 'hello from epoch 0'
  (source_dir / 'prompt.txt').write_text(original_content, encoding='utf-8')

  store.snapshot(experiment_id, 0)

  mutated_content = 'MUTATED content that should be reverted'
  (source_dir / 'prompt.txt').write_text(mutated_content, encoding='utf-8')
  assert (source_dir / 'prompt.txt').read_text(encoding='utf-8') == mutated_content

  store.checkout(experiment_id, 0)

  restored = (source_dir / 'prompt.txt').read_text(encoding='utf-8')
  assert restored == original_content, (
    f'checkout did not restore epoch 0 content: got {restored!r}, expected {original_content!r}'
  )


def test_store_checkout_latest_epoch_tip(tmp_path: Path) -> None:
  """Multi-epoch snapshots: checkout epoch 0 vs 1 yields distinct payloads.

  AP-2: snapshot at epoch 0, modify file, snapshot at epoch 1. Checking
  out epoch 0 restores epoch-0 content; checking out epoch 1 restores
  epoch-1 content.
  """
  store, _param, source_dir = _setup_store_with_param(tmp_path)
  experiment_id = 'exp-tip'

  epoch_0_content = 'version zero'
  (source_dir / 'prompt.txt').write_text(epoch_0_content, encoding='utf-8')
  store.snapshot(experiment_id, 0)

  epoch_1_content = 'version one -- improved prompt'
  (source_dir / 'prompt.txt').write_text(epoch_1_content, encoding='utf-8')
  store.snapshot(experiment_id, 1)

  (source_dir / 'prompt.txt').write_text('garbage -- should be overwritten', encoding='utf-8')

  store.checkout(experiment_id, 0)
  assert (source_dir / 'prompt.txt').read_text(encoding='utf-8') == epoch_0_content

  store.checkout(experiment_id, 1)
  assert (source_dir / 'prompt.txt').read_text(encoding='utf-8') == epoch_1_content


def test_store_checkout_multiple_files(tmp_path: Path) -> None:
  """Checkout restores multiple files under a directory parameter."""
  store, _param, source_dir = _setup_store_with_param(tmp_path)
  experiment_id = 'exp-multi'

  (source_dir / 'a.txt').write_text('file-a-original', encoding='utf-8')
  (source_dir / 'b.txt').write_text('file-b-original', encoding='utf-8')
  store.snapshot(experiment_id, 0)

  (source_dir / 'a.txt').write_text('file-a-mutated', encoding='utf-8')
  (source_dir / 'b.txt').write_text('file-b-mutated', encoding='utf-8')

  store.checkout(experiment_id, 0)

  assert (source_dir / 'a.txt').read_text(encoding='utf-8') == 'file-a-original'
  assert (source_dir / 'b.txt').read_text(encoding='utf-8') == 'file-b-original'
