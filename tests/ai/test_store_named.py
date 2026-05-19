"""Tests for FileStore named parameters, schema embedding, and reorder resilience."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.snapshot import ParameterSchema
from pathlib import Path
import pytest


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
  """Workspace root with two parameter directories."""
  ws = tmp_path / 'project'
  ws.mkdir()
  prompts = ws / 'prompts'
  prompts.mkdir()
  (prompts / 'system.txt').write_text('you are helpful', encoding='utf-8')
  config_dir = ws / 'config'
  config_dir.mkdir()
  (config_dir / 'settings.toml').write_text('key = "val"', encoding='utf-8')
  return ws


@pytest.fixture
def config(workspace: Path) -> AutoPilotConfig:
  """AutoPilotConfig rooted in the workspace."""
  return AutoPilotConfig(workspace=workspace)


class TestFileStoreRegisterNamed:
  """register_parameters stores the mapping internally."""

  def test_filestore_register_named(self, config: AutoPilotConfig, workspace: Path) -> None:
    store = FileStore(config)
    param = PathParameter(source=str(workspace / 'prompts'), pattern='*')
    store.register_parameters({'prompts': param})
    assert 'prompts' in store._param_names
    assert store._param_names['prompts'] is param


class TestFileStoreSnapshotNamedKeys:
  """Snapshot entries use named keys like 'prompts/system.txt'."""

  def test_filestore_snapshot_named_keys(self, config: AutoPilotConfig, workspace: Path) -> None:
    store = FileStore(config)
    param = PathParameter(source=str(workspace / 'prompts'), pattern='*')
    store.register_parameters({'prompts': param})
    manifest = store.snapshot('exp-1', 0)
    assert any(k.startswith('prompts/') for k in manifest.entries)
    assert not any(k.startswith('param_') for k in manifest.entries)


class TestFileStoreCheckoutNamed:
  """Checkout restores content to the correct named parameter."""

  def test_filestore_checkout_named(self, config: AutoPilotConfig, workspace: Path) -> None:
    store = FileStore(config)
    prompts_dir = workspace / 'prompts'
    param = PathParameter(source=str(prompts_dir), pattern='*')
    store.register_parameters({'prompts': param})

    store.snapshot('exp-1', 0)
    (prompts_dir / 'system.txt').write_text('modified', encoding='utf-8')
    store.checkout('exp-1', 0)
    assert (prompts_dir / 'system.txt').read_text(encoding='utf-8') == 'you are helpful'


class TestFileStoreReorderResilience:
  """Reordering registration dict does not break old snapshot checkout."""

  def test_filestore_reorder_resilience(self, config: AutoPilotConfig, workspace: Path) -> None:
    prompts_dir = workspace / 'prompts'
    config_dir = workspace / 'config'
    p_prompts = PathParameter(source=str(prompts_dir), pattern='*')
    p_config = PathParameter(source=str(config_dir), pattern='*')

    store = FileStore(config)
    store.register_parameters({'prompts': p_prompts, 'config': p_config})
    store.snapshot('exp-1', 0)

    (prompts_dir / 'system.txt').write_text('changed', encoding='utf-8')
    (config_dir / 'settings.toml').write_text('changed', encoding='utf-8')

    store2 = FileStore(config)
    store2.register_parameters({'config': p_config, 'prompts': p_prompts})
    store2.checkout('exp-1', 0)

    assert (prompts_dir / 'system.txt').read_text(encoding='utf-8') == 'you are helpful'
    assert (config_dir / 'settings.toml').read_text(encoding='utf-8') == 'key = "val"'


class TestFileStoreAddParameter:
  """Adding a third parameter does not break old two-param snapshots."""

  def test_filestore_add_parameter(self, config: AutoPilotConfig, workspace: Path) -> None:
    prompts_dir = workspace / 'prompts'
    config_dir = workspace / 'config'
    p_prompts = PathParameter(source=str(prompts_dir), pattern='*')
    p_config = PathParameter(source=str(config_dir), pattern='*')

    store = FileStore(config)
    store.register_parameters({'prompts': p_prompts, 'config': p_config})
    store.snapshot('exp-1', 0)

    extra_dir = workspace / 'extra'
    extra_dir.mkdir()
    (extra_dir / 'data.txt').write_text('extra', encoding='utf-8')
    p_extra = PathParameter(source=str(extra_dir), pattern='*')

    store2 = FileStore(config)
    store2.register_parameters({'prompts': p_prompts, 'config': p_config, 'extra': p_extra})
    (prompts_dir / 'system.txt').write_text('changed', encoding='utf-8')
    store2.checkout('exp-1', 0)
    assert (prompts_dir / 'system.txt').read_text(encoding='utf-8') == 'you are helpful'


class TestFileStoreSchemaEmbedded:
  """Manifest schema embeds parameter names, types, and sources."""

  def test_filestore_schema_embedded(self, config: AutoPilotConfig, workspace: Path) -> None:
    prompts_dir = workspace / 'prompts'
    param = PathParameter(source=str(prompts_dir), pattern='*')
    store = FileStore(config)
    store.register_parameters({'prompts': param})
    manifest = store.snapshot('exp-1', 0)

    assert manifest.schema is not None
    assert isinstance(manifest.schema, ParameterSchema)
    assert len(manifest.schema.parameters) == 1
    schema_entry = manifest.schema.parameters[0]
    assert schema_entry.name == 'prompts'
    assert schema_entry.type_name == 'PathParameter'
    assert schema_entry.source == str(prompts_dir)
    assert schema_entry.pattern == '*'

  def test_schema_round_trips_through_disk(self, config: AutoPilotConfig, workspace: Path) -> None:
    prompts_dir = workspace / 'prompts'
    param = PathParameter(source=str(prompts_dir), pattern='*')
    store = FileStore(config)
    store.register_parameters({'prompts': param})
    store.snapshot('exp-1', 0)

    loaded = store.load_snapshot('exp-1', 0)
    assert loaded.schema is not None
    assert loaded.schema.parameters[0].name == 'prompts'


class TestFileStoreSchemaValidation:
  """Checkout validates schema against registered params."""

  def test_total_mismatch_raises(self, config: AutoPilotConfig, workspace: Path) -> None:
    prompts_dir = workspace / 'prompts'
    param = PathParameter(source=str(prompts_dir), pattern='*')
    store = FileStore(config)
    store.register_parameters({'prompts': param})
    store.snapshot('exp-1', 0)

    store2 = FileStore(config)
    store2.register_parameters({'different_name': param})
    with pytest.raises(StoreError, match='no registered parameters match'):
      store2.checkout('exp-1', 0)

  def test_partial_mismatch_warning(
    self, config: AutoPilotConfig, workspace: Path, caplog: pytest.LogCaptureFixture
  ) -> None:
    prompts_dir = workspace / 'prompts'
    param = PathParameter(source=str(prompts_dir), pattern='*')
    extra_dir = workspace / 'extra'
    extra_dir.mkdir(parents=True, exist_ok=True)
    (extra_dir / 'data.txt').write_text('extra')
    extra_param = PathParameter(source=str(extra_dir), pattern='*')
    store = FileStore(config)
    store.register_parameters({'prompts': param, 'extra': extra_param})
    store.snapshot('exp-1', 0)

    store2 = FileStore(config)
    store2.register_parameters({'prompts': param})
    with caplog.at_level('WARNING'):
      store2.checkout('exp-1', 0)
    assert 'schema mismatch' in caplog.text

  def test_partial_mismatch_strict_raises(self, config: AutoPilotConfig, workspace: Path) -> None:
    prompts_dir = workspace / 'prompts'
    param = PathParameter(source=str(prompts_dir), pattern='*')
    extra_dir = workspace / 'extra'
    extra_dir.mkdir(parents=True, exist_ok=True)
    (extra_dir / 'data.txt').write_text('extra')
    extra_param = PathParameter(source=str(extra_dir), pattern='*')
    store = FileStore(config)
    store.register_parameters({'prompts': param, 'extra': extra_param})
    store.snapshot('exp-1', 0)

    store2 = FileStore(config)
    store2.register_parameters({'prompts': param})
    with pytest.raises(StoreError, match='schema mismatch'):
      store2.checkout('exp-1', 0, strict_schema=True)
