"""Tests for stabilize --parameter-prefix.

Covers:
  - parameter_prefix limits merged parameters
  - without parameter_prefix merges all
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from pathlib import Path


class TestStabilizeParameterPrefix:
  def _setup_two_params(self, tmp_path: Path) -> AutoPilotConfig:
    """Create a store with two parameters and snapshot them."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'main.txt').write_text('prompt content')

    configs_dir = tmp_path / 'configs'
    configs_dir.mkdir()
    (configs_dir / 'settings.txt').write_text('config content')

    config = AutoPilotConfig(workspace=tmp_path)
    prompt_param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    config_param = PathParameter(source=str(configs_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'prompts': prompt_param, 'configs': config_param})
    store.snapshot('exp-001', 0)

    (prompts_dir / 'main.txt').write_text('MODIFIED')
    (configs_dir / 'settings.txt').write_text('MODIFIED')

    return config

  def test_prefix_limits_merged_parameters(self, tmp_path: Path) -> None:
    """Only names matching prefix are merged when flag set."""
    config = self._setup_two_params(tmp_path)

    copied = config.stabilize('exp-001', parameter_prefix='prompts')

    assert len(copied) == 1
    assert any('prompts' in str(p) for p in copied)
    assert (tmp_path / 'prompts' / 'main.txt').read_text() == 'prompt content'
    assert (tmp_path / 'configs' / 'settings.txt').read_text() == 'MODIFIED'

  def test_without_prefix_merges_all(self, tmp_path: Path) -> None:
    """Without prefix flag, all parameters are merged."""
    config = self._setup_two_params(tmp_path)

    copied = config.stabilize('exp-001')

    assert len(copied) == 2
    assert (tmp_path / 'prompts' / 'main.txt').read_text() == 'prompt content'
    assert (tmp_path / 'configs' / 'settings.txt').read_text() == 'config content'
