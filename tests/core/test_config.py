"""Tests for FilePath descriptor, Config base, and AutoPilotConfig."""

from autopilot.core.config import AutoPilotConfig, Config, FilePath
from autopilot.core.environment import Environment, LocalEnvironment
from autopilot.core.errors import ConfigError
from pathlib import Path
import pytest


class TestFilePath:
  def test_static_path_returns_path(self):
    class C(Config):
      autopilot_path = FilePath('workspace', '.autopilot')

    c = C(workspace=Path('/ws'))
    result = c.autopilot_path
    assert isinstance(result, Path)
    assert result == Path('/ws/.autopilot')

  def test_parameterized_path_returns_callable(self):
    class C(Config):
      autopilot_path = FilePath('workspace', '.autopilot')
      experiments_path = FilePath('autopilot_path', 'experiments')
      experiment_path = FilePath('experiments_path', '{slug}')

    c = C(workspace=Path('/ws'))
    result = c.experiment_path
    assert callable(result)
    resolved = result(slug='exp-001')
    assert isinstance(resolved, Path)
    assert resolved == Path('/ws/.autopilot/experiments/exp-001')

  def test_descriptor_on_class_returns_descriptor(self):
    class C(Config):
      autopilot_path = FilePath('workspace', '.autopilot')

    assert isinstance(C.__dict__['autopilot_path'], FilePath)
    assert isinstance(C.autopilot_path, FilePath)

  def test_auto_registration_in_file_paths(self):
    class C(Config):
      a_path = FilePath('workspace', 'a')
      b_path = FilePath('workspace', 'b')

    assert 'a_path' in C.file_paths
    assert 'b_path' in C.file_paths

  def test_override_via_set(self):
    class C(Config):
      autopilot_path = FilePath('workspace', '.autopilot')

    c = C(workspace=Path('/ws'))
    assert c.autopilot_path == Path('/ws/.autopilot')
    c.autopilot_path = Path('/custom')
    assert c.autopilot_path == Path('/custom')

  def test_cascading_after_override(self):
    class C(Config):
      autopilot_path = FilePath('workspace', '.autopilot')
      projects_path = FilePath('autopilot_path', 'projects')

    c = C(workspace=Path('/ws'))
    assert c.projects_path == Path('/ws/.autopilot/projects')
    c.autopilot_path = Path('/new')
    assert c.projects_path == Path('/new/projects')

  def test_parameterized_with_multiple_fields(self):
    class C(Config):
      autopilot_path = FilePath('workspace', '.autopilot')
      data_path = FilePath('autopilot_path', '{kind}_{version}')

    c = C(workspace=Path('/ws'))
    result = c.data_path(kind='train', version='v2')
    assert result == Path('/ws/.autopilot/train_v2')

  def test_static_child_of_parameterized_parent(self):
    class C(Config):
      autopilot_path = FilePath('workspace', '.autopilot')
      experiments_path = FilePath('autopilot_path', 'experiments')
      experiment_path = FilePath('experiments_path', '{slug}')
      config_file = FilePath('experiment_path', 'config.json')

    c = C(workspace=Path('/ws'))
    result = c.config_file
    assert callable(result)
    resolved = result(slug='exp-001')
    assert resolved == Path('/ws/.autopilot/experiments/exp-001/config.json')


class TestFilePathMerge:
  def test_subclass_inherits_parent_file_paths(self):
    class Base(Config):
      base_path = FilePath('workspace', 'base')

    class Child(Base):
      child_path = FilePath('workspace', 'child')

    assert 'base_path' in Child.file_paths
    assert 'child_path' in Child.file_paths

  def test_subclass_can_override_parent_file_path(self):
    class Base(Config):
      data_path = FilePath('workspace', 'data')

    class Child(Base):
      data_path = FilePath('workspace', 'custom_data')

    c = Child(workspace=Path('/ws'))
    assert c.data_path == Path('/ws/custom_data')

  def test_init_subclass_merges_across_mro(self):
    class A(Config):
      a_path = FilePath('workspace', 'a')

    class B(A):
      b_path = FilePath('workspace', 'b')

    class C(B):
      c_path = FilePath('workspace', 'c')

    assert 'a_path' in C.file_paths
    assert 'b_path' in C.file_paths
    assert 'c_path' in C.file_paths


class TestConfig:
  def test_workspace_and_project(self):
    c = AutoPilotConfig(workspace=Path('/ws'), project='myproj')
    assert c.workspace == Path('/ws')
    assert c.project == 'myproj'

  def test_root_with_project(self):
    c = AutoPilotConfig(workspace=Path('/ws'), project='myproj')
    assert c.root == c.projects_path / 'myproj'

  def test_root_without_project(self):
    c = AutoPilotConfig(workspace=Path('/ws'))
    assert c.root == c.autopilot_path

  def test_root_setter_override(self):
    c = AutoPilotConfig(workspace=Path('/ws'))
    c.root = Path('/custom/root')
    assert c.root == Path('/custom/root')

  def test_root_setter_overrides_project_logic(self):
    c = AutoPilotConfig(workspace=Path('/ws'), project='proj')
    c.root = Path('/override')
    assert c.root == Path('/override')


class TestConfigEnvironment:
  def test_config_default_environment_is_local(self) -> None:
    c = Config(workspace=Path('/ws'))
    assert isinstance(c.environment, LocalEnvironment)

  def test_autopilot_config_default_environment_is_local(self, tmp_path: Path) -> None:
    c = AutoPilotConfig(workspace=tmp_path)
    assert isinstance(c.environment, LocalEnvironment)

  def test_config_accepts_custom_environment(self, tmp_path: Path) -> None:
    class _StubEnv(Environment):
      def setup(self, experiment, store, module):
        return tmp_path

    env = _StubEnv()
    c = AutoPilotConfig(workspace=tmp_path, environment=env)
    assert c.environment is env


class TestConfigDeletedApi:
  def test_config_has_no_environment_hooks(self) -> None:
    assert not hasattr(Config, 'has_environment_hooks')
    assert not hasattr(Config, 'setup_environment')
    assert not hasattr(Config, 'teardown_environment')

  def test_autopilot_config_has_no_ignore_patterns(self) -> None:
    assert not hasattr(AutoPilotConfig, 'ignore_patterns')

  def test_autopilot_config_has_no_symlink_as_unit(self) -> None:
    assert not hasattr(AutoPilotConfig, 'symlink_as_unit')

  def test_autopilot_config_has_no_core_files(self) -> None:
    assert not hasattr(AutoPilotConfig, 'core_files')

  def test_autopilot_config_has_no_manifest_file(self) -> None:
    assert not hasattr(AutoPilotConfig, 'manifest_file')

  def test_autopilot_config_has_no_events_file(self) -> None:
    assert not hasattr(AutoPilotConfig, 'events_file')


class TestConfigStubs:
  def test_config_stabilize_raises(self):
    c = Config(workspace=Path('/ws'))
    with pytest.raises(NotImplementedError):
      c.stabilize('exp-1')

  def test_config_init_workspace_raises(self):
    c = Config(workspace=Path('/ws'))
    with pytest.raises(NotImplementedError):
      c.init_workspace()

  def test_config_init_project_raises(self):
    c = Config(workspace=Path('/ws'))
    with pytest.raises(NotImplementedError):
      c.init_project()

  def test_autopilot_stabilize_returns_empty_for_missing(self, tmp_path: Path):
    c = AutoPilotConfig(workspace=tmp_path)
    assert c.stabilize('exp-1') == []

  def test_stabilize_signature_returns_list_path(self):
    import inspect

    sig = inspect.signature(AutoPilotConfig.stabilize)
    assert sig.return_annotation == list[Path]


class TestAutoPilotConfig:
  def test_all_static_paths_resolve(self):
    c = AutoPilotConfig(workspace=Path('/ws'))
    assert c.autopilot_path == Path('/ws/.autopilot')
    assert c.projects_path == Path('/ws/.autopilot/projects')
    assert c.templates_path == Path('/ws/templates')

  def test_paths_with_project(self):
    c = AutoPilotConfig(workspace=Path('/ws'), project='proj')
    assert c.root == c.projects_path / 'proj'
    assert c.experiments_path == c.root / 'experiments'
    assert c.datasets_path == c.root / 'datasets'
    assert c.records_path == c.root / 'records'
    assert c.cli_file == c.root / 'cli.py'
    assert c.store_path == c.root / 'store'
    assert c.objects_path == c.store_path / 'objects'
    assert c.snapshots_path == c.store_path / 'snapshots'
    assert c.worktrees_path == c.store_path / 'worktrees'
    assert c.forest_file == c.store_path / 'forest.json'
    assert c.refs_file == c.store_path / 'refs.json'
    assert c.store_experiments_path == c.store_path / 'experiments'

  def test_parameterized_experiment_path(self):
    c = AutoPilotConfig(workspace=Path('/ws'), project='proj')
    result = c.experiment_path(slug='exp-001')
    assert result == c.experiments_path / 'exp-001'

  def test_parameterized_epoch_path(self):
    c = AutoPilotConfig(workspace=Path('/ws'), project='proj')
    result = c.epoch_path(slug='exp-001', epoch=3)
    assert 'exp-001' in str(result)
    assert 'epoch_3' in str(result)

  def test_parameterized_store_experiment_path(self):
    c = AutoPilotConfig(workspace=Path('/ws'), project='proj')
    result = c.store_experiment_path(experiment_id='abc')
    assert result == c.store_experiments_path / 'abc'

  def test_parameterized_store_epoch_path(self):
    c = AutoPilotConfig(workspace=Path('/ws'), project='proj')
    result = c.store_epoch_path(experiment_id='abc', epoch=2)
    assert 'abc' in str(result)
    assert 'epoch_2' in str(result)

  def test_parameterized_result_file(self):
    c = AutoPilotConfig(workspace=Path('/ws'), project='proj')
    result = c.result_file(experiment_id='abc', epoch=0)
    assert str(result).endswith('result.json')

  def test_store_path_override_cascades(self):
    c = AutoPilotConfig(workspace=Path('/ws'), project='proj')
    c.store_path = Path('/custom/store')
    assert c.objects_path == Path('/custom/store/objects')
    assert c.snapshots_path == Path('/custom/store/snapshots')
    assert c.forest_file == Path('/custom/store/forest.json')


class TestAutoPilotConfigInit:
  def test_init_workspace(self, tmp_path):
    c = AutoPilotConfig(workspace=tmp_path)
    c.init_workspace()
    assert c.autopilot_path.exists()
    assert c.projects_path.exists()

  def test_init_workspace_idempotent(self, tmp_path):
    c = AutoPilotConfig(workspace=tmp_path)
    c.init_workspace()
    c.init_workspace()
    assert c.autopilot_path.exists()

  def test_init_project(self, tmp_path):
    c = AutoPilotConfig(workspace=tmp_path, project='myproj')
    c.init_workspace()
    c.init_project()
    assert (c.projects_path / 'myproj').exists()

  def test_init_project_no_project_raises(self, tmp_path):
    c = AutoPilotConfig(workspace=tmp_path)
    with pytest.raises(ConfigError, match='no project'):
      c.init_project()

  def test_init_project_idempotent(self, tmp_path):
    c = AutoPilotConfig(workspace=tmp_path, project='proj')
    c.init_workspace()
    c.init_project()
    c.init_project()
    assert (c.projects_path / 'proj').exists()


class TestSubclassFilePaths:
  def test_custom_subclass_adds_file_paths(self):
    class Custom(AutoPilotConfig):
      custom_path = FilePath('workspace', 'custom')

    c = Custom(workspace=Path('/ws'))
    assert c.custom_path == Path('/ws/custom')
    assert 'custom_path' in Custom.file_paths
    assert 'autopilot_path' in Custom.file_paths
