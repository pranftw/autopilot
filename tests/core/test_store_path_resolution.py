"""BUG-002/BUG-014 regression: store path resolution must not double-nest.

Validates that store_path resolves to ``workspace/.autopilot/store/`` (no project)
or ``workspace/.autopilot/projects/<project>/store/`` (with project), and that
all downstream paths chain correctly from store_path.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from pathlib import Path


class TestStorePathResolution:
  """Section 4.1: path resolution correctness."""

  def test_store_path_resolves_under_root(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    assert config.store_path == tmp_path / '.autopilot' / 'store'

  def test_store_path_with_project_scoping(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path, project='myproj')
    assert config.store_path == tmp_path / '.autopilot' / 'projects' / 'myproj' / 'store'

  def test_store_path_no_double_nesting(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    paths_to_check = [
      str(config.store_path),
      str(config.forest_file),
      str(config.refs_file),
      str(config.objects_path),
      str(config.snapshots_path),
      str(config.worktrees_path),
    ]
    for path_str in paths_to_check:
      assert '.autopilot/.autopilot' not in path_str, f'double nesting in: {path_str}'

  def test_store_path_no_double_nesting_with_project(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path, project='proj')
    paths_to_check = [
      str(config.store_path),
      str(config.forest_file),
      str(config.refs_file),
      str(config.objects_path),
    ]
    for path_str in paths_to_check:
      assert '.autopilot/.autopilot' not in path_str, f'double nesting in: {path_str}'

  def test_forest_file_resolves_under_store(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    assert config.forest_file == tmp_path / '.autopilot' / 'store' / 'forest.json'

  def test_refs_file_resolves_under_store(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    assert config.refs_file == tmp_path / '.autopilot' / 'store' / 'refs.json'

  def test_objects_path_chains_correctly(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    assert config.objects_path == config.store_path / 'objects'

  def test_snapshots_path_chains_correctly(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    assert config.snapshots_path == config.store_path / 'snapshots'

  def test_worktrees_path_chains_correctly(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    assert config.worktrees_path == config.store_path / 'worktrees'


class TestRoundTripIntegration:
  """Section 4.2: workspace init -> tree create -> reload integration."""

  def test_store_dir_created_on_bootstrap(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    config.init_workspace()
    config.store_path.mkdir(parents=True, exist_ok=True)
    assert config.store_path.is_dir()
    assert (tmp_path / '.autopilot' / 'store').is_dir()

  def test_workspace_init_then_tree_create_reload(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    config.init_workspace()
    config.store_path.mkdir(parents=True, exist_ok=True)

    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('test-tree', description='integration test')
    forest.save()

    store2 = FileStore(config)
    forest2 = FileForest(store2)
    trees = forest2.list_trees()
    assert any(t.name == 'test-tree' for t in trees)

  def test_store_path_override_via_config(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    custom_store = tmp_path / 'custom' / 'store'
    config.store_path = custom_store
    assert config.store_path == custom_store
    assert config.objects_path == custom_store / 'objects'
    assert config.forest_file == custom_store / 'forest.json'

  def test_store_path_override_preserves_descendants(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    custom_store = tmp_path / 'alt_store'
    config.store_path = custom_store
    assert config.objects_path == custom_store / 'objects'
    assert config.snapshots_path == custom_store / 'snapshots'
    assert config.forest_file == custom_store / 'forest.json'
    assert config.refs_file == custom_store / 'refs.json'

  def test_round_trip_with_project(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path, project='alpha')
    config.init_workspace()
    config.init_project()
    config.store_path.mkdir(parents=True, exist_ok=True)

    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('proj-tree', description='project scoped')
    forest.save()

    config2 = AutoPilotConfig(workspace=tmp_path, project='alpha')
    store2 = FileStore(config2)
    forest2 = FileForest(store2)
    trees = forest2.list_trees()
    assert any(t.name == 'proj-tree' for t in trees)
