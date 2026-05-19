"""Shared pytest fixtures for integration tests.

Provides the standard workspace + params seed file + AutoPilotConfig +
PathParameter + FileStore + FileForest tuple so tests stop reimplementing
_setup / _setup_workspace.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from tests.integration.doubles import minimal_trainer_stack
import pytest


@pytest.fixture
def integration_workspace_with_store(
  tmp_path,
) -> tuple[AutoPilotConfig, PathParameter, FileStore, FileForest]:
  """Build standard workspace + config + store + forest for integration tests.

  Creates:
    tmp_path/workspace/
    tmp_path/workspace/params/
    tmp_path/workspace/params/seed.txt  (satisfies pattern='*.txt')

  Returns:
    (config, path_param, store, forest) tuple. Tests create trees in the
    forest as needed -- tree creation stays in test bodies because tree
    slugs and descriptions differ per file.
  """
  workspace = tmp_path / 'workspace'
  workspace.mkdir()
  params_dir = workspace / 'params'
  params_dir.mkdir()
  (params_dir / 'seed.txt').write_text('seed', encoding='utf-8')

  config = AutoPilotConfig(workspace=workspace)
  path_param = PathParameter(source=str(params_dir), pattern='*.txt')
  store = FileStore(config)
  store.register_parameters({'source': path_param})
  forest = FileForest(store=store)
  return config, path_param, store, forest


@pytest.fixture
def minimal_trainer_stack_fixture(
  integration_workspace_with_store,
  request,
) -> tuple:
  """Build minimal module + datamodule from the workspace fixture.

  Supports parametrize via ``@pytest.mark.parametrize('minimal_trainer_stack_fixture',
  [0.91], indirect=True)`` for custom accuracy values.
  """
  _, path_param, _, _ = integration_workspace_with_store
  accuracy = getattr(request, 'param', 0.5)
  return minimal_trainer_stack(path_param, accuracy=accuracy)
