"""Project-specific Config subclass.

Pass ``environment=...`` into the constructor for non-local execution
isolation (e.g. ``IsolatedEnvironment`` from ``autopilot.ai.environment``
when worktree-based isolation is needed).

Config is the single place for path computation and environment injection.
All path resolution flows through Config.

Example (opt-in worktree isolation)::

    from autopilot.ai.environment import IsolatedEnvironment

    cfg = {name}Config(...)
    env = IsolatedEnvironment(
      cfg,
      ignore_patterns=(
        '.autopilot/',
        '__pycache__/',
        '.pytest_cache/',
        '*.pyc',
        '.git/',
        '.ruff_cache/',
        '*.egg-info/',
      ),
      symlink_as_unit=('.venv', 'node_modules'),
      core_files=('pyproject.toml', 'README.md'),
    )
    cfg.environment = env

Tailor ignore_patterns, symlink_as_unit, and core_files to your project
layout. The framework supplies no defaults -- all three are required.
"""

from autopilot.core.config import AutoPilotConfig


class {name}Config(AutoPilotConfig):
  """Config for the {name} project.

  Uses ``LocalEnvironment`` by default (no isolation). Pass a custom
  ``Environment`` subclass to the constructor for worktree isolation
  or other execution strategies.
  """
