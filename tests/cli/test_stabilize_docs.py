"""Tests stabilize command hazard documentation (BUG-045 / Plan 10 backlog)."""

from autopilot.cli.commands.stabilize import StabilizeCommand
import autopilot.cli.commands.stabilize as stabilize_mod


def test_stabilize_help_documents_cross_project_overwrite_hazard() -> None:
  """Module or class docstring notes cross-workspace overwrite / prefix hazard."""
  mod_doc = stabilize_mod.__doc__ or ''
  cls_doc = StabilizeCommand.__doc__ or ''
  blob = f'{mod_doc}\n{cls_doc}'.lower()
  assert 'overwrite' in blob or 'cross-project' in blob or 'parameter-prefix' in blob


def test_stabilize_module_docstring_notes_shared_parameters_destination() -> None:
  """Module doc mentions original_path or overwrite for shared destinations."""
  mod_doc = stabilize_mod.__doc__ or ''
  assert 'original_path' in mod_doc or 'overwrite' in mod_doc
