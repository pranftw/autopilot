from autopilot.cli.command import CLI, Command
from unittest.mock import MagicMock, patch


def test_cli_init_has_datamodule_none():
  cli = CLI()
  assert cli.datamodule is None


def test_cli_datamodule_set():
  cli = CLI()
  dm = MagicMock()
  cli.datamodule = dm
  assert cli.datamodule is dm


def test_cli_datamodule_not_registered_as_command():
  cli = CLI()
  cli.datamodule = MagicMock()
  assert 'datamodule' not in cli._commands


def test_run_direct_passes_datamodule_to_context():
  cli = CLI()
  dm = MagicMock()
  cli.datamodule = dm

  dummy_cmd = Command()
  dummy_cmd.name = 'test'
  cli.test = dummy_cmd

  captured_ctx = {}

  def fake_dispatch(ctx, args, **kwargs):
    captured_ctx['ctx'] = ctx

  with (
    patch.object(cli, 'dispatch', side_effect=fake_dispatch),
  ):
    cli.run_direct(argv=['test'])

  assert captured_ctx['ctx'].datamodule is dm
