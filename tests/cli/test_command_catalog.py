"""Tests for autopilot debug commands --json catalog.

Validates the machine-readable command catalog builder, JSON output,
context enforcement alignment, argument metadata, and drift guards.
"""

from autopilot.cli.commands.debug_commands import (
  JSON_CAPABLE_COMMANDS,
  CommandArgumentEntry,
  CommandCatalogEntry,
  CommandsCatalog,
  _infer_action_type,
  _serialize_default,
)
from autopilot.cli.main import AutoPilotCLI
from pathlib import Path
from tests.cli.conftest import collect_leaf_commands, run_cli_no_context
import argparse
import pytest


@pytest.fixture
def catalog() -> CommandsCatalog:
  """Build a catalog from the default CLI."""
  return CommandsCatalog.build(AutoPilotCLI())


@pytest.fixture
def catalog_dict(catalog: CommandsCatalog) -> dict:
  """Serialized catalog payload."""
  return catalog.to_dict()


def _command_by_name(catalog_dict: dict, name: str) -> dict:
  """Find a command entry by name in catalog dict.

  Args:
    catalog_dict: Serialized catalog.
    name: Command path to find.

  Returns:
    The matching command dict.

  Raises:
    AssertionError: When the command is not found.
  """
  for entry in catalog_dict['commands']:
    if entry['name'] == name:
      return entry
  msg = f'command {name!r} not found in catalog'
  raise AssertionError(msg)


# ---------------------------------------------------------------------------
# 4.1 Catalog structure and JSON output
# ---------------------------------------------------------------------------


class TestCatalogStructureAndJsonOutput:
  """Catalog structure and JSON output tests."""

  def test_catalog_cli_returns_valid_json(self, tmp_path: Path) -> None:
    """CLI dispatch returns parseable JSON envelope with ok: True."""
    result = run_cli_no_context(tmp_path, ['debug', 'commands'])
    assert result['ok'] is True
    assert 'result' in result
    assert 'commands' in result['result']
    assert isinstance(result['result']['commands'], list)
    assert result['result']['command_count'] >= 100

  def test_catalog_commands_sorted_by_name(self, catalog_dict: dict) -> None:
    """Catalog commands are sorted lexicographically by name."""
    names = [entry['name'] for entry in catalog_dict['commands']]
    assert names == sorted(names)

  def test_catalog_entry_schema(self, catalog_dict: dict) -> None:
    """Each entry has the expected top-level keys and argument schema."""
    entry = catalog_dict['commands'][0]
    assert 'name' in entry
    assert 'help' in entry
    assert 'requires_context' in entry
    assert 'supports_json' in entry
    assert 'arguments' in entry
    assert isinstance(entry['arguments'], list)
    if entry['arguments']:
      arg = entry['arguments'][0]
      assert 'name' in arg
      assert 'flags' in arg
      assert 'type' in arg
      assert 'required' in arg
      assert 'default' in arg
      assert 'help' in arg
      assert 'global' in arg


# ---------------------------------------------------------------------------
# 4.2 Known command spot-checks
# ---------------------------------------------------------------------------


class TestKnownCommandSpotChecks:
  """Spot-check specific well-known commands."""

  def test_catalog_includes_experiment_add(self, catalog_dict: dict) -> None:
    """'experiment add' is present, requires context, no JSON."""
    entry = _command_by_name(catalog_dict, 'experiment add')
    assert entry['requires_context'] is True
    assert entry['supports_json'] is False

  def test_catalog_includes_query(self, catalog_dict: dict) -> None:
    """'query' is present, no context required, supports JSON."""
    entry = _command_by_name(catalog_dict, 'query')
    assert entry['requires_context'] is False
    assert entry['supports_json'] is True

  def test_catalog_includes_store_checkout(self, catalog_dict: dict) -> None:
    """'store checkout' is present, requires context, supports JSON."""
    entry = _command_by_name(catalog_dict, 'store checkout')
    assert entry['requires_context'] is True
    assert entry['supports_json'] is True


# ---------------------------------------------------------------------------
# 4.3 Context enforcement alignment
# ---------------------------------------------------------------------------


class TestContextEnforcementAlignment:
  """Validate catalog context flags match live CLI enforcement."""

  def test_catalog_requires_context_matches_cli(self, catalog_dict: dict) -> None:
    """Every catalog entry's requires_context matches CLI.requires_context."""
    cli = AutoPilotCLI()
    for entry in catalog_dict['commands']:
      expected = cli.requires_context(entry['name'])
      assert entry['requires_context'] == expected, (
        f'{entry["name"]}: catalog={entry["requires_context"]}, cli={expected}'
      )

  def test_catalog_read_only_commands_exempt(self, catalog_dict: dict) -> None:
    """Sample of read-only commands all have requires_context=False."""
    sample = [
      'debug commands',
      'query',
      'status',
      'experiment show',
      'store doctor',
      'tree list',
      'workspace status',
      'policy check',
      'recommend',
      'report summary',
    ]
    for name in sample:
      entry = _command_by_name(catalog_dict, name)
      assert entry['requires_context'] is False, f'{name} should be context-exempt'

  def test_debug_commands_is_context_exempt(self, tmp_path: Path) -> None:
    """'debug commands' does not require --context and exits 0."""
    cli = AutoPilotCLI()
    assert cli.requires_context('debug commands') is False
    result = run_cli_no_context(tmp_path, ['debug', 'commands'])
    assert result['ok'] is True


# ---------------------------------------------------------------------------
# 4.4 Argument metadata
# ---------------------------------------------------------------------------


class TestArgumentMetadata:
  """Validate argument extraction and metadata."""

  def test_experiment_add_has_spec_version_argument(self, catalog_dict: dict) -> None:
    """'experiment add' includes --spec-version argument."""
    entry = _command_by_name(catalog_dict, 'experiment add')
    spec_args = [a for a in entry['arguments'] if a['name'] == 'spec_version']
    assert len(spec_args) == 1
    arg = spec_args[0]
    assert '--spec-version' in arg['flags']
    assert arg['type'] == 'str'

  def test_query_has_global_json_flag(self, catalog_dict: dict) -> None:
    """'query' has use_json argument marked global."""
    entry = _command_by_name(catalog_dict, 'query')
    json_args = [a for a in entry['arguments'] if a['name'] == 'use_json']
    assert len(json_args) == 1
    assert json_args[0]['global'] is True

  def test_global_flags_marked_global(self, catalog_dict: dict) -> None:
    """Every catalog entry includes project, use_json, context as global."""
    global_names = {'project', 'use_json', 'context'}
    for entry in catalog_dict['commands']:
      arg_names = {a['name'] for a in entry['arguments']}
      for gname in global_names:
        assert gname in arg_names, f'{entry["name"]} missing global flag {gname}'
      for arg in entry['arguments']:
        if arg['name'] in global_names:
          assert arg['global'] is True, f'{entry["name"]}.{arg["name"]} should be global=True'


# ---------------------------------------------------------------------------
# 4.5 Drift guards
# ---------------------------------------------------------------------------


class TestDriftGuards:
  """Ensure catalog stays aligned with parser and test oracles."""

  def test_catalog_covers_all_parser_leaves(self, catalog_dict: dict) -> None:
    """Catalog command names match the full set from parser walk."""
    cli = AutoPilotCLI()
    parser = cli.build_parser()
    expected = set(collect_leaf_commands(parser))
    actual = {entry['name'] for entry in catalog_dict['commands']}
    assert actual == expected

  def test_json_capable_commands_aligned_with_test_oracle(self) -> None:
    """JSON_CAPABLE_COMMANDS equals EXPECTED_JSON_COMMANDS from test oracle."""
    from tests.cli.test_json_contract_matrix import EXPECTED_JSON_COMMANDS

    assert JSON_CAPABLE_COMMANDS == EXPECTED_JSON_COMMANDS


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
  """Unit tests for _infer_action_type and _serialize_default."""

  def test_serialize_default_suppress_is_none(self) -> None:
    """argparse.SUPPRESS serializes as None."""
    assert _serialize_default(argparse.SUPPRESS) is None

  def test_serialize_default_none_stays_none(self) -> None:
    """None stays None."""
    assert _serialize_default(None) is None

  def test_serialize_default_preserves_primitives(self) -> None:
    """Strings, ints, floats, bools pass through."""
    assert _serialize_default('hello') == 'hello'
    assert _serialize_default(42) == 42
    assert _serialize_default(2.5) == 2.5
    verbose_default = True
    assert _serialize_default(verbose_default) is True

  def test_serialize_default_list(self) -> None:
    """Lists pass through."""
    assert _serialize_default([1, 2]) == [1, 2]

  def test_serialize_default_other_types_str(self) -> None:
    """Non-primitive types become their str representation."""
    from pathlib import Path

    result = _serialize_default(Path('/tmp'))
    assert result == '/tmp'

  def test_infer_action_type_store_true_is_flag(self) -> None:
    """store_true action infers as 'flag'."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    action = next(a for a in parser._actions if a.dest == 'verbose')
    assert _infer_action_type(action) == 'flag'

  def test_infer_action_type_int(self) -> None:
    """int-typed argument infers as 'int'."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int)
    action = next(a for a in parser._actions if a.dest == 'count')
    assert _infer_action_type(action) == 'int'

  def test_infer_action_type_float(self) -> None:
    """float-typed argument infers as 'float'."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate', type=float)
    action = next(a for a in parser._actions if a.dest == 'rate')
    assert _infer_action_type(action) == 'float'

  def test_infer_action_type_positional(self) -> None:
    """Positional argument infers as 'positional'."""
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    action = next(a for a in parser._actions if a.dest == 'name')
    assert _infer_action_type(action) == 'positional'

  def test_infer_action_type_append(self) -> None:
    """append action infers as 'append'."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--items', action='append')
    action = next(a for a in parser._actions if a.dest == 'items')
    assert _infer_action_type(action) == 'append'

  def test_infer_action_type_str_default(self) -> None:
    """String-typed option infers as 'str'."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    action = next(a for a in parser._actions if a.dest == 'name')
    assert _infer_action_type(action) == 'str'


# ---------------------------------------------------------------------------
# Dataclass round-trip tests
# ---------------------------------------------------------------------------


class TestDataclassRoundTrips:
  """Verify to_dict serialization of catalog dataclasses."""

  def test_command_argument_entry_to_dict(self) -> None:
    """CommandArgumentEntry serializes with 'global' key."""
    entry = CommandArgumentEntry(
      name='verbose',
      flags=['--verbose', '-v'],
      type='flag',
      required=False,
      default=False,
      help='enable verbosity',
      global_=True,
    )
    d = entry.to_dict()
    assert d['name'] == 'verbose'
    assert d['flags'] == ['--verbose', '-v']
    assert d['type'] == 'flag'
    assert d['required'] is False
    assert d['default'] is False
    assert d['help'] == 'enable verbosity'
    assert d['global'] is True
    assert 'global_' not in d

  def test_command_catalog_entry_to_dict(self) -> None:
    """CommandCatalogEntry serializes with nested arguments."""
    arg = CommandArgumentEntry(
      name='id',
      flags=[],
      type='positional',
      required=True,
      default=None,
      help='experiment id',
      global_=False,
    )
    entry = CommandCatalogEntry(
      name='experiment add',
      help='Add an experiment',
      requires_context=True,
      supports_json=False,
      arguments=[arg],
    )
    d = entry.to_dict()
    assert d['name'] == 'experiment add'
    assert d['requires_context'] is True
    assert d['supports_json'] is False
    assert len(d['arguments']) == 1
    assert d['arguments'][0]['name'] == 'id'

  def test_commands_catalog_to_dict(self) -> None:
    """CommandsCatalog.to_dict includes commands and command_count."""
    entry = CommandCatalogEntry(
      name='query',
      help='Run queries',
      requires_context=False,
      supports_json=True,
      arguments=[],
    )
    catalog = CommandsCatalog([entry])
    d = catalog.to_dict()
    assert d['command_count'] == 1
    assert len(d['commands']) == 1
    assert d['commands'][0]['name'] == 'query'


# ---------------------------------------------------------------------------
# Build test
# ---------------------------------------------------------------------------


class TestBuild:
  """Test CommandsCatalog.build()."""

  def test_commands_catalog_build_returns_all_leaves(self) -> None:
    """build() returns the same count as collect_leaf_commands."""
    cli = AutoPilotCLI()
    parser = cli.build_parser()
    expected_count = len(collect_leaf_commands(parser))
    catalog = CommandsCatalog.build(cli)
    assert catalog.to_dict()['command_count'] == expected_count

  def test_build_with_explicit_cli(self) -> None:
    """build() with explicit CLI instance produces valid catalog."""
    catalog = CommandsCatalog.build(AutoPilotCLI())
    assert catalog.to_dict()['command_count'] > 100
