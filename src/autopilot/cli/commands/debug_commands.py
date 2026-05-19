"""Machine-readable CLI command catalog builder.

Introspects the live argparse parser tree and produces structured metadata
about every registered leaf command: name, help text, context requirement,
JSON support, and argument details. Agents consume this via
``autopilot debug commands --json`` to discover the command surface without
parsing ``--help`` text.

Output schema (via ``CommandsCatalog.to_dict()``)::

    {
      "commands": [
        {
          "name": "experiment add",
          "help": "...",
          "requires_context": true,
          "supports_json": false,
          "arguments": [
            {"name": "...", "flags": [...], "type": "...", ...}
          ]
        },
        ...
      ],
      "command_count": 114
    }

Maintenance: when a new command gains ``--json`` support, add it to
``JSON_CAPABLE_COMMANDS`` below. The drift-guard test
``test_json_capable_commands_aligned_with_test_oracle`` ensures this set
stays aligned with the test oracle in ``test_json_contract_matrix.py``.
"""

from autopilot.cli.command import CLI
from dataclasses import dataclass, field
from typing import Any
import argparse

GLOBAL_FLAG_DESTS = frozenset(
  {
    'project',
    'workspace',
    'experiment',
    'dataset',
    'split',
    'epoch',
    'hyperparams',
    'dry_run',
    'verbose',
    'no_color',
    'use_json',
    'expose',
    'context',
    'wait',
    'retry',
  }
)

JSON_CAPABLE_COMMANDS = frozenset(
  {
    'ai generate run',
    'ai generate resume',
    'ai generate dry-run',
    'ai judge run',
    'ai judge resume',
    'ai judge summarize',
    'ai judge distribution',
    'dataset list',
    'dataset show',
    'dataset split',
    'debug commands',
    'debug profiler',
    'debug trend',
    'debug store reflog',
    'diagnose run',
    'diagnose heatmap',
    'execute',
    'experiment cancel',
    'experiment compare',
    'experiment deploy',
    'experiment deploy-log',
    'experiment undeploy',
    'experiment complete',
    'experiment fail',
    'experiment impact',
    'experiment lineage',
    'experiment timeline',
    'experiment invalidate',
    'experiment list',
    'experiment metadata get',
    'experiment metadata set',
    'experiment metadata show',
    'experiment notes show',
    'experiment notes write',
    'experiment show',
    'experiment status',
    'optimize preflight',
    'policy check',
    'policy explain',
    'project doctor',
    'project list',
    'propose create',
    'propose list',
    'propose revert',
    'propose verify',
    'query',
    'recommend',
    'report compare',
    'report narrative',
    'report summary',
    'report trend',
    'status',
    'store branch',
    'store checkout',
    'store copy-epoch',
    'store doctor',
    'store diff',
    'store log',
    'store merge',
    'store merge-analysis',
    'store merge-apply',
    'store merge-preview',
    'store merge-resolve',
    'store recover',
    'store reflog expire',
    'store reflog list',
    'store snapshot',
    'store stash',
    'store stash-list',
    'store stash-pop',
    'store status',
    'store tag create',
    'store tag list',
    'store tag verify',
    'store worktree list',
    'trace collect',
    'trace inspect',
    'trace verify',
    'track',
    'tree create',
    'tree describe',
    'tree list',
    'tree remove',
    'tree show',
    'undo-guide',
    'workspace doctor',
    'workspace journal',
    'workspace status',
    'workspace tree',
  }
)


@dataclass
class CommandArgumentEntry:
  """Single argparse action serialized for agents.

  Attributes:
    name: Argument dest name.
    flags: Option strings (empty for positionals).
    type: Inferred type label.
    required: Whether the argument is required.
    default: Serialized default value.
    help: Help text or None.
    global_: True when this is a global flag present on every command.
  """

  name: str
  flags: list[str]
  type: str
  required: bool
  default: Any
  help: str | None
  global_: bool = field(metadata={'alias': 'global'})

  def to_dict(self) -> dict[str, Any]:
    """Serialize to JSON-safe dict.

    Returns:
      Dict with ``global`` key (not ``global_``).
    """
    return {
      'name': self.name,
      'flags': self.flags,
      'type': self.type,
      'required': self.required,
      'default': self.default,
      'help': self.help,
      'global': self.global_,
    }


@dataclass
class CommandCatalogEntry:
  """One leaf CLI command.

  Attributes:
    name: Space-joined command path (e.g. ``'experiment add'``).
    help: Command help text or None.
    requires_context: Whether ``--context`` is required.
    supports_json: Whether the handler emits structured JSON.
    arguments: List of argument entries for this command.
  """

  name: str
  help: str | None
  requires_context: bool
  supports_json: bool
  arguments: list[CommandArgumentEntry]

  def to_dict(self) -> dict[str, Any]:
    """Serialize to JSON-safe dict.

    Returns:
      Dict representation of this catalog entry.
    """
    return {
      'name': self.name,
      'help': self.help,
      'requires_context': self.requires_context,
      'supports_json': self.supports_json,
      'arguments': [arg.to_dict() for arg in self.arguments],
    }


class CommandsCatalog:
  """Build a machine-readable catalog from a live CLI instance.

  Attributes:
    commands: Sorted list of catalog entries.
  """

  def __init__(self, commands: list[CommandCatalogEntry]) -> None:
    """Store catalog entries.

    Args:
      commands: Pre-built catalog entries.
    """
    self.commands = commands

  @classmethod
  def build(cls, cli: CLI, parser: argparse.ArgumentParser | None = None) -> 'CommandsCatalog':
    """Walk the CLI parser tree and build catalog entries.

    Args:
      cli: CLI instance for context enforcement lookups.
      parser: Pre-built root parser. Built from ``cli`` if not provided.

    Returns:
      Populated ``CommandsCatalog`` with entries sorted by ``name`` lexicographically.
    """
    if parser is None:
      parser = cli.build_parser()
    leaves = _walk_parser_leaves(parser)
    entries: list[CommandCatalogEntry] = []
    for command_path, leaf_parser in leaves:
      requires_context = cli.requires_context(command_path)
      supports_json = command_path in JSON_CAPABLE_COMMANDS
      help_text = leaf_parser.description or _extract_parser_help(leaf_parser)
      arguments = _extract_arguments(leaf_parser)
      entries.append(
        CommandCatalogEntry(
          name=command_path,
          help=help_text,
          requires_context=requires_context,
          supports_json=supports_json,
          arguments=arguments,
        )
      )
    entries.sort(key=lambda e: e.name)
    return cls(entries)

  def to_dict(self) -> dict[str, Any]:
    """Return catalog payload for ``ctx.output.result()``.

    Returns:
      Dict with ``commands`` list and ``command_count``.
    """
    return {
      'commands': [entry.to_dict() for entry in self.commands],
      'command_count': len(self.commands),
    }


_FLAG_ACTION_CLASSES = (
  argparse._StoreTrueAction,
  argparse._StoreFalseAction,
  argparse._CountAction,
)


def _infer_action_type(action: argparse.Action) -> str:
  """Map argparse action to a simple type label.

  Args:
    action: An argparse action instance.

  Returns:
    One of ``'flag'``, ``'int'``, ``'float'``, ``'str'``, ``'append'``,
    ``'positional'``.
  """
  if isinstance(action, _FLAG_ACTION_CLASSES):
    return 'flag'
  if isinstance(action, argparse._AppendAction):
    return 'append'
  if not action.option_strings:
    return 'positional'
  if action.type is int:
    return 'int'
  if action.type is float:
    return 'float'
  return 'str'


def _serialize_default(default: Any) -> Any:
  """Convert argparse defaults to JSON-safe values.

  Args:
    default: Raw default from an argparse action.

  Returns:
    JSON-serializable value. ``argparse.SUPPRESS`` becomes ``None``.
  """
  if default is argparse.SUPPRESS:
    return None
  if default is None:
    return None
  if isinstance(default, (str, int, float, bool)):
    return default
  if isinstance(default, list):
    return default
  return str(default)


def _walk_parser_leaves(
  parser: argparse.ArgumentParser,
  path: tuple[str, ...] = (),
) -> list[tuple[str, argparse.ArgumentParser]]:
  """Yield (command_path, leaf_parser) pairs.

  Recursively walks the parser tree via ``_SubParsersAction`` choices.
  Leaf parsers are those with no subparsers action.

  Args:
    parser: Root or sub-parser to walk.
    path: Accumulated command path tokens.

  Returns:
    List of (space-joined command path, leaf parser) tuples.
  """
  leaves: list[tuple[str, argparse.ArgumentParser]] = []
  has_subparsers = False
  for action in parser._actions:
    if isinstance(action, argparse._SubParsersAction):
      has_subparsers = True
      for name, sub in action.choices.items():
        leaves.extend(_walk_parser_leaves(sub, (*path, name)))
  if not has_subparsers and path:
    leaves.append((' '.join(path), parser))
  return leaves


def _extract_arguments(parser: argparse.ArgumentParser) -> list[CommandArgumentEntry]:
  """Extract non-help, non-subparser actions from a leaf parser.

  Args:
    parser: Leaf argument parser.

  Returns:
    List of ``CommandArgumentEntry`` for the parser's actions.
  """
  entries: list[CommandArgumentEntry] = []
  for action in parser._actions:
    if isinstance(action, argparse._HelpAction):
      continue
    if isinstance(action, argparse._SubParsersAction):
      continue
    name = action.dest
    flags = list(action.option_strings)
    action_type = _infer_action_type(action)
    required = action.required if action.option_strings else True
    default = _serialize_default(action.default)
    help_text = action.help
    is_global = name in GLOBAL_FLAG_DESTS
    entries.append(
      CommandArgumentEntry(
        name=name,
        flags=flags,
        type=action_type,
        required=required,
        default=default,
        help=help_text,
        global_=is_global,
      )
    )
  return entries


def _extract_parser_help(parser: argparse.ArgumentParser) -> str | None:
  """Extract the help string from a leaf parser.

  Args:
    parser: ArgumentParser to extract help from.

  Returns:
    The parser's help text if available, else None.
  """
  return getattr(parser, 'help', None)
