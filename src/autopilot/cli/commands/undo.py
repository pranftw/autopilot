"""Undo guide command -- read-only reversal suggestions.

Safety: this command **never executes** undo operations. It inspects the
latest mutating execution record and suggests a reversal CLI string with a
confidence level. Agents and humans decide whether to act on the suggestion.

Suggestions are best-effort: some mutations have no safe automatic reverse
(e.g. terminal status transitions), and the confidence field communicates
this.
"""

from autopilot.cli.command import CLI, Command
from autopilot.cli.context import CLIContext
from autopilot.core.errors import TrackingError
from autopilot.tracking.executions import ExecutionRecord, load_executions
from autopilot.tracking.io import read_jsonl
from dataclasses import dataclass
from typing import Any, Literal
import argparse


@dataclass(frozen=True)
class UndoSuggestion:
  """Structured reversal suggestion for a mutating CLI command.

  Attributes:
    suggested_undo: CLI string to reverse the operation, or None when no
      safe reversal exists.
    confidence: How reliable the suggestion is.
    notes: Optional explanation or caveats.
  """

  suggested_undo: str | None
  confidence: Literal['high', 'medium', 'low']
  notes: str | None


def _recipe_experiment_deploy(record: ExecutionRecord) -> UndoSuggestion:
  """Suggest undeploy for experiment deploy.

  Args:
    record: Execution record with deploy args.

  Returns:
    Suggestion with label-based undeploy command.
  """
  args = record.args
  label = None
  for i, arg in enumerate(args):
    if arg == '--as' and i + 1 < len(args):
      label = args[i + 1]
      break
    if arg.startswith('--as='):
      label = arg[len('--as=') :]
      break
  if label is not None:
    return UndoSuggestion(
      suggested_undo=f'experiment undeploy {label}',
      confidence='high',
      notes=None,
    )
  return UndoSuggestion(
    suggested_undo=None,
    confidence='low',
    notes='could not extract deployment label from args',
  )


def _recipe_tree_switch(
  record: ExecutionRecord,
  all_records: list[ExecutionRecord],
  cli: CLI,
) -> UndoSuggestion:
  """Suggest switch back to prior tree.

  Algorithm: find the two most recent tree switch mutations in execution
  history. The prior switch's target is the tree to switch back to.

  Args:
    record: The latest tree switch record (used for consistent interface).
    all_records: Full execution history for prior-switch lookup.
    cli: CLI instance for requires_context checks.

  Returns:
    Suggestion to switch back to the previously active tree.
  """
  _ = record
  switch_records: list[ExecutionRecord] = []
  for rec in reversed(all_records):
    if rec.command == 'tree switch' and cli.requires_context(rec.command):
      switch_records.append(rec)
    if len(switch_records) >= 2:
      break

  if len(switch_records) < 2:
    return UndoSuggestion(
      suggested_undo=None,
      confidence='low',
      notes='need prior tree switch in execution log to infer previous tree',
    )

  prior_rec = switch_records[1]
  prior_target = _extract_switch_target(prior_rec.args)
  if prior_target is None:
    return UndoSuggestion(
      suggested_undo=None,
      confidence='low',
      notes='could not extract target tree from prior switch record',
    )
  return UndoSuggestion(
    suggested_undo=f'tree switch {prior_target}',
    confidence='medium',
    notes=None,
  )


def _extract_switch_target(args: list[str]) -> str | None:
  """Extract the positional tree name from tree switch args.

  Args:
    args: Argv tokens after the 'tree switch' prefix.

  Returns:
    First non-flag token, or None when no positional found.
  """
  for token in args:
    if not token.startswith('-'):
      return token
  return None


def _recipe_experiment_add(record: ExecutionRecord) -> UndoSuggestion:
  """Suggest cancel for experiment add.

  Args:
    record: Execution record with experiment add args.

  Returns:
    Suggestion with cancel command containing experiment id.
  """
  exp_id = None
  for token in record.args:
    if not token.startswith('-'):
      exp_id = token
      break
  if exp_id is not None:
    return UndoSuggestion(
      suggested_undo=f'experiment cancel {exp_id}',
      confidence='high',
      notes=None,
    )
  return UndoSuggestion(
    suggested_undo=None,
    confidence='low',
    notes='could not extract experiment id from args',
  )


def _recipe_store_checkout(record: ExecutionRecord) -> UndoSuggestion:
  """Suggest recover for store checkout.

  Args:
    record: Execution record (unused; interface consistency).

  Returns:
    Suggestion pointing to store recover command.
  """
  _ = record
  return UndoSuggestion(
    suggested_undo='store recover --reflog-entry 0',
    confidence='medium',
    notes='recover restores branch tip metadata (refs only, not working files)',
  )


def _recipe_store_snapshot(record: ExecutionRecord) -> UndoSuggestion:
  """Snapshots have no safe automatic undo.

  Args:
    record: Execution record (unused; interface consistency).

  Returns:
    Low-confidence suggestion with no reversal command.
  """
  _ = record
  return UndoSuggestion(
    suggested_undo=None,
    confidence='low',
    notes='snapshots are append-only; no safe automatic undo',
  )


def _recipe_terminal_operation(record: ExecutionRecord) -> UndoSuggestion:
  """Terminal status transitions cannot be reversed.

  Args:
    record: Execution record (unused; interface consistency).

  Returns:
    Low-confidence suggestion noting terminal operation.
  """
  _ = record
  return UndoSuggestion(
    suggested_undo=None,
    confidence='low',
    notes='terminal operation',
  )


RECIPE_TABLE: dict[str, Any] = {
  'experiment deploy': _recipe_experiment_deploy,
  'experiment add': _recipe_experiment_add,
  'store checkout': _recipe_store_checkout,
  'store snapshot': _recipe_store_snapshot,
  'experiment fail': _recipe_terminal_operation,
  'experiment complete': _recipe_terminal_operation,
  'experiment invalidate': _recipe_terminal_operation,
}


def _lookup_recipe(
  command: str,
  record: ExecutionRecord,
  all_records: list[ExecutionRecord],
  cli: CLI,
) -> UndoSuggestion:
  """Look up and invoke the recipe for a command, or return a fallback.

  Args:
    command: Resolved command string from the execution record.
    record: The execution record to generate a suggestion for.
    all_records: Full execution history (needed for tree switch algorithm).
    cli: CLI instance for requires_context checks.

  Returns:
    Structured undo suggestion with confidence level.
  """
  if command == 'tree switch':
    return _recipe_tree_switch(record, all_records, cli)

  for prefix, handler in RECIPE_TABLE.items():
    if command == prefix or command.startswith(prefix + ' '):
      return handler(record)

  return UndoSuggestion(
    suggested_undo=None,
    confidence='low',
    notes='manual review required',
  )


class UndoGuideCommand(Command):
  """Suggest reversal for the last mutating CLI command (read-only)."""

  name = 'undo-guide'
  help = 'suggest undo for last mutating command'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Inspect execution log and suggest reversal for last mutation."""
    exec_path = ctx.config.executions_path
    if not exec_path.exists():
      result: dict[str, Any] = {
        'last_mutation': None,
        'suggested_undo': None,
        'confidence': None,
        'notes': 'no execution history found',
      }
      ctx.output.result(result)
      return

    try:
      records = load_executions(exec_path)
    except TrackingError:
      raw_dicts = read_jsonl(exec_path, strict=False)
      records = [ExecutionRecord.from_dict(d) for d in raw_dicts]
    if not records:
      result = {
        'last_mutation': None,
        'suggested_undo': None,
        'confidence': None,
        'notes': 'no execution history found',
      }
      ctx.output.result(result)
      return

    cli = CLI()
    last_mutation: ExecutionRecord | None = None
    for rec in reversed(records):
      if cli.requires_context(rec.command):
        last_mutation = rec
        break

    if last_mutation is None:
      ctx.fail('only read-only commands in execution history; nothing to undo')
      return

    suggestion = _lookup_recipe(last_mutation.command, last_mutation, records, cli)

    result = {
      'last_mutation': {
        'command': last_mutation.command,
        'args': last_mutation.args,
        'timestamp': last_mutation.timestamp,
        'context': last_mutation.context,
      },
      'suggested_undo': suggestion.suggested_undo,
      'confidence': suggestion.confidence,
      'notes': suggestion.notes,
    }
    ctx.output.result(result)
