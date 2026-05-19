"""Query command: composable filter over experiment tree nodes.

autopilot query [--completed] [--failed] [--running] [--pending] [--terminal]
  [--cancelled] [--filter key=value] [--metric-gt name:value]
  [--metric-lt name:value] [--metric-between name:low:high]
  [--best metric] [--higher] [--lower] [--sort metric]
  [--asc] [--all-trees] [--spec-version VERSION] [--context-contains TEXT]
  [--context-source SOURCE] [--context-after ISO8601]
  [--created-after ISO8601] [--created-before ISO8601]
  [--case-sensitive] [--compact] [--json]

Builds a QueryBuilder chain from CLI flags. Multiple flags compose (AND).
Default output is a table; --json gives structured JSON.

Active tree requirement: ``--all-trees`` queries do not require an active tree;
they route through ``Forest.query()`` which aggregates nodes from every tree in
the forest. Default single-tree queries require an active tree (set via
``tree switch`` or ``tree create``). ``experiment list`` is an alias for
``query`` and follows the same rules.

``--compact`` omits ``context_log`` from JSON rows. ``--metric-gt`` uses
colon separator (``name:value``), not ``=``; the CLI detects ``=`` misuse
and emits guidance. ``--metric-between`` uses ``name:low:high`` syntax for
inclusive range filters (``low <= value <= high``). JSON rows include
``created_at`` and ``started_at`` timestamps. Each row also includes
``metrics_trusted`` (bool): True only when experiment status is
``completed``; False for all other statuses.

JSON rows also include ``deployed_as`` (``str | null``) from the node's
deployment label (``null`` when not deployed) and ``has_notes`` (``bool``)
indicating whether the experiment has non-empty notes.

``--best`` runs after all filters including ``--metric-gt`` / ``--metric-lt`` /
``--metric-between``. ``--sort`` applies only to list mode (not combined with
``--best`` ordering). ``--best`` with ``--all-trees`` includes a ``tree`` field
on the ``best`` JSON object for tree attribution (same semantics as list-mode
``tree``).

``--sort`` uses descending order by default (highest first). ``--asc``
requests ascending order (lowest first, e.g. cheapest-first for ``cost_usd``).
``--asc`` is ignored when ``--sort`` is omitted.

``--deployed`` filters to deployed experiments on the active tree only.
When no deployments match and ``--all-trees`` is not set, the CLI emits an
advisory: *No deployments in active tree. Use --all-trees to search all
trees.* (buffered into JSON ``messages`` in JSON mode).

``--context-contains`` matches context log reasons **or** experiment notes
(case rules unchanged; ``--case-sensitive`` applies to both).

Flag-to-QueryBuilder mapping:
  --completed              -> .completed()
  --failed                 -> .failed()
  --running                -> .running()
  --pending                -> .pending()
  --terminal               -> .terminal()
  --cancelled              -> .filter(status=cancelled)
  --filter key=value       -> .filter(**kwargs)
  --metric-gt n:v          -> .metric_gt(n, v)
  --metric-lt n:v          -> .metric_lt(n, v)
  --metric-between n:l:h   -> .metric_between(n, l, h)
  --best metric            -> .best(metric, higher_is_better) (after all filters)
  --higher / --lower       -> controls direction for --best
  --sort metric            -> .order_by_metric(metric) descending (list mode only)
  --asc                    -> ascending sort order with --sort
  --all-trees              -> query all trees in forest (Forest.query())
  --spec-version VERSION   -> .where(experiment.spec_version == VERSION)
  --context-contains TEXT  -> case-insensitive match on context log or notes
  --case-sensitive         -> makes --context-contains case-sensitive
  --context-source SOURCE  -> .where(context_log.filter_by_source(SOURCE) non-empty)
  --context-after ISO8601  -> .where(context_log.after(ISO8601) non-empty)
  --created-after ISO8601  -> .where(experiment.created_at >= timestamp)
  --created-before ISO8601 -> .where(experiment.created_at < timestamp)

Default scope is active tree only. Use --all-trees for cross-tree queries
that aggregate nodes from every tree in the forest (equivalent to
``Forest.query()``).

``--best`` resolves metric names with a val-first prefix strategy: given
``--best accuracy``, the query tries ``val_accuracy``, then ``train_accuracy``,
then ``accuracy`` (matching framework convention where val metrics are favored).
When none of those match, a single-pass ``train_``/``val_`` normalization is
applied to existing metric keys (no iterative strip) to handle legacy
double-prefixed keys like ``train_train_accuracy``.

When ``--all-trees`` is active, output rows include a ``tree`` field indicating
which tree produced each experiment record.

Path resolution via ctx.config (no paths.* calls).

Import terminal modules directly (e.g. ``from autopilot.core.module.module import Module``),
not package facade -- there is no ``__init__.py``.
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest, parse_metric_threshold_spec, require_active_tree
from autopilot.cli.primitives import Argument, Flag
from autopilot.core.enums import Status
from autopilot.core.experiment import Experiment
from autopilot.core.forest import Forest
from autopilot.core.metric_utils import metric_base_name
from autopilot.core.node import Node
from autopilot.tracking.io import parse_timestamp
from collections.abc import Sequence
from typing import Any
import argparse


def experiment_attr_equals(key: str, expected: str):
  """Build a predicate that compares a stringified experiment field to the expected value.

  Args:
    key: Attribute name on ``node.experiment``.
    expected: Expected string form of the field value.

  Returns:
    Callable taking a ``Node`` and returning whether the field matches.
  """

  def predicate(node):
    d = vars(node.experiment)
    actual = d.get(key)
    return str(actual) == expected

  return predicate


def resolve_metric_name(nodes: Sequence[Node], name: str) -> str:
  """Resolve a user-provided metric name using val-first prefix strategy.

  Tries ``val_{name}`` first, then ``train_{name}``, then ``name`` verbatim.
  When none of those match, applies single-pass ``train_``/``val_`` normalization
  to each node's metric keys and checks whether any key's stripped base matches
  the requested name (handles legacy double-prefixed keys like
  ``train_train_accuracy``).

  Falls back to the original name when nothing matches.

  Args:
    nodes: Candidate nodes to inspect for metric key presence.
    name: User-provided bare metric name (e.g. ``'accuracy'``).

  Returns:
    Resolved metric key string.
  """
  candidates = [f'val_{name}', f'train_{name}', name]
  for candidate in candidates:
    for node in nodes:
      if candidate in node.experiment.metrics:
        return candidate
  for node in nodes:
    for key in node.experiment.metrics:
      base = metric_base_name(key)
      if base in {name, f'val_{name}', f'train_{name}'}:
        return key
  return name


def _experiment_has_notes(experiment: Experiment) -> bool:
  """Return True when experiment.notes is present and non-empty."""
  notes = experiment.notes
  return notes is not None and len(notes) > 0


def _build_tree_map(forest: Forest) -> dict[str, str]:
  """Build a mapping from experiment id to tree name across all trees.

  First occurrence of each experiment id wins, matching :meth:`Forest.query`
  deduplication (same ``_trees`` iteration order; later duplicates are ignored).

  Args:
    forest: Forest instance whose trees to scan.

  Returns:
    Dict mapping experiment id to owning tree name.
  """
  tree_map: dict[str, str] = {}
  for tree in forest.list_trees():
    for node in tree.query().all():
      exp_id = node.experiment.id
      if exp_id in tree_map:
        continue
      tree_map[exp_id] = tree.name
  return tree_map


class QueryCommand(Command):
  """Query experiments with composable filters."""

  name = 'query'
  help = 'Query experiments with composable filters'

  completed_flag = Flag('--completed', help='filter completed experiments')
  failed_flag = Flag('--failed', help='filter failed experiments')
  running_flag = Flag('--running', help='filter running experiments')
  pending_flag = Flag('--pending', help='filter pending experiments')
  terminal_flag = Flag('--terminal', help='filter terminal experiments')
  filter_arg = Argument(
    '--filter',
    action='append',
    default=None,
    metavar='KEY=VALUE',
    help='filter by key=value (repeatable)',
  )
  metric_gt = Argument(
    '--metric-gt',
    action='append',
    default=None,
    metavar='NAME:VALUE',
    help='metric greater than threshold (repeatable)',
  )
  metric_lt = Argument(
    '--metric-lt',
    action='append',
    default=None,
    metavar='NAME:VALUE',
    help='metric less than threshold (repeatable)',
  )
  metric_between = Argument(
    '--metric-between',
    action='append',
    default=None,
    metavar='NAME:LOW:HIGH',
    help='metric within inclusive range (repeatable)',
  )
  best = Argument('--best', default=None, metavar='METRIC', help='return best by metric')
  higher = Flag('--higher', help='higher is better for --best (default)')
  lower = Flag('--lower', help='lower is better for --best')
  sort = Argument(
    '--sort',
    default=None,
    metavar='METRIC',
    help='sort results by metric (descending by default; use --asc for ascending)',
  )
  asc = Flag('--asc', help='sort ascending (lowest first); default is descending')
  all_trees = Flag('--all-trees', help='query all trees in forest (cross-tree mode)')
  context_contains = Argument(
    '--context-contains',
    default=None,
    metavar='TEXT',
    help='filter by substring match in context log reasons or experiment notes',
  )
  context_source = Argument(
    '--context-source',
    default=None,
    metavar='SOURCE',
    help='filter by context log entry source',
  )
  context_after = Argument(
    '--context-after',
    default=None,
    metavar='ISO8601',
    help='filter by context log entries at or after ISO 8601 timestamp',
  )
  created_after = Argument(
    '--created-after',
    default=None,
    metavar='ISO8601',
    help='filter experiments created at or after ISO 8601 timestamp',
  )
  created_before = Argument(
    '--created-before',
    default=None,
    metavar='ISO8601',
    help='filter experiments created before ISO 8601 timestamp',
  )
  cancelled = Flag('--cancelled', help='filter cancelled experiments')
  case_sensitive = Flag('--case-sensitive', help='make --context-contains case-sensitive')
  compact = Flag('--compact', help='omit context_log from JSON rows')
  include_invalidated = Flag(
    '--include-invalidated', help='include invalidated experiments in results'
  )
  deployed = Flag('--deployed', help='only show deployed experiments')
  spec_version = Argument(
    '--spec-version',
    default=None,
    metavar='VERSION',
    dest='spec_version',
    help='filter experiments with this spec_version',
  )
  metadata_contains = Argument(
    '--metadata-contains',
    action='append',
    default=None,
    metavar='KEY:VALUE',
    help='filter by metadata key:value (splits on first colon; repeatable)',
  )

  def _render_best(
    self,
    ctx: CLIContext,
    node: Node,
    *,
    compact: bool = False,
    tree_map: dict[str, str] | None = None,
    tree_name: str | None = None,
  ) -> None:
    """Render a single best-match node.

    Args:
      ctx: CLI context for output rendering.
      node: Best-match node to render.
      compact: When true, omit ``context_log`` from JSON output.
      tree_map: When provided (``--all-trees`` mode), includes ``tree``
        field on the ``best`` JSON object for tree attribution.
      tree_name: Active tree name for single-tree queries (when
        ``tree_map`` is None).
    """
    exp = node.experiment
    if ctx.output.use_json:
      payload: dict[str, Any] = {
        'id': exp.id,
        'status': exp.status.value,
        'hypothesis': exp.hypothesis,
        'metrics': exp.metrics,
        'spec_version': exp.spec_version,
        'created_at': exp.created_at,
        'started_at': exp.started_at,
        'metrics_trusted': exp.status == Status.completed,
        'deployed_as': node.deployed_as,
      }
      if not compact:
        payload['context_log'] = exp.context_log.to_list()
      if tree_map is not None:
        payload['tree'] = tree_map.get(exp.id)
      else:
        payload['tree'] = tree_name
      ctx.output.result({'best': payload})
    else:
      ctx.output.info(f'Best: {exp.id} ({exp.status.value})')
      ctx.output.data({'metrics': exp.metrics, 'hypothesis': exp.hypothesis})

  def _render_all(
    self,
    ctx: CLIContext,
    nodes: Sequence[Node],
    tree_map: dict[str, str] | None = None,
    *,
    compact: bool = False,
  ) -> None:
    """Render a list of matched nodes.

    Args:
      ctx: CLI context for output rendering.
      nodes: Query result nodes.
      tree_map: When provided (--all-trees mode), maps experiment id to tree name
        for attribution in output rows.
      compact: When true, omit ``context_log`` from JSON rows.
    """
    if ctx.output.use_json:
      results = []
      for n in nodes:
        exp = n.experiment
        entry: dict[str, Any] = {
          'id': exp.id,
          'status': exp.status.value,
          'hypothesis': exp.hypothesis,
          'metrics': exp.metrics,
          'spec_version': exp.spec_version,
          'created_at': exp.created_at,
          'started_at': exp.started_at,
          'dataset_fingerprint': exp.dataset_meta.get('dataset_fingerprint'),
          'metrics_trusted': exp.status == Status.completed,
          'deployed_as': n.deployed_as,
          'has_notes': _experiment_has_notes(exp),
        }
        if not compact:
          entry['context_log'] = exp.context_log.to_list()
        if tree_map is not None:
          entry['tree'] = tree_map.get(exp.id)
        results.append(entry)
      ctx.output.result({'experiments': results, 'count': len(results)})
      return
    if not nodes:
      ctx.output.info('(no matching experiments)')
      return
    rows = []
    for n in nodes:
      exp = n.experiment
      metrics_str = ', '.join(f'{k}={v}' for k, v in exp.metrics.items())
      row: dict[str, str] = {
        'id': exp.id,
        'status': exp.status.value,
        'hypothesis': '' if exp.hypothesis is None else exp.hypothesis,
        'metrics': metrics_str,
      }
      if tree_map is not None:
        row['tree'] = tree_map.get(exp.id) or ''
      rows.append(row)
    columns = ['id', 'status', 'hypothesis', 'metrics']
    if tree_map is not None:
      columns.append('tree')
    ctx.output.table(rows, columns)

  def _apply_status_filters(self, qb: Any, args: argparse.Namespace) -> Any:
    """Apply status convenience filters from CLI flags.

    Returns:
      Updated ``QueryBuilder`` with status predicates applied.
    """
    if args.completed:
      qb = qb.completed()
    if args.failed:
      qb = qb.failed()
    if args.running:
      qb = qb.running()
    if args.pending:
      qb = qb.pending()
    if args.terminal:
      qb = qb.terminal()
    if args.cancelled:
      qb = qb.filter(status=Status.cancelled)
    return qb

  def _apply_context_filters(self, qb: Any, args: argparse.Namespace) -> Any:
    """Apply --context-contains, --context-source, --context-after filters.

    ``--context-contains`` is case-insensitive by default and searches both
    context log entry reasons **and** experiment notes. Pass
    ``--case-sensitive`` to require exact case matching (applies to both
    notes and context log). Other context filters delegate to ContextLog
    methods (DRY-02). All supplied filters compose with AND.

    Returns:
      Updated ``QueryBuilder`` with context predicates applied.
    """
    if args.context_contains is not None:
      text = args.context_contains
      if args.case_sensitive:
        qb = qb.where(
          lambda n, t=text: (
            len(n.experiment.context_log.search(t)) > 0
            or (n.experiment.notes is not None and t in n.experiment.notes)
          ),
        )
      else:
        needle = text.lower()
        qb = qb.where(
          lambda n, nd=needle: (
            any(nd in e.reason.lower() for e in n.experiment.context_log.entries)
            or (
              n.experiment.notes is not None
              and len(n.experiment.notes) > 0
              and nd in n.experiment.notes.lower()
            )
          ),
        )
    if args.context_source is not None:
      source = args.context_source
      qb = qb.where(
        lambda n, s=source: len(n.experiment.context_log.filter_by_source(s)) > 0,
      )
    if args.context_after is not None:
      ts = args.context_after
      qb = qb.where(
        lambda n, t=ts: len(n.experiment.context_log.after(t)) > 0,
      )
    return qb

  def _apply_temporal_filters(self, ctx: CLIContext, qb: Any, args: argparse.Namespace) -> Any:
    """Apply --created-after and --created-before temporal filters.

    Thresholds are parsed via ``parse_timestamp()`` (CLAUDE.md: never compare
    ISO strings directly). Experiments missing ``created_at`` are dropped when
    any temporal bound is active.

    Returns:
      Updated ``QueryBuilder`` with temporal predicates applied.
    """
    if args.created_after is not None:
      try:
        threshold = parse_timestamp(args.created_after)
      except ValueError as exc:
        ctx.fail(
          f'invalid --created-after timestamp {args.created_after!r}: {exc}; '
          f'expected ISO 8601 format (e.g. 2026-01-01T00:00:00Z)'
        )
      qb = qb.where(
        lambda n, t=threshold: (
          n.experiment.created_at is not None and parse_timestamp(n.experiment.created_at) >= t
        ),
      )
    if args.created_before is not None:
      try:
        threshold = parse_timestamp(args.created_before)
      except ValueError as exc:
        ctx.fail(
          f'invalid --created-before timestamp {args.created_before!r}: {exc}; '
          f'expected ISO 8601 format (e.g. 2026-01-01T00:00:00Z)'
        )
      qb = qb.where(
        lambda n, t=threshold: (
          n.experiment.created_at is not None and parse_timestamp(n.experiment.created_at) < t
        ),
      )
    return qb

  def _apply_metric_between(self, ctx: CLIContext, qb: Any, args: argparse.Namespace) -> Any:
    """Apply --metric-between flags.

    Each spec must be ``name:low:high`` with exactly three colon-separated
    segments. Calls ``ctx.fail`` with actionable guidance on parse errors.

    Returns:
      Updated ``QueryBuilder`` with metric-between predicates applied.
    """
    if not args.metric_between:
      return qb
    for spec in args.metric_between:
      parts = spec.split(':')
      if len(parts) != 3:
        ctx.fail(
          f'invalid --metric-between {spec!r}: expected name:low:high '
          f'format with exactly 3 colon-separated segments'
        )
      metric_name, low_str, high_str = parts
      if not metric_name:
        ctx.fail(f'invalid --metric-between {spec!r}: metric name must not be empty')
      try:
        low = float(low_str)
        high = float(high_str)
      except ValueError:
        ctx.fail(
          f'invalid --metric-between {spec!r}: low and high must be '
          f'numeric values (got {low_str!r} and {high_str!r})'
        )
      qb = qb.metric_between(metric_name, low, high)
    return qb

  def _apply_field_and_metric_filters(
    self, ctx: CLIContext, qb: Any, args: argparse.Namespace
  ) -> Any:
    """Apply --filter, --metric-gt, --metric-lt, and --metric-between flags.

    Detects ``=`` misuse in ``--metric-gt`` / ``--metric-lt`` specs
    (should use ``name:value`` colon separator) and calls ``ctx.fail``
    with guidance.

    Returns:
      Updated ``QueryBuilder`` with field and metric predicates applied.
    """
    if args.filter:
      for f in args.filter:
        key, _, value = f.partition('=')
        if key == 'status':
          status_val = Status(value.lower())
          qb = qb.filter(status=status_val)
        else:
          qb = qb.where(experiment_attr_equals(key, value))

    if args.metric_gt:
      for spec in args.metric_gt:
        if '=' in spec and ':' not in spec:
          ctx.fail(
            f'invalid --metric-gt {spec!r}: got {type(spec).__name__} with '
            f"'=' separator; use name:value format (colon separator)"
          )
        name, value = parse_metric_threshold_spec(ctx, spec, '--metric-gt')
        qb = qb.metric_gt(name, value)

    if args.metric_lt:
      for spec in args.metric_lt:
        if '=' in spec and ':' not in spec:
          ctx.fail(
            f'invalid --metric-lt {spec!r}: got {type(spec).__name__} with '
            f"'=' separator; use name:value format (colon separator)"
          )
        name, value = parse_metric_threshold_spec(ctx, spec, '--metric-lt')
        qb = qb.metric_lt(name, value)

    qb = self._apply_metric_between(ctx, qb, args)
    return qb

  def _apply_metadata_filters(self, ctx: CLIContext, qb: Any, args: argparse.Namespace) -> Any:
    """Apply --metadata-contains flags.

    Each spec is ``key:value`` split on the **first** colon only.
    Keys must not contain colons. Values may contain colons
    (e.g. ``url:https://example.com``).

    Returns:
      Updated ``QueryBuilder`` with metadata predicates applied.
    """
    if not args.metadata_contains:
      return qb
    experiments_path = ctx.config.experiments_path
    for spec in args.metadata_contains:
      colon_idx = spec.find(':')
      if colon_idx < 0:
        ctx.fail(
          f'invalid --metadata-contains {spec!r}: expected key:value '
          f'format with at least one colon separator'
        )
      key = spec[:colon_idx]
      value = spec[colon_idx + 1 :]
      if not key:
        ctx.fail(f'invalid --metadata-contains {spec!r}: key must not be empty')
      qb = qb.metadata_contains(key, value, experiments_path)
    return qb

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run a composed tree query and render best match or all matches.

    Default scope is active tree only. Use ``--all-trees`` for cross-tree
    queries that span all trees in the forest.

    By default, invalidated experiments are excluded from results. Use
    ``--include-invalidated`` to include them.

    ``--deployed`` filters to only deployed experiments (``deployed_as`` set).
    When ``--deployed`` yields no matches on the active tree (without
    ``--all-trees``), emits an advisory suggesting ``--all-trees``.
    ``--spec-version VERSION`` filters to experiments whose ``spec_version``
    matches the given string exactly (``None`` experiments never match).

    ``--best METRIC`` runs after all filters (including ``--metric-gt`` /
    ``--metric-lt`` / ``--metric-between``) and resolves the metric name using val-first prefix
    strategy (``val_*`` first, then ``train_*``, then bare name). With
    ``--all-trees``, includes ``tree`` on the ``best`` JSON object.

    ``--sort METRIC`` applies only to list mode (not combined with ``--best``).
    Default order is descending (highest first); ``--asc`` requests ascending
    order (lowest first, e.g. cheapest-first for ``cost_usd``).

    ``--all-trees`` includes a ``tree`` field in output for attribution.
    ``--context-contains`` matches context log reasons or experiment notes.

    JSON rows include ``deployed_as`` (deployment label or ``null``) and
    ``has_notes`` (bool, true when experiment has non-empty notes).
    """
    forest = load_forest(ctx)
    tree_map: dict[str, str] | None = None
    active_tree_name: str | None = None

    if args.all_trees:
      qb = forest.query()
      tree_map = _build_tree_map(forest)
    else:
      tree = require_active_tree(ctx, forest)
      active_tree_name = tree.name
      qb = tree.query()

    if not args.include_invalidated:
      qb = qb.exclude(status=Status.invalidated)

    if args.deployed:
      qb = qb.where(lambda n: n.deployed_as is not None)

    if args.spec_version is not None:
      qb = qb.where(lambda n, v=args.spec_version: n.experiment.spec_version == v)

    qb = self._apply_status_filters(qb, args)
    qb = self._apply_field_and_metric_filters(ctx, qb, args)
    qb = self._apply_context_filters(qb, args)
    qb = self._apply_temporal_filters(ctx, qb, args)
    qb = self._apply_metadata_filters(ctx, qb, args)
    compact = args.compact

    if args.best:
      higher_is_better = not args.lower
      resolved_metric = resolve_metric_name(qb.all(), args.best)
      result = qb.best(resolved_metric, higher_is_better=higher_is_better)
      if result is None:
        if args.deployed and not args.all_trees:
          ctx.output.info('No deployments in active tree. Use --all-trees to search all trees.')
        if ctx.output.use_json:
          ctx.output.result({'best': None})
        else:
          ctx.output.info('(no matching experiments)')
        return
      self._render_best(ctx, result, compact=compact, tree_map=tree_map, tree_name=active_tree_name)
      return

    if args.sort is not None:
      qb = qb.order_by_metric(args.sort, ascending=args.asc)

    nodes = qb.all()
    if len(nodes) == 0 and args.deployed and not args.all_trees:
      ctx.output.info('No deployments in active tree. Use --all-trees to search all trees.')
    self._render_all(ctx, nodes, tree_map=tree_map, compact=compact)
