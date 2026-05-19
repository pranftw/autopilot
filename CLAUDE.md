# AutoPilot

CLAUDE.md and AGENTS.md contain identical content. CLAUDE.md is the canonical source; AGENTS.md mirrors it for non-Claude agents. Edits must be applied to both files.

PyTorch/Lightning-inspired optimization framework for non-differentiable systems.
forward -> loss -> backward -> optimizer.step() for prompts, configs, code, pipelines.

## Architecture

- **core**: Module (`core.module.module`), AutoPilotModule (`core.module.autopilot_module`), Parameter, ScalarParameter (`core.parameter`), Gradient, Loss, Optimizer, Metric, Trainer (`core.trainer.trainer`), Config, FilePath,
  Scheduler, LambdaScheduler (`core.scheduler`), IncompatibleKeys (`core.module.module`),
  Experiment, Node, Tree, Forest, Store (`core.store.base`), QueryBuilder, Environment, LocalEnvironment,
  Delta (`core.comparison`), MetricsComparator, ComparatorMetric,
  ConstraintResult (`core.constraint`), gate_to_constraint (`core.constraint`),
  Recommender, Recommendation (`core.recommend`),
  Callback (`core.callbacks.callback`), ContextLogCallback (`core.callbacks.context`), OnExceptionCallback (`core.callbacks.on_exception`),
 Loop (`core.loops.loop`), EpochLoop (`core.loops.epoch`),
 Operator, Context, OperatorNode (`core.operator`), RemovableHandle (`core.graph`), EvalDatum,
 CheckpointIO, JSONCheckpointIO, CheckpointCallback, Logger, JSONLogger,
 Profiler, SimpleProfiler (`core.profiler`),
 ParameterSchema, ParameterSchemaEntry,
  MergeStrategy, ConflictEntry, MergeAnalysisResult, MergeIndex,
  MergeClassification, DiffKind (all in `core.store.types`),
  BranchHandle, RefsView (`core.branch`),
  DiagnosticEntry (`core.diagnostic`),
  DecisionEntry (`core.decision`),
  TrendAnalyzer, TrendResult (`core.trend`),
  MetadataArtifact (`core.metadata`),
  ContextEntry, ContextLog (`core.context`), Traceable (`core.traceable`),
 TraceReport, TraceDimension, verify_trace_completeness (`core.trace`).
  `Traceable` is the base class for `Experiment` (`Experiment(Traceable)`) providing `context_log`, `add_context()`, and `create_context_log()` to all lifetime objects.
  Public customization hooks on the context system: `Traceable.create_context_log()` (factory for custom log types), `Traceable.add_context()` (pre-process entries), `ContextLog.accept()` (pre-append validation gate), `ContextEntry.create()` (canonical factory via `cls`), `ContextLogCallback.should_record()` (filter entries), `Trainer.emit_context()` (callback-driven emission), `Callback.on_context_emit()` (hook for custom recording)
 - `TraceReport` / `verify_trace_completeness()` (`core.trace`) audits context+reflog completeness (policy gates, gradient journals, store context, optional cost); distinct from `build_timeline()` merge in `core.timeline`. CLI: `trace verify` (read-only, context-exempt).
 - `Node` is the experiment tree / project graph entry -- **not** the autograd `OperatorNode`.
 - `Store` merge lifecycle is three-step: `merge_analysis` (cheap classification) -> `merge_preview` (materialize `MergeIndex` with `ConflictEntry` triples) -> `merge_apply` (persist as new epoch). `MergeStrategy` enum: normal, ours, theirs, union. Union strategy concatenates text content line-by-line when ancestor exists, or raw concatenation otherwise. `FileStore.merge_and_apply` is the convenience wrapper for automation. Legacy `Store.merge()` and `MergeResult` are removed.
  - `EvalDatum` (in `core/types.py`) is the standard evaluation/training data container with
  success, metrics, feedback, metadata, split, epoch, and error_message fields. Base `Datum` has only `items`, `_id`, `grad_fn`.
  - `Operator`, `Context`, and `OperatorNode` live in `core/operator.py`; `Gradient` (and `NumericGradient`) live in `core/gradient.py`.
  - `CheckpointIO` / `JSONCheckpointIO` (`core/checkpoint.py`) are the training checkpoint storage backend (save/load/remove/exists). `Trainer.save_checkpoint` assembles full state; `fit(ckpt_path=...)` resumes. `CheckpointCallback` (`core/callbacks/checkpoint.py`) saves per-epoch. `StoreCheckpointCallback` is separate (parameter versioning via `Store.snapshot`). The `ai.evaluation.checkpoints.CheckpointIO` is unrelated (eval JSONL progress).
 - `Store.__init__(config)` is config-only; parameters registered separately via `Store.register_parameters(dict)`. Named keys (module attribute names) replace positional `param_0` indexing.
 - `Store.branch(experiment_id)` **creates** a new branch. `Store.branch_handle(experiment_id)` returns a `BranchHandle` that **curries** per-branch operations (snapshot, checkout, log, diff, latest_epoch) without repeating the experiment_id. `Store.refs_view` returns a `RefsView` — iterable read-only view over all branches with `__getitem__` (returns `BranchHandle`), `__contains__`, `__iter__`, `__len__`. `BranchHandle.latest_epoch()` reads `store.load_refs()['branches'][experiment_id]['latest_epoch']`; missing branch raises `StoreError`. Base `Store` raises `NotImplementedError` for both; `FileStore` provides the concrete implementation.
 - `Parameter.schema_entry()` returns `ParameterSchemaEntry` with type metadata. `Parameter.load_from_dict(data)` applies serialized state into a live instance (used by `Module.load_state_dict`).
  - `ParameterSchema` / `ParameterSchemaEntry` live in `core/snapshot.py`; embedded in `SnapshotManifest.schema`.
 - `SnapshotManifest.context` (`str | None`) is an optional reason/provenance string for audit traceability. `Store.snapshot(..., context=)` threads it through to the persisted manifest. Older manifests on disk that lack the key deserialize as `None`.
- **ai**: PathParameter (`ai.parameter`), FileStore (`ai.store.file_store`), AgentOptimizer (`ai.optimizer`),
 JudgeLoss (`ai.loss`), ClaudeCodeAgent (`ai.agents.claude_code`), TextGradient,
 AutoPilotExperiment, FileForest, IsolatedEnvironment, MergeAgent,
 DeploymentEvent, DeploymentLog (`ai.deployment`),
 ForestRecommender (`ai.recommend`),
 StoreTransaction (`ai.transaction`)
  - `PathParameter.working_root` is the effective I/O root (ephemeral, never serialized). `bind(root)` / `unbind()` switch between worktree and canonical `source`. Trainer owns bind/unbind lifecycle.
  - `PathParameter.from_dict` intentionally does not use `_hydrate_datum_base` because `source`/`pattern` are constructor-only arguments, not dataclass fields.
  - `FileStore(config)` config-only constructor; `register_parameters(dict)` sets named parameter keys. Snapshot manifests embed `ParameterSchema`. `Store.store_blob(digest, data)` is abstract on the base class; `FileStore` implements it as the content-addressed blob write (used by `MergeAgent` and CLI `merge-resolve --content`).
 - `FileStore` maintains an append-only `reflog.jsonl` for auditability. Every mutating operation (snapshot, checkout, branch, reset_branch, merge_apply, materialize, copy_epoch, tag, stash, stash_pop) appends an entry. `store doctor` detects reflog gaps via `reflog_gaps` field. CLI: `debug store reflog`, `store reflog list` (identical JSON output, both read-only).
 - `Store.checkout(..., context=)` threads context to reflog entries (parity with `Store.snapshot(..., context=)`). CLI callers forward `ctx.context`; programmatic callers (Experiment.rollback, Trainer checkpoint resume) supply descriptive fixed strings. `BranchHandle.checkout(epoch, context=)` and `Tree.checkout(experiment_id, context=)` also accept context for parity.
 - `FileStore.iter_reflog()` yields reflog entries in chronological order (oldest first). Corrupt JSONL lines are skipped with a stderr message. `FileStore.expire_reflog(older_than: timedelta) -> int` drops entries older than the cutoff and returns the count removed; rewrites atomically under the store lock. Corrupt JSON lines and entries with missing or unparseable timestamps are retained (not expired). `FileStore.recover_from_reflog(entry_index: int)` restores branch tip metadata (latest_epoch + HEAD) from the Nth valid reflog entry (0-based); does not run checkout or touch working-tree files. CLI: `store reflog expire --older-than Nd`, `store recover --reflog-entry N`.
 - `Store.tag(name, experiment_id, epoch)` / `get_tag(name)` / `list_tags()` for named epoch references. Tags are immutable (duplicate name raises `StoreError`). `FileStore` stores tags in `refs.json`. CLI: `store tag create`, `store tag list`.
 - `Store.copy_epoch(source_experiment_id, source_epoch, target_experiment_id, *, context)` copies a snapshot manifest from one branch to another as the next sequential epoch. Content-addressed blobs are shared by digest (no byte duplication). Self-copy is allowed. Validates both branches exist, source epoch exists, and all blob digests are present in the object store. Appends a `copy_epoch` reflog entry with `source_experiment_id` and `source_epoch` metadata. Sets HEAD to the target branch. CLI: `store copy-epoch <source_exp> <source_epoch> <target_exp>`.
 - `Store.stash()` / `stash_list()` / `stash_pop()` provide Git-stash-like WIP capture. Stash manifests live under `{store_path}/stash/` as dense-numbered `0000.json`, `0001.json`, etc. `epoch = -1` sentinel distinguishes stash from real snapshots. Capture-only: working files unchanged after `stash()`. LIFO default on `stash_pop()`; explicit index supported. Renumber-after-pop keeps indices dense. `_walk_manifests` includes stash files so `prune_orphans` treats stash blobs as reachable. Stash operations append reflog entries with `operation` in `{stash, stash_pop}`. `stash_pop(index, *, context)` accepts an optional `context` string threaded into the reflog entry for audit provenance (parity with `snapshot`/`checkout`/`tag` context recording). CLI `store stash-pop` forwards `--context` to this parameter. `stash_pop` raises `StoreError` when registered parameters have no entry in the stash manifest (parameters registered after the stash was created). `stash_list` raises `StoreError` on corrupt manifests.
 - `MergeAgent(agent, store)` drives agent-based conflict resolution: `build_resolution_prompt` -> `resolve_conflicts` -> `apply_resolution`. Rejects binary/non-UTF-8 conflicts (requires `--content` manual path). Uses `hash_content` from `store_lock` for canonical hashing.
 - `DeploymentEvent(DictMixin)` (`ai/deployment.py`) records a single deploy/undeploy/replace action. `DeploymentLog` provides append-only JSONL read/write at `{workspace}/.autopilot/deployment_events.jsonl`. `emit_deployment_event` is the convenience emitter. `deployment_log_for_workspace(workspace)` resolves the canonical path. CLI deploy/undeploy handlers emit events and mirror them into the experiment context log with `source='deployment'`.
 - `DatasetFingerprint` (`ai/fingerprint.py`) provides dataset content/version fingerprinting for drift detection and cross-experiment data lineage. `Trainer._complete_experiment_success` auto-attaches `DataModule.dataset_fingerprint` (when set) into `experiment.dataset_meta['dataset_fingerprint']`, skipping if already present (avoids duplicate writes on resume).
 - `FileForest.save()` acquires `AutopilotFileLock` at `store_path/forest.lock` to prevent concurrent writers from corrupting `forest.json`. Lock is fail-fast by default (`timeout_s=None`). On contention, raises `ConcurrentMutationError` (from `tracking/file_lock.py`) with `retry_after_ms` hint. `FileForest.lock_timeout_s` can be set to enable waiting (threaded from CLI `--wait` via `load_forest`). Load does not require locking (`atomic_write_json` ensures readers see old-or-new, never torn bytes).
 - `StoreTransaction` (`ai/transaction.py`) is the atomic multi-resource context manager for `FileStore` writes. Coordinates refs.json, snapshot manifests, and reflog.jsonl under a single lock scope. `FileStore.transaction(context=...)` is the factory. Writes are buffered in memory; on success exit they are flushed to disk via atomic helpers. On failure: discards buffered writes and releases lock. Nesting raises `RuntimeError`. `merge_apply` uses `StoreTransaction` internally for atomic persist.
- **ai.evaluation**: GeneratorAgent (`autopilot.ai.evaluation.generator`),
  JudgeAgent (`autopilot.ai.evaluation.judge`),
  PythonStep (`autopilot.ai.evaluation.steps`),
  EvalRunContext, hash_eval_config (`autopilot.ai.evaluation.pipeline`)
- **data**: Dataset, DataLoader, DataModule, Stage,
  Sampler, SequentialSampler, RandomSampler, BatchSampler, SubsetSampler, WeightedSampler, EpochAwareSamplerMixin,
  IncrementalSplitter, SplitAssignment
  - `DataLoader` uses sampler-based ordering; `shuffle` parameter is removed (use `sampler=RandomSampler(dataset)`). `batch_size=None` raises `ValueError`.
  - `DataModule.setup(stage: Stage)` takes a `Stage` enum, not a string. `state_dict()` / `load_state_dict()` for checkpoint round-trips.
  - `Stage` enum: `fit`, `validate`, `test`, `predict` (in `data/datamodule.py`).
  - `IncrementalSplitter` / `SplitAssignment` (`data/splitter.py`) provide incremental train/val splitting with per-sample labeling.
- **policy**: Policy (`policy.policy`), Gate hierarchy, MonotonicGate, BudgetGate (`policy/gates.py`), ConstraintResult (`core.constraint`)
  - `MonotonicGate(metric, *, direction='non_decreasing', required=True, epsilon=0.0)`: requires a metric to never decrease (or never increase) across epochs. Compares current value against `_prev_<metric>` key injected by `EpochLoop`. First epoch (no history) passes. Direction: `'non_decreasing'` (current >= prev - epsilon) or `'non_increasing'` (current <= prev + epsilon). `epsilon` is an absolute tolerance for noisy metrics; must be non-negative; default `0.0` preserves strict comparison. Negative epsilon raises `ValueError`.
  - `BudgetGate(max_usd, *, required=True)`: rejects epochs where cumulative `cost_usd` exceeds `max_usd`. Opt-in only -- never auto-attached by Trainer. Wire via policy composition (e.g. `QualityFirstPolicy(gates=[..., BudgetGate(max_usd=50.0)])`). Missing `cost_usd` metric fails closed. `BudgetGate.max_usd` public property returns the configured budget ceiling.
 - `MonotonicGate.direction` and `MonotonicGate.epsilon` public properties return the configured direction (`'non_decreasing'` or `'non_increasing'`) and absolute tolerance respectively.
 - `Gate.state_dict()` / `Gate.from_dict(data)`: base class defines the serialization protocol (raises `NotImplementedError`). Each concrete gate (`MinGate`, `MaxGate`, `RangeGate`, `MonotonicGate`, `BudgetGate`) implements `state_dict()` returning a dict with `'type'` key and configuration, and `from_dict(data)` as a classmethod for deserialization.
 - `Policy.state_dict()` / `Policy.load_state_dict(state)`: serialization protocol on base `Policy` (raises `NotImplementedError`). `QualityFirstPolicy` implements concrete versions: `state_dict()` returns `{'gates': [...], 'human_review_on_warn': bool}`; `load_state_dict(state)` reconstructs gates via `GATE_TYPE_MAP` dispatch.
 - `QualityFirstPolicy.forward()` populates `result.gates` with a `list[ConstraintResult]` for trainer policy-gate journaling. Gate evaluation happens exactly once per gate (reuses cached results). `_gate_threshold_str` produces operator-prefixed thresholds: `'>= 0.8'` (MinGate), `'<= 1.0'` (MaxGate), `'[0, 1.0]'` (RangeGate), `'non_decreasing (epsilon=0.1)'` (MonotonicGate), `'50.0 USD'` (BudgetGate).
  - `Gate.hint` (`str | None`): ephemeral diagnostic string populated via `difflib` closest-match suggestions when the configured metric is not found in `Result.metrics`. Cleared at the start of each `forward()` call. Not serialized. `format_missing_explanation()` produces `'{GateName}({metric}): missing -> FAIL; {hint}'`. `_format_metric_unavailable_hint` lists all available keys and "did you mean" suggestions. `_suggest_closest_metrics_tied` returns all equally-close names above cutoff `0.4`.
  - `ConstraintResult(DictMixin)` (`core/constraint.py`): structured pass/fail per policy constraint. Fields: `name`, `passed`, `metric`, `value` (float | None), `threshold` (str), `message` (str | None). Constructor validates all six field types (TypeError on invalid). Bool is explicitly rejected for `value` despite being an int subclass. `from_dict` validates required keys and reports all missing keys at once (KeyError on missing).
  - `Result.gates` is `list[ConstraintResult]` (breaking change from `dict[str, str]`). `Result.passed` is a computed `@property`: `all(c.passed for c in self.gates)`. Passing a dict for `gates` raises `TypeError`. `Result.to_dict()` includes the computed `passed` key. `Result.from_dict()` hydrates nested `ConstraintResult` dicts and strips legacy `passed` keys.
  - `gate_to_constraint(gate_name, gate_result, metric, value, threshold) -> ConstraintResult` maps `GateResult` enum to structured constraint. `PASSED` -> `passed=True`; `FAIL`/`WARN`/`SKIP` -> `passed=False`.
  - `QualityFirstMetric.to_result()` produces `list[ConstraintResult]` for `Result.gates` via `gate_to_constraint`. Each gate's threshold is extracted as a human-readable string.
- **gradients** (`gradients.py`): Convenience re-export module for all gradient types. `from autopilot.gradients import Gradient, NumericGradient, TextGradient` — single import path. Original layer-specific paths (`core.gradient`, `ai.gradient`) remain valid.
- **tracking**: I/O primitives (`utc_now_iso`, `parse_timestamp`, `read_json_dict`, `iter_jsonl_lines`, `exclusive_create`, atomic/append helpers),
  `AutopilotFileLock` (`tracking/file_lock.py`) -- thin wrapper around `filelock` (PyPI) that raises `ConcurrentMutationError` (subclass of `TrackingError`) on contention. Uses `fcntl.flock` on POSIX (kernel-enforced, auto-released on crash). Fail-fast default (`timeout_s=None`). `ConcurrentMutationError` carries `operation` label and `retry_after_ms` hint (`LOCK_RETRY_AFTER_MS = 100`). Timeout semantics: `None` = fail-fast, positive float = wait seconds, `-1.0` = block forever.
  unified execution tracking (`executions.jsonl` dispatch-level records, TeeWriter, capture_output)
- **cli**: workspace/project/experiment/optimize/ai/tree/query/checkout/stabilize/execute/debug/dataset/propose/store/report/policy/status/diagnose/trace/track/recommend commands
 - CLI commands fall into two families: **forest-backed** commands whose primary object is the experiment tree and metadata (`experiment`, `tree`, `query`, related read paths) -- they reason about nodes, metric slugs, and HEAD without necessarily touching bytes on disk; and **source-backed** commands that materialize or capture file content relative to a tracked `PathParameter.source` root and store blobs (`store checkout`, `store snapshot`, `store merge-*` persistence). `store log` bridges both: it defaults to forest inference for the experiment but accepts `--source` for explicit source-backed usage.
 - `cli/primitives.py` owns low-level CLI framework primitives: `ArgparseCLIError`, `AutopilotArgumentParser`, `Argument`, `Flag`, `SubcommandMeta`, `subcommand`, `argument`, `collect_arguments`, `collect_subcommands`. `cli/command.py` contains `_BASE_CONTEXT_EXEMPT`, `Command`, `CLI`.
 - `cli/helpers.py` owns shared forest/tree/experiment bootstrap helpers (`load_forest`, `require_active_tree`, `require_experiment_node`, `store_vcs_arguments`, `journal_user_context`, `resolve_command_epoch`) used across CLI commands.
 - `CLI._project_registry` (`cli/command.py`) maps project slugs to concrete `CLI` subclasses for `autopilot -p <name>` dispatch (intentional exception to the no-registries rule).
 - `cli/commands/propose.py` uses `ProposalVerdict` (from `ai/proposal.py`) for structured verify results. `VerdictKind` (StrEnum in `cli/commands/propose.py`) for verdict classification (`improved`/`regressed`/`inconclusive`).
 - `cli/messages.py` owns shared CLI message constants (`MSG_*` prefix) used across command handlers.
 - `cli/helpers.py` also provides `resolve_epoch` for epoch argument resolution shared across commands.
 - `debug commands` emits machine-readable catalog via `CommandsCatalog.build()` (`cli/commands/debug_commands.py`). Agents discover commands, required flags, context requirements, and JSON support without parsing help text.
 - Store merge CLI: `store merge-analysis`, `store merge-preview`, `store merge-apply`, `store merge-resolve`. Three-step workflow: `merge-analysis` (classify) -> `merge-preview` (materialize conflicts, emit `preview_token`) -> `merge-resolve` (per-key resolution via `--ours`/`--theirs`/`--content`) -> `merge-apply` (persist). `--token` couples preview/resolve/apply. JSON envelopes include conflict side metadata (`{digest, size}`), `preview_token`, strategy, and epoch on apply.
 - Global `--context` flag: mutating commands require `--context 'reason'`; read-only commands in `_BASE_CONTEXT_EXEMPT` (`command.py`) are exempt. `_BASE_CONTEXT_EXEMPT` covers all read-only commands and entries must stay aligned with registered command names. Enforcement uses `CLI.requires_context(command)` instance method. Whitespace-only / empty values are rejected identically to omission. Project CLI subclasses extend exemptions via `CLI(context_exempt_commands=frozenset({...}))` constructor parameter which merges with the base set. `--dry-run` is globally exempt from `--context` enforcement via the dispatch gate (`if not ctx.dry_run and self.requires_context(command)`): dry-run previews perform no durable mutation and should not require a reason string.
 - Global `--wait` flag: `--wait TIMEOUT_MS` controls lock contention behavior. Absent = fail-fast (default). `--wait 0` = block forever. `--wait N` (N > 0) = wait up to N milliseconds. Wired to `CLIContext.wait_timeout_ms` and threaded through `load_forest` to `FileForest.lock_timeout_s` and `FileStore.lock_timeout_s`. On contention with `--json`, emits `{"ok": false, "error": "...", "error_code": "concurrent_mutation", "retry_after_ms": 100}`.
 - Global `--retry` flag: `--retry N` reruns the entire command up to N times on `ConcurrentMutationError` with exponential backoff starting from `LOCK_RETRY_AFTER_MS` (100ms, 200ms, 400ms, ...). Default `0` = fail-fast (preserves current behavior). Wired to `CLIContext.retry_max`. On success after retries, JSON envelope includes `retry_attempts` (int) at the top level. `--retry` and `--wait` are mutually exclusive (both solve contention differently: block inside lock acquisition vs rerun whole command). Combining both with `--retry N > 0` triggers `ctx.fail` with guidance. Retry loop is disabled during `--dry-run`.
 - DRY-04 split: `--context` flows to `ExecutionRecord.context` from `CLI.dispatch` (execution log); to `experiment.add_context(source='user')` from handlers via `journal_user_context` (experiment context log). Never both in the same layer.
 - `journal_user_context(ctx, experiment, args)` (`cli/helpers.py`) is the single call site for user-string journaling on experiments. At most one call per CLI invocation (DRY-07). No-op when `ctx.context is None`.
 - Query enhancements: `--sort <metric>` orders results by metric descending (highest first) via `QueryBuilder.order_by_metric`; `--sort` + `--asc` requests ascending order (lowest first). `--sort` applies only to list mode (not combined with `--best` ordering). `--best <metric>` runs after all filters including `--metric-gt` / `--metric-lt` and resolves metric names with val-first prefix strategy (`val_*` -> `train_*` -> bare). `--metric-between <name>:<low>:<high>` is sugar for inclusive range filters (`low <= value <= high`) wired directly to `QueryBuilder.metric_between`. `--all-trees` includes a `tree` field in output for tree attribution. `--best` with `--all-trees` includes `tree` on the `best` JSON object. `--context-contains` matches context log reasons **or** experiment notes (case rules unchanged; `--case-sensitive` applies to both). `experiment list` is an alias for `query`. Cross-tree `Forest.query()` deduplicates by `experiment_id`, keeping the first occurrence in tree iteration order (BUG-044 fix). `QueryBuilder` methods return new instances (immutable chain contract); callers must reassign (`b = b.metric_gt(...)`).
 - `tree remove <name>` removes a tree from the forest (irreversible). Persists under forest lock via `FileForest.remove_tree` -> `save()`.
 - `Forest.find_experiment(experiment_id) -> tuple[Node, Tree] | None` is the canonical cross-tree experiment lookup. Checks the active tree first, then remaining trees in iteration order; first match wins. Returns `None` when absent (callers translate via `ctx.fail`). CLI commands (`experiment compare`, `experiment show`, `metadata`, `stabilize`, `experiment deploy`, `experiment undeploy`) delegate to this method. No duplicate cross-tree scan helpers exist after plan 02.
 - `MetadataArtifact(JSONArtifact)` (`core/metadata.py`) stores durable key-value experiment metadata at `{experiment_path}/metadata.json`. API: `set(key, value, base_dir)`, `get(key, base_dir)`, `show(base_dir)`. Empty key raises `ValueError`. Missing key returns `None`. Missing file returns `{}`. Distinct from `dataset_meta` (data lineage vs configuration tags).
 - `QueryBuilder.metadata_contains(key, value, experiments_path)` filters nodes whose experiment metadata has `key` equal to `value` (string equality after `str()` coercion). Immutable chain contract. CLI: `query --metadata-contains key:value` (splits on first colon; values may contain colons).
 - `experiment metadata set <id> <key> <value>` (mutating, requires `--context`), `experiment metadata get <id> <key>` (read-only, context-exempt), `experiment metadata show <id>` (read-only, context-exempt). All support `--json`.
 - `experiment compare` cross-tree lookup: when an experiment is not found in the active tree, all trees in the forest are searched before failing (BUG-019 fix). Metric prefix normalization aligns `val_*`/`train_*`/bare keys by base name. JSON output includes a direction-aware `verdict` field (`improved`/`regressed`/`inconclusive`). Verdict uses `infer_direction` heuristic (substring matching for long patterns via `LOWER_IS_BETTER_PATTERNS`: loss, error, latency, cost, perplexity; segment matching for short patterns via `LOWER_IS_BETTER_SEGMENT_PATTERNS`: cer, wer -- to avoid false positives on English words like 'answer' or 'concern') to determine whether higher or lower values are better. Every delta dict includes a mandatory `higher_is_better` boolean. `--higher-metric NAME` / `--lower-metric NAME` (repeatable) override the heuristic for specific metrics. Conflicting overrides (same name in both) and unknown metric names are hard errors. When both experiments declare a `spec_version` and they differ, emits a warning. JSON always includes `spec_version: {baseline, candidate}`. `dataset_fingerprint_drift` is tri-state: `True` (drift detected), `False` (both present, no drift), `None`/`null` (either fingerprint missing or empty -- lineage unknown). Text mode prints `unknown (fingerprint missing on one side)` when `None`; no output when `False`. `--weights 'metric:weight,...'` enables weighted multi-metric aggregation: weights are normalized internally (non-negative, positive sum required), each metric's contribution is `normalized_weight * direction_sign * delta` where `direction_sign` is `+1` (higher-is-better) or `-1` (lower-is-better) matching per-delta direction resolution. JSON adds `weighted_verdict` (`improved`/`regressed`/`inconclusive`) and `weighted_score_delta` (float) only when `--weights` is provided. Omitting `--weights` leaves default JSON unchanged. Every weighted metric must exist as a numeric metric in both experiments (unknown or non-numeric metric is a hard error). `WEIGHTED_VERDICT_EPSILON = 1e-9` is the inconclusive band constant.
 - `report compare --all-trees --metric <name>` compares the best experiment per tree by the specified metric. `--higher` (default) or `--lower` controls direction. JSON output includes per-tree best experiment with metrics and a `winner` field. Does not accept positional slugs.
 - `report compare` classic path (positional slugs) is baseline-centric: first slug is baseline, each subsequent slug is compared pairwise against it. Direction (higher-is-better vs lower-is-better) resolved per metric via `infer_direction` heuristic, overridden by `--lower-metric` flags. Delta `higher_is_better` fields reflect resolved direction.
 - `report summary` supports aggregate mode: when `--experiment` is omitted, aggregates the active tree by default, or all trees with `--all-trees`. JSON result includes `scope` (`'tree'` or `'workspace'`), `experiments_count` (by status), `metric_summary` (min/max/mean per metric key), and `best_experiment`. Only **completed** experiments contribute to `metric_summary` and `best_experiment`. `best_experiment` selects the completed experiment with the highest value on the first metric key in lexicographic order. Single-experiment mode (with `--experiment`) is unchanged.
 - `report trend --all-trees` runs `TrendAnalyzer.analyze()` once per tree and returns a JSON `result.trees` dict keyed by tree name. Each value is a `TrendResult.to_dict()` or `None` (when the tree has zero analyzable experiments). Without `--all-trees`, behavior is unchanged (active-tree-only). Text mode shows per-tree blocks with `[tree_name]` prefixes.
 - `recommend --metric-gt NAME:NUMBER` and `--metric-lt NAME:NUMBER` (repeatable) pre-filter candidates before ranking. AND semantics: a candidate must satisfy all supplied predicates. `metric_gt` requires `metrics[name] > threshold`; `metric_lt` requires `metrics[name] < threshold`. Missing metric keys exclude the candidate. When all candidates are filtered out, returns a sentinel `Recommendation` with `action='investigate'` and `experiment_id=None` (exit 0, not `ctx.fail`). Uses `parse_metric_threshold_spec` from `cli/helpers.py` (shared with `query`).

## Agent execution interface

Four tiers for agent-driven execution, from lightest to heaviest:

1. **Parameterized scripts** -- example scripts accept argparse flags (`--max-epochs`, `--threshold`, `--json`, etc.). The agent varies behavior without editing files:
   ```bash
   uv run python run_trainer.py --max-epochs 10 --threshold 0.5 --json
   ```

2. **`autopilot execute`** -- execute Python code/files/modules with full tracking. Four modes: `-c` (inline), file path, `-m` (module), or stdin pipe. Extra args forwarded to the code's `sys.argv`. All commands are automatically tracked at the dispatch level:
   ```bash
   autopilot execute -c "
   from textmatch.module import TextMatchModule
   m = TextMatchModule('rules')
   print(list(m.parameters()))
   "
   ```
   File mode (first positional arg is a script path; remaining args forwarded):
   ```bash
   autopilot execute run.py --max-epochs 2
   ```
   Module mode:
   ```bash
   autopilot execute -m textmatch.module
   ```
   Stdin pipe:
   ```bash
   echo 'print("hello")' | autopilot execute
   ```
   With forwarded argparse args:
   ```bash
   autopilot execute -c "
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--epochs', type=int, default=5)
   args = parser.parse_args()
   print(args.epochs)
   " --epochs 10
   ```

3. **`autopilot track`** -- run and audit an arbitrary shell command. Tokens after `--` are executed via `subprocess.run(argv, shell=False)`. The dispatch-level `ExecutionRecord` logging persists one JSONL row with stdout/stderr, timing, and exit code. Requires `--context`. After the subprocess finishes, raises `SystemExit(returncode)` so the JSONL row matches the child's exit code:
   ```bash
   autopilot track --context 'run linter' -- ruff check src/
   autopilot track --context 'deploy build' -- make deploy
   ```
   Unlike `execute` (Python-only), `track` runs any shell command. Both use `ExecutionRecord` via the dispatch wrapper. Use `track` for non-Python tooling (linters, builds, deploys) that needs audit trail coverage.

   With `--json`, emits a structured result before exit:
   ```json
   {"ok": true, "result": {"exit_code": 0, "argv": ["ruff", "check", "src/"]}, "messages": []}
   ```
   With `--dry-run`, reports what would run without executing:
   ```json
   {"ok": true, "result": {"argv": ["ruff", "check", "src/"], "dry_run": true}, "messages": []}
   ```

4. **File editing** -- when primitives themselves need to change (new Module, Loss, Optimizer). Reserved for capability changes, not routine parameterization.

### Shell escaping guidance for agents

- For `autopilot execute -c "CODE"`, prefer single-quoted code for `$`, `{}`, or f-strings: `autopilot execute -c 'print(f"x={42}")'`
- Use stdin pipe for complex multi-line code -- avoids all escaping issues: `echo '...' | autopilot execute`
- Use `--` to forward flags that collide with autopilot globals (e.g. `autopilot execute -m mod -- --help` forwards `--help` to the module instead of showing execute's help)
- `sys.argv[0]` is `'-c'` in the subprocess (the command runs `uv run python -c` internally)
- Autopilot global flags (`--experiment`, `--workspace`, `--json`, `--dry-run`) are consumed by autopilot and NOT forwarded to the subprocess -- use different flag names in your code's argparse
- Global flags go before the `execute` subcommand's own arguments

## Execution tracking

- All CLI commands are automatically tracked at the dispatch level (`CLI.dispatch()`).
- `ExecutionRecord` captures: `timestamp`, `command`, `args`, `duration_ms`, `exit_code`, `stdout`, `stderr`, `experiment`, `project`, `extra`, `context` (optional reason/provenance string for traceability).
- Storage: `{root}/executions.jsonl` (project-scoped via `Config` / `FilePath` on `AutoPilotConfig`).
- Execute modes: `autopilot execute -c "code"`, `autopilot execute script.py`, `autopilot execute -m module`, stdin pipe.
- Inspection: `autopilot debug executions list`, `show`, `tail` (with `--json` where applicable). `list` supports `--context-contains <substring>` for filtering records by context field.
- Experiment journal: `autopilot experiment show <id> --context-log` displays the decision journal. `--context-source <source>` filters by source, `--limit N` shows the N most recent entries. `experiment compare` includes context summary in text and full `context_log` arrays in JSON.
- `experiment show --json` includes lineage and trust fields: `parent` (str | None, parent experiment id), `baseline` (str | None, baseline experiment id), `dataset_fingerprint` (Any | None, same shape as query), `metrics_trusted` (bool, True only when status is `completed`).
- `query --json` experiment rows include `metrics_trusted` (bool): True only when status is `completed`; False for all other statuses including `invalidated`. Rows also include `deployed_as` (`str | null`, deployment label from the node) and `has_notes` (`bool`, true when the experiment has non-empty notes).
- `--deployed` filters to deployed experiments on the active tree only. When no matches and `--all-trees` is not set, emits advisory: *No deployments in active tree. Use --all-trees to search all trees.* (buffered into JSON `messages`).
- `experiment fail --json` `error` field uses a fallback chain: `--error` value > `--context` flag > trainer failure context log entry. The fallback is display-only and does not mutate `experiment.error`.
 - `experiment notes write <id> --body 'text'` or `--file <path>`: mutually exclusive flags for inline text or UTF-8 file content. Positional text argument is removed (clean break). Binary files (NUL byte in first 8192 bytes) and non-UTF-8 files are rejected with actionable errors.

## Design principles

- Usability over performance. Simple over easy. Progressive disclosure.
- Layering: `autopilot.ai.evaluation` emits progress and results through
  `EvaluationOutputProtocol` (`ai/evaluation/protocols.py`), not through
  `autopilot.cli` types. CLI `Output` (`autopilot.cli.output.Output`) satisfies
  that protocol structurally at application boundaries.
- `Trainer` intentionally depends on `autopilot.policy.policy.Policy`: epoch-loop code
  uses policy gates for post-validation acceptance (`GateResult`) on merged train/val metrics before advancing
  last accepted epoch — policy is a first-class training concept, not an accidental CLI leak.
- isinstance on core classes only (Module, Parameter, Gradient, Datum, Experiment, Metric, Loss, EvalDatum, NumericGradient), never concrete leaves.
- No Python `warnings` module anywhere in `src/` or `tests/`: either raise a specific exception or handle silently. Ambiguous signals via `warnings.warn` are prohibited. `pytest.warns` is banned because the framework does not emit warnings.
- Store interacts with parameters only through snapshot()/restore().
- All customization hooks are public methods (never underscore-prefixed).
- Components are Python objects, not string-key lookups. No registries in library code except the intentional CLI project registry: `CLI._project_registry` maps `project` slugs to concrete `CLI` subclasses for `autopilot -p <name>` dispatch (`cli/command.py`). Do not add parallel string-key registries elsewhere.
- Workflows in code, not config files.
- Callbacks: all hooks receive `(trainer, module, ...)` matching Lightning convention.
- Epochs: 0-based throughout the framework (0 to max_epochs-1), matching Lightning's current_epoch.
- Trainer has public `store` parameter (`Trainer(..., store=store)`), not a private `_store`.
- Trainer has public `datamodule` property (`trainer.datamodule`), not a private `_datamodule`. EpochLoop uses `trainer.datamodule` (not `getattr`).
- Trainer.fit() requires `AutoPilotModule`, not plain `Module`. The type annotation is `fit(module: AutoPilotModule)`.
- Trainer metric discovery excludes children of MetricCollection to prevent double-update.
- `Metric.compute()` is cached across repeated calls until `update()` or `reset()` invalidates the memo. Raises `RuntimeError` when called without prior `update()`.
- Experiment has `store`, `last_accepted_epoch`, `should_rollback`, `rollback()`, `strict_snapshot_after_complete`, `spec_version` as base-class attributes with proper defaults.
- `Experiment.spec_version` (`str | None`, default `None`): optional metric/evaluation schema version string. Set at experiment creation via `experiment add --spec-version`. Serialized in `state_dict()` / `load_state_dict()` (missing key in legacy dicts deserializes as `None`). Surfaced in `experiment status`, `experiment show`, `query`, and `experiment compare` JSON payloads. `query --spec-version VERSION` filters by exact match. No runtime enforcement -- metadata only.
- Metrics vs Metadata boundary:
  - `experiment.metrics` (`dict[str, float]`): numeric values produced by training/validation that are **queryable**, **comparable**, and **sortable**. Examples: accuracy, loss, latency_ms, cost_usd. These flow through `Trainer._complete_experiment_success`, appear in `query --json`, and drive `experiment compare` deltas.
  - `MetadataArtifact` (via `experiment metadata set/get/show`): non-numeric or configuration key-value pairs that **tag** an experiment for filtering but are not mathematically comparable. Examples: model_name, dataset_version, prompt_template_hash, hyperparameter_config_id. These are queried via `--metadata-contains key:value`.
  - `experiment.dataset_meta`: data lineage specifically (fingerprint, split info). Distinct from both metrics and metadata.
  - Rule of thumb: if you would plot it on a chart or compute a delta, it's a metric. If you would use it as a filter label or lookup key, it's metadata.
- `Experiment.rollback(epoch)` checks out store at `epoch` and sets `experiment.epoch = epoch`. Appends a `ContextEntry` via `add_context` (DRY-07: no `emit_context` since no Trainer is in scope) with `source='trainer'`, `epoch=self.epoch` (the new epoch), and `metadata` built via `DecisionEntry.rollback(target_epoch=epoch, reason=...)` (contains `_type='rollback'`, `target_epoch`, `reason`). No-op when `store` is None or `epoch` is None. Missing epoch propagates `StoreError`.
- `Experiment` is a context manager (`with experiment:`) -- `__enter__` calls `start()`, `__exit__` calls `complete()` or `fail()` with a status guard to prevent double-finalization. `__exit__` never suppresses exceptions. `Trainer.fit` uses `with self._experiment:` when an experiment is configured.
- `Status.invalidated` is a terminal status. `Experiment.invalidate(reason: str)` transitions from `completed` only; rejected from other states. Sets `invalidated_at` timestamp. Query excludes invalidated experiments by default (`--include-invalidated` to include). CLI: `experiment invalidate`.
- `Node.deployed_as: str | None` labels an experiment's deployment target. CLI: `experiment deploy <id> --as <name> [--replace]`. Uniqueness enforced per forest via `Forest.deploy(node, label, replace=bool)` / `Forest.undeploy(label)`. `--replace` transfers the label from the current holder (cross-tree). `experiment undeploy <label>` clears the label. Both commands resolve experiments cross-tree via `Forest.find_experiment` (no tree switch required), journal `source='deployment'` context entries on affected experiments, and emit deployment events strictly after successful `forest.save()` (no event written on persistence failure). Query: `--deployed` filter.
- Configuration resolution failures raise `ConfigError` and experiment lifecycle failures raise `ExperimentError`, both from `autopilot.core.errors`.
- Epoch counters: (1) `store` refs `latest_epoch` = persisted tip for snapshots/materialize/checkouts, (2) `Trainer.current_epoch` = loop cursor during `fit`, (3) `Experiment.epoch` = logical experiment cursor, must be explicitly aligned on rollback/checkout. `Tree.add()` auto-branches in Store when a store is attached (BUG-035). `Tree.remove` does not prune store branches (BUG-041 documented).
- Experiment lifecycle CLI: `experiment complete`, `experiment fail`, `experiment cancel` transition experiment status from the CLI. `complete` and `fail` accept both `pending` and `running` experiments (for CLI-only workflows without Trainer). `cancel` accepts any non-terminal status. All persist the forest and journal user context.
- `experiment impact <id>` shows direct and transitive dependents via reverse dependency graph BFS. Also includes direct tree children via `Node.parent` pointers (forest-wide scan). JSON payload: `experiment_id`, `dependents`, `direct_dependents`, `children`. Children are orthogonal to dependency edges.
- `experiment add` auto-sets tree HEAD to the new experiment id immediately (FRICTION-003). `Tree.head` has a public setter for in-memory HEAD updates without triggering store checkout. `--no-parent` skips HEAD auto-parenting (creates orphan experiment). `--parent <id>` and `--no-parent` are mutually exclusive; conflict triggers `ctx.fail` with guidance.
- `Store.reset_branch(experiment_id)` resets a branch's `latest_epoch` to -1, enabling re-run from epoch 0. Existing snapshot manifests are retained. HEAD is unchanged. Refs-only; use `reset_and_restore` when working-tree files must sync in the same operation. CLI: `autopilot --experiment <id> store branch --reset`. Next `snapshot()` call succeeds at epoch 0 (BUG-008).
 - `Store.reset_and_restore(experiment_id, epoch, *, context)` atomically resets branch tip and syncs working-tree parameter files; appends both `reset_branch` and `checkout` reflog entries under one lock. `epoch=None` sets tip to `-1` and clears tracked files. `epoch=N` sets tip to N and restores that epoch's snapshot content. CLI: `autopilot --experiment <id> store branch --reset --restore [--epoch N]`.
- `tree switch` auto-checkouts the HEAD experiment's latest snapshot by default so that working tree files are synced to the new active tree. Pass `--no-checkout` to skip the automatic checkout; in that case a disk-state advisory is emitted recommending a manual `store checkout` or environment activation. When the HEAD experiment's branch has no snapshots (`latest_epoch < 0`), checkout is skipped as a benign no-op (exit 0) with an informational advisory.
- `IsolatedEnvironment.setup` requires a prior snapshot or parent branch. A branch with `latest_epoch=-1` and no parent returns empty snapshot content, meaning parameter files will not be materialized in the worktree.
- `Trainer.fit()` wraps `Config.environment.activate` when an `IsolatedEnvironment` is present -- binds `PathParameter`s to the worktree and unbinds them in `finally`. Requires an experiment; raises `ConfigError` otherwise.
- `Trainer._complete_experiment_success` merges train/val metrics with prefixed keys (`train_*`/`val_*`) when both splits exist, using `strip_metric_prefix` (from `core.metric_utils`) to avoid double-prefixing (e.g. `train_train_accuracy` becomes `train_accuracy`). When only validation metrics are present, they are stored unprefixed.
- `Trainer.should_stop_at` scans callback hook results (a list of dicts) for any dict containing `{'stop': True}`. Non-dict entries are silently skipped.
- `Trainer._ensure_agent_optimizer_context` auto-wires context (`experiment_id`, `epoch`, `metrics`, `trainer`) into `AgentOptimizer` after `configure_optimizers()` when no explicit context was provided. The `'trainer'` key enables optional traceability: `AgentOptimizer` may call `trainer.emit_context()` after successful agentic steps.
- `Trainer._restore_path_parameter_files` triggers `store.checkout` on checkpoint resume when the module has `PathParameter`s and the checkpoint carries an experiment epoch.
- `Trainer.validate(module, dataloaders=None, datamodule=None) -> dict[str, float]` for standalone validation without fit. `Trainer.test(module, dataloaders=None, datamodule=None) -> dict[str, float]` for standalone test evaluation. Both delegate to `run_eval_phase` internally.
- `num_sanity_val_steps=2` (default): capped validation run before the first training epoch as a pre-flight check. Skipped when no val loader or when value is 0. Hooks: `on_sanity_check_start`/`on_sanity_check_end` dispatched.
- `Trainer(profiler=SimpleProfiler())` enables wall-clock timing of training sections. `Profiler` is the base class (`core/profiler.py`); `SimpleProfiler` records wall-clock durations via `time.perf_counter()`. `describe()` returns `{action: {'count': int, 'total_ms': float, 'mean_ms': float}}`. Sections profiled: `training_step`, `validation_step`, `backward`, `optimizer_step`, `store_snapshot`, `store_checkout`. On `fit` completion, `profiler_summary.json` is written to the experiment directory. Profiler errors are isolated (never abort training). CLI: `debug profiler --experiment <id>` (read-only, context-exempt).
- `debug trend <metric> [--all-trees] [--json]` runs `TrendAnalyzer.analyze()` on completed experiments in the active tree (or all trees with `--all-trees`). Returns trend direction, slope, and data points. Read-only, context-exempt.
- `Trainer.emit_context(reason, *, source, metadata)` builds a `ContextEntry` via `ContextEntry.create()` (DRY-01) with `epoch=self.current_epoch` and dispatches to all callbacks via `on_context_emit`. Always dispatches regardless of experiment presence; `ContextLogCallback` silently no-ops when `trainer.experiment` is None.
- `Trainer._attach_default_callbacks()` follows Lightning's enable_* pattern: when `enable_context_log=True` (default) and an experiment is present, appends `ContextLogCallback`. A user-registered callback with `_is_context_log_callback = True` suppresses the default (replacement). Setting `enable_context_log=False` disables auto-attach; combining `False` with a user context callback raises `ConfigError` (conflict). Detection uses the flag (DRY-06), not `isinstance`.
- `Trainer._complete_experiment_success` emits `'experiment completed successfully'` context with `source='trainer'` and `metadata={'final_metrics': ...}` before calling `experiment.complete()`.
- `Trainer._fit_failure_path` emits `'experiment failed: {error}'` context with `source='trainer'` and `metadata={'error': str(exc)}` before marking the experiment as failed.
- `EpochLoop` injects prior-epoch accepted metrics into the gate result under `_prev_<metric>` keys (single leading underscore). This enables `MonotonicGate` to compare against the last accepted epoch without maintaining its own history. When no prior accepted metrics exist (first epoch), `_prev_` keys are absent and `MonotonicGate` passes (baseline).
- `EpochLoop._check_policy_gate` runs **after validation** in `_finalize_epoch`, receiving merged train/val metrics (prefixed `train_*`/`val_*` when both splits exist, unprefixed when train-only). Emits context on both accept and reject paths with `DecisionEntry.POLICY_GATE_TYPE` as `_type` discriminator and a `gates` list of `ConstraintResult.to_dict()` payloads. Reject path: `metadata={'_type': ..., 'gates': [...]}` (no `metrics` or `gate_result`). Accept path: `metadata={'_type': ..., 'gates': [...], 'metrics': ...}`. Reason strings: `f'policy gate rejected epoch {epoch}'` / `f'epoch {epoch} accepted by policy gate'`. Emission uses `trainer.emit_context` (DRY-07).
- `EpochLoop.run()` sets `trainer.current_epoch = epoch` at the top of each loop iteration, before any hooks or callbacks. `_run_epoch` also sets it (idempotent) for direct callers. All `emit_context` calls within an epoch observe the correct 0-based epoch index.
- `EpochLoop.run()` calls `_set_sampler_epoch_for_loader(loader, epoch)` on both train and val loaders immediately after setting `trainer.current_epoch`, before `_should_stop_before_epoch`. The helper checks **`batch_sampler` first**, then `sampler`, and unwraps `BatchSampler` inner samplers in both paths. When `loader.sampler` is `None`, falls back to `loader.batch_sampler` (the common `DataLoader(..., batch_sampler=...)` construction). The extracted `_set_epoch_on_sampler` helper handles `BatchSampler` unwrap and `EpochAwareSamplerMixin` isinstance check. This ensures `RandomSampler` and `WeightedSampler` produce deterministic per-epoch sequences.
- `Trainer._fit_success_path` handles gate-reject lifecycle: when `stopped_by_gate` and `last_accepted_epoch is None` (all epochs rejected), the experiment is failed via `experiment.fail()` -- not completed successfully. When `stopped_by_gate` with partial acceptance, `_complete_experiment_success` runs normally. Only when all planned epochs ran without gate stop does it emit `'training completed: max_epochs reached'` context with `source='trainer'`.
- `Trainer.capture_gradient_summaries()` snapshots parameter gradients as structured `list[dict[str, str]]` after backward and before zero_grad. Defers to the optimizer when `owns_step_gradient_context` is True (no Trainer-side capture). Each dict has keys `param_name` (from `named_parameters()`), `param_type` (`type(param).__name__`), `gradient_type` (`type(param.grad).__name__`), `summary` (truncated `grad.render()` at `GRAD_SUMMARY_MAX_CHARS = 200`). `emit_epoch_gradient_journal(trainer, *, epoch)` emits cached summaries per accepted epoch after gate accept (metadata includes `epoch` + `gradient_summaries`); `_emit_gradient_journal()` retained as final summary in `_complete_experiment_success`; all-rejected path emits journal before fail. `build_gradient_journal_row(param_name, param, max_chars)` in `core/trainer/journal.py` is the shared row constructor used by both Trainer and AgentOptimizer.
- `EpochLoop._finalize_epoch` emits `DecisionEntry.optimizer_step` context after a policy-gate-accepted epoch (for non-agentic optimizers). Metadata: `{'_type': 'optimizer_step', 'epoch': int, 'param_summaries': list[dict]}`. Each param summary has keys `param_name`, `param_type`, `value_preview` (truncated at `PARAM_SUMMARY_MAX_CHARS = 200`). Skipped when `optimizer.owns_step_gradient_context` is True (agentic optimizer manages its own context).
- `AgentOptimizer._agentic_step` emits `'optimizer step failed'` context with `source='agent-optimizer'` when the agent produces no output.
- `EpochLoop.run()` emits `'early stopping triggered before epoch {epoch}'` context with `source='early-stopping'` when `_should_stop_before_epoch` returns True.
- `propose verify` uses `MetricsComparator` against baseline/candidate experiment metrics. Returns structured JSON with per-metric deltas and an overall verdict (`improved`/`regressed`/`inconclusive`). After `record_verdict`, emits `DecisionEntry.comparison` context with `source='proposal'` when experiment found in forest (metadata includes `proposal_id`, `baseline_epoch`, `candidate_epoch`, `verdict`, `deltas`).
- `Gradient.todo_items() -> list[str]` extracts actionable items from `render()` output (lines > 15 chars, non-headers). `TextGradient` overrides to return `[self.attribution]` when set.
- `AgentOptimizer` supports path constraints: `allowed_paths` restricts the agent to a set of path prefixes, `forbidden_paths` blocks specific subtrees (forbidden always wins on overlap). Both are normalized to POSIX strings resolved against the config root. Empty `allowed_paths` means unrestricted; empty `forbidden_paths` means nothing forbidden. `validate_paths_after_step=True` scans filesystem after each agent step and raises `ConfigError` on violations.
- `AgentOptimizer.agentic` flag (default `True`) enables file-based feedback: `write_epoch_feedback()` writes `.optimization/epoch_N.md`, `update_todo()` tracks gradient-derived action items, `build_task_brief()` produces concise agent prompts with inline todos and file pointers. When `agentic=False`, uses the legacy `build_prompt()` path. `feedback_dir` is auto-wired from `config.root` by Trainer when available, or must be passed explicitly; raises `ConfigError` when unresolvable. After a successful agentic step, emits a context entry via `trainer.emit_context()` with structured gradient summaries as `list[dict[str, str]]` (same schema as Trainer journal: `param_name`, `param_type`, `gradient_type`, `summary`) built via `build_gradient_journal_row` from `core/trainer/journal.py`. Source is `'agent-optimizer'`. Silent no-op when no trainer ref or no gradients.
- Experiment metrics favor val-prefixed keys: when both train and validation metrics exist, keys become `train_*` / `val_*`. Tree nodes and Forest queries see these prefixed keys via `experiment.metrics`.
- Multi-loss: Trainer uses `module.training_step()` return value; multiple losses compose via explicit wiring in `training_step` (call each loss, combine gradients). There is no hidden multi-loss registry or automatic fan-out; future per-loss routing is deferred.
- Datum batching: when `batch_size > 1`, the object passed to `AutoPilotModule.training_step` is a `Datum` whose `items` contains one `EvalDatum` per sample (e.g. `Datum(items=[EvalDatum, EvalDatum, ...])`). Metrics and training code must iterate `datum.items` (or otherwise handle list-shaped batches) -- do not assume the top-level object is a single `EvalDatum` with scalar fields representing the whole batch.
- Loss ownership: do not call `loss.forward()` or `loss.backward()` inside `training_step` -- the Trainer drives the `Loss` lifecycle in `_process_batch` (forward, backward, reset). If the autograd graph is consumed before the Trainer's backward call, `_process_batch` raises `RuntimeError` with guidance to remove manual loss calls.
- Gradient routing: semantic `Gradient` objects are broadcast to all parameters during `backward()`. Per-parameter gradient routing (directing specific gradients to specific parameters) is not yet implemented; current model is broadcast-then-accumulate. Future work may add routing keys on `Gradient` and filter predicates on `Parameter`.
- Module and parameter authoring is Python-only: CLI cannot invent new `Parameter` subclasses without code generation. Use `project init` templates to scaffold new modules; extend `Module` and `Parameter` in Python code.
- `Optimizer` uses PyTorch-style `param_groups`: constructor accepts `params: list[Parameter] | list[dict]`, keyword-only `lr: float = 1.0`, and `**defaults`. A flat list is auto-wrapped into one group. Each group dict has `'params': list[Parameter]` plus per-group hyperparameters seeded from `self.defaults`. `self.state: dict[int, dict]` holds per-parameter optimizer state keyed by `id(param)` at runtime. Checkpoint `state_dict()` serializes state under `Parameter.id` strings (stable hex). No `_parameters` attribute, no top-level `self.lr` -- LR lives per-group in `param_groups[i]['lr']`.
- `Optimizer.owns_step_gradient_context` (`@property`, default `False`): when True, the optimizer emits gradient context entries during `step()` and Trainer skips its own gradient capture and completion-time gradient journal emission. `AgentOptimizer` overrides to return `self._agentic` (True only when agentic mode is active).
- Non-agent optimizers are first-class: plain `Optimizer` subclasses work with `Trainer` via `configure_optimizers()` and explicit `step()` loops. `AgentOptimizer` is one concrete implementation, not the only path. For non-agent workflows: subclass `Optimizer`, override `step()`, wire via `Module.configure_optimizers()`.
- `Scheduler` (`core.scheduler`) is the base LR scheduler. `LambdaScheduler(optimizer, lr_lambda)` is the built-in scheduler using a user-supplied `Callable[[int], float]`. Scheduler state persists via `state_dict()` / `load_state_dict()`. Trainer wires scheduler from `configure_optimizers()` dict return `{'optimizer': opt, 'scheduler': sched}` and calls `scheduler.step(epoch)` after `on_train_epoch_end` each epoch.
- `Module.register_buffer(name, value, persistent=True)` stores non-parameter state. Persistent buffers appear in `state_dict()`; non-persistent do not. `Module.requires_grad_(requires_grad=True)` sets `requires_grad` on all parameters (returns `self` for chaining). `Module.zero_grad()` clears `grad` and `grad_accumulator` on all parameters.
- `Module.load_state_dict(state_dict, strict=True)` returns `IncompatibleKeys(missing_keys, unexpected_keys)`. When `strict=True` (default), raises `RuntimeError` on mismatches. When `strict=False`, loads matching keys silently and returns mismatches in the result.
- Backward hooks on Module are observation-only: `register_full_backward_hook(hook)` fires after `backward()` with `(module, grad_input, grad_output)` but hooks cannot modify gradients (non-differentiable domain).
- Per-sample failure analysis: `EvalDatum.success` (bool) and `EvalDatum.error_message` fields carry per-sample pass/fail status. Aggregate by filtering `[d for d in results if not d.success]` and inspecting `error_message`. Evaluation JSONL logs (`evaluation.jsonl`) contain per-sample records for post-hoc analysis.
- `CostTrackerCallback` writes `cost_summary.json` to the experiment directory on loop end. `debug cost` CLI reads this file. CostTracker itself is read-only reporting. Budget enforcement is opt-in via `BudgetGate` in the policy layer (see policy section); `BudgetGate` reads `cost_usd` from metrics and rejects epochs that exceed the budget.
- No `autopilot agent` CLI: outer coding agents (Claude, Codex) drive workflows through `optimize`, `execute`, `experiment`, and other existing commands. There is no in-CLI agent command surface.
- `FileStore.doctor()` returns `list[DiagnosticEntry]` — structured findings with diagnostic codes, severity, repair metadata. `FileStore.doctor_report()` converts entries to the legacy dict shape (`healthy`, `manifest_errors`, `missing_blobs`, `orphan_blobs`, `orphan_count`, `refs_issues`, `forest_errors`, `reflog_gaps`, `diagnostics`). CLI: `autopilot store doctor`.
- `FileStore.repair_diagnostics(entries, *, dry_run, context)` applies repairs for entries with `repairable=True`. Repair actions: `orphan_blob` -> delete, `stale_lock` -> delete (PID liveness check), `broken_ref` -> reset to last valid epoch, `reflog_gap` -> backfill synthetic reflog entry, `ghost_epoch` -> delete manifest file. `manifest_error` and `missing_blob` are fail-closed (never auto-repaired). Raises `StoreError` when `context` is None and repairable entries exist (mutating operation requires provenance).
- `ghost_epoch` entries appear in the `diagnostics` list returned by `doctor()` and in the `diagnostics` key of `doctor_report()`. There is no separate top-level `ghost_epochs` key -- unlike `reflog_gaps`, ghost epochs are surfaced via the structured `DiagnosticEntry` list only. Ghost epochs do not affect the `healthy` flag (they are `severity='warning'`).
- `store doctor` reports `healthy=True` even when orphan blobs exist. Orphans do not cause correctness issues (they are unreferenced content blobs). `orphan_count` and `orphan_blobs` fields report orphans for informational purposes. Use `debug store prune-orphans` or `store doctor --repair` to clean them up. `reflog_gaps` lists branches present in refs but absent from reflog (informational only). Only `manifest_errors`, `missing_blobs`, `refs_issues`, and `forest_errors` affect the `healthy` flag. Missing `forest.json` produces a `forest_missing` info diagnostic (`healthy` unaffected); malformed `forest.json` produces `forest_corrupt` error. `doctor_report()` adds `forest_missing: bool`.
- `store doctor --repair` and `workspace doctor --repair` apply safe repairs. `--repair` requires `--context` (mutating). `--repair --dry-run` previews repairs without applying. JSON output includes `diagnostics` (list of `DiagnosticEntry` dicts), `repaired`, and `dry_run` fields. Without `--repair`, both commands remain read-only and context-exempt.
- `Store.prune_orphans()` removes orphaned object blobs not reachable from any snapshot manifest. Base `Store` is a no-op; `FileStore` walks the object store against manifest edges. Fail-closed: raises `StoreError` if any snapshot manifest is corrupt or unparseable (prevents accidental data loss). CLI: `debug store prune-orphans`.
- `stabilize --parameter-prefix <prefix>` filters parameters by name prefix before the copy step. Default (flag omitted): all parameters participate.
- Mandatory `--context` on mutating CLI commands: every command that creates or mutates state requires a reason string via `--context`. Read-only commands (in `_BASE_CONTEXT_EXEMPT`) are exempt; `_BASE_CONTEXT_EXEMPT` covers all read-only commands. Project CLIs extend via `CLI(context_exempt_commands=...)`. This ensures agents and humans always record why an action was taken.
- New CLI debug subcommands default to requiring `--context`. To exempt a read-only debug command, add it to `_BASE_CONTEXT_EXEMPT` in `cli/command.py` and add a test in `tests/cli/test_context_exemptions.py`.
- Temporal awareness: temporal queries and comparisons use `parse_timestamp()` from `tracking/io.py` to parse ISO 8601 strings into `datetime` objects. String comparison of timestamps is never correct; always compare `datetime` objects.

### Autograd

- Operators act on `Datum` (and subclasses); define-by-run graph built during forward, traversed during backward, reset after.
- Graph preservation in `Module.forward()`: NEVER create a new `Datum(items=[...])` inside `forward()` — this detaches from the autograd graph built by upstream operators. Instead:
  - Return the input datum (possibly with modified items via operators like `select`, `merge`, `broadcast`).
  - Use `datum.clone()` if a copy is needed (preserves `grad_fn` lineage).
  - Use framework operators (`select`, `merge`, `broadcast`) which correctly wire `OperatorNode` into the graph.
  - If you must construct output from scratch (e.g., aggregation), ensure the result's `grad_fn` is set by using an operator that handles graph wiring.
  - Creating `EvalDatum(...)` inside forward for return values is fine ONLY when the module is a leaf that does not need backward propagation (e.g., an evaluation-only step).
- `AccumulateGrad` at leaves; fan-in uses homogeneous gradient types (cross-type raises `TypeError`).
- Gradient accumulation detail: when multiple operators feed gradients to the same parameter during `backward()`:
  1. Each upstream operator's `backward()` produces a gradient (or `None` to skip).
  2. During graph traversal, partial gradients fan in via `pending_grads` accumulation on predecessor nodes.
  3. At parameter leaves, an `AccumulateGrad` operator node (stored on `parameter.grad_accumulator`) combines incoming gradients:
     - `TextGradient.accumulate()`: merges texts with `'; '` (semicolon + space) separator.
     - `NumericGradient.accumulate()`: sums numeric values.
  4. The final combined gradient is assigned to `parameter.grad`.
  5. All gradients in an accumulation batch MUST be the same type. Mixing `TextGradient` + `NumericGradient` on the same parameter raises `TypeError` at fan-in time.
  6. Example: if a parameter receives 3 `TextGradient` objects with texts "improve clarity", "reduce length", "fix grammar", the final `parameter.grad` is a single `TextGradient` with text "improve clarity; reduce length; fix grammar".
  7. `Optimizer.zero_grad()` clears both `parameter.grad` and `parameter.grad_accumulator` for the next backward pass.
- `backward` may return `None` per input; fan-in via `pending_grads` accumulation.
- `Datum` is subclassable; operators preserve subclass where `clone()` / copy paths allow.
- `Module`: explicit forward wiring (no composition sugar); `ModuleCallOperator` records calls.
- `Loss.forward()` returns `None` (not a `Datum`); Loss is the gradient source, not a graph node.
- `graph.py` is a leaf module: zero `autopilot.core.*` imports. Nodes are duck-typed.
  Verification: `rg 'from autopilot' src/autopilot/core/graph.py` must return zero matches.

### Argument order / data-first operators

- `select(datum, index)` — datum first, index second. Returns `Datum` containing `datum.items[index]`.
- `broadcast(datum, n)` — datum first, count second.
- `merge(d1, d2, ...)` — all datum operands, variadic.
- `select` no longer accepts variadic datums. For multi-datum selection, use `merge(d0, d1, ...)` then `select(merged, index)`.
- Passing a non-Datum as first argument to `select` raises `TypeError` with message including `argument order changed` (migration signal for old `select(index, *datums)` callers).

### Intentional PyTorch divergences

- Non-tensor domain: Datum carries structured data, not numeric arrays.
- `Loss.forward()` returns `None` (PyTorch losses return a scalar tensor).
- Gradients are primarily semantic text/structured feedback, not numeric dL/dx. `NumericGradient` provides a numeric lane for testing and programmatic losses.
- `ModuleCallOperator` is coarser than PyTorch per-op recording: one graph node per Module.__call__.
- Call modules as `module(datum)` or `self(datum)`, not `module.forward(datum)`, when graph recording is required. `Module.__call__` wraps `forward()` and records a `ModuleCallOperator` node in the autograd graph. Calling `forward()` directly bypasses graph recording -- use only when intentionally avoiding graph capture (e.g. inside an operator that manages its own graph wiring).

### Intentional Lightning divergences

- Epoch callback order: AutoPilot invokes `on_train_epoch_end` after the validation pass completes (including `on_validation_epoch_end`). PyTorch Lightning calls `on_train_epoch_end` before validation. Rely on `on_validation_epoch_end` or experiment hooks when you need post-val metrics inside the same epoch.

### Intentional API divergences

Naming choices that differ from what an agent might first guess. These are documented and tested, not bugs:
- **`ConflictEntry.ancestor`** — field is `ancestor`, not `base`. Reflects three-way merge terminology.
- **`Delta.metric`** — field is `metric`, not `metric_name`. Shorter, consistent with `Result.metrics` keys.
- **`broadcast`** returns a **`Datum`** (with `.items` containing `n` clones), not a `list`. All operators return `Datum`.
- **`RangeGate`**: uses `min_value` / `max_value`, not `low` / `high`.
- **`MonotonicGate`**: uses `direction='non_decreasing'` (or `'non_increasing'`), not `'increasing'` / `'decreasing'`.
- **`select`**: datum-first argument order: `select(datum, index)`, not `select(index, datum)`.
- **`TextGradient.text`** — primary content field is `text`, not `direction`. Legacy `direction=` kwarg raises `TypeError` with migration guidance.
- **`ContextLog.append`** — accepts both a reason string and a pre-built `ContextEntry`. When passed a `ContextEntry`, runs `accept()` and appends directly (same gating as `record()`) and returns the entry or `None` if rejected. `record()` always returns `None`; `append()` returns the entry on success. Keyword args are ignored when a `ContextEntry` is passed.

### Callback dispatch

- `Trainer.dispatch_callbacks(hook_name: str, **kwargs)` resolves `hook_name` with `getattr(cb, hook_name)` and invokes matching Lightning-style methods. String-based dispatch is intentional and aligns with PyTorch Lightning's hook naming; prefer adding real hook methods on `Callback` subclasses over dynamic attribute tricks elsewhere in the codebase.
- `on_context_emit(trainer, module, entry)` is the context traceability hook on `Callback`. Default is no-op. `ContextLogCallback` (`core/callbacks/context.py`) records entries to `experiment.context_log` via `record()`. Detection uses `_is_context_log_callback` class flag (DRY-06), not `isinstance`. Override `should_record(entry)` for filtering, `on_context_emit()` for custom recording.
- Callback lifecycle hooks: `setup(trainer, module, stage)` / `teardown(trainer, module, stage)` bracket each Trainer entry point (fit/validate/test/predict). `on_exception(trainer, module, exception)` fires before `_fit_failure_path`. `on_save_checkpoint(trainer, module, checkpoint)` mutates the checkpoint dict in-place before IO write. `on_load_checkpoint(trainer, module, checkpoint)` observes the raw dict before `_restore_from_checkpoint`. Sanity check hooks (`on_sanity_check_start/end`) are default no-ops dispatched by `_run_sanity_check`.
 - Batch and predict hooks: `on_validation_batch_start/end` and `on_test_batch_start/end` are dispatched per-batch by `Trainer.run_eval_phase`. `on_predict_start/end` and `on_predict_batch_start/end` are dispatched by `Trainer.predict()`. All are default no-ops on `Callback`.
 - `Trainer.predict(module, dataloaders=None, datamodule=None) -> list[Any]` iterates a predict dataloader, calling `module.predict_step(batch, batch_idx)` under `no_grad()`, collecting outputs. Does not call `configure_optimizers`, metrics, or loss. Dispatches predict lifecycle/batch callbacks. Restores prior `_module`/`_datamodule`/`module.trainer` refs in `finally`.
 - `Trainer.run_eval_phase(module, dataloader, *, step_method, hook_prefix, max_batches, epoch_arg)` is the canonical eval runner for validate, test, and sanity check. Dispatches epoch- and batch-level callbacks, discovers and computes metrics from the module, toggles `module.eval()`/`module.train()`.
 - `AutoPilotModule.predict_step(batch, batch_idx) -> Any` defines per-batch prediction logic. Default raises `NotImplementedError`. `DataModule.predict_dataloader() -> DataLoader` resolves the predict data loader (default raises `NotImplementedError`).

## Operational guidance

### Agent prompts and tool validation

- Validate all tool/agent outputs before trusting them: check for hallucinated tool names, phantom function calls, and fabricated metrics. Agents may invent tools that do not exist (BUG-006).
- When using `JudgeLoss` or evaluation judges, ensure the judge prompt has access to the actual tool call transcript, not just final output. Judges cannot assess tool usage they cannot see (BUG-007).
- Rebaseline prompts after each optimization round. Prompt drift accumulates: the optimizer may embed assumptions from prior epochs that no longer hold (FRICTION-005).

### Callbacks and cost

- `OnExceptionCallback` (`core/callbacks/on_exception.py`) saves a crash checkpoint and optional store snapshot on unhandled exception during `Trainer.fit`. Best-effort: save failures are caught (narrow `OSError`/`RuntimeError`/`StoreError`) and never mask the original exception. Default crash path: `{config.root}/crash_checkpoint.json` when experiment is set, else `crash_checkpoint.json` in CWD. Store snapshot context: `f'crash: {type(exception).__name__}'`. On clean teardown (no exception fired), removes stale crash checkpoint. After exception, teardown preserves the crash file for recovery. Opt-in: `Trainer(callbacks=[OnExceptionCallback()])`.
- `CheckpointCallback` must be explicitly attached to `Trainer(callbacks=[...])` for `.ckpt` / JSON checkpoint files to be saved per epoch. It is not auto-attached (GAP-010). Supports `monitor` parameter for best-checkpoint tracking: `CheckpointCallback(directory=..., monitor='val_accuracy')`. Properties: `last_checkpoint_path` (latest epoch), `best_checkpoint_path` (highest monitored metric, higher-is-better, ties favor later epoch), `best_metric_value`.
- `Trainer.fit(module, ckpt_path=...)` accepts `Path | str | None`. String tokens: `'last'` resolves to the latest-epoch checkpoint (not filesystem mtime); `'best'` resolves to the checkpoint with the highest monitored metric value (requires `CheckpointCallback(monitor=...)`). Resolution order: (1) `CheckpointCallback`'s in-memory tracked paths (primary, set during same process), (2) disk scan of `epoch-NNNN.json` files under the callback's directory when in-memory path is `None` (fallback for crash recovery / fresh process -- corrupt files skipped silently), (3) `ConfigError` when neither resolves. Unknown strings raise `ConfigError` listing valid tokens. `Path('last')` is treated as a literal file path, not a token. Multiple `CheckpointCallback` instances with resume tokens raises `ConfigError` (ambiguous).
- `CostTrackerCallback` must be explicitly attached to `Trainer(callbacks=[...])` for cost summaries (`cost_summary.json`). It is not auto-attached (GAP-011). `CostTrackerCallback(emit_context=True)` emits structured context entries each epoch via `trainer.emit_context` with `source='cost'` and metadata `{'_type': COST_ATTRIBUTION_TYPE, 'epoch': int, 'cost_usd': float, 'cumulative': float}`. Default `emit_context=False` avoids context log noise.
- `StoreCheckpointCallback` passes a `context` string to `store.snapshot(...)` including epoch index and up to three metrics (sorted lexicographically by key, numeric values only) formatted as `key=value` tokens. Format: `'epoch N checkpoint (metrics: k1=v1, k2=v2)'`. Absent metrics produce `'epoch N checkpoint'`.
- `ConfigSnapshotCallback` (`core/callbacks/config_snapshot.py`) captures `module.state_dict()` at fit start via `emit_context`. Opt-in only -- large file-based parameters may bloat the context log.
- `Trainer._fit_failure_path` emits a consolidated failure context entry including `traceback` (full `traceback.format_exc()`) and `exception_type` in metadata. One entry per failure (no duplicate with the error-only entry).
- `Trainer(accumulate_grad_batches=N)` accumulates gradients across N batches before calling `backward()`. Default is 1 (no accumulation). For finer-grained control, implement accumulation logic explicitly in the module's `training_step` or via a custom callback.

### Stabilize behavior

- `stabilize` copies all parameter files from the source experiment, including optimizer-mutated artifacts (e.g. `.optimization/` feedback files, todo lists). Use `--parameter-prefix` to filter if only a subset of parameters should transfer (GAP-015).

### Streaming and logging

- Set `PYTHONUNBUFFERED=1` in the agent's environment when streamed progress output is needed in logs. Python buffers stdout by default, which delays real-time visibility of training progress (FRICTION-008).

### PathParameter safety

- `PathParameter.restore()` writes text content only; file permissions (e.g., executable bit) and symlinks are NOT preserved. Executable scripts must be re-permissioned after `store checkout`. Binary files are skipped during snapshot and protected from deletion on checkout.
- `PathParameter.restore()` rejects absolute keys, empty keys, and keys resolving outside the parameter root (path traversal protection). Raises `ValueError` with actionable guidance.
- `PathParameter.snapshot()` skips symlinks resolving outside the parameter root with a `logger.warning` (stdlib `logging`, not `warnings.warn`). Broken symlinks are also skipped with a warning. The warning includes the relative path and reason (`'outside_root'` or `'broken'`). Uses `Path.resolve()` for containment checks (required on macOS where `/var` symlinks to `/private/var`).
- `FileStore.checkout` skips extraneous-file cleanup for parameters with zero snapshot entries, preventing mass deletion when a snapshot was taken on an empty directory.

### Merge operations

- Merge conflict keys are manifest-relative paths (e.g. `prompt.txt`, `rules/main.py`), not absolute filesystem paths. Key names match the parameter's `pattern` within the store's object layout (FRICTION-010).
- Merge positional ordering: `merge-analysis <experiment_id> <from_experiment_id>` and `merge-preview <experiment_id> <from_experiment_id>` take the target (ours) as the first positional argument and source (theirs) as the second. `merge-apply` and `merge-resolve` reference a cached preview via `--token`. In conflict resolution, `--ours` refers to the target (first positional) side and `--theirs` refers to the source (second positional) side (FRICTION-011).
- `store merge-preview` writes an ephemeral cache file consumed by `merge-resolve` and `merge-apply`. Abandoned previews (never applied) are not auto-cleaned. The command is already context-exempt in `_BASE_CONTEXT_EXEMPT` because it is ephemeral staging, not a durable workspace mutation.

### Store snapshot and stash CLI

- `store snapshot` is idempotent by default: when file-entry digests are identical to the latest epoch, no new epoch is written and the prior manifest is returned. Use `--force` to record a new epoch even when files are unchanged (e.g. for context-only provenance markers). JSON output includes `skipped: true/false`.
- `store snapshot` and `store create` forward `--context` to `FileStore.snapshot(..., context=)` so manifests and reflog carry audit context.
- `store stash` and `store stash-pop` rehydrate parameter registration from the latest snapshot manifest's `ParameterSchema` for the active experiment. This requires at least one prior snapshot with a schema; attempting stash on an experiment with no snapshots raises `StoreError` with guidance to run `store snapshot` first.
- Tag names allow ASCII letters (a-z, A-Z), digits (0-9), hyphen (-), underscore (_), and dot (.). Slashes (`/`), spaces, and other punctuation are not permitted. Max length: 128 characters. The error message from `validate_tag_name` explicitly lists allowed characters.
- `TagEntry.manifest_digest` (`str | None`): SHA-256 hex digest of the canonical manifest JSON at tag creation time. `None` for pre-attestation tags (created before digest computation was added). Deserialization tolerance: missing key in `refs.json` maps to `None`.
- `FileStore.tag()` computes `manifest_digest` at tag creation: loads the manifest for `(experiment_id, epoch)`, canonicalizes via `json.dumps(manifest.to_dict(), sort_keys=True, separators=(',', ':'))`, and hashes with `hash_content`.
- `FileStore.verify_tag(name) -> dict[str, Any]` recomputes the digest from the current on-disk manifest and compares to the stored `manifest_digest`. Returns `{'verified': True}` on match, `{'verified': False, 'reason': 'digest mismatch', 'expected': ..., 'actual': ...}` on tamper, or `{'verified': False, 'reason': 'no digest available'}` for pre-attestation tags. Unknown tag raises `StoreError`.
- CLI: `store tag verify <name>` (read-only, context-exempt, `--json` supported). Exit code 0 on verified, non-zero on mismatch or missing digest.

### Concurrent mutation handling

- When agents encounter `ConcurrentMutationError` (error_code `concurrent_mutation` in JSON envelope), retry after `retry_after_ms` milliseconds. Two built-in contention strategies: `--wait <ms>` blocks inside lock acquisition (e.g. `--wait 5000` waits up to 5s, `--wait 0` blocks indefinitely); `--retry N` reruns the entire command up to N times with exponential backoff (100ms, 200ms, 400ms, ...). The two flags are mutually exclusive. Without either flag, failure is immediate on contention. Prefer `--retry` for agent workflows (bounded, observable via `retry_attempts` in JSON) and `--wait` for scripted pipelines expecting brief contention.

### Execution tracking scope

- Only CLI-invoked commands are logged to `executions.jsonl`. Direct Python API usage (e.g. `Trainer.fit()` called from a script without `autopilot execute`) is not tracked. Use `autopilot execute script.py` for Python or `autopilot track -- <command>` for non-Python commands to ensure tracking (Ctx-4).

### EpochOrchestrator plateau detection

- `OrchestratorConfig.monitor` must match the **actual key present** in per-epoch metrics. When Trainer merges train/val splits, keys are prefixed (`train_*`/`val_*`), so `monitor` must use the prefixed name (e.g. `'val_accuracy'`, not `'accuracy'`). When only one split exists, metrics are unprefixed.
- `OrchestratorConfig` raises `ConfigError` at construction time when `plateau_window > 0` and `monitor is None`. When `plateau_window == 0` and `monitor is None`, construction succeeds (plateau detection is disabled). `EpochOrchestrator()` without an explicit config defaults to `plateau_window=0` (plateau disabled).
- `_detect_plateau` compares the last `plateau_window` values of the monitored metric. If the range (max - min) relative to max is below `plateau_threshold`, the orchestrator stops training. Missing `monitor` key in any epoch's metrics causes that epoch to be skipped in plateau detection.
- `EpochOrchestrator.stop_reason` (read-only property): exposes the internal `_stop_reason` string without widening the mutation surface. Recognized values: `'callback_stop'`, `'policy_fail'`, `'plateau'`, or `None` (normal completion / not yet run). The same value appears in the `run()` result dict under `stop_reason`.
- Plateau stop emits `'plateau detected after epoch {epoch}'` context with `source='plateau'` and metadata `_type='plateau_stop'` (via `DecisionEntry.plateau_stop()`). Metadata includes `monitor`, `epoch`, `plateau_window`, `plateau_threshold`, and `values` (window metric values). Filter: `entry.metadata.get('_type') == DecisionEntry.PLATEAU_STOP_TYPE`.

### Optimization loop ordering

- The per-epoch execution order is: `training_step` (forward + loss + backward) -> validation -> policy gate -> optimizer step. The gate sees merged train/val metrics (prefixed `train_*`/`val_*`). The optimizer receives gradients only after the gate accepts the epoch (TS-ordering).
- With a single-epoch configuration (`max_epochs=1`), the policy gate runs after the sole epoch's validation. If the gate rejects, no optimizer step occurs and the experiment ends with `last_accepted_epoch=None`. Design single-epoch workflows with permissive gates or no gate (TS-loop-kill).

### PathParameter patterns for software projects

- For software projects where parameters are directories (not single files), use `PathParameter(source=Path('src/module'), pattern='**/*.py')` to capture all files under a directory. The store snapshots every file matching the pattern, enabling per-file diff and merge (TS-per-file).

### DataModule for single-gate workflows

- For workflows that only need a validation gate (no training data), implement `DataModule.setup(Stage.fit)` to populate `val_dataloader` only. Set `train_dataloader` to return an empty loader or a single-item loader that triggers one `training_step` per epoch. The Trainer requires at least one training batch to advance the epoch (TS-dataset).

### Context journaling best practices

- For sparse-signal optimization (e.g. weekly human feedback), emit context entries with `source='user'` and include the feedback text in `metadata`. This creates an audit trail even when metrics are not updated every epoch. Use `experiment.add_context(reason, source='user', metadata={...})` for non-Trainer contexts (TS-sparse).
- Use `DecisionEntry` factory methods (`deployment`, `rollback`, `comparison`, `policy_gate`, `plateau_stop`, `optimizer_step`) to produce typed metadata dicts with a `_type` discriminator for machine-filterable context log entries. Filter by `entry.metadata.get('_type') == DecisionEntry.DEPLOYMENT_TYPE` (or `ROLLBACK_TYPE`, `COMPARISON_TYPE`, `POLICY_GATE_TYPE`, `PLATEAU_STOP_TYPE`, `OPTIMIZER_STEP_TYPE`).
- Experiment-creation rationale: when creating an experiment as a result of comparing prior experiments (e.g., "experiment A was better than B on metric X, so we branch from A"), emit a comparison context entry on the new experiment immediately after creation:
  ```python
  experiment = Experiment(experiment_id='new-exp', hypothesis='improve on A')
  experiment.start()
  experiment.add_context(
    'created based on comparison of baseline-exp vs candidate-exp',
    source='user',
    metadata=DecisionEntry.comparison(
      baseline_id='baseline-exp',
      candidate_id='candidate-exp',
      verdict='improved',
      deltas=[{'metric': 'accuracy', 'delta': 0.05}],
    ),
  )
  ```
  This links the new experiment's provenance to the decision that spawned it. Query context logs with `_type == 'comparison'` to trace decision chains across the experiment tree.
- `tree switch` auto-checkouts by default. Pass `--no-checkout` to skip disk sync, in which case the advisory recommends running `store checkout` manually. The `--no-checkout` escape hatch is for advanced workflows that manage disk state independently.

### Workspace metadata and status

- `workspace init` persists `--context` to `.autopilot/workspace.json` as `description`. Pre-existing workspaces without this file report `description: null` in status.
- `workspace status` exposes `description` (str or null) as a top-level key in the JSON payload. Text mode shows `Purpose: <description>` when set.
- `workspace status` exits non-zero (`SystemExit(1)`) when composite health is unhealthy, so agents and scripts can rely on the exit code.
- `workspace status` JSON payload includes `deployments` (`list[dict]` with `label`, `experiment_id`, `tree` per entry, sorted by `(tree, label)`, empty list when none) and `trees.detail` (`list[dict]` with `name`, `experiment_count`, `active`, `description` per tree). `trees.detail` supplements the existing `trees.count` and `trees.active` keys.
- `tree create` uses global `--context` as the tree description when `--description` is omitted. Explicit `--description` always wins.

### Workspace doctor

- `workspace doctor` checks the full directory layout created by `workspace init` (experiments, records, datasets, projects directories under `.autopilot/`). API-only workspaces that use custom store paths (e.g. `examples/textmatch`) will report unhealthy until `workspace init` is run. This is expected; `doctor` validates the canonical CLI layout.
- `workspace doctor` validates `forest.json` parse integrity when the file exists. Invalid JSON or non-dict root marks health as unhealthy with a `forest_error` string in the result. The same check runs inside `workspace status` health section via shared `_run_workspace_checks` helper.

## CLI command matrix

Commands, context requirements, and JSON support. Read-only commands are exempt from `--context`.

| Command | Mutating | `--context` required | `--json` support |
|---------|----------|---------------------|------------------|
| `ai generate run` | Yes | Yes | Yes |
| `ai generate resume` | Yes | Yes | Yes |
| `ai generate dry-run` | Yes | Yes | Yes |
| `ai judge run` | Yes | Yes | Yes |
| `ai judge resume` | Yes | Yes | Yes |
| `ai judge summarize` | No | No | Yes |
| `ai judge distribution` | No | No | Yes |
| `checkout` | Yes | Yes | No |
| `dataset list` | No | No | Yes |
| `dataset seed` | Yes | Yes | No |
| `dataset show` | No | No | Yes |
| `dataset split` | No | No | Yes |
| `debug` (most subcommands) | No | No | Yes |
| `debug commands` | No | No | Yes |
| `debug profiler` | No | No | Yes |
| `debug trend` | No | No | Yes |
| `debug store prune-orphans` | Yes | Yes | No |
| `debug store reflog` | No | No | Yes |
| `diagnose run` | No | No | Yes |
| `diagnose heatmap` | No | No | Yes |
| `execute` | Yes | Yes | Yes |
| `experiment add` | Yes | Yes | No |
| `experiment cancel` | Yes | Yes | Yes |
| `experiment compare` | No | No | Yes |
| `experiment deploy` | Yes | Yes | Yes |
| `experiment deploy-log` | No | No | Yes |
| `experiment undeploy` | Yes | Yes | Yes |
| `experiment complete` | Yes | Yes | Yes |
| `experiment fail` | Yes | Yes | Yes |
| `experiment impact` | No | No | Yes |
| `experiment lineage` | No | No | Yes |
| `experiment timeline` | No | No | Yes |
| `experiment invalidate` | Yes | Yes | Yes |
| `experiment list` | No | No | Yes |
| `experiment notes show` | No | No | Yes |
| `experiment notes write` | Yes | Yes | Yes |
| `experiment metadata set` | Yes | Yes | Yes |
| `experiment metadata get` | No | No | Yes |
| `experiment metadata show` | No | No | Yes |
| `experiment remove` | Yes | Yes | No |
| `experiment show` | No | No | Yes |
| `experiment status` | No | No | Yes |
| `optimize train` | Yes | Yes | No |
| `optimize deploy` | Yes | Yes | No |
| `optimize validate` | Yes | Yes | No |
| `optimize test` | Yes | Yes | No |
| `optimize resume` | Yes | Yes | No |
| `optimize preflight` | Yes | Yes | Yes |
| `optimize set-hparams` | Yes | Yes | No |
| `optimize loop` | Yes | Yes | No |
| `policy check` | No | No | Yes |
| `policy explain` | No | No | Yes |
| `project doctor` | No | No | Yes |
| `project init` | Yes | Yes | No |
| `project list` | No | No | Yes |
| `propose create` | Yes | Yes | Yes |
| `propose list` | No | No | Yes |
| `propose revert` | Yes | Yes | Yes |
| `propose verify` | Yes | Yes | Yes |
| `query` | No | No | Yes |
| `recommend` | No | No | Yes |
| `report compare` | No | No | Yes |
| `report narrative` | No | No | Yes |
| `report summary` | No | No | Yes |
| `report trend` | No | No | Yes |
| `stabilize` | Yes | Yes | No |
| `status` | No | No | Yes |
| `store branch --reset [--restore]` | Yes | Yes | Yes |
| `store checkout` | Yes | Yes | Yes |
| `store copy-epoch` | Yes | Yes | Yes |
| `store create` | Yes | Yes | No |
| `store doctor` | No | No | Yes |
| `store doctor --repair` | Yes | Yes | Yes |
| `store diff` | No | No | Yes |
| `store log` | No | No | Yes |
| `store merge` | No | No | Yes |
| `store merge-analysis` | No | No | Yes |
| `store merge-apply` | Yes | Yes | Yes |
| `store merge-preview` | No | No | Yes |
| `store merge-resolve` | Yes | Yes | Yes |
| `store promote` | Yes | Yes | No |
| `store snapshot` | Yes | Yes | Yes |
| `store stash` | Yes | Yes | Yes |
| `store stash-list` | No | No | Yes |
| `store stash-pop` | Yes | Yes | Yes |
| `store status` | No | No | Yes |
| `store recover` | Yes | Yes | Yes |
| `store reflog expire` | Yes | Yes | Yes |
| `store reflog list` | No | No | Yes |
| `store tag create` | Yes | Yes | Yes |
| `store tag list` | No | No | Yes |
| `store tag verify` | No | No | Yes |
| `store worktree create` | Yes | Yes | No |
| `store worktree list` | No | No | Yes |
| `trace collect` | No | No | Yes |
| `trace inspect` | No | No | Yes |
| `trace verify` | No | No | Yes |
| `track` | Yes | Yes | Yes |
| `tree create` | Yes | Yes | Yes |
| `tree describe` | No | No | Yes |
| `tree list` | No | No | Yes |
| `tree remove` | Yes | Yes | Yes |
| `tree show` | No | No | Yes |
| `tree switch` | Yes | Yes | No |
| `undo-guide` | No | No | Yes |
| `workspace doctor` | No | No | Yes |
| `workspace doctor --repair` | Yes | Yes | Yes |
| `workspace init` | Yes | Yes | No |
| `workspace journal` | No | No | Yes |
| `workspace status` | No | No | Yes |
| `workspace tree` | No | No | Yes |

## Style rules

- Google Python Style Guide baseline
- 2-space indentation (ruff enforced)
- Single quotes everywhere
- Absolute imports only, from-imports first, no blank lines between
- No relative imports, no dynamic imports, no deferred imports in `src/` (tests may use inner imports)
- No `if TYPE_CHECKING:` blocks
- No `__init__.py` files -- all imports from terminal files directly
- Line length: 100
- `PLR6301` (no-staticmethod) globally ignored; use normal methods or module-level functions
- Module-level UPPERCASE variables are allowed for: (1) compiled regexes, (2) TypeVars/ParamSpecs, (3) frozenset/tuple *literals* (not computed from function calls), and (4) named numeric/string constants that replace magic values in logic (e.g., `RPM_WINDOW_SECONDS = 60.0`, `SHORT_ID_HEX_LEN = 12`). Even single-use constants are legitimate when they name a magic value. Prohibited: `DEFAULT_*` naming for hardcoded assumptions that should be constructor parameters, and module-level constants caching function calls (e.g., `frozenset(X.__fields__) - frozenset(Y.__fields__)`).
- Magic values: numbers other than 0/1/-1/2 and non-obvious string literals in logic MUST be named constants, regardless of usage count. A named constant communicates intent. Acceptable inline without a name: loop bounds derived from len(), standard API arguments (indent=2, stacklevel=2), and values whose meaning is immediately obvious from surrounding code (e.g., `json.dumps(..., indent=2)`).
- `_is_context_log_callback` is an allowed class-level flag pattern for callback discrimination (DRY-06). Use this boolean flag on callback classes instead of `isinstance` checks to detect the built-in context log callback. This is the sole exception to the "no underscore-prefixed hooks" rule -- it is a detection flag, not an extension point.

## Prohibited patterns

- No `getattr(args, 'x', default)` on declared argparse arguments
- No fake fallback objects on precondition failure
- No inline file-content strings for templates
- No module-level UPPERCASE constants caching function calls or set operations (use inline literals or compute inside the function). Exceptions: compiled regexes (`re.compile`), TypeVars, and `frozenset` literals are allowed.
- No `str = ''` defaults (use `str | None = None`)
- No `.get('key', '')` (use `d['key']` or `.get('key')`)
- No `except Exception: pass` (sole exception: the silent wrapper around `log_execution()` in `CLI.dispatch()` to prevent tracking failures from crashing commands)
- No `except Exception: return {}` (runner must propagate errors)
- No `trainer._store` (use `trainer.store` public property)
- No `getattr(experiment, 'store')` or `getattr(experiment, 'rollback')` (attributes exist on base class)
- No `# noqa` comments
- No os.environ/os.getenv/load_dotenv inside src/autopilot/
- No `import warnings`, `warnings.warn()`, `warnings.catch_warnings()`, or `pytest.warns()` anywhere in `src/` or `tests/`. Raise a specific exception or handle silently. The framework does not emit Python warnings. Enforced by 6 ast-grep rules (`no-warnings-warn`, `no-import-warnings`, `no-import-warnings-aliased`, `no-from-warnings-import`, `no-catch-warnings`, `no-pytest-warns`).
- No private extension/customization methods (all hooks are public)
- No deferred imports in `src/` (tests may use inner imports for isolation); module layout eliminates all import cycles
- No `SequenceModule` or `ParallelModule` (use `Sequential` from `core/ops.py` or explicit wiring)
- No direct `param.grad =` assignment outside `core/operator.py` (except `= None` in `Optimizer.zero_grad()`). `AgentOptimizer` must call `zero_grad()`, not ad-hoc per-parameter loops. `no-param-grad-assign` applies to `src/` only; tests legitimately set `.grad` for testing
- No `ctx.output.error(...); return` in CLI command handlers (use `ctx.fail(message)` which emits error, flushes JSON envelope, and exits non-zero)
- No `subprocess.run(['uv', 'run', 'pytest', ...])` inside test files (recursive test invocation doubles suite time)
- No `subprocess.run(['uv', 'run', 'python', '-c', ...])` in tests for import validation (use in-process `exec()` or direct import)
- No `shutil.copytree` of `examples/` directories in tests to run them via `uv run` subprocess (test logic in-process instead)
- No duplicated test doubles: when a stub Module, Loss, Optimizer, DataModule, or factory function is needed in tests, check `tests/doubles.py` first, then check subtree `conftest.py` files (`tests/data/conftest.py`, `tests/core/conftest.py`). Only define a local stub when behavior is genuinely unique to that test (graph semantics, counting, specific side effects).
- No duplicated test fixtures: experiment factories, dataset classes, and RunConfig builders must be imported from shared locations.

## DRY rules

- One canonical implementation per concern
- I/O goes through tracking/io.py
- Path computation goes through Config (`core/config.py`)
- Serialization uses DictMixin from core/serialization.py
- Timestamps use `utc_now_iso()` from `tracking/io.py` (no inline `datetime.now(UTC).isoformat()`)
- JSON files that must be a single object use `read_json_dict()` from `tracking/io.py`
- Command handlers orchestrate only; no duplicated backend logic

## Test DRY

- Shared test doubles live in `tests/doubles.py` (cross-cutting) and subtree `conftest.py` files such as `tests/data/conftest.py` (data layer) and `tests/core/conftest.py` (experiment factories, tree/forest fixtures).
- Before defining a new test stub, check if an existing shared double covers the need. Only create local stubs when the test requires unique behavior (specific metric values, call counting, graph semantics, file side effects).
- `NoopEvalModule`, `DirectNumericLoss`, `NoOpOptimizer`, `SizedDataset`, `make_experiment` are canonical shared doubles from plan 10; `make_run_config` from plan 11 (plus experiment-state fixtures in `tests/core/conftest.py` where listed in plans 10-11).
- CLI test helpers: use `run_cli` (injects `--context 'test'`) for mutating commands; use `run_cli_no_context` for read-only / context-exempt commands. Never use `run_cli` for tests validating context enforcement -- it would mask missing exemptions.
- Shared workspace fixtures: `workspace_with_store_and_forest` (tests/conftest.py) for full workspace + store + forest; `multi_tree_forest` (tests/cli/conftest.py) for cross-tree CLI tests; `concurrent_forest_writer` (tests/conftest.py) for lock contention tests.
- No autouse checkout mocks in checkout-focused test files. Tests that verify `FileStore.checkout` behavior must exercise real code paths. Non-checkout tests may use per-test `patch` for checkout when snapshots are not seeded.

## Testing

- Unit tests for every function: normal, edge, error cases
- Dataclass round-trips: to_dict -> from_dict -> assert equal
- Integration tests for end-to-end flows
- Use tmp_path fixture, not filesystem mocking
- Tests must be fast: full suite target < 15s, no individual test > 1s
- Prefer in-process testing over subprocess invocation. Validate imports with `exec()` or direct import, not `uv run python -c`
- Never re-run `pytest` from inside a test (recursive invocation)
- `tests/dogfood_regressions/test_v8_regressions.py` — permanent guards for dogfood V4-V7 findings (MonotonicGate epsilon E2E, BudgetGate threshold, policy gates, checkout context, plateau context, batch_sampler epoch, trace completeness, tree name validation, command catalog)

## Google Style Guide practices

- Docstring bar: module, class, and every public method/function. Google style (summary line, blank, body with Args/Returns/Raises).
- Explicit truthiness: use `if x is not None:` instead of `if x:` when `None` vs empty matters. Avoid `or []` / `or {}` for mutable defaults.
- Naming: snake_case functions/variables, CamelCase classes. No single-char variables outside loop iterators.
- Exceptions: always chain with `raise X from exc`. Use `finally` for cleanup. Narrow except clauses.
- Function length: ~60 lines soft limit, ~80 hard limit. Extract helpers for readability.
- Comprehensions and lambdas: prefer named helpers when logic exceeds one expression.
- Comments: lowercase sentence fragments (unless referencing a class/type/proper noun/acronym). No fancy separators.
- Typing: `X | None` union syntax (not `Optional[X]`). Full annotations on all public API.
- Error messages: include type names (`type(...).__name__`), offending values, and "what to do next" guidance.

## Verification

Canonical quality gate (all must exit 0):

```bash
uv run ruff check src/ tests/                              # lint (46 families, preview=true)
uv run ruff format --check src/ tests/                     # formatting (2-space, single quotes, LF)
uv run ty check                                             # type checking (semantic correctness)
uv run ast-grep scan --config sgconfig.yml src/ tests/     # 20 custom rules
uv run pytest -x -v                                        # tests
```

The ast-grep rules enforce all prohibited patterns listed above deterministically.
No manual `rg` is needed for patterns covered by ast-grep (see `rules/`).
DRY rules (one canonical implementation, I/O through tracking/io.py, etc.) are guidelines enforced through code review and documentation, not ast-grep patterns. AST-based enforcement covers prohibited code patterns only.

Remaining `rg` checks (not covered by tooling):
- `rg 'from autopilot' src/autopilot/core/graph.py` -- must return 0 (graph.py isolation)
- `find src/ -name '__init__.py'` -- must return 0 (no init files)
- `rg 'import warnings' src/autopilot/` -- must return 0 (no warnings module in src)

## Safety

Never commit .env, API tokens, raw execution logs, or large generated outputs.
