"""AgentOptimizer: wraps an Agent to apply code changes based on param.grad.

Supports two modes controlled by the ``agentic`` flag:

  ``agentic=True`` (default): file-based feedback. Each ``step()`` call writes
  per-epoch feedback to ``{feedback_dir}/epoch_N.md``, maintains an in-memory
  todo list from gradient ``todo_items()``, builds a short task brief with
  file pointers, and passes that to the agent. The agent reads epoch files
  selectively via its file tools.

  ``agentic=False``: prompt-based feedback. Uses the original ``build_prompt()``
  to assemble all gradients into the agent prompt directly.
"""

from autopilot.ai.agents.agent import Agent
from autopilot.core.errors import ConfigError
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.journal import build_gradient_journal_row
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any


def _normalize_paths(
  paths: Sequence[str | Path] | None,
  anchor: Path,
) -> list[str]:
  """Resolve and normalize paths to POSIX strings against an anchor.

  Args:
    paths: Raw path inputs from the caller, or None.
    anchor: Absolute directory used as the resolution root.

  Returns:
    Sorted list of resolved POSIX path strings.
  """
  if paths is None:
    return []
  result: list[str] = []
  for p in paths:
    resolved = (anchor / Path(p)).resolve()
    result.append(PurePosixPath(resolved).as_posix())
  return sorted(result)


def _list_files_under_parameters(parameters: list[Parameter]) -> dict[str, float]:
  """Snapshot mtime+size for files under PathParameter patterns.

  Only inspects parameters that have a ``matched_files`` method
  (i.e. PathParameter instances).

  Args:
    parameters: List of parameters to inspect.

  Returns:
    Dict mapping resolved POSIX path to ``(mtime, size)`` encoded as a
    single float ``mtime + size * 1e-15`` for cheap comparison.
  """
  files: dict[str, float] = {}
  for param in parameters:
    matched = getattr(param, 'matched_files', None)
    if matched is None:
      continue
    for f in matched():
      key = PurePosixPath(f.resolve()).as_posix()
      try:
        stat = f.stat()
        files[key] = stat.st_mtime + stat.st_size * 1e-15
      except OSError:
        pass
  return files


def _check_path_violations(
  before: dict[str, float],
  after: dict[str, float],
  allowed: list[str],
  forbidden: list[str],
) -> str | None:
  """Find the first file that violates allow/deny path constraints.

  A file is checked when it is new or modified (mtime/size changed).
  Forbidden takes precedence: a path matching both allowed and forbidden
  is treated as forbidden.

  Args:
    before: Pre-step file snapshot from ``_list_files_under_parameters``.
    after: Post-step file snapshot.
    allowed: Normalized allowed path prefixes (empty = unrestricted).
    forbidden: Normalized forbidden path prefixes.

  Returns:
    First offending POSIX path string, or None when all edits are valid.
  """
  changed = set()
  for path, fingerprint in after.items():
    if path not in before or before[path] != fingerprint:
      changed.add(path)

  for path in sorted(changed):
    if _is_forbidden(path, forbidden):
      return path
    if allowed and not _is_allowed(path, allowed):
      return path
  return None


def _is_forbidden(path: str, forbidden: list[str]) -> bool:
  """Check whether a POSIX path falls under any forbidden prefix.

  Returns:
    True when ``path`` equals or is a child of any forbidden prefix.
  """
  return any(path == prefix or path.startswith(prefix + '/') for prefix in forbidden)


def _is_allowed(path: str, allowed: list[str]) -> bool:
  """Check whether a POSIX path falls under any allowed prefix.

  Returns:
    True when ``path`` equals or is a child of any allowed prefix.
  """
  return any(path == prefix or path.startswith(prefix + '/') for prefix in allowed)


class _TodoItem:
  """In-memory optimization todo item."""

  def __init__(self, text: str, epoch: int) -> None:
    self.text = text
    self.epoch = epoch
    self.addressed = False

  def __repr__(self) -> str:
    mark = 'x' if self.addressed else ' '
    return f'[{mark}] (epoch {self.epoch}) {self.text}'


class AgentOptimizer(Optimizer):
  """Optimizer that uses an Agent to apply updates from gradients.

  Calls param.render() and param.grad.render() to build the prompt.
  Zero isinstance checks against concrete parameter or gradient types.

  When used with ``Trainer``, ``feedback_dir`` is auto-wired from
  ``config.root`` via ``Trainer._ensure_agent_optimizer_context``.
  When used standalone with ``agentic=True``, ``feedback_dir`` must be
  passed explicitly or the agent must have a ``_cwd`` attribute; otherwise
  ``ConfigError`` is raised on first access.

  Anti-gaming path constraints (FR-009a):
    ``allowed_paths`` restricts the agent to a set of path prefixes.
    ``forbidden_paths`` blocks specific subtrees even if they fall under an
    allowed parent (forbidden always wins on overlap). Both are normalized
    to POSIX strings resolved against the config root anchor at init time
    and injected into ``build_context()`` so agents always see the rules.
    When ``validate_paths_after_step=True``, a filesystem scan after each
    successful agent step raises ``ConfigError`` if edits violate the
    constraints. Empty ``allowed_paths`` means unrestricted (no allow
    filtering); empty ``forbidden_paths`` means nothing forbidden.

  Public extension methods:
    build_prompt() -> str       -- assemble the full optimization prompt
    build_context() -> dict     -- provide context dict to the Agent
    update_context(**kwargs)    -- refresh epoch/metrics/collation_context between epochs
    write_epoch_feedback(epoch) -- write per-epoch feedback file (agentic mode)
    update_todo()               -- maintain in-memory todo from gradients
    build_task_brief() -> str   -- short task prompt with file pointers

  step() flow (agentic=True):
    1. Check if any parameter has a gradient
    2. write_epoch_feedback(epoch) -> writes .optimization/epoch_N.md
    3. update_todo() -> maintains in-memory todo from gradient todo_items()
    4. build_task_brief() -> short prompt with file pointers and inline todo
    5. agent.run(brief)
    6. (optional) validate_paths_after_step -> scan for violations
    7. Store prev_metrics, zero_grad()

  step() flow (agentic=False):
    1. Check if any parameter has a gradient
    2. build_prompt() -> full prompt with all gradients inline
    3. build_context() -> context dict
    4. agent.run(prompt, context=ctx)
    5. (optional) validate_paths_after_step -> scan for violations
    6. zero_grad()

  Checkpoint hooks (override base Optimizer):
    state_dict()            -- includes ``context`` key; agentic state is ephemeral
    load_state_dict(state)  -- restores base fields plus ``context`` when present

  Context keys used by build_prompt():
    epoch, metrics, collation_context (from CollationResult.context)
  """

  def __init__(
    self,
    agent: Agent,
    params: list[Parameter],
    lr: float = 1.0,
    context: dict[str, Any] | None = None,
    agentic: bool = True,
    feedback_dir: str | None = None,
    *,
    allowed_paths: Sequence[str | Path] | None = None,
    forbidden_paths: Sequence[str | Path] | None = None,
    validate_paths_after_step: bool = False,
  ) -> None:
    """Initializes the optimizer with an agent and parameter list.

    Args:
      agent: Agent instance used to apply gradient-driven updates.
      params: Parameters to optimize (passed to Optimizer base).
      lr: Learning rate (reserved for future use; passed to base class).
      context: Initial context dict for prompt building (epoch, metrics, etc.).
      agentic: When True (default), use file-based feedback. When False,
        use prompt-based feedback via build_prompt().
      feedback_dir: Override for the feedback file directory. When None,
        falls back to agent ``_cwd`` or raises ``ConfigError``.
      allowed_paths: Paths the agent may edit. Empty or None means all paths
        allowed (unless forbidden). Normalized to POSIX strings resolved
        against the config root anchor.
      forbidden_paths: Paths the agent must never edit. Takes precedence over
        ``allowed_paths`` on overlap. Empty or None means nothing forbidden.
      validate_paths_after_step: When True, scan filesystem after each agent
        step and raise ``ConfigError`` if edits violate allow/deny rules.
    """
    super().__init__(params, lr=lr)
    self._agent = agent
    self._context = context if context is not None else {}
    self._agentic = agentic
    self._feedback_dir_override = feedback_dir
    self._todo_items: list[_TodoItem] = []
    self._prev_metrics: dict[str, float] = {}
    anchor = self._resolve_anchor()
    self._allowed_paths = _normalize_paths(allowed_paths, anchor)
    self._forbidden_paths = _normalize_paths(forbidden_paths, anchor)
    self._validate_paths_after_step = validate_paths_after_step

  @property
  def owns_step_gradient_context(self) -> bool:
    """True when agentic mode is active (emits gradient context per step)."""
    return self._agentic

  @property
  def context(self) -> dict[str, Any]:
    """Current optimizer context dict for prompt building."""
    return self._context

  @context.setter
  def context(self, value: dict[str, Any]) -> None:
    self._context = value

  @property
  def feedback_dir(self) -> str | None:
    """Configured feedback directory override, or None."""
    return self._feedback_dir_override

  @feedback_dir.setter
  def feedback_dir(self, value: str | None) -> None:
    """Set the feedback directory override."""
    self._feedback_dir_override = value

  @property
  def _feedback_dir(self) -> str:
    """Directory for per-epoch feedback files.

    Returns configured override, agent cwd-based path, or raises
    ConfigError if neither is available.

    Raises:
      ConfigError: When agentic mode has no resolvable feedback directory.
    """
    if self._feedback_dir_override is not None:
      return self._feedback_dir_override
    cwd = getattr(self._agent, '_cwd', None)
    if cwd is not None:
      return str(Path(cwd) / '.optimization')
    msg = (
      'AgentOptimizer requires feedback_dir when agentic=True and agent '
      'has no _cwd. Pass feedback_dir= to the constructor or set agent._cwd.'
    )
    raise ConfigError(msg)

  def _resolve_anchor(self) -> Path:
    """Resolve the normalization anchor for allow/deny path prefixes.

    Uses ``Config.root`` from the context when available, else falls back
    to the current working directory.

    Returns:
      Absolute resolved directory path.
    """
    config = self._context.get('config')
    if config is not None:
      root = getattr(config, 'root', None)
      if root is not None:
        return Path(root).resolve()
    return Path.cwd().resolve()

  def _validate_post_step(self, before: dict[str, float]) -> None:
    """Validate filesystem changes against allow/deny constraints.

    Compares pre-step file snapshot with current state and raises
    ``ConfigError`` on the first violation. Called when
    ``validate_paths_after_step`` is True.

    Args:
      before: Pre-step snapshot from ``_list_files_under_parameters``.

    Raises:
      ConfigError: When a new or modified file violates path constraints,
        with the offending path in the message.
    """
    after = _list_files_under_parameters(self.parameters)
    offender = _check_path_violations(
      before,
      after,
      self._allowed_paths,
      self._forbidden_paths,
    )
    if offender is not None:
      msg = (
        f'post-step path violation: {offender} is outside allowed paths '
        f'or matches a forbidden path. Review allowed_paths and '
        f'forbidden_paths on AgentOptimizer.'
      )
      raise ConfigError(msg)

  def step(self) -> None:
    """Applies one optimization step via the wrapped Agent.

    No-ops when no parameter has a non-``None`` gradient. On success,
    clears gradients via :meth:`zero_grad`.

    Raises:
      Propagates any exception raised by ``agent.run``.
    """
    has_grads = False
    for param in self.parameters:
      if param.requires_grad and param.grad is not None:
        has_grads = True
        break
    if not has_grads:
      return

    if self._agentic:
      self._agentic_step()
    else:
      self._prompt_step()

  def _prompt_step(self) -> None:
    """Original prompt-based step: all gradients inline in the prompt."""
    prompt = self.build_prompt()
    ctx = self.build_context()
    before = (
      _list_files_under_parameters(self.parameters) if self._validate_paths_after_step else None
    )
    if self._agent.limiter:
      self._agent.limiter.acquire()
    result = self._agent.run(prompt, context=ctx)
    if result and result.output:
      if before is not None:
        self._validate_post_step(before)
      self.zero_grad()

  def _agentic_step(self) -> None:
    """File-based agentic step: write feedback, build brief, run agent.

    After a successful agent run (agent produces output), emits a context
    entry via ``trainer.emit_context()`` with truncated gradient summaries
    in metadata. Emission requires ``'trainer'`` in the context dict and
    at least one parameter with a non-``None`` gradient. Silent skip when
    either condition is unmet.

    When ``validate_paths_after_step`` is True, scans the filesystem after
    the agent run and raises ``ConfigError`` if edits leave the allowed
    region or touch forbidden paths.
    """
    epoch = self._context.get('epoch', 0)
    self.write_epoch_feedback(epoch)
    self.update_todo()
    brief = self.build_task_brief()

    before = (
      _list_files_under_parameters(self.parameters) if self._validate_paths_after_step else None
    )
    if self._agent.limiter:
      self._agent.limiter.acquire()
    result = self._agent.run(brief, context=self.build_context())

    if result and result.output:
      if before is not None:
        self._validate_post_step(before)
      self._emit_step_context()
      self._prev_metrics = dict(self._context.get('metrics', {}))
      self.zero_grad()
    else:
      self._emit_failure_context()

  def _emit_step_context(self) -> None:
    """Emit a context entry after a successful agentic step.

    Builds structured gradient summaries as ``list[dict[str, str]]`` with
    keys ``param_name``, ``param_type``, ``gradient_type``, ``summary``
    (capped at ``GRAD_SUMMARY_MAX_CHARS`` per summary) via
    :func:`~autopilot.core.trainer.journal.build_gradient_journal_row`.
    Calls ``trainer.emit_context()`` when a trainer reference is available
    and at least one gradient summary exists. Silent no-op when
    ``'trainer'`` is missing from context or no gradients to summarize.

    Uses ``trainer._module.named_parameters()`` for stable module-relative
    names. Falls back to positional naming when no module is available.
    """
    trainer_ref = self._context.get('trainer')
    if trainer_ref is None:
      return
    module = getattr(trainer_ref, '_module', None)
    if module is not None:
      grad_summaries: list[dict[str, str]] = [
        build_gradient_journal_row(name, param)
        for name, param in module.named_parameters()
        if param.requires_grad and param.grad is not None
      ]
    else:
      grad_summaries = [
        build_gradient_journal_row(f'param_{i}', param)
        for i, param in enumerate(self.parameters)
        if param.requires_grad and param.grad is not None
      ]
    if not grad_summaries:
      return
    trainer_ref.emit_context(
      'optimizer applied changes based on gradient feedback',
      source='agent-optimizer',
      metadata={'gradient_summaries': grad_summaries},
    )

  def _emit_failure_context(self) -> None:
    """Emit a context entry when an agentic step fails to produce output.

    Silent no-op when ``'trainer'`` is missing from context.
    """
    trainer_ref = self._context.get('trainer')
    if trainer_ref is None:
      return
    epoch = self._context.get('epoch', 0)
    trainer_ref.emit_context(
      f'optimizer step failed: agent produced no output at epoch {epoch}',
      source='agent-optimizer',
      metadata={'epoch': epoch},
    )

  def write_epoch_feedback(self, epoch: int) -> Path:
    """Write current epoch gradients and context to a markdown file.

    Args:
      epoch: Current epoch number.

    Returns:
      Path to the written epoch file.
    """
    parts: list[str] = [f'# Epoch {epoch}']

    metrics = self._context.get('metrics')
    if metrics:
      parts.append('\n## Metrics')
      for key, val in metrics.items():
        parts.append(f'{key}: {val}')

    collation_context = self._context.get('collation_context')
    if collation_context:
      parts.append(f'\n## Direction\n{collation_context}')

    for param in self.parameters:
      if not param.requires_grad or param.grad is None:
        continue
      parts.extend([f'\n## Parameter: {param.id}', param.grad.render()])

    opt_dir = Path(self._feedback_dir)
    opt_dir.mkdir(parents=True, exist_ok=True)
    path = opt_dir / f'epoch_{epoch}.md'
    path.write_text('\n'.join(parts), encoding='utf-8')
    return path

  def update_todo(self) -> None:
    """Update the in-memory optimization todo from current gradients.

    Marks addressed items when metrics improve, adds new items via
    ``param.grad.todo_items()``. The todo is inlined in ``build_task_brief()``
    rather than written to a file.
    """
    epoch = self._context.get('epoch', 0)
    if self._prev_metrics and self._context:
      self._mark_addressed_items(self._context.get('metrics', {}))
    new_items = self._extract_todo_items(epoch)
    self._merge_todo_items(new_items)

  def build_task_brief(self) -> str:
    """Build a short task prompt with inline todo and epoch file pointers.

    Returns:
      Concise prompt with goal, current metrics, inline todo, and file
      pointers. The agent reads detailed feedback from epoch files.
    """
    parts: list[str] = ['Apply improvements based on optimization feedback.']
    epoch = self._context.get('epoch')
    metrics = self._context.get('metrics')
    if epoch is not None:
      parts.append(f'\nCurrent epoch: {epoch}')
    if metrics:
      parts.append(f'Current metrics: {metrics}')

    todo_text = self._render_todo()
    if todo_text:
      parts.append(f'\n## Todo\n{todo_text}')

    parts.extend(
      [
        f'\nDetailed per-epoch feedback is in {self._feedback_dir}/epoch_*.md',
        'Read what you need and make targeted improvements to the parameter files.',
      ]
    )
    return '\n'.join(parts)

  def _render_todo(self) -> str:
    """Render in-memory todo items as a checklist string.

    Returns:
      Newline-joined checklist, or empty string when no items exist.
    """
    if not self._todo_items:
      return ''
    lines: list[str] = []
    for item in self._todo_items:
      mark = 'x' if item.addressed else ' '
      lines.append(f'- [{mark}] {item.text}')
    return '\n'.join(lines)

  def _mark_addressed_items(self, current_metrics: dict[str, float]) -> None:
    """Mark todo items as addressed when metrics improve."""
    if not self._prev_metrics or not current_metrics:
      return
    improved = False
    for key, value in current_metrics.items():
      prev = self._prev_metrics.get(key)
      if prev is not None and value > prev:
        improved = True
        break
    if improved:
      for item in self._todo_items:
        if not item.addressed:
          item.addressed = True

  def _extract_todo_items(self, epoch: int) -> list[_TodoItem]:
    """Extract new todo items from current parameter gradients.

    Returns:
      List of new todo items derived from gradient ``todo_items()``.
    """
    items: list[_TodoItem] = []
    for param in self.parameters:
      if not param.requires_grad or param.grad is None:
        continue
      items.extend(_TodoItem(text=text, epoch=epoch) for text in param.grad.todo_items() if text)
    return items

  def _merge_todo_items(self, new_items: list[_TodoItem]) -> None:
    """Merge new items into the existing todo, avoiding exact duplicates."""
    existing_texts = {item.text for item in self._todo_items}
    for item in new_items:
      if item.text not in existing_texts:
        self._todo_items.append(item)
        existing_texts.add(item.text)

  def build_prompt(self) -> str:
    """Assembles the full optimization prompt from gradients and context.

    Iterates parameters with non-``None`` gradients, renders each via
    ``param.render()`` and ``param.grad.render()``, and prepends epoch /
    metrics / collation context if present.

    Returns:
      Newline-joined prompt string for the agent.
    """
    parts: list[str] = ['Apply the following improvements based on feedback:']

    epoch = self._context.get('epoch')
    metrics = self._context.get('metrics')
    collation_context = self._context.get('collation_context')

    if epoch is not None:
      parts.append(f'\nCurrent epoch: {epoch}')
    if metrics:
      parts.append(f'Current metrics: {metrics}')
    if collation_context:
      parts.append(f'\n## Overall Direction\n{collation_context}')

    for param in self.parameters:
      if not param.requires_grad or param.grad is None:
        continue
      parts.append(f'\n--- Parameter {param.id} ---')
      desc = param.render()
      if desc:
        parts.append(desc)
      parts.append(param.grad.render())

    return '\n'.join(parts)

  def build_context(self) -> dict[str, Any]:
    """Returns a shallow copy of the current optimizer context.

    Includes ``allowed_paths`` and ``forbidden_paths`` keys so agents
    always see the active path constraints (empty lists when unconfigured).

    Returns:
      Dict with keys such as ``epoch``, ``metrics``, ``collation_context``,
      ``allowed_paths``, ``forbidden_paths``.
    """
    ctx = dict(self._context)
    ctx['allowed_paths'] = list(self._allowed_paths)
    ctx['forbidden_paths'] = list(self._forbidden_paths)
    return ctx

  def update_context(self, **kwargs: Any) -> None:
    """Update optimizer context between epochs (e.g. new metrics)."""
    self._context.update(kwargs)

  def state_dict(self) -> dict[str, Any]:
    """Serialize optimizer state including agent context.

    Agentic state (in-memory todo, prev_metrics) is ephemeral and not
    included. After load_state_dict(), the todo starts empty.

    Returns:
      Base optimizer dict plus ``context`` key with the current context mapping.
    """
    state = super().state_dict()
    state['context'] = dict(self._context)
    return state

  def load_state_dict(self, state: dict[str, Any]) -> None:
    """Restore optimizer state including agent context.

    Args:
      state: Dict previously returned by :meth:`state_dict`. Base fields are
        restored by ``super()``. The ``context`` key, when present, replaces
        the entire context mapping.
    """
    super().load_state_dict(state)
    if 'context' in state:
      self._context = dict(state['context'])
