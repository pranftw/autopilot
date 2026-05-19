"""ClaudeCodeAgent: wraps Claude Code Agent SDK CLI via subprocess."""

from autopilot.ai.agents.agent import Agent, AgentResult
from autopilot.ai.runtime import RateLimiter
from autopilot.core.errors import AgentError
from typing import Any
import asyncio
import json
import subprocess

STDOUT_ERROR_SNIPPET_LEN = 200


class ClaudeCodeAgent(Agent):
  """Agent that runs Claude Code via the Agent SDK CLI (claude -p).

  The working directory (``cwd``) for the subprocess is resolved at call time:
  when a ``PathParameter`` is bound to a worktree (via ``bind()``), the agent
  should be configured with the worktree path so file edits land in the
  isolated directory. Use ``set_cwd(path)`` to update after bind, or pass
  ``cwd`` at construction time for a static working directory.
  """

  def __init__(
    self,
    allowed_tools: list[str] | None = None,
    model: str | None = None,
    permission_mode: str | None = None,
    append_system_prompt: str | None = None,
    cwd: str | None = None,
    limiter: RateLimiter | None = None,
    num_parallel: int = 1,
  ) -> None:
    """Configure Claude Code CLI flags and optional rate limiting.

    Args:
      allowed_tools: Default tools allow-list when context does not override.
      model: Default ``--model`` value when not overridden per call.
      permission_mode: Optional ``--permission-mode`` string for the CLI.
      append_system_prompt: Default appended system prompt unless context overrides.
      cwd: Working directory for the ``claude`` subprocess. When ``None``,
        inherits the parent process cwd. Updated via ``set_cwd()`` when
        PathParameters are bound to a worktree.
      limiter: Optional shared rate limiter for concurrent calls.
      num_parallel: Parallelism hint forwarded to :class:`Agent`.
    """
    super().__init__(limiter=limiter, num_parallel=num_parallel)
    self._allowed_tools = allowed_tools
    self._model = model
    self._permission_mode = permission_mode
    self._append_system_prompt = append_system_prompt
    self._cwd = cwd

  def set_cwd(self, cwd: str | None) -> None:
    """Update the working directory for subsequent subprocess calls.

    Typically called after PathParameter.bind() to align the agent's
    cwd with the worktree path.

    Args:
      cwd: New working directory, or ``None`` to inherit parent process cwd.
    """
    self._cwd = cwd

  def run(self, prompt: str, context: dict[str, Any] | None = None) -> AgentResult:
    """Sync execution via subprocess. Renamed from forward().

    Returns:
      Parsed :class:`AgentResult` from JSON stdout.

    Raises:
      AgentError: When the CLI is missing, exits non-zero, or returns invalid JSON.
    """
    ctx = context if context is not None else {}
    cmd = self._build_command(prompt, ctx)
    try:
      proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=self._cwd,
        check=False,
      )
    except FileNotFoundError as exc:
      msg = 'claude binary not found'
      raise AgentError(msg) from exc

    if proc.returncode != 0:
      msg = f'claude exited with code {proc.returncode}: {proc.stderr}'
      raise AgentError(msg)

    try:
      data = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
      msg = f'failed to parse claude output: {proc.stdout[:STDOUT_ERROR_SNIPPET_LEN]}'
      raise AgentError(msg) from exc

    return AgentResult(
      output=data['result'],
      session_id=data.get('session_id'),
      metadata={k: v for k, v in data.items() if k not in {'result', 'session_id'}},
    )

  async def async_run(self, prompt: str, context: dict[str, Any] | None = None) -> AgentResult:
    """Async wrapper -- runs sync subprocess in executor.

    Returns:
      Same :class:`AgentResult` as :meth:`run`.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, self.run, prompt, context)

  def _build_command(self, prompt: str, ctx: dict[str, Any]) -> list[str]:
    cmd = ['claude', '-p', prompt, '--output-format', 'json']

    session_id = ctx.get('session_id')
    if session_id:
      cmd.extend(['--resume', session_id])

    allowed = ctx.get('allowed_tools') if 'allowed_tools' in ctx else self._allowed_tools
    if allowed is not None:
      cmd.extend(['--allowedTools', ','.join(allowed) if allowed else ''])

    if self._permission_mode:
      cmd.extend(['--permission-mode', self._permission_mode])

    sp = ctx.get('system_prompt')
    system_prompt = self._append_system_prompt if sp is None else sp
    if system_prompt:
      cmd.extend(['--append-system-prompt', system_prompt])

    if self._model:
      cmd.extend(['--model', self._model])

    return cmd
