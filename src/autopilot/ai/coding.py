"""ClaudeCodeAgent: wraps Claude Code Agent SDK CLI via subprocess."""

from autopilot.ai.agent import Agent, AgentResult
from autopilot.core.errors import AgentError
from typing import Any
import json
import subprocess


class ClaudeCodeAgent(Agent):
  """Agent that runs Claude Code via the Agent SDK CLI (claude -p)."""

  def __init__(
    self,
    allowed_tools: list[str] | None = None,
    model: str | None = None,
    permission_mode: str | None = None,
    append_system_prompt: str | None = None,
    cwd: str | None = None,
  ) -> None:
    self._allowed_tools = allowed_tools
    self._model = model
    self._permission_mode = permission_mode
    self._append_system_prompt = append_system_prompt
    self._cwd = cwd

  def forward(self, prompt: str, context: dict[str, Any] | None = None) -> AgentResult:
    ctx = context or {}
    cmd = self._build_command(prompt, ctx)
    try:
      proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=self._cwd,
      )
    except FileNotFoundError:
      raise AgentError('claude binary not found')

    if proc.returncode != 0:
      raise AgentError(f'claude exited with code {proc.returncode}: {proc.stderr}')

    try:
      data = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
      raise AgentError(f'failed to parse claude output: {proc.stdout[:200]}')

    return AgentResult(
      output=data.get('result', ''),
      session_id=data.get('session_id'),
      metadata={k: v for k, v in data.items() if k not in ('result', 'session_id')},
    )

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

    system_prompt = ctx.get('system_prompt') or self._append_system_prompt
    if system_prompt:
      cmd.extend(['--append-system-prompt', system_prompt])

    if self._model:
      cmd.extend(['--model', self._model])

    return cmd
