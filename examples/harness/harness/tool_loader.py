"""Dynamic tool loading and registration for the harness agent.

Tools are defined in ``harness/tools/retail_tools.py`` -- a PathParameter
target that the optimizer rewrites between epochs.  ``load_tools`` ``exec``s
the file in a sandboxed namespace exposing ``RetailDB``, ``HarnessDeps``,
and ``RunContext`` so tool code can reference these types without imports.
Partial files (only some tool names defined) are handled gracefully; missing
tools are simply omitted from the returned dict.

TOOL_NAMES coupling
-------------------

``TOOL_NAMES`` is the authoritative registry of tool function names that
``load_tools`` will extract from ``retail_tools.py``.  Adding a new public
tool function to ``harness/tools/retail_tools.py`` is **not sufficient** on
its own -- the function name **must also be added** to the ``TOOL_NAMES``
tuple in this module.  The loader only extracts functions whose names appear
in ``TOOL_NAMES``; unlisted functions are silently ignored.

This explicit-registration design is intentional: it keeps the boundary
between optimizable file content (which the ``AgentOptimizer`` rewrites)
and the application loader (which is application code outside the versioned
``PathParameter`` artifact) clearly defined.  Auto-discovery via
``inspect`` is deliberately not used.
"""

from dataclasses import dataclass, field
from harness.database import RetailDB
from pathlib import Path
from pydantic_ai import RunContext
from typing import Any, Callable


@dataclass
class HarnessDeps:
  """Pydantic AI dependency injection container.

  Attributes:
    db: The in-memory retail database for the current scenario run.
    tool_log: Accumulates structured dicts for each tool call made
      during the conversation (tool name, arguments, result metadata).
  """

  db: RetailDB
  tool_log: list[dict] = field(default_factory=list)


TOOL_NAMESPACE: dict[str, Any] = {
  'RetailDB': RetailDB,
  'HarnessDeps': HarnessDeps,
  'RunContext': RunContext,
}

# any new public tool added to harness/tools/retail_tools.py must also be
# listed here; the loader only extracts functions whose names appear in this
# tuple (see module docstring "TOOL_NAMES coupling" section)
TOOL_NAMES: tuple[str, ...] = (
  'calculate',
  'cancel_pending_order',
  'exchange_delivered_order_items',
  'find_user_id_by_email',
  'find_user_id_by_name_zip',
  'get_order_details',
  'get_product_details',
  'get_user_details',
  'list_all_product_types',
  'modify_pending_order_address',
  'modify_pending_order_items',
  'modify_pending_order_payment',
  'modify_user_address',
  'return_delivered_order_items',
  'think',
  'transfer_to_human_agents',
)


def load_tools(tools_path: Path) -> dict[str, Callable]:
  """Load tool functions from a Python file (PathParameter target).

  Exec's the file in a sandboxed namespace seeded with ``TOOL_NAMESPACE``,
  then extracts callable objects matching known ``TOOL_NAMES``.

  Args:
    tools_path: Path to the retail_tools.py file.

  Returns:
    Dict mapping tool name to callable for each defined tool.

  Raises:
    RuntimeError: If the file has syntax errors.
  """
  code = tools_path.read_text(encoding='utf-8')
  namespace = dict(TOOL_NAMESPACE)
  try:
    exec(code, namespace)
  except SyntaxError as exc:
    raise RuntimeError(f'Tool file has syntax error: {exc}') from exc
  return {name: namespace[name] for name in TOOL_NAMES if name in namespace}


def register_tools(agent: Any, tools: dict[str, Callable]) -> None:
  """Register loaded tool functions on a Pydantic AI agent.

  Uses ``agent.tool(fn)`` for each tool so that the first parameter
  receives ``RunContext[HarnessDeps]`` with dependency injection.

  Args:
    agent: A ``pydantic_ai.Agent`` instance.
    tools: Dict of tool name -> callable (from ``load_tools``).
  """
  for fn in tools.values():
    agent.tool(fn)
