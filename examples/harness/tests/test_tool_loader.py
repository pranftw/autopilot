"""Tests for harness.tool_loader."""

from harness.database import RetailDB
from harness.tool_loader import TOOL_NAMES, HarnessDeps, load_tools, register_tools
from pathlib import Path
from unittest.mock import MagicMock
import pytest


class TestLoadToolsValidFile:
  def test_returns_callable_for_defined_tools(self, tmp_path: Path) -> None:
    tool_file = tmp_path / 'tools.py'
    tool_file.write_text(
      'def calculate(ctx, expression):\n  return str(eval(expression))\n'
      'def think(ctx, thought):\n  return ""\n',
      encoding='utf-8',
    )
    tools = load_tools(tool_file)
    assert 'calculate' in tools
    assert 'think' in tools
    assert callable(tools['calculate'])
    assert callable(tools['think'])

  def test_returns_only_known_tool_names(self, tmp_path: Path) -> None:
    tool_file = tmp_path / 'tools.py'
    tool_file.write_text(
      'def calculate(ctx, expr):\n  return "1"\ndef unknown_function():\n  pass\n',
      encoding='utf-8',
    )
    tools = load_tools(tool_file)
    assert 'calculate' in tools
    assert 'unknown_function' not in tools


class TestLoadToolsSyntaxError:
  def test_raises_runtime_error_on_bad_syntax(self, tmp_path: Path) -> None:
    tool_file = tmp_path / 'bad.py'
    tool_file.write_text('def broken(\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='syntax error'):
      load_tools(tool_file)


class TestLoadToolsMissingFunction:
  def test_partial_file_returns_subset(self, tmp_path: Path) -> None:
    tool_file = tmp_path / 'partial.py'
    tool_file.write_text(
      'def think(ctx, thought):\n  return ""\n',
      encoding='utf-8',
    )
    tools = load_tools(tool_file)
    assert 'think' in tools
    assert len(tools) == 1


class TestLoadToolsNamespaceAvailable:
  def test_tool_code_can_reference_namespace_types(self, tmp_path: Path) -> None:
    tool_file = tmp_path / 'ns_tools.py'
    tool_file.write_text(
      'def calculate(ctx: "RunContext[HarnessDeps]", expr: str) -> str:\n'
      '  assert RetailDB is not None\n'
      '  assert HarnessDeps is not None\n'
      '  assert RunContext is not None\n'
      '  return "ok"\n',
      encoding='utf-8',
    )
    tools = load_tools(tool_file)
    assert 'calculate' in tools
    result = tools['calculate'](MagicMock(), '2+2')
    assert result == 'ok'


class TestRegisterTools:
  def test_calls_agent_tool_for_each_function(self) -> None:
    mock_agent = MagicMock()
    tools = {
      'calculate': lambda ctx, e: '1',
      'think': lambda ctx, t: '',
    }
    register_tools(mock_agent, tools)
    assert mock_agent.tool.call_count == 2

  def test_passes_callables_through(self) -> None:
    mock_agent = MagicMock()

    def my_tool(ctx, x):
      return 'result'

    register_tools(mock_agent, {'calculate': my_tool})
    mock_agent.tool.assert_called_once_with(my_tool)

  def test_empty_tools_dict_no_calls(self) -> None:
    mock_agent = MagicMock()
    register_tools(mock_agent, {})
    mock_agent.tool.assert_not_called()


class TestHarnessDeps:
  def test_defaults(self) -> None:
    db = RetailDB()
    deps = HarnessDeps(db=db)
    assert deps.db is db
    assert deps.tool_log == []

  def test_tool_log_accumulates(self) -> None:
    deps = HarnessDeps(db=RetailDB())
    deps.tool_log.append({'tool': 'calculate', 'args': {'expression': '1+1'}})
    assert len(deps.tool_log) == 1


class TestToolNames:
  def test_sixteen_tools(self) -> None:
    assert len(TOOL_NAMES) == 16

  def test_sorted_order(self) -> None:
    assert TOOL_NAMES == tuple(sorted(TOOL_NAMES))

  def test_expected_names(self) -> None:
    expected = {
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
    }
    assert set(TOOL_NAMES) == expected


class TestLoadRealTools:
  def test_real_retail_tools_loads_sixteen(self) -> None:
    tools_path = Path(__file__).parent.parent / 'harness' / 'tools' / 'retail_tools.py'
    if not tools_path.exists():
      pytest.skip('retail_tools.py not found')
    tools = load_tools(tools_path)
    assert len(tools) == 16
    for name in TOOL_NAMES:
      assert name in tools, f'missing tool: {name}'
      assert callable(tools[name])


class TestToolLoaderDocumentation:
  def test_tool_loader_tool_names_documented(self) -> None:
    """Module docstring documents TOOL_NAMES coupling with retail_tools.py."""
    import harness.tool_loader as tl

    assert tl.__doc__ is not None
    assert 'TOOL_NAMES' in tl.__doc__
    assert 'retail_tools.py' in tl.__doc__
    assert 'must' in tl.__doc__.lower()
    assert 'added' in tl.__doc__.lower() or 'update' in tl.__doc__.lower()
