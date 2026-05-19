"""Pretty-print formatting for ``Module`` instances."""

from typing import Any


def format_module_repr(module: Any) -> str:
  """Render a Lightning/nn.Module-style multi-line repr.

  Returns:
    Either a compact ``Type()`` line or an indented subtree listing.
  """
  extra = module.extra_repr()
  lines = [f'{type(module).__name__}(']
  if extra:
    lines[0] = f'{type(module).__name__}({extra}'
    if not module._modules and not module._parameters:
      return f'{type(module).__name__}({extra})'
  two_sp = '  '
  lines.extend(
    f'  ({name}): {repr(child).replace(chr(10), chr(10) + two_sp)}'
    for name, child in module._modules.items()
  )
  lines.extend(f'  ({name}): Parameter' for name in module._parameters)
  if len(lines) == 1:
    return f'{type(module).__name__}()'
  lines.append(')')
  return '\n'.join(lines)
