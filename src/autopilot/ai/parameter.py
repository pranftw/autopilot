"""PathParameter: file-system parameters for optimizer scoping.

Declares which files/directories the optimizer is allowed to modify.
"""

from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from pathlib import Path
from typing import Any


class PathParameter(Parameter):
  """File-system parameter declaring mutable scope.

  source: path on disk (directory or file)
  pattern: glob for which files within source are mutable

  Example::

    class MyModule(Module):
      def __init__(self):
        super().__init__()
        self.prompts = PathParameter(source='~/project/prompts', pattern='**/*.md')
        self.config = PathParameter(source='~/project/config', pattern='*.tf')
  """

  source: str
  pattern: str = '**/*'

  def __init__(
    self,
    source: str,
    pattern: str = '**/*',
    **kwargs: Any,
  ) -> None:
    super().__init__(**kwargs)
    object.__setattr__(self, 'source', source)
    object.__setattr__(self, 'pattern', pattern)

  def matched_files(self) -> list[Path]:
    """List files matching the pattern within source."""
    source_path = Path(self.source).expanduser()
    if not source_path.exists():
      return []
    if source_path.is_file():
      return [source_path]
    return sorted(source_path.glob(self.pattern))

  def render(self) -> str:
    files = self.matched_files()
    if not files:
      return ''
    parts = [f'Editable files ({self.source}):']
    for f in files[:20]:
      parts.append(f'  - {f}')
    return '\n'.join(parts)

  def snapshot(self) -> dict[str, str]:
    result: dict[str, str] = {}
    root = Path(self.source)
    for f in self.matched_files():
      key = str(f.relative_to(root))
      result[key] = f.read_text(encoding='utf-8')
    return result

  def restore(self, content: dict[str, str]) -> None:
    root = Path(self.source)
    for rel_path, text in content.items():
      target = root / rel_path
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_text(text, encoding='utf-8')

  def to_dict(self) -> dict[str, Any]:
    d = super().to_dict()
    d['source'] = self.source
    d['pattern'] = self.pattern
    return d

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'PathParameter':
    data = dict(data)
    stored_id = data.pop('id', None)
    source = data.pop('source', '')
    pattern = data.pop('pattern', '**/*')
    requires_grad = data.pop('requires_grad', True)
    items_raw = data.pop('items', [])
    items = [Datum.from_dict(item) for item in items_raw]
    param = cls(source=source, pattern=pattern, **data, items=items)
    param.requires_grad = requires_grad
    if stored_id:
      object.__setattr__(param, '_id', stored_id)
    return param
