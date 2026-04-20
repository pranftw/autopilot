"""PathParameter: file-system parameters for optimizer scoping.

Declares which files/directories the optimizer is allowed to modify.
"""

from autopilot.core.models import Datum
from autopilot.core.parameter import Parameter
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

  source: str = ''
  pattern: str = '**/*'

  def __init__(
    self,
    source: str = '',
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

  def to_dict(self) -> dict[str, Any]:
    d = super().to_dict()
    d['source'] = self.source
    d['pattern'] = self.pattern
    return d

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'PathParameter':
    data = dict(data)
    source = data.pop('source', '')
    pattern = data.pop('pattern', '**/*')
    requires_grad = data.pop('requires_grad', True)
    items_raw = data.pop('items', [])
    items = [Datum.from_dict(item) for item in items_raw]
    param = cls(source=source, pattern=pattern, **data, items=items)
    param.requires_grad = requires_grad
    return param
