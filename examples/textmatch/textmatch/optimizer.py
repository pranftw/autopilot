from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from pathlib import Path
import json


class RuleOptimizer(Optimizer):
  def __init__(self, parameters: list[Parameter], rules_dir: str, lr: float = 1.0):
    super().__init__(parameters, lr)
    self._rules_dir = rules_dir

  def step(self) -> None:
    for param in self._parameters:
      if not param.requires_grad or param.grad is None:
        continue
      grad = param.grad
      rules = self._load_rules()
      modified = False

      for entry in grad.missing_patterns:
        expected = entry.get('expected_category', '')
        text = entry.get('text', '')
        if expected and text:
          words = [w for w in text.lower().split() if len(w) > 3]
          if words:
            pattern = '|'.join(words[:3])
            rules.append(
              {
                'pattern': pattern,
                'category': expected,
                'priority': len(rules) + 1,
              }
            )
            modified = True

      for entry in grad.wrong_category:
        rule_idx = entry.get('rule_index', -1)
        expected = entry.get('expected', '')
        if 0 <= rule_idx < len(rules) and expected:
          current = rules[rule_idx]
          parts = current['pattern'].split('|')
          if len(parts) > 1:
            rules[rule_idx]['pattern'] = '|'.join(parts[:-1])
            modified = True

      if modified:
        self._save_rules(rules)

  def _load_rules(self) -> list[dict]:
    rules_path = Path(self._rules_dir) / 'rules.json'
    if not rules_path.exists():
      return []
    with open(rules_path) as f:
      return json.load(f)

  def _save_rules(self, rules: list[dict]) -> None:
    rules_path = Path(self._rules_dir) / 'rules.json'
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rules_path, 'w') as f:
      json.dump(rules, f, indent=2)
