from autopilot.ai.parameter import PathParameter
from autopilot.core.gradient import Gradient
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.types import Datum
from autopilot.core.module import AutoPilotModule
from autopilot.core.parameter import Parameter
from dataclasses import dataclass, field
from pathlib import Path
from textmatch.optimizer import RuleOptimizer
from typing import Any
import json
import re


@dataclass
class RuleGradient(Gradient):
  missing_patterns: list[dict] = field(default_factory=list)
  wrong_category: list[dict] = field(default_factory=list)
  ambiguous: list[dict] = field(default_factory=list)

  def accumulate(self, other: 'RuleGradient') -> 'RuleGradient':
    combined_metadata = {**self.metadata, **other.metadata}
    combined_metadata['total_errors'] = self.metadata.get('total_errors', 0) + other.metadata.get(
      'total_errors', 0
    )
    return RuleGradient(
      missing_patterns=self.missing_patterns + other.missing_patterns,
      wrong_category=self.wrong_category + other.wrong_category,
      ambiguous=self.ambiguous + other.ambiguous,
      metadata=combined_metadata,
    )

  def render(self) -> str:
    parts: list[str] = [f'Total errors: {self.metadata.get("total_errors", 0)}']
    if self.missing_patterns:
      parts.append(f'Missing patterns ({len(self.missing_patterns)}):')
      for m in self.missing_patterns:
        parts.append(f'  - {m.get("expected_category")}: {m.get("text", "")[:80]}')
    if self.wrong_category:
      parts.append(f'Wrong category ({len(self.wrong_category)}):')
      for w in self.wrong_category:
        got, exp = w.get('got'), w.get('expected')
        parts.append(f'  - rule {w.get("rule_index")}: got {got}, expected {exp}')
    if self.ambiguous:
      parts.append(f'Ambiguous ({len(self.ambiguous)}):')
      for a in self.ambiguous:
        parts.append(f'  - {a.get("id")}: {len(a.get("matched_rules", []))} rules matched')
    return '\n'.join(parts)


class TextMatchLoss(Loss):
  def __init__(self, parameters: list[Parameter] | None = None):
    super().__init__(parameters)
    self._errors: list[Datum] = []

  def forward(self, data: Datum, targets: Any = None) -> None:
    if not data.success:
      self._errors.append(data)

  def backward(self) -> None:
    missing = []
    wrong = []
    ambiguous = []
    for err in self._errors:
      meta = err.metadata
      failure_type = meta.get('failure_type', '')
      if failure_type == 'no_match':
        missing.append(
          {
            'text': meta.get('text', ''),
            'expected_category': meta.get('expected', ''),
            'ids': [err.id],
          }
        )
      elif failure_type == 'wrong_category':
        wrong.append(
          {
            'rule_index': meta.get('matched_rule_index', -1),
            'got': meta.get('predicted', ''),
            'expected': meta.get('expected', ''),
            'id': err.id,
          }
        )
      elif failure_type == 'ambiguous':
        ambiguous.append(
          {
            'id': err.id,
            'matched_rules': meta.get('matched_rules', []),
          }
        )
    grad = RuleGradient(
      missing_patterns=missing,
      wrong_category=wrong,
      ambiguous=ambiguous,
      metadata={
        'total_errors': len(self._errors),
        'by_type': {
          'missing': len(missing),
          'wrong': len(wrong),
          'ambiguous': len(ambiguous),
        },
      },
    )
    for param in self._loss_parameters:
      if param.requires_grad:
        param.grad = grad

  def reset(self) -> None:
    self._errors = []


class AccuracyMetric(Metric):
  higher_is_better = True

  def __init__(self):
    super().__init__()
    self.add_state('_correct', 0)
    self.add_state('_total', 0)
    self.add_state('_per_category', dict)

  def update(self, datum: Datum) -> None:
    self._total += 1
    expected = datum.metadata.get('expected', '')
    if datum.success:
      self._correct += 1
    if expected:
      cat = self._per_category.setdefault(expected, {'correct': 0, 'total': 0})
      cat['total'] += 1
      if datum.success:
        cat['correct'] += 1

  def compute(self) -> dict[str, float]:
    accuracy = self._correct / self._total if self._total > 0 else 0.0
    return {
      'accuracy': accuracy,
      'total': float(self._total),
      'correct': float(self._correct),
    }


class TextMatchModule(AutoPilotModule):
  def __init__(self, rules_dir: str):
    super().__init__()
    self.rules = PathParameter(source=rules_dir, pattern='*.json')
    self.loss = TextMatchLoss([self.rules])
    self.accuracy = AccuracyMetric()
    self._rules_dir = rules_dir

  def _load_rules(self) -> list[dict]:
    rules_path = Path(self._rules_dir) / 'rules.json'
    if not rules_path.exists():
      return []
    with open(rules_path) as f:
      return json.load(f)

  def _classify(self, batch: Datum, rules: list[dict]) -> Datum:
    text = batch.metadata.get('text', '')
    expected = batch.metadata.get('expected', '')
    matched = []
    for i, rule in enumerate(rules):
      if re.search(rule['pattern'], text, re.IGNORECASE):
        matched.append((i, rule))

    if not matched:
      return Datum(
        success=False,
        metadata={
          'text': text,
          'predicted': '',
          'expected': expected,
          'failure_type': 'no_match',
        },
      )

    if len(matched) > 1:
      matched.sort(key=lambda x: x[1].get('priority', 999))

    best_idx, best_rule = matched[0]
    predicted = best_rule['category']

    if predicted == expected:
      return Datum(
        success=True,
        metadata={
          'text': text,
          'predicted': predicted,
          'expected': expected,
        },
      )

    return Datum(
      success=False,
      metadata={
        'text': text,
        'predicted': predicted,
        'expected': expected,
        'matched_rule_index': best_idx,
        'failure_type': 'wrong_category',
      },
    )

  def forward(self, batch: Datum) -> Datum:
    rules = self._load_rules()
    return self._classify(batch, rules)

  def training_step(self, batch: Any) -> Datum:
    return self.forward(batch)

  def validation_step(self, batch: Any) -> Datum:
    return self.forward(batch)

  def configure_optimizers(self):
    return RuleOptimizer([self.rules], rules_dir=self._rules_dir)
