from autopilot.ai.parameter import PathParameter
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.models import Datum
from autopilot.core.module import AutoPilotModule
from autopilot.core.parameter import Parameter
from dataclasses import dataclass
from pathlib import Path
from textmatch.optimizer import RuleOptimizer
from typing import Any
import json
import re


@dataclass
class RuleGradient:
  missing_patterns: list[dict]
  wrong_category: list[dict]
  ambiguous: list[dict]
  summary: dict


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
            'item_ids': [err.item_id],
          }
        )
      elif failure_type == 'wrong_category':
        wrong.append(
          {
            'rule_index': meta.get('matched_rule_index', -1),
            'got': meta.get('predicted', ''),
            'expected': meta.get('expected', ''),
            'item_id': err.item_id,
          }
        )
      elif failure_type == 'ambiguous':
        ambiguous.append(
          {
            'item_id': err.item_id,
            'matched_rules': meta.get('matched_rules', []),
          }
        )
    grad = RuleGradient(
      missing_patterns=missing,
      wrong_category=wrong,
      ambiguous=ambiguous,
      summary={
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
  def __init__(self):
    super().__init__()
    self._correct = 0
    self._total = 0
    self._per_category: dict[str, dict[str, int]] = {}

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

  def reset(self) -> None:
    self._correct = 0
    self._total = 0
    self._per_category = {}


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
    item_id = batch.item_id

    matched = []
    for i, rule in enumerate(rules):
      if re.search(rule['pattern'], text, re.IGNORECASE):
        matched.append((i, rule))

    if not matched:
      return Datum(
        success=False,
        item_id=item_id,
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
        item_id=item_id,
        metadata={
          'text': text,
          'predicted': predicted,
          'expected': expected,
        },
      )

    return Datum(
      success=False,
      item_id=item_id,
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
