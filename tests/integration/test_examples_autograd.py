"""Integration tests for examples migration to graph-based autograd.

Validates:
  - TextMatch E2E training with graph backward (no external API required).
  - Protim gradient flow with mocked agent (no Claude CLI required).
  - Gradients reach parameters through AccumulateGrad, not direct assignment.
  - Banned pattern regression guards (no param.grad=, no self.forward(batch),
    no Datum(success=) in examples/).
"""

from autopilot.ai.parameter import PathParameter
from autopilot.core.gradient import Gradient
from autopilot.core.types import EvalDatum
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import re
import sys

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / 'examples'
PROTIM_DIR = EXAMPLES_DIR / 'protim'
TEXTMATCH_DIR = EXAMPLES_DIR / 'textmatch'


def _read_python_files(directory: Path) -> dict[str, str]:
  """Read all .py files under directory, excluding .venv and __pycache__."""
  result = {}
  for p in directory.rglob('*.py'):
    if '.venv' in p.parts or '__pycache__' in p.parts:
      continue
    result[str(p.relative_to(directory))] = p.read_text(encoding='utf-8')
  return result


class TestTextMatchGradientFlow:
  """TextMatch module can run forward -> loss -> backward with graph-based gradients."""

  def test_textmatch_params_get_gradient(self, tmp_path):
    rules_dir = tmp_path / 'rules'
    rules_dir.mkdir()
    rules_json = rules_dir / 'rules.json'
    rules_json.write_text(
      json.dumps(
        [
          {'pattern': 'hello', 'category': 'greeting', 'priority': 1},
        ]
      )
    )

    sys.path.insert(0, str(TEXTMATCH_DIR))
    try:
      from textmatch.module import TextMatchModule

      module = TextMatchModule(str(rules_dir))
      param = module.rules
      assert param.grad is None

      batch = EvalDatum(metadata={'text': 'goodbye world', 'expected': 'farewell'})

      module.train()
      data = module(batch)
      assert data.grad_fn is not None, 'self(batch) should produce Datum with grad_fn'

      loss = module.loss
      loss(data, batch)
      loss.backward()

      assert param.grad is not None, 'parameter should have gradient after backward'
      assert isinstance(param.grad, Gradient)
    finally:
      sys.path.pop(0)

  def test_textmatch_training_runs(self, tmp_path):
    rules_dir = tmp_path / 'rules'
    rules_dir.mkdir()
    rules_json = rules_dir / 'rules.json'
    rules_json.write_text(
      json.dumps(
        [
          {'pattern': 'urgent|asap', 'category': 'high_priority', 'priority': 1},
          {'pattern': 'question|help', 'category': 'support', 'priority': 2},
        ]
      )
    )

    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()
    train_data = [
      {'text': 'urgent request', 'expected_category': 'high_priority'},
      {'text': 'need help please', 'expected_category': 'support'},
      {'text': 'asap delivery', 'expected_category': 'high_priority'},
    ]
    (datasets_dir / 'train.jsonl').write_text('\n'.join(json.dumps(item) for item in train_data))
    (datasets_dir / 'val.jsonl').write_text('\n'.join(json.dumps(item) for item in train_data[:1]))

    sys.path.insert(0, str(TEXTMATCH_DIR))
    try:
      from textmatch.data import TextMatchDataModule
      from textmatch.module import TextMatchModule

      module = TextMatchModule(str(rules_dir))
      dm = TextMatchDataModule(str(datasets_dir))

      loss = module.loss
      metric = module.accuracy

      module.train()
      train_loader = dm.train_dataloader()
      metric.reset()
      loss.reset()

      for batch in train_loader:
        data = module(batch)
        loss(data, batch)
        metric.update(data)

      loss.backward()
      train_metrics = metric.compute()

      assert 'accuracy' in train_metrics
      assert module.rules.grad is not None
      assert isinstance(module.rules.grad, Gradient)
    finally:
      sys.path.pop(0)

  def test_textmatch_multiple_batches_accumulate(self, tmp_path):
    rules_dir = tmp_path / 'rules'
    rules_dir.mkdir()
    (rules_dir / 'rules.json').write_text(
      json.dumps(
        [
          {'pattern': 'hello', 'category': 'greeting', 'priority': 1},
        ]
      )
    )

    sys.path.insert(0, str(TEXTMATCH_DIR))
    try:
      from textmatch.module import TextMatchModule

      module = TextMatchModule(str(rules_dir))

      module.train()
      batches = [
        EvalDatum(metadata={'text': 'hello friend', 'expected': 'greeting'}),
        EvalDatum(metadata={'text': 'goodbye world', 'expected': 'farewell'}),
        EvalDatum(metadata={'text': 'unknown input', 'expected': 'other'}),
      ]

      loss = module.loss
      loss.reset()
      for batch in batches:
        data = module(batch)
        loss(data, batch)

      loss.backward()
      assert module.rules.grad is not None
    finally:
      sys.path.pop(0)


class TestProtimGradientFlow:
  """Protim module gradient flow with mocked ClaudeCodeAgent."""

  def test_protim_params_get_gradient(self, tmp_path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.txt').write_text('You are a QA assistant.')

    sys.path.insert(0, str(PROTIM_DIR))
    try:
      mock_result = MagicMock()
      mock_result.output = 'Paris'

      with patch(
        'protim.module.ClaudeCodeAgent',
        return_value=MagicMock(run=MagicMock(return_value=mock_result)),
      ):
        from protim.module import PromptModule

        module = PromptModule(str(prompts_dir))
        param = module.prompt
        assert param.grad is None

        batch = EvalDatum(
          metadata={'question': 'Capital of France?', 'expected': 'Paris'},
        )

        module.train()
        data = module(batch)
        assert data.grad_fn is not None

        loss = module.loss
        loss(data, batch)
        loss.backward()

        assert param.grad is not None
        assert isinstance(param.grad, Gradient)
    finally:
      sys.path.pop(0)

  def test_protim_failure_produces_gradient_with_direction(self, tmp_path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.txt').write_text('You are a QA assistant.')

    sys.path.insert(0, str(PROTIM_DIR))
    try:
      mock_result = MagicMock()
      mock_result.output = 'London'

      with patch(
        'protim.module.ClaudeCodeAgent',
        return_value=MagicMock(run=MagicMock(return_value=mock_result)),
      ):
        from protim.module import PromptModule

        module = PromptModule(str(prompts_dir))

        batch = EvalDatum(
          metadata={'question': 'Capital of France?', 'expected': 'Paris'},
        )

        module.train()
        data = module(batch)

        assert not data.success

        loss = module.loss
        loss(data, batch)
        loss.backward()

        grad = module.prompt.grad
        assert isinstance(grad, Gradient)
        rendered = grad.render()
        assert 'incorrectly' in rendered
    finally:
      sys.path.pop(0)

  def test_protim_training_runs(self, tmp_path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.txt').write_text('You are a QA assistant.')

    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()
    train_data = [
      {'question': 'Capital of France?', 'expected': 'Paris'},
      {'question': 'Capital of Germany?', 'expected': 'Berlin'},
    ]
    (datasets_dir / 'train.jsonl').write_text('\n'.join(json.dumps(item) for item in train_data))
    (datasets_dir / 'val.jsonl').write_text('\n'.join(json.dumps(item) for item in train_data[:1]))

    sys.path.insert(0, str(PROTIM_DIR))
    try:
      mock_result = MagicMock()
      mock_result.output = 'Paris'

      with patch(
        'protim.module.ClaudeCodeAgent',
        return_value=MagicMock(run=MagicMock(return_value=mock_result)),
      ):
        from protim.data import QADataModule
        from protim.module import PromptModule

        module = PromptModule(str(prompts_dir))
        dm = QADataModule(str(datasets_dir))

        loss = module.loss
        metric = module.accuracy

        module.train()
        train_loader = dm.train_dataloader()
        metric.reset()
        loss.reset()

        for batch in train_loader:
          data = module(batch)
          loss(data, batch)
          metric.update(data)

        loss.backward()
        train_metrics = metric.compute()

        assert 'accuracy' in train_metrics
        assert module.prompt.grad is not None
        assert isinstance(module.prompt.grad, Gradient)
    finally:
      sys.path.pop(0)


class TestBannedPatterns:
  """Regression guards: no direct param.grad assignment, no self.forward(batch)
  in step methods, no Datum(success=) in examples/."""

  def test_protim_no_direct_grad_assignment(self):
    files = _read_python_files(PROTIM_DIR)
    pattern = re.compile(r'param\.grad\s*=')
    violations = []
    for fname, content in files.items():
      for i, line in enumerate(content.splitlines(), 1):
        if pattern.search(line):
          violations.append(f'{fname}:{i}: {line.strip()}')
    assert not violations, 'Direct param.grad assignment found in protim:\n' + '\n'.join(violations)

  def test_textmatch_no_direct_grad_assignment(self):
    files = _read_python_files(TEXTMATCH_DIR)
    pattern = re.compile(r'param\.grad\s*=')
    violations = []
    for fname, content in files.items():
      for i, line in enumerate(content.splitlines(), 1):
        if pattern.search(line):
          violations.append(f'{fname}:{i}: {line.strip()}')
    assert not violations, 'Direct param.grad assignment found in textmatch:\n' + '\n'.join(
      violations
    )

  def test_examples_use_self_call_not_forward(self):
    pattern = re.compile(r'self\.forward\(batch\)')
    violations = []
    for example_dir in [PROTIM_DIR, TEXTMATCH_DIR]:
      files = _read_python_files(example_dir)
      for fname, content in files.items():
        in_step = False
        for i, line in enumerate(content.splitlines(), 1):
          stripped = line.strip()
          if stripped.startswith(('def training_step', 'def validation_step')):
            in_step = True
          elif stripped.startswith('def ') and in_step:
            in_step = False
          if in_step and pattern.search(line):
            rel = example_dir.name + '/' + fname
            violations.append(f'{rel}:{i}: {stripped}')
    assert not violations, 'self.forward(batch) in step methods:\n' + '\n'.join(violations)

  def test_no_bare_datum_success_constructor(self):
    pattern = re.compile(r'Datum\(success=')
    violations = []
    for example_dir in [PROTIM_DIR, TEXTMATCH_DIR]:
      files = _read_python_files(example_dir)
      for fname, content in files.items():
        for i, line in enumerate(content.splitlines(), 1):
          if pattern.search(line):
            rel = example_dir.name + '/' + fname
            violations.append(f'{rel}:{i}: {line.strip()}')
    assert not violations, 'Datum(success=...) found (use EvalDatum instead):\n' + '\n'.join(
      violations
    )


class TestLossBaseClassIntegration:
  """Verify Loss subclass contracts: super().forward() tracks _last_data,
  compute_seed_gradient() returns a Gradient seed, reset() clears all state."""

  def test_prompt_loss_tracks_last_data(self, tmp_path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.txt').write_text('test')

    sys.path.insert(0, str(PROTIM_DIR))
    try:
      from protim.module import PromptLoss

      param = PathParameter(source=str(prompts_dir), pattern='*.txt')
      loss = PromptLoss([param])

      datum = EvalDatum(success=False, metadata={'question': 'q', 'expected': 'a', 'actual': 'b'})
      loss.forward(datum)

      assert loss._last_data is datum
      assert len(loss._accumulated) == 1
      assert len(loss._failures) == 1
    finally:
      sys.path.pop(0)

  def test_prompt_loss_compute_seed_gradient(self, tmp_path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.txt').write_text('test')

    sys.path.insert(0, str(PROTIM_DIR))
    try:
      from protim.module import PromptLoss

      param = PathParameter(source=str(prompts_dir), pattern='*.txt')
      loss = PromptLoss([param])

      datum = EvalDatum(success=False, metadata={'question': 'q', 'expected': 'a', 'actual': 'b'})
      loss.forward(datum)

      seed = loss.compute_seed_gradient()
      assert isinstance(seed, Gradient)
      rendered = seed.render()
      assert 'incorrectly' in rendered
    finally:
      sys.path.pop(0)

  def test_prompt_loss_reset_clears_all(self, tmp_path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.txt').write_text('test')

    sys.path.insert(0, str(PROTIM_DIR))
    try:
      from protim.module import PromptLoss

      param = PathParameter(source=str(prompts_dir), pattern='*.txt')
      loss = PromptLoss([param])

      datum = EvalDatum(success=False, metadata={'question': 'q', 'expected': 'a', 'actual': 'b'})
      loss.forward(datum)
      loss.reset()

      assert loss._last_data is None
      assert loss._accumulated == []
      assert loss._failures == []
    finally:
      sys.path.pop(0)

  def test_textmatch_loss_tracks_last_data(self, tmp_path):
    rules_dir = tmp_path / 'rules'
    rules_dir.mkdir()

    sys.path.insert(0, str(TEXTMATCH_DIR))
    try:
      from textmatch.module import TextMatchLoss

      param = PathParameter(source=str(rules_dir), pattern='*.json')
      loss = TextMatchLoss([param])

      datum = EvalDatum(
        success=False,
        metadata={'text': 'x', 'expected': 'y', 'failure_type': 'no_match'},
      )
      loss.forward(datum)

      assert loss._last_data is datum
      assert len(loss._accumulated) == 1
      assert len(loss._errors) == 1
    finally:
      sys.path.pop(0)

  def test_textmatch_loss_compute_seed_gradient(self, tmp_path):
    rules_dir = tmp_path / 'rules'
    rules_dir.mkdir()

    sys.path.insert(0, str(TEXTMATCH_DIR))
    try:
      from textmatch.module import TextMatchLoss

      param = PathParameter(source=str(rules_dir), pattern='*.json')
      loss = TextMatchLoss([param])

      datum = EvalDatum(
        success=False,
        metadata={'text': 'x', 'expected': 'y', 'failure_type': 'no_match'},
      )
      loss.forward(datum)

      seed = loss.compute_seed_gradient()
      assert isinstance(seed, Gradient)
      rendered = seed.render()
      assert 'Missing patterns' in rendered
    finally:
      sys.path.pop(0)

  def test_textmatch_loss_reset_clears_all(self, tmp_path):
    rules_dir = tmp_path / 'rules'
    rules_dir.mkdir()

    sys.path.insert(0, str(TEXTMATCH_DIR))
    try:
      from textmatch.module import TextMatchLoss

      param = PathParameter(source=str(rules_dir), pattern='*.json')
      loss = TextMatchLoss([param])

      datum = EvalDatum(
        success=False,
        metadata={'text': 'x', 'expected': 'y', 'failure_type': 'no_match'},
      )
      loss.forward(datum)
      loss.reset()

      assert loss._last_data is None
      assert loss._accumulated == []
      assert loss._errors == []
    finally:
      sys.path.pop(0)
