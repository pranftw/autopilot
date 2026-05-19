"""Import and behavioral tests for example run_trainer.py scripts.

Validates that edited run_trainer.py files remain importable and
syntactically valid without spawning full training runs. Also verifies
callback behavior for context emission.

No subprocess invocations; uses in-process importlib and AST inspection.
"""

from pathlib import Path
from unittest.mock import MagicMock
import ast
import compileall
import importlib.util
import sys

REPO = Path(__file__).resolve().parents[2]


def _import_run_trainer(example_rel: str):
  """Import run_trainer.py from an example directory.

  Adds the example directory to sys.path, imports the module, and
  removes it from sys.path afterwards. Returns the imported module.
  """
  root = REPO / 'examples' / example_rel
  path_str = str(root)
  module_name = 'run_trainer_' + example_rel.replace('/', '_')

  sys.path.insert(0, path_str)
  try:
    spec = importlib.util.spec_from_file_location(
      module_name,
      root / 'run_trainer.py',
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
  finally:
    if path_str in sys.path:
      sys.path.remove(path_str)


# --- 4.1 In-process import and syntax tests ---


def test_harness_run_trainer_imports() -> None:
  """examples/harness/run_trainer.py loads via importlib."""
  mod = _import_run_trainer('harness')
  assert hasattr(mod, 'main')
  assert callable(mod.main)


def test_textmatch_run_trainer_imports() -> None:
  """examples/textmatch/run_trainer.py loads via importlib."""
  mod = _import_run_trainer('textmatch')
  assert hasattr(mod, 'main')
  assert callable(mod.main)


def test_protim_run_trainer_imports() -> None:
  """examples/protim/run_trainer.py loads via importlib."""
  mod = _import_run_trainer('protim')
  assert hasattr(mod, 'main')
  assert callable(mod.main)


def test_multi_module_run_trainer_imports() -> None:
  """examples/multi_module/run_trainer.py compiles and imports."""
  root = REPO / 'examples' / 'multi_module'
  assert compileall.compile_file(str(root / 'run_trainer.py'), quiet=1)
  mod = _import_run_trainer('multi_module')
  assert hasattr(mod, 'main')
  assert callable(mod.main)


# --- 4.2 Behavioral verification tests ---


def _import_harness_callbacks():
  """Import harness callbacks module via importlib. Returns the module.

  Inserts examples/harness on sys.path for the duration of loading
  (required for harness package imports within callbacks.py), then
  removes it.
  """
  harness_dir = REPO / 'examples' / 'harness'
  path_str = str(harness_dir)
  sys.path.insert(0, path_str)
  try:
    spec = importlib.util.spec_from_file_location(
      'harness.callbacks',
      harness_dir / 'harness' / 'callbacks.py',
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
  finally:
    if path_str in sys.path:
      sys.path.remove(path_str)


def test_harness_optimizer_context_callback_emits() -> None:
  """OptimizerContextCallback emits context when val metric improves."""
  callbacks_mod = _import_harness_callbacks()
  callback_cls = callbacks_mod.OptimizerContextCallback

  callback = callback_cls()

  trainer = MagicMock()
  trainer.optimizer = None
  module = MagicMock()

  result_mock = MagicMock()
  result_mock.metrics = {'val_task_success_rate': 0.8}

  callback.on_epoch_end(trainer, module, epoch=0, result=result_mock)

  trainer.emit_context.assert_called_once()
  call_args = trainer.emit_context.call_args
  assert call_args[0][0] == 'harness optimization decision: val improved vs prior best'
  assert call_args[1]['source'] == 'harness'
  assert call_args[1]['metadata']['epoch'] == 0
  assert call_args[1]['metadata']['metric'] == 0.8
  assert call_args[1]['metadata']['prior_best'] == float('-inf')


def test_harness_optimizer_context_callback_no_emit_below_threshold() -> None:
  """OptimizerContextCallback does not emit when metric does not improve."""
  callbacks_mod = _import_harness_callbacks()
  callback_cls = callbacks_mod.OptimizerContextCallback

  callback = callback_cls()
  callback._best = 0.9

  trainer = MagicMock()
  trainer.optimizer = None
  module = MagicMock()

  result_mock = MagicMock()
  result_mock.metrics = {'val_task_success_rate': 0.8}

  callback.on_epoch_end(trainer, module, epoch=1, result=result_mock)

  trainer.emit_context.assert_not_called()


def test_harness_optimizer_context_callback_no_emit_no_metric() -> None:
  """OptimizerContextCallback does not emit when metric key is absent."""
  callbacks_mod = _import_harness_callbacks()
  callback_cls = callbacks_mod.OptimizerContextCallback

  callback = callback_cls()

  trainer = MagicMock()
  trainer.optimizer = None
  module = MagicMock()

  result_mock = MagicMock()
  result_mock.metrics = {'some_other_metric': 0.5}

  callback.on_epoch_end(trainer, module, epoch=0, result=result_mock)

  trainer.emit_context.assert_not_called()


def test_run_trainer_add_context_calls() -> None:
  """Harness run_trainer.py contains add_context calls at decision points."""
  script_path = REPO / 'examples' / 'harness' / 'run_trainer.py'
  source = script_path.read_text()
  tree = ast.parse(source, filename=str(script_path))

  add_context_calls = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      func = node.func
      if isinstance(func, ast.Attribute) and func.attr == 'add_context':
        add_context_calls.append(node)

  assert len(add_context_calls) >= 1, 'run_trainer.py should contain at least one add_context call'


def test_textmatch_run_trainer_add_context_calls() -> None:
  """Textmatch run_trainer.py contains add_context calls."""
  script_path = REPO / 'examples' / 'textmatch' / 'run_trainer.py'
  source = script_path.read_text()
  tree = ast.parse(source, filename=str(script_path))

  add_context_calls = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      func = node.func
      if isinstance(func, ast.Attribute) and func.attr == 'add_context':
        add_context_calls.append(node)

  assert len(add_context_calls) >= 1, (
    'textmatch/run_trainer.py should contain at least one add_context call'
  )


def test_protim_run_trainer_add_context_calls() -> None:
  """Protim run_trainer.py contains add_context calls."""
  script_path = REPO / 'examples' / 'protim' / 'run_trainer.py'
  source = script_path.read_text()
  tree = ast.parse(source, filename=str(script_path))

  add_context_calls = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      func = node.func
      if isinstance(func, ast.Attribute) and func.attr == 'add_context':
        add_context_calls.append(node)

  assert len(add_context_calls) >= 1, (
    'protim/run_trainer.py should contain at least one add_context call'
  )


def test_multi_module_run_trainer_add_context_comment() -> None:
  """Multi_module run_trainer.py documents opt-in experiment context wiring."""
  script_path = REPO / 'examples' / 'multi_module' / 'run_trainer.py'
  source = script_path.read_text()

  assert 'add_context' in source, (
    'multi_module/run_trainer.py should reference add_context '
    '(at minimum in a comment explaining opt-in wiring)'
  )
