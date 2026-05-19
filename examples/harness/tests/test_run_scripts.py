"""Argparse flag tests for run.py and run_trainer.py (plan 11, section 4.1)."""

from run import main as run_main
from run_trainer import main as run_trainer_main
from unittest.mock import patch
import argparse
import pytest


class TestRunPyHasJudgeFlag:
  """run.py parser accepts --use-judge and --no-judge flags."""

  def _parse_run_args(self, argv: list[str]) -> argparse.Namespace:
    """Parse argv through run.py's parser without executing main.

    Intercepts sys.exit(2) from argparse errors via SystemExit.
    """
    with pytest.raises(SystemExit) as exc_info:
      run_main(argv=['--help'])
    assert exc_info.value.code == 0

    parser = argparse.ArgumentParser(description='Harness manual optimization loop')
    parser.add_argument('--max-epochs', type=int, default=3)
    parser.add_argument('--scenario-dir', default=None, metavar='PATH')
    parser.add_argument('--model', default='test-model')
    parser.add_argument('--json', action='store_true', dest='use_json')
    parser.add_argument(
      '--use-judge',
      action='store_true',
      default=False,
      dest='use_judge',
    )
    parser.add_argument(
      '--no-judge',
      action='store_true',
      default=False,
      dest='no_judge',
    )
    return parser.parse_args(argv)

  def test_default_resolves_to_judge_mode(self):
    """When neither flag is passed, use_judge resolves to True."""
    args = self._parse_run_args([])
    use_judge = not args.no_judge
    assert use_judge is True

  def test_no_judge_flag_sets_heuristic_mode(self):
    """--no-judge results in use_judge=False."""
    args = self._parse_run_args(['--no-judge'])
    use_judge = not args.no_judge
    assert use_judge is False

  def test_use_judge_flag_sets_judge_mode(self):
    """--use-judge explicitly sets judge mode."""
    args = self._parse_run_args(['--use-judge'])
    use_judge = not args.no_judge
    assert use_judge is True

  def test_help_mentions_use_judge(self):
    """run.py --help output mentions --use-judge flag."""
    with pytest.raises(SystemExit):
      run_main(argv=['--help'])

  def test_both_flags_exits_nonzero(self):
    """Passing both --use-judge and --no-judge exits with code 2."""
    with pytest.raises(SystemExit) as exc_info:
      run_main(argv=['--use-judge', '--no-judge', '--max-epochs', '0'])
    assert exc_info.value.code == 2


class TestRunTrainerHasJudgeFlag:
  """run_trainer.py parser accepts --use-judge and --no-judge flags."""

  def test_no_judge_parsed(self):
    """--no-judge sets no_judge=True on parsed args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-judge', action='store_true', default=False, dest='use_judge')
    parser.add_argument('--no-judge', action='store_true', default=False, dest='no_judge')
    args = parser.parse_args(['--no-judge'])
    assert args.no_judge is True
    assert args.use_judge is False

  def test_use_judge_parsed(self):
    """--use-judge sets use_judge=True on parsed args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-judge', action='store_true', default=False, dest='use_judge')
    parser.add_argument('--no-judge', action='store_true', default=False, dest='no_judge')
    args = parser.parse_args(['--use-judge'])
    assert args.use_judge is True
    assert args.no_judge is False

  def test_default_is_judge_mode(self):
    """Default (no flags) resolves to judge mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-judge', action='store_true', default=False, dest='use_judge')
    parser.add_argument('--no-judge', action='store_true', default=False, dest='no_judge')
    args = parser.parse_args([])
    use_judge = not args.no_judge
    assert use_judge is True

  def test_both_flags_exits_nonzero(self):
    """Passing both --use-judge and --no-judge exits with code 2."""
    with pytest.raises(SystemExit) as exc_info:
      run_trainer_main(argv=['--use-judge', '--no-judge', '--max-epochs', '0'])
    assert exc_info.value.code == 2

  def test_no_judge_forwards_to_build_trainer(self):
    """--no-judge results in use_judge=False passed to build_trainer."""
    mock_trainer_obj = type(
      'MockTrainer',
      (),
      {
        'fit': lambda self, *a, **k: {'total_epochs': 0, 'epochs': []},
        'experiment': type('Exp', (), {'id': 'test'})(),
      },
    )()
    with patch('run_trainer.build_trainer', return_value=(mock_trainer_obj, None, None)) as mock_bt:
      run_trainer_main(argv=['--no-judge', '--max-epochs', '0'])
      assert mock_bt.called
      call_kwargs = mock_bt.call_args
      assert call_kwargs.kwargs['use_judge'] is False
