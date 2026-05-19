"""Tests for sub-plan 17: truthiness cleanup.

Covers explicit None checks, len-to-truthiness, mutable defaults,
.keys() removal, and section comment normalization.
"""

from autopilot.ai.data import split_names_and_normalized_ratios
from autopilot.ai.evaluation.schemas import CheckpointEvent, CheckpointHeader
from autopilot.cli.command import Command
from autopilot.cli.commands.execute import ExecuteCommand
from autopilot.cli.expose import ExposeCollector, inject_expose
from autopilot.cli.output import Output
from autopilot.core.callbacks.callback import Callback
from autopilot.core.diagnostics import Diagnostics
from autopilot.core.experiment import Experiment
from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.models import DatasetSnapshot
from autopilot.core.module.module import Module
from autopilot.core.node import Node
from autopilot.core.query import QueryBuilder
from autopilot.core.snapshot import SnapshotManifest
from autopilot.core.store.types import DiffResult, MergeIndex, StatusResult
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.tree import Tree
from autopilot.core.types import EvalDatum
from autopilot.policy.quality_first import QualityFirstMetric, QualityFirstPolicy
from autopilot.tracking.executions import create_execution_record, resolve_command
from tests.doubles import NoopEvalModule
from unittest.mock import MagicMock
import argparse
import pytest

# 2.1 trainer callbacks and fit context


class TestTrainerCallbacksNoneVsEmpty:
  def test_callbacks_none_yields_empty_list(self) -> None:
    trainer = Trainer(callbacks=None)
    assert trainer.callbacks == []

  def test_callbacks_empty_list_yields_empty_list(self) -> None:
    trainer = Trainer(callbacks=[])
    assert trainer.callbacks == []

  def test_callbacks_none_and_empty_have_distinct_identity(self) -> None:
    t1 = Trainer(callbacks=None)
    t2 = Trainer(callbacks=[])
    assert t1.callbacks is not t2.callbacks

  def test_callbacks_explicit_list_preserved(self) -> None:
    cb = Callback()
    trainer = Trainer(callbacks=[cb])
    assert trainer.callbacks == [cb]

  def test_fit_ctx_none_yields_empty_dict(self) -> None:
    mod = NoopEvalModule()
    trainer = Trainer()
    trainer.fit(mod, max_epochs=1, ctx=None)
    assert trainer.fit_context == {}

  def test_fit_ctx_empty_dict_yields_empty_dict(self) -> None:
    mod = NoopEvalModule()
    trainer = Trainer()
    trainer.fit(mod, max_epochs=1, ctx={})
    assert trainer.fit_context == {}

  def test_fit_ctx_with_data_preserved(self) -> None:
    mod = NoopEvalModule()
    trainer = Trainer()
    trainer.fit(mod, max_epochs=1, ctx={'key': 'val'})
    assert trainer.fit_context == {'key': 'val'}


# 2.2 policy gate lists


class TestPolicyGatesNoneVsEmpty:
  def test_quality_first_gates_none(self) -> None:
    policy = QualityFirstPolicy(gates=None)
    assert policy._gates == []

  def test_quality_first_gates_empty(self) -> None:
    policy = QualityFirstPolicy(gates=[])
    assert policy._gates == []

  def test_quality_first_metric_gates_none(self) -> None:
    metric = QualityFirstMetric(gates=None)
    assert metric._gates == []

  def test_quality_first_metric_gates_empty(self) -> None:
    metric = QualityFirstMetric(gates=[])
    assert metric._gates == []


# 2.3 store from_dict


class TestStoreFromDictTruthiness:
  def test_snapshot_manifest_entries_missing(self) -> None:
    result = SnapshotManifest.from_dict({'epoch': 0, 'timestamp': 't'})
    assert result.entries == {}

  def test_snapshot_manifest_entries_null(self) -> None:
    result = SnapshotManifest.from_dict({'epoch': 0, 'timestamp': 't', 'entries': None})
    assert result.entries == {}

  def test_snapshot_manifest_entries_empty(self) -> None:
    result = SnapshotManifest.from_dict({'epoch': 0, 'timestamp': 't', 'entries': {}})
    assert result.entries == {}

  def test_diff_result_entries_missing(self) -> None:
    result = DiffResult.from_dict({})
    assert result.entries == []

  def test_diff_result_entries_null(self) -> None:
    result = DiffResult.from_dict({'entries': None})
    assert result.entries == []

  def test_diff_result_entries_empty(self) -> None:
    result = DiffResult.from_dict({'entries': []})
    assert result.entries == []

  def test_merge_index_conflicts_null(self) -> None:
    result = MergeIndex.from_dict({'conflicts': None})
    assert result.conflicts == {}

  def test_merge_index_conflicts_empty(self) -> None:
    result = MergeIndex.from_dict({'conflicts': {}})
    assert result.conflicts == {}

  def test_merge_index_resolved_null(self) -> None:
    result = MergeIndex.from_dict({'resolved': None})
    assert result.resolved == {}

  def test_status_result_entries_missing(self) -> None:
    result = StatusResult.from_dict({})
    assert result.entries == []

  def test_status_result_entries_null(self) -> None:
    result = StatusResult.from_dict({'entries': None})
    assert result.entries == []


# 2.4 models truthiness


class TestModelsTruthiness:
  def test_dataset_snapshot_entries_null(self) -> None:
    result = DatasetSnapshot.from_dict({'created_at': 'now', 'entries': None})
    assert result.entries == []

  def test_dataset_snapshot_entries_empty(self) -> None:
    result = DatasetSnapshot.from_dict({'created_at': 'now', 'entries': []})
    assert result.entries == []

  def test_dataset_snapshot_entries_missing(self) -> None:
    result = DatasetSnapshot.from_dict({'created_at': 'now'})
    assert result.entries == []


# 2.5 module prefix


class TestModulePrefixTruthiness:
  def test_named_modules_prefix_none(self) -> None:
    m = Module()
    names = [name for name, _ in m.named_modules(prefix=None)]
    assert names == ['']

  def test_named_modules_prefix_empty_string(self) -> None:
    m = Module()
    names = [name for name, _ in m.named_modules(prefix='')]
    assert names == ['']

  def test_named_parameters_prefix_none(self) -> None:
    m = Module()
    names = [name for name, _ in m.named_parameters(prefix=None)]
    assert names == []

  def test_named_parameters_prefix_empty_string(self) -> None:
    m = Module()
    names = [name for name, _ in m.named_parameters(prefix='')]
    assert names == []


# 2.8 diagnostics


class TestDiagnosticsErrorMessage:
  def test_error_message_empty_string_preserved(self) -> None:
    diag = Diagnostics(path=None)
    items = [{'success': False, 'error_message': ''}]
    samples = diag.select_samples(items)
    assert samples == ['']

  def test_error_message_missing_yields_fallback(self) -> None:
    diag = Diagnostics(path=None)
    items = [{'success': False}]
    samples = diag.select_samples(items)
    assert samples == ['failed (no message)']

  def test_error_message_present(self) -> None:
    diag = Diagnostics(path=None)
    items = [{'success': False, 'error_message': 'timeout'}]
    samples = diag.select_samples(items)
    assert samples == ['timeout']


# 2.8 resolve_command


class TestResolveCommand:
  def test_empty_parts_yields_unknown(self) -> None:
    args = argparse.Namespace()
    assert resolve_command(args) == 'unknown'

  def test_command_present(self) -> None:
    args = argparse.Namespace(command='tree')
    assert resolve_command(args) == 'tree'


# 2.9 metric collection


class TestMetricCollectionKeysTruthiness:
  def test_metric_keys_from_dict_without_keys_call(self) -> None:
    class Acc(Metric):
      higher_is_better = True

      def compute(self) -> dict[str, float]:
        return {'acc': 1.0}

      def update(self, datum) -> None:
        pass

    coll = MetricCollection({'acc': Acc()})
    assert coll._metric_keys == ['acc']

  def test_compute_with_none_prefix_postfix(self) -> None:
    class Acc(Metric):
      higher_is_better = True

      def compute(self) -> dict[str, float]:
        return {'acc': 1.0}

      def update(self, datum) -> None:
        pass

    coll = MetricCollection({'acc': Acc()}, prefix=None, postfix=None)
    coll.update(EvalDatum(success=True))
    result = coll.compute()
    assert 'acc' in result

  def test_compute_with_prefix(self) -> None:
    class Acc(Metric):
      higher_is_better = True

      def compute(self) -> dict[str, float]:
        return {'acc': 1.0}

      def update(self, datum) -> None:
        pass

    coll = MetricCollection({'acc': Acc()}, prefix='train_')
    coll.update(EvalDatum(success=True))
    result = coll.compute()
    assert 'train_acc' in result


# 2.10 & 2.11 output and expose collector


class TestOutputExposeCollector:
  def test_expose_collector_empty_no_inject(self) -> None:
    collector = ExposeCollector()
    result = {'data': 1}
    injected = inject_expose(dict(result), collector)
    assert '_commands' not in injected

  def test_expose_collector_nonempty_injects(self) -> None:
    collector = ExposeCollector()
    collector.add(description='test', command='echo hi')
    result = {'data': 1}
    injected = inject_expose(dict(result), collector)
    assert '_commands' in injected
    assert len(injected['_commands']) == 1

  def test_output_result_with_empty_collector(self, capsys) -> None:
    collector = ExposeCollector()
    output = Output(use_json=False, expose_collector=collector)
    output.result({'ok': True})
    captured = capsys.readouterr()
    assert 'OK' in captured.out


# 2.12 query exists


class TestQueryExists:
  def test_exists_false_for_empty(self) -> None:
    qb = QueryBuilder([], lambda x: None)
    assert qb.exists() is False

  def test_exists_true_for_nonempty(self) -> None:
    exp = Experiment(experiment_id='e1')
    node = Node(experiment=exp)
    qb = QueryBuilder([node], lambda x: node)
    assert qb.exists() is True


# 2.13 merge resolved


class TestMergeIndexResolvedFlag:
  def test_no_conflicts_is_resolved(self) -> None:
    idx = MergeIndex(conflicts={})
    assert idx.is_resolved() is True

  def test_with_conflicts_not_resolved(self) -> None:
    from autopilot.core.store.types import ConflictEntry

    idx = MergeIndex(conflicts={'file.txt': ConflictEntry(key='file.txt')})
    assert idx.is_resolved() is False


# 2.16 execute mutable default


class TestExecuteMutableDefault:
  def test_parse_twice_independent_extra_args(self) -> None:
    cmd = ExecuteCommand()
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()
    cmd.register(subs)

    args1 = parser.parse_args(['execute'])
    args2 = parser.parse_args(['execute'])
    _mode1, _, extra1 = cmd.parse_mode(args1)
    _mode2, _, extra2 = cmd.parse_mode(args2)
    assert extra1 is not extra2
    extra1.append('sentinel')
    assert 'sentinel' not in extra2

  def test_default_not_shared_mutable(self) -> None:
    cmd = ExecuteCommand()
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()
    cmd.register(subs)

    args1 = parser.parse_args(['execute'])
    args2 = parser.parse_args(['execute'])
    assert args1.extra_args is not args2.extra_args


# 2.17 pydantic checkpoint defaults


class TestPydanticDefaults:
  def test_checkpoint_header_distinct_args(self) -> None:
    a = CheckpointHeader(subsystem='gen', config_hash='h1', created_at='now')
    b = CheckpointHeader(subsystem='gen', config_hash='h2', created_at='now')
    assert a.args is not b.args
    assert a.args == {}
    assert b.args == {}

  def test_checkpoint_event_distinct_payload(self) -> None:
    a = CheckpointEvent(type='result', id='1', timestamp='now')
    b = CheckpointEvent(type='result', id='2', timestamp='now')
    assert a.payload is not b.payload
    assert a.payload == {}
    assert b.payload == {}

  def test_mutating_one_does_not_affect_other(self) -> None:
    a = CheckpointHeader(subsystem='gen', config_hash='h', created_at='now')
    b = CheckpointHeader(subsystem='gen', config_hash='h', created_at='now')
    a.args['key'] = 'value'
    assert 'key' not in b.args


# 2.18 command .keys() removal


class TestCommandRepr:
  def test_command_repr_lists_children(self) -> None:
    class Parent(Command):
      name = 'parent'
      help = 'parent'

    class Child(Command):
      name = 'child'
      help = 'child'

    parent = Parent()
    parent._commands['child'] = Child()
    r = repr(parent)
    assert 'child' in r


# 2.19 tree kwargs .keys()


class TestTreeKwargsKeys:
  def test_update_rejects_unexpected_kwargs(self) -> None:
    store = MagicMock()
    tree = Tree(name='t', store=store)
    exp = Experiment(experiment_id='e1')
    node = Node(experiment=exp)
    tree.add(node)
    with pytest.raises(TypeError, match='unknown keyword argument'):
      tree.update('e1', bad_key='val')


# 2.21 data split names


class TestDataSplitNames:
  def test_split_names_order(self) -> None:
    ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    names, _ = split_names_and_normalized_ratios(ratios)
    assert names == ['train', 'val', 'test']


# 2.8 execution record extra


class TestExecutionRecordExtra:
  def test_extra_none_yields_empty_dict(self) -> None:
    record = create_execution_record(
      command='test',
      args=[],
      duration_ms=100.0,
      exit_code=0,
      extra=None,
    )
    assert record.extra == {}

  def test_extra_dict_preserved(self) -> None:
    record = create_execution_record(
      command='test',
      args=[],
      duration_ms=100.0,
      exit_code=0,
      extra={'key': 'val'},
    )
    assert record.extra == {'key': 'val'}
