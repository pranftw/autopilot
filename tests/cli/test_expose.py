"""Tests for --expose mechanism."""

from autopilot.cli.expose import ExposeCollector, ExposeRecord, expose_command, inject_expose
from autopilot.cli.output import Output
import json


class TestExposeRecord:
  def test_to_dict(self):
    r = ExposeRecord(description='test', command='echo hi', exit_code=0, duration_s=1.5)
    d = r.to_dict()
    assert d['description'] == 'test'
    assert d['command'] == 'echo hi'
    assert d['exit_code'] == 0
    assert d['duration_s'] == 1.5

  def test_from_dict(self):
    d = {'description': 'x', 'command': 'y', 'exit_code': 1, 'duration_s': 2.0, 'stderr': 'err'}
    r = ExposeRecord.from_dict(d)
    assert r.description == 'x'
    assert r.stderr == 'err'

  def test_timestamp_auto(self):
    r = ExposeRecord(description='test')
    assert r.timestamp != ''


class TestExposeCollector:
  def test_empty(self):
    c = ExposeCollector()
    assert c.to_list() == []
    assert len(c) == 0

  def test_three_commands(self):
    c = ExposeCollector()
    c.add('first', 'cmd1', exit_code=0, duration_s=1.0)
    c.add('second', 'cmd2', exit_code=0, duration_s=2.0)
    c.add('third', 'cmd3', exit_code=1, duration_s=0.5, stderr='fail')
    result = c.to_list()
    assert len(result) == 3
    assert result[0]['description'] == 'first'
    assert result[2]['stderr'] == 'fail'

  def test_record_has_all_fields(self):
    c = ExposeCollector()
    c.add('desc', 'cmd', exit_code=0, duration_s=1.0, stderr='')
    record = c.to_list()[0]
    assert 'description' in record
    assert 'command' in record
    assert 'exit_code' in record
    assert 'duration_s' in record
    assert 'stderr' in record
    assert 'timestamp' in record


class TestInjectExpose:
  def test_empty_collector_unchanged(self):
    c = ExposeCollector()
    result = {'data': 'hello'}
    out = inject_expose(result, c)
    assert '_commands' not in out

  def test_non_empty_adds_commands(self):
    c = ExposeCollector()
    c.add('test', 'echo', exit_code=0, duration_s=0.1)
    result = {'data': 'hello'}
    out = inject_expose(result, c)
    assert '_commands' in out
    assert len(out['_commands']) == 1


class TestExposeCommand:
  def test_timing_and_exit_code(self):
    c = ExposeCollector()
    with expose_command(c, 'test op', 'test cmd') as state:
      state['exit_code'] = 0
    records = c.to_list()
    assert len(records) == 1
    assert records[0]['description'] == 'test op'
    assert records[0]['duration_s'] >= 0

  def test_exception_sets_exit_code(self):
    c = ExposeCollector()
    try:
      with expose_command(c, 'failing', 'bad cmd'):
        raise ValueError('boom')
    except ValueError:
      pass
    records = c.to_list()
    assert records[0]['exit_code'] == 1
    assert 'boom' in records[0]['stderr']


class TestOutputExposeIntegration:
  def test_result_includes_commands_when_expose_active(self, capsys):
    collector = ExposeCollector()
    collector.add(description='step1', command='echo hi', exit_code=0, duration_s=0.1)
    output = Output(use_json=True, expose_collector=collector)
    output.result({'status': 'done'})
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert '_commands' in envelope['result']
    assert len(envelope['result']['_commands']) == 1
    assert envelope['result']['_commands'][0]['description'] == 'step1'

  def test_result_no_commands_when_collector_empty(self, capsys):
    collector = ExposeCollector()
    output = Output(use_json=True, expose_collector=collector)
    output.result({'status': 'done'})
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert '_commands' not in envelope['result']

  def test_result_no_commands_when_no_collector(self, capsys):
    output = Output(use_json=True)
    output.result({'status': 'done'})
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert '_commands' not in envelope['result']
