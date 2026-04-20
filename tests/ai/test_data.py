from autopilot.ai.data import SlotPlanner, StratifiedSplitter
from autopilot.ai.models import ConversationTurn, DataItem, VarDef
from autopilot.data.dataset import ListDataset
from pydantic import BaseModel
import pytest


class SimpleCustom(BaseModel):
  domain: str
  difficulty: str


def make_item(id: str, domain: str = 'math', difficulty: str = 'easy') -> DataItem[SimpleCustom]:
  return DataItem(
    id=id,
    turns=[ConversationTurn(role='user', content='test')],
    custom=SimpleCustom(domain=domain, difficulty=difficulty),
  )


def ds_items(ds: ListDataset) -> list:
  return [ds[i] for i in range(len(ds))]


class TestListDataset:
  def test_getitem(self):
    a = make_item('a')
    b = make_item('b')
    ds = ListDataset([a, b])
    assert ds[0] is a
    assert ds[1] is b

  def test_len(self):
    ds = ListDataset([make_item('1'), make_item('2')])
    assert len(ds) == 2

  def test_from_jsonl(self, tmp_path):
    p = tmp_path / 'data.jsonl'
    items = [make_item('x', domain='physics'), make_item('y', domain='math')]
    p.write_text('\n'.join(i.model_dump_json() for i in items), encoding='utf-8')
    loaded = ListDataset.from_jsonl(p, DataItem[SimpleCustom])
    assert len(loaded) == 2
    assert loaded[0].id == 'x'
    assert loaded[0].custom.domain == 'physics'
    assert loaded[1].id == 'y'

  def test_to_jsonl(self, tmp_path):
    p = tmp_path / 'out.jsonl'
    items = [make_item('a'), make_item('b')]
    ListDataset(items).to_jsonl(p)
    lines = p.read_text(encoding='utf-8').strip().split('\n')
    assert len(lines) == 2
    assert DataItem[SimpleCustom].model_validate_json(lines[0]).id == 'a'

  def test_from_to_round_trip(self, tmp_path):
    p = tmp_path / 'round.jsonl'
    orig = [make_item('1', domain='d1'), make_item('2', domain='d2')]
    ListDataset(orig).to_jsonl(p)
    loaded = ListDataset.from_jsonl(p, DataItem[SimpleCustom])
    assert len(loaded) == 2
    for i in range(len(loaded)):
      assert loaded[i].model_dump() == orig[i].model_dump()

  def test_subset(self):
    items = [make_item(str(i)) for i in range(5)]
    ds = ListDataset(items)
    sub = ds.subset([1, 3])
    assert len(sub) == 2
    assert sub[0].id == '1'
    assert sub[1].id == '3'

  def test_subset_preserves_items(self):
    items = [make_item('a'), make_item('b')]
    ds = ListDataset(items)
    sub = ds.subset([0])
    assert sub[0] is items[0]

  def test_empty_dataset(self):
    ds = ListDataset([])
    assert len(ds) == 0

  def test_index_out_of_range(self):
    ds = ListDataset([make_item('only')])
    with pytest.raises(IndexError):
      _ = ds[1]


class TestStratifiedSplitter:
  def test_deterministic_with_seed(self):
    items = [make_item(f'id{i}', domain='d') for i in range(20)]
    ds = ListDataset(items)
    ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    s1 = StratifiedSplitter(ratios, lambda it: it.custom.domain, seed=42).split(ds)
    s2 = StratifiedSplitter(ratios, lambda it: it.custom.domain, seed=42).split(ds)
    for name in ratios:
      assert [x.id for x in ds_items(s1[name])] == [x.id for x in ds_items(s2[name])]

  def test_different_seeds_differ(self):
    items = [make_item(f'id{i}', domain='d') for i in range(30)]
    ds = ListDataset(items)
    ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    s1 = StratifiedSplitter(ratios, lambda it: it.custom.domain, seed=1).split(ds)
    s2 = StratifiedSplitter(ratios, lambda it: it.custom.domain, seed=2).split(ds)
    t1 = {x.id for x in ds_items(s1['train'])}
    t2 = {x.id for x in ds_items(s2['train'])}
    assert t1 != t2

  def test_ratios_respected(self):
    items = [make_item(f'id{i}', domain='x') for i in range(100)]
    ds = ListDataset(items)
    ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    out = StratifiedSplitter(ratios, lambda it: it.custom.domain, seed=0).split(ds)
    assert abs(len(out['train']) / 100 - 0.8) <= 0.05
    assert abs(len(out['val']) / 100 - 0.1) <= 0.05
    assert abs(len(out['test']) / 100 - 0.1) <= 0.05

  def test_matched_distributions(self):
    items = []
    for i in range(50):
      items.append(make_item(f'a{i}', domain='A'))
    for i in range(50):
      items.append(make_item(f'b{i}', domain='B'))
    ds = ListDataset(items)
    ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    out = StratifiedSplitter(ratios, lambda it: it.custom.domain, seed=7).split(ds)
    for name in ratios:
      items_in_split = ds_items(out[name])
      a_ids = [x for x in items_in_split if x.custom.domain == 'A']
      b_ids = [x for x in items_in_split if x.custom.domain == 'B']
      assert len(a_ids) + len(b_ids) == len(out[name])
      if len(out[name]) > 0:
        pa = len(a_ids) / len(items_in_split)
        pb = len(b_ids) / len(items_in_split)
        assert abs(pa - 0.5) <= 0.11
        assert abs(pb - 0.5) <= 0.11

  def test_all_items_assigned(self):
    items = [make_item(f'id{i}') for i in range(37)]
    ds = ListDataset(items)
    ratios = {'train': 0.5, 'val': 0.25, 'test': 0.25}
    out = StratifiedSplitter(ratios, lambda it: it.custom.difficulty, seed=3).split(ds)
    total = sum(len(out[k]) for k in ratios)
    assert total == 37

  def test_no_overlap(self):
    items = [make_item(f'id{i}') for i in range(40)]
    ds = ListDataset(items)
    ratios = {'train': 0.5, 'val': 0.25, 'test': 0.25}
    out = StratifiedSplitter(ratios, lambda it: it.id, seed=9).split(ds)
    seen: set[str] = set()
    for name in ratios:
      for x in ds_items(out[name]):
        assert x.id not in seen
        seen.add(x.id)
    assert len(seen) == 40

  def test_single_item_group(self):
    items = [make_item('only', domain='solo')]
    ds = ListDataset(items)
    ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    out = StratifiedSplitter(ratios, lambda it: it.custom.domain, seed=0).split(ds)
    sizes = {k: len(out[k]) for k in ratios}
    assert sum(sizes.values()) == 1
    assert max(sizes.values()) == 1

  def test_empty_dataset(self):
    ds = ListDataset([])
    ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    out = StratifiedSplitter(ratios, lambda it: it.custom.domain, seed=0).split(ds)
    assert all(len(out[k]) == 0 for k in ratios)

  def test_two_splits(self):
    items = [make_item(f'id{i}') for i in range(50)]
    ds = ListDataset(items)
    ratios = {'train': 0.8, 'test': 0.2}
    out = StratifiedSplitter(ratios, lambda it: it.custom.domain, seed=11).split(ds)
    assert len(out['train']) + len(out['test']) == 50

  def test_three_splits(self):
    items = [make_item(f'id{i}') for i in range(60)]
    ds = ListDataset(items)
    ratios = {'train': 0.5, 'val': 0.25, 'test': 0.25}
    out = StratifiedSplitter(ratios, lambda it: it.custom.domain, seed=5).split(ds)
    assert len(out['train']) + len(out['val']) + len(out['test']) == 60

  def test_custom_key_fn(self):
    items = [make_item('a', domain='math'), make_item('b', domain='code')]
    ds = ListDataset(items)
    ratios = {'train': 1.0}
    out = StratifiedSplitter(ratios, lambda item: item.custom.domain, seed=0).split(ds)
    assert len(out['train']) == 2


class TestSlotPlanner:
  def test_deterministic_with_seed(self):
    v = {'x': VarDef(choices=['a', 'b'], distribution=[0.5, 0.5])}
    s1 = SlotPlanner(v, seed=99).create_slots(10)
    s2 = SlotPlanner(v, seed=99).create_slots(10)
    assert s1 == s2

  def test_different_seeds_differ(self):
    v = {'x': VarDef(choices=['a', 'b', 'c'], distribution=[0.34, 0.33, 0.33])}
    s1 = SlotPlanner(v, seed=1).create_slots(50)
    s2 = SlotPlanner(v, seed=2).create_slots(50)
    assert s1 != s2

  def test_correct_count(self):
    v = {'x': VarDef(choices=['a'], distribution=[1.0])}
    slots = SlotPlanner(v, seed=0).create_slots(100)
    assert len(slots) == 100

  def test_weighted_distribution(self):
    v = {
      'c': VarDef(
        choices=['r', 'g', 'b'],
        distribution=[0.5, 0.3, 0.2],
      )
    }
    slots = SlotPlanner(v, seed=123).create_slots(10000)
    counts = {'r': 0, 'g': 0, 'b': 0}
    for s in slots:
      counts[s['c']] += 1
    assert abs(counts['r'] / 10000 - 0.5) < 0.05
    assert abs(counts['g'] / 10000 - 0.3) < 0.05
    assert abs(counts['b'] / 10000 - 0.2) < 0.05

  def test_all_choices_covered(self):
    v = {'x': VarDef(choices=['p', 'q', 'r', 's'], distribution=[0.25, 0.25, 0.25, 0.25])}
    slots = SlotPlanner(v, seed=0).create_slots(1000)
    seen = {s['x'] for s in slots}
    assert seen == {'p', 'q', 'r', 's'}

  def test_id_prefix_applied(self):
    v = {'x': VarDef(choices=['a'], distribution=[1.0])}
    slots = SlotPlanner(v, seed=0).create_slots(3, id_prefix='TAX')
    assert slots[0]['id'].startswith('TAX')

  def test_metadata_included(self):
    v = {
      'k': VarDef(
        choices=['a', 'b'],
        distribution=[0.5, 0.5],
        metadata=[{'meta_a': 1}, {'meta_b': 2}],
      )
    }
    slots = SlotPlanner(v, seed=0).create_slots(20)
    found_a = any(s.get('meta_a') == 1 for s in slots)
    found_b = any(s.get('meta_b') == 2 for s in slots)
    assert found_a or found_b

  def test_single_var(self):
    v = {'only': VarDef(choices=['z'], distribution=[1.0])}
    slots = SlotPlanner(v, seed=0).create_slots(5)
    assert all(s['only'] == 'z' for s in slots)

  def test_multiple_vars(self):
    v = {
      'u': VarDef(choices=['1', '2'], distribution=[0.5, 0.5]),
      'w': VarDef(choices=['x', 'y'], distribution=[0.5, 0.5]),
    }
    slots = SlotPlanner(v, seed=0).create_slots(8)
    for s in slots:
      assert 'u' in s and 'w' in s

  def test_weighted_pick(self):
    var = VarDef(choices=['a', 'b'], distribution=[0.5, 0.5])
    planner = SlotPlanner({}, seed=0)
    choice, meta = planner.weighted_pick(var)
    assert choice in ('a', 'b')
    assert meta is None
