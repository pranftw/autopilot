"""Tests for harness.database.RetailDB."""

from harness.database import RetailDB
from pathlib import Path
import json
import pytest

# -- fixtures --


def _minimal_retail_data() -> dict:
  """Return a minimal tau-bench-shaped retail data dict."""
  return {
    'products': {
      'prod_1': {
        'name': 'Widget',
        'product_id': 'prod_1',
        'variants': {
          'item_a': {
            'item_id': 'item_a',
            'options': {'color': 'red'},
            'available': True,
            'price': 10.00,
          },
          'item_b': {
            'item_id': 'item_b',
            'options': {'color': 'blue'},
            'available': True,
            'price': 15.00,
          },
          'item_c': {
            'item_id': 'item_c',
            'options': {'color': 'green'},
            'available': False,
            'price': 12.00,
          },
        },
      },
    },
    'users': {
      'user_1': {
        'user_id': 'user_1',
        'name': {'first_name': 'Alice', 'last_name': 'Smith'},
        'address': {
          'address1': '123 Main St',
          'address2': '',
          'city': 'Portland',
          'state': 'OR',
          'country': 'USA',
          'zip': '97201',
        },
        'email': 'alice@example.com',
        'payment_methods': {
          'cc_1': {
            'source': 'credit_card',
            'id': 'cc_1',
            'brand': 'visa',
            'last_four': '4242',
          },
          'gc_1': {
            'source': 'gift_card',
            'id': 'gc_1',
            'balance': 100.00,
          },
        },
        'orders': ['#W0001', '#W0002'],
      },
    },
    'orders': {
      '#W0001': {
        'order_id': '#W0001',
        'user_id': 'user_1',
        'address': {
          'address1': '123 Main St',
          'address2': '',
          'city': 'Portland',
          'state': 'OR',
          'country': 'USA',
          'zip': '97201',
        },
        'items': [
          {
            'name': 'Widget',
            'product_id': 'prod_1',
            'item_id': 'item_a',
            'price': 10.00,
            'options': {'color': 'red'},
          },
        ],
        'status': 'pending',
        'fulfillments': [],
        'payment_history': [
          {
            'transaction_type': 'payment',
            'amount': 10.00,
            'payment_method_id': 'cc_1',
          },
        ],
      },
      '#W0002': {
        'order_id': '#W0002',
        'user_id': 'user_1',
        'address': {
          'address1': '123 Main St',
          'address2': '',
          'city': 'Portland',
          'state': 'OR',
          'country': 'USA',
          'zip': '97201',
        },
        'items': [
          {
            'name': 'Widget',
            'product_id': 'prod_1',
            'item_id': 'item_a',
            'price': 10.00,
            'options': {'color': 'red'},
          },
        ],
        'status': 'delivered',
        'fulfillments': [
          {'tracking_id': ['TRK1'], 'item_ids': ['item_a']},
        ],
        'payment_history': [
          {
            'transaction_type': 'payment',
            'amount': 10.00,
            'payment_method_id': 'cc_1',
          },
        ],
      },
    },
  }


@pytest.fixture
def retail_json(tmp_path: Path) -> Path:
  """Write minimal retail.json to tmp_path and return the path."""
  path = tmp_path / 'retail.json'
  path.write_text(json.dumps(_minimal_retail_data()), encoding='utf-8')
  return path


@pytest.fixture
def db(retail_json: Path) -> RetailDB:
  """Load a RetailDB from the minimal fixture."""
  return RetailDB.from_file(retail_json)


# -- tests --


class TestFromFile:
  def test_products_loaded(self, db: RetailDB) -> None:
    assert len(db.products) == 1
    assert 'prod_1' in db.products

  def test_users_loaded(self, db: RetailDB) -> None:
    assert len(db.users) == 1
    assert 'user_1' in db.users

  def test_orders_loaded(self, db: RetailDB) -> None:
    assert len(db.orders) == 2
    assert '#W0001' in db.orders
    assert '#W0002' in db.orders

  def test_list_format_normalization(self, tmp_path: Path) -> None:
    """from_file handles both list and dict formats for top-level keys."""
    data = {
      'products': [
        {'product_id': 'p1', 'name': 'A', 'variants': {}},
      ],
      'users': [
        {'user_id': 'u1', 'name': {}, 'address': {}, 'email': 'a@b.com'},
      ],
      'orders': [
        {'order_id': '#O1', 'user_id': 'u1', 'status': 'pending'},
      ],
    }
    path = tmp_path / 'list_format.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    loaded = RetailDB.from_file(path)
    assert 'p1' in loaded.products
    assert 'u1' in loaded.users
    assert '#O1' in loaded.orders


class TestClone:
  def test_clone_independence(self, db: RetailDB) -> None:
    clone = db.clone()
    clone.products['prod_1']['name'] = 'MODIFIED'
    assert db.products['prod_1']['name'] == 'Widget'

  def test_clone_has_same_data(self, db: RetailDB) -> None:
    clone = db.clone()
    assert set(clone.products.keys()) == set(db.products.keys())
    assert set(clone.users.keys()) == set(db.users.keys())
    assert set(clone.orders.keys()) == set(db.orders.keys())


class TestFindUser:
  def test_find_user_by_email_found(self, db: RetailDB) -> None:
    result = db.find_user_by_email('alice@example.com')
    assert result == 'user_1'

  def test_find_user_by_email_case_insensitive(self, db: RetailDB) -> None:
    result = db.find_user_by_email('Alice@Example.COM')
    assert result == 'user_1'

  def test_find_user_by_email_not_found(self, db: RetailDB) -> None:
    result = db.find_user_by_email('nobody@example.com')
    assert result is None

  def test_find_user_by_name_zip_found(self, db: RetailDB) -> None:
    result = db.find_user_by_name_zip('Alice', 'Smith', '97201')
    assert result == 'user_1'

  def test_find_user_by_name_zip_case_insensitive(self, db: RetailDB) -> None:
    result = db.find_user_by_name_zip('ALICE', 'smith', '97201')
    assert result == 'user_1'

  def test_find_user_by_name_zip_not_found(self, db: RetailDB) -> None:
    result = db.find_user_by_name_zip('Alice', 'Smith', '00000')
    assert result is None


class TestGetters:
  def test_get_order_found(self, db: RetailDB) -> None:
    order = db.get_order('#W0001')
    assert order['order_id'] == '#W0001'
    assert order['status'] == 'pending'

  def test_get_order_not_found(self, db: RetailDB) -> None:
    with pytest.raises(ValueError, match='Order not found'):
      db.get_order('#WXXX')

  def test_get_product(self, db: RetailDB) -> None:
    product = db.get_product('prod_1')
    assert product['name'] == 'Widget'
    assert 'item_a' in product['variants']

  def test_get_product_not_found(self, db: RetailDB) -> None:
    with pytest.raises(ValueError, match='Product not found'):
      db.get_product('no_such')

  def test_get_user_found(self, db: RetailDB) -> None:
    user = db.get_user('user_1')
    assert user['email'] == 'alice@example.com'

  def test_get_user_not_found(self, db: RetailDB) -> None:
    with pytest.raises(ValueError, match='User not found'):
      db.get_user('no_user')

  def test_get_variant(self, db: RetailDB) -> None:
    variant = db.get_variant('prod_1', 'item_a')
    assert variant['price'] == 10.00

  def test_get_variant_not_found(self, db: RetailDB) -> None:
    with pytest.raises(ValueError, match='Variant not found'):
      db.get_variant('prod_1', 'no_variant')

  def test_list_all_product_types(self, db: RetailDB) -> None:
    result = db.list_all_product_types()
    assert result == {'Widget': 'prod_1'}


class TestCancelOrder:
  def test_cancel_order_success(self, db: RetailDB) -> None:
    order = db.cancel_order('#W0001', 'no longer needed')
    assert order['status'] == 'cancelled'
    assert order['cancel_reason'] == 'no longer needed'
    assert any(p['transaction_type'] == 'refund' for p in order['payment_history'])

  def test_cancel_non_pending_raises(self, db: RetailDB) -> None:
    with pytest.raises(ValueError, match='Non-pending order cannot be cancelled'):
      db.cancel_order('#W0002', 'no longer needed')

  def test_cancel_invalid_reason_raises(self, db: RetailDB) -> None:
    with pytest.raises(ValueError, match='Invalid reason'):
      db.cancel_order('#W0001', 'just because')

  def test_cancel_refunds_gift_card_immediately(self, db: RetailDB) -> None:
    db.orders['#W0001']['payment_history'] = [
      {'transaction_type': 'payment', 'amount': 50.0, 'payment_method_id': 'gc_1'},
    ]
    initial_balance = db.users['user_1']['payment_methods']['gc_1']['balance']
    db.cancel_order('#W0001', 'ordered by mistake')
    new_balance = db.users['user_1']['payment_methods']['gc_1']['balance']
    assert new_balance == initial_balance + 50.0


class TestMutationHelpers:
  def test_modify_order_address(self, db: RetailDB) -> None:
    order = db.modify_order_address(
      '#W0001',
      '999 New St',
      'Apt 2',
      'Denver',
      'CO',
      'USA',
      '80202',
    )
    assert order['address']['city'] == 'Denver'
    assert order['address']['zip'] == '80202'

  def test_modify_order_address_non_pending_raises(self, db: RetailDB) -> None:
    with pytest.raises(ValueError, match='Non-pending order cannot be modified'):
      db.modify_order_address(
        '#W0002',
        '999 New St',
        '',
        'Denver',
        'CO',
        'USA',
        '80202',
      )

  def test_modify_order_items(self, db: RetailDB) -> None:
    order = db.modify_order_items('#W0001', ['item_a'], ['item_b'], 'cc_1')
    assert order['status'] == 'pending (item modified)'
    assert order['items'][0]['item_id'] == 'item_b'
    assert order['items'][0]['price'] == 15.00

  def test_modify_order_payment(self, db: RetailDB) -> None:
    order = db.modify_order_payment('#W0001', 'gc_1')
    assert len(order['payment_history']) == 3

  def test_exchange_delivered_order(self, db: RetailDB) -> None:
    order = db.exchange_order_items('#W0002', ['item_a'], ['item_b'], 'cc_1')
    assert order['status'] == 'exchange requested'
    assert order['exchange_items'] == ['item_a']
    assert order['exchange_new_items'] == ['item_b']
    assert order['exchange_price_difference'] == 5.0

  def test_return_delivered_order(self, db: RetailDB) -> None:
    order = db.return_order_items('#W0002', ['item_a'], 'cc_1')
    assert order['status'] == 'return requested'
    assert order['return_items'] == ['item_a']

  def test_modify_user_address(self, db: RetailDB) -> None:
    user = db.modify_user_address(
      'user_1',
      '555 Elm',
      '',
      'Austin',
      'TX',
      'USA',
      '73301',
    )
    assert user['address']['city'] == 'Austin'
