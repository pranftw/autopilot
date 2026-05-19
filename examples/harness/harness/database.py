"""In-memory retail database loaded from tau-bench-compatible JSON.

Data provenance: the seed file ``harness/db/retail.json`` mirrors the structure
used by `tau-bench <https://github.com/sierra-research/tau-bench>`_ retail
environment.  Products, users, and orders are stored as dicts keyed by their
respective IDs for O(1) lookup.  All mutation methods mirror the semantics of
the corresponding tau-bench tool ``invoke`` bodies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import copy
import json


@dataclass
class RetailDB:
  """In-memory retail database loaded from tau-bench-compatible JSON.

  Attributes:
    products: Dict of product_id -> product dict (with nested variants dict).
    users: Dict of user_id -> user dict.
    orders: Dict of order_id -> order dict.
  """

  products: dict[str, dict] = field(default_factory=dict)
  users: dict[str, dict] = field(default_factory=dict)
  orders: dict[str, dict] = field(default_factory=dict)

  @classmethod
  def from_file(cls, path: Path) -> RetailDB:
    """Load database from a tau-bench-shaped retail JSON file.

    The JSON is expected to have top-level ``products``, ``users``, and
    ``orders`` keys, each mapping IDs to record dicts.

    Args:
      path: Path to the JSON file.

    Returns:
      A populated RetailDB instance.
    """
    raw = json.loads(path.read_text(encoding='utf-8'))
    products = raw.get('products', {})
    users = raw.get('users', {})
    orders = raw.get('orders', {})
    if isinstance(products, list):
      products = {str(p['product_id']): p for p in products}
    if isinstance(users, list):
      users = {str(u['user_id']): u for u in users}
    if isinstance(orders, list):
      orders = {str(o['order_id']): o for o in orders}
    return cls(products=products, users=users, orders=orders)

  def clone(self) -> RetailDB:
    """Return a deep copy for isolated scenario runs."""
    return RetailDB(
      products=copy.deepcopy(self.products),
      users=copy.deepcopy(self.users),
      orders=copy.deepcopy(self.orders),
    )

  # -- lookup helpers --

  def get_user(self, user_id: str) -> dict:
    """Look up a user by ID.

    Args:
      user_id: The user identifier (e.g. ``'sara_doe_496'``).

    Returns:
      The user dict.

    Raises:
      ValueError: If the user is not found.
    """
    if user_id not in self.users:
      raise ValueError('User not found')
    return self.users[user_id]

  def get_order(self, order_id: str) -> dict:
    """Look up an order by ID.

    Args:
      order_id: The order identifier (e.g. ``'#W0001'``).

    Returns:
      The order dict.

    Raises:
      ValueError: If the order is not found.
    """
    if order_id not in self.orders:
      raise ValueError('Order not found')
    return self.orders[order_id]

  def get_product(self, product_id: str) -> dict:
    """Look up a product by ID.

    Args:
      product_id: The product identifier (e.g. ``'6086499569'``).

    Returns:
      The product dict.

    Raises:
      ValueError: If the product is not found.
    """
    if product_id not in self.products:
      raise ValueError('Product not found')
    return self.products[product_id]

  def get_variant(self, product_id: str, variant_id: str) -> dict:
    """Look up a specific variant of a product.

    Args:
      product_id: The product identifier.
      variant_id: The variant (item) identifier within the product.

    Returns:
      The variant dict.

    Raises:
      ValueError: If the product or variant is not found.
    """
    product = self.get_product(product_id)
    variants = product.get('variants', {})
    if variant_id not in variants:
      raise ValueError('Variant not found')
    return variants[variant_id]

  def find_user_by_email(self, email: str) -> str | None:
    """Find a user ID by email address (case-insensitive).

    Args:
      email: The email address to search for.

    Returns:
      The user_id string, or None if no match.
    """
    lower_email = email.lower()
    for user_id, user in self.users.items():
      if user.get('email', '').lower() == lower_email:
        return user_id
    return None

  def find_user_by_name_zip(
    self,
    first_name: str,
    last_name: str,
    zip_code: str,
  ) -> str | None:
    """Find a user ID by first name, last name, and zip (case-insensitive names).

    Args:
      first_name: First name to match.
      last_name: Last name to match.
      zip_code: Zip code to match (exact).

    Returns:
      The user_id string, or None if no match.
    """
    for user_id, user in self.users.items():
      name = user.get('name', {})
      address = user.get('address', {})
      if (
        name.get('first_name', '').lower() == first_name.lower()
        and name.get('last_name', '').lower() == last_name.lower()
        and address.get('zip', '') == zip_code
      ):
        return user_id
    return None

  def list_all_product_types(self) -> dict[str, str]:
    """Return a dict mapping product name -> product_id, sorted by name.

    Returns:
      Sorted dict of {name: product_id}.
    """
    product_dict = {p['name']: p['product_id'] for p in self.products.values()}
    return dict(sorted(product_dict.items()))

  # -- mutation helpers --

  def cancel_order(self, order_id: str, reason: str) -> dict:
    """Cancel a pending order and process refunds.

    Args:
      order_id: The order to cancel.
      reason: Must be ``'no longer needed'`` or ``'ordered by mistake'``.

    Returns:
      The updated order dict.

    Raises:
      ValueError: If the order is not pending or the reason is invalid.
    """
    order = self.get_order(order_id)
    if order['status'] != 'pending':
      raise ValueError('Non-pending order cannot be cancelled')
    valid_reasons = {'no longer needed', 'ordered by mistake'}
    if reason not in valid_reasons:
      raise ValueError('Invalid reason')

    refunds = []
    for payment in order.get('payment_history', []):
      payment_id = payment['payment_method_id']
      refund = {
        'transaction_type': 'refund',
        'amount': payment['amount'],
        'payment_method_id': payment_id,
      }
      refunds.append(refund)
      user = self.get_user(order['user_id'])
      pm = user.get('payment_methods', {}).get(payment_id)
      if pm is not None and pm.get('source') == 'gift_card':
        pm['balance'] = round(pm['balance'] + payment['amount'], 2)

    order['status'] = 'cancelled'
    order['cancel_reason'] = reason
    order['payment_history'].extend(refunds)
    return order

  def modify_order_items(
    self,
    order_id: str,
    item_ids: list[str],
    new_item_ids: list[str],
    payment_method_id: str,
  ) -> dict:
    """Modify items in a pending order to new variants of the same product.

    Args:
      order_id: The order to modify.
      item_ids: Current item IDs to replace.
      new_item_ids: Replacement item IDs (same length, same product types).
      payment_method_id: Payment method for price difference.

    Returns:
      The updated order dict.

    Raises:
      ValueError: On validation failures (non-pending, missing items, etc.).
    """
    order = self.get_order(order_id)
    if order['status'] != 'pending':
      raise ValueError('Non-pending order cannot be modified')

    all_item_ids = [item['item_id'] for item in order['items']]
    for item_id in item_ids:
      if item_ids.count(item_id) > all_item_ids.count(item_id):
        raise ValueError(f'{item_id} not found')

    if len(item_ids) != len(new_item_ids):
      raise ValueError('The number of items to be exchanged should match')

    diff_price = 0.0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
      if item_id == new_item_id:
        raise ValueError('The new item id should be different from the old item id')
      item = next((i for i in order['items'] if i['item_id'] == item_id), None)
      if item is None:
        raise ValueError(f'Item {item_id} not found')
      variant = self.get_variant(item['product_id'], new_item_id)
      if not variant.get('available', False):
        raise ValueError(f'New item {new_item_id} not found or available')
      diff_price += variant['price'] - item['price']

    diff_price = round(diff_price, 2)
    user = self.get_user(order['user_id'])
    pm = user.get('payment_methods', {}).get(payment_method_id)
    if pm is None:
      raise ValueError('Payment method not found')
    if pm.get('source') == 'gift_card' and pm.get('balance', 0) < diff_price:
      raise ValueError('Insufficient gift card balance to pay for the new item')

    order['payment_history'].append(
      {
        'transaction_type': 'payment' if diff_price > 0 else 'refund',
        'amount': abs(diff_price),
        'payment_method_id': payment_method_id,
      }
    )
    if pm.get('source') == 'gift_card':
      pm['balance'] = round(pm['balance'] - diff_price, 2)

    for item_id, new_item_id in zip(item_ids, new_item_ids):
      item = next((i for i in order['items'] if i['item_id'] == item_id), None)
      if item is None:
        raise ValueError(f'Item {item_id} not found')
      variant = self.get_variant(item['product_id'], new_item_id)
      item['item_id'] = new_item_id
      item['price'] = variant['price']
      item['options'] = variant.get('options', {})
    order['status'] = 'pending (item modified)'
    return order

  def modify_order_address(
    self,
    order_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip_code: str,
  ) -> dict:
    """Modify the shipping address of a pending order.

    Args:
      order_id: The order to modify.
      address1: Primary address line.
      address2: Secondary address line.
      city: City.
      state: State.
      country: Country.
      zip_code: Postal code.

    Returns:
      The updated order dict.

    Raises:
      ValueError: If the order is not pending.
    """
    order = self.get_order(order_id)
    if 'pending' not in order['status']:
      raise ValueError('Non-pending order cannot be modified')
    order['address'] = {
      'address1': address1,
      'address2': address2,
      'city': city,
      'state': state,
      'country': country,
      'zip': zip_code,
    }
    return order

  def modify_order_payment(
    self,
    order_id: str,
    payment_method_id: str,
  ) -> dict:
    """Modify the payment method of a pending order.

    Args:
      order_id: The order to modify.
      payment_method_id: New payment method ID.

    Returns:
      The updated order dict.

    Raises:
      ValueError: On validation failures.
    """
    order = self.get_order(order_id)
    if 'pending' not in order['status']:
      raise ValueError('Non-pending order cannot be modified')

    user = self.get_user(order['user_id'])
    pm = user.get('payment_methods', {}).get(payment_method_id)
    if pm is None:
      raise ValueError('Payment method not found')

    history = order.get('payment_history', [])
    if len(history) != 1 or history[0].get('transaction_type') != 'payment':
      raise ValueError('There should be exactly one payment for a pending order')
    if history[0]['payment_method_id'] == payment_method_id:
      raise ValueError('The new payment method should be different from the current one')

    amount = history[0]['amount']
    if pm.get('source') == 'gift_card' and pm.get('balance', 0) < amount:
      raise ValueError('Insufficient gift card balance to pay for the order')

    order['payment_history'].extend(
      [
        {
          'transaction_type': 'payment',
          'amount': amount,
          'payment_method_id': payment_method_id,
        },
        {
          'transaction_type': 'refund',
          'amount': amount,
          'payment_method_id': history[0]['payment_method_id'],
        },
      ]
    )

    if pm.get('source') == 'gift_card':
      pm['balance'] = round(pm['balance'] - amount, 2)
    old_pm = user.get('payment_methods', {}).get(history[0]['payment_method_id'])
    if old_pm is not None and old_pm.get('source') == 'gift_card':
      old_pm['balance'] = round(old_pm['balance'] + amount, 2)

    return order

  def exchange_order_items(
    self,
    order_id: str,
    item_ids: list[str],
    new_item_ids: list[str],
    payment_method_id: str,
  ) -> dict:
    """Exchange items in a delivered order to new variants of the same product.

    Args:
      order_id: The order to exchange items in.
      item_ids: Item IDs to exchange.
      new_item_ids: Replacement item IDs.
      payment_method_id: Payment method for price difference.

    Returns:
      The updated order dict.

    Raises:
      ValueError: On validation failures.
    """
    order = self.get_order(order_id)
    if order['status'] != 'delivered':
      raise ValueError('Non-delivered order cannot be exchanged')

    all_item_ids = [item['item_id'] for item in order['items']]
    for item_id in item_ids:
      if item_ids.count(item_id) > all_item_ids.count(item_id):
        raise ValueError(f'Number of {item_id} not found.')

    if len(item_ids) != len(new_item_ids):
      raise ValueError('The number of items to be exchanged should match.')

    diff_price = 0.0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
      item = next((i for i in order['items'] if i['item_id'] == item_id), None)
      if item is None:
        raise ValueError(f'Item {item_id} not found')
      variant = self.get_variant(item['product_id'], new_item_id)
      if not variant.get('available', False):
        raise ValueError(f'New item {new_item_id} not found or available')
      diff_price += variant['price'] - item['price']

    diff_price = round(diff_price, 2)

    user = self.get_user(order['user_id'])
    pm = user.get('payment_methods', {}).get(payment_method_id)
    if pm is None:
      raise ValueError('Payment method not found')
    if pm.get('source') == 'gift_card' and pm.get('balance', 0) < diff_price:
      raise ValueError('Insufficient gift card balance to pay for the price difference')

    order['status'] = 'exchange requested'
    order['exchange_items'] = sorted(item_ids)
    order['exchange_new_items'] = sorted(new_item_ids)
    order['exchange_payment_method_id'] = payment_method_id
    order['exchange_price_difference'] = diff_price
    return order

  def return_order_items(
    self,
    order_id: str,
    item_ids: list[str],
    payment_method_id: str,
  ) -> dict:
    """Return items from a delivered order.

    Args:
      order_id: The order to return items from.
      item_ids: Item IDs to return.
      payment_method_id: Payment method for the refund.

    Returns:
      The updated order dict.

    Raises:
      ValueError: On validation failures.
    """
    order = self.get_order(order_id)
    if order['status'] != 'delivered':
      raise ValueError('Non-delivered order cannot be returned')

    user = self.get_user(order['user_id'])
    pm = user.get('payment_methods', {}).get(payment_method_id)
    if pm is None:
      raise ValueError('Payment method not found')
    if (
      pm.get('source') != 'gift_card'
      and payment_method_id != order['payment_history'][0]['payment_method_id']
    ):
      raise ValueError('Payment method should be the original payment method')

    all_item_ids = [item['item_id'] for item in order['items']]
    for item_id in item_ids:
      if item_ids.count(item_id) > all_item_ids.count(item_id):
        raise ValueError('Some item not found')

    order['status'] = 'return requested'
    order['return_items'] = sorted(item_ids)
    order['return_payment_method_id'] = payment_method_id
    return order

  def modify_user_address(
    self,
    user_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip_code: str,
  ) -> dict:
    """Modify the default address of a user.

    Args:
      user_id: The user whose address to modify.
      address1: Primary address line.
      address2: Secondary address line.
      city: City.
      state: State.
      country: Country.
      zip_code: Postal code.

    Returns:
      The updated user dict.

    Raises:
      ValueError: If the user is not found.
    """
    user = self.get_user(user_id)
    user['address'] = {
      'address1': address1,
      'address2': address2,
      'city': city,
      'state': state,
      'country': country,
      'zip': zip_code,
    }
    return user
