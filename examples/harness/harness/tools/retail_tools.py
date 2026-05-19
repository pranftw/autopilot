"""Optimizable retail tool functions -- PathParameter target.

This file is exec'd by ``load_tools`` in a sandbox namespace that provides
``RetailDB``, ``HarnessDeps``, and ``RunContext``.  The AgentOptimizer
rewrites tool docstrings, validation logic, and error messages across epochs.
Tool names must remain in ``TOOL_NAMES`` from ``harness.tool_loader``.
"""

import json


def calculate(ctx: 'RunContext[HarnessDeps]', expression: str) -> str:
  """Calculate the result of a mathematical expression.

  Use this tool to compute numeric results for expressions involving prices,
  quantities, or totals. Always communicate the final result to the customer.

  Args:
    ctx: Pydantic AI run context with dependencies.
    expression: Math expression using +, -, *, /, parentheses (e.g. '2 + 2').

  Returns:
    A numeric result as a string (e.g. '4.0'), or error message starting with
    'Error:' describing what went wrong (invalid characters, syntax error, etc).
  """
  ctx.deps.tool_log.append({'tool': 'calculate', 'args': {'expression': expression}})
  if not all(c in '0123456789+-*/(). ' for c in expression):
    return 'Error: Invalid characters in expression. Use only digits, +, -, *, /, parentheses.'
  try:
    result = round(float(eval(expression, {'__builtins__': None}, {})), 2)
  except Exception as exc:
    return f'Error: Invalid expression: {exc}. Please provide a valid math expression.'
  return str(result)


def cancel_pending_order(
  ctx: 'RunContext[HarnessDeps]',
  order_id: str,
  reason: str,
) -> str:
  """Cancel a pending order. The reason must be 'no longer needed' or 'ordered by mistake'.

  REQUIRED: Get explicit customer confirmation before calling this tool. Tell the
  customer about refund timing (gift cards: immediate, other methods: 5-7 business days).
  Always communicate the cancellation confirmation and refund details to the customer.

  Args:
    ctx: Pydantic AI run context with dependencies.
    order_id: The order ID (e.g. '#W0001').
    reason: One of: 'no longer needed' or 'ordered by mistake' (must be exact).

  Returns:
    JSON string of cancelled order details, or error message starting with 'Error:'
    (e.g. 'Error: Order not found', 'Error: Cannot cancel delivered order').
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'cancel_pending_order',
      'args': {'order_id': order_id, 'reason': reason},
    }
  )
  try:
    order = ctx.deps.db.cancel_order(order_id, reason)
    return json.dumps(order, indent=2)
  except ValueError as exc:
    return f'Error: Cannot cancel order: {exc}'


def exchange_delivered_order_items(
  ctx: 'RunContext[HarnessDeps]',
  order_id: str,
  item_ids: list[str],
  new_item_ids: list[str],
  payment_method_id: str,
) -> str:
  """Exchange items in a delivered order for new items of the same product type.

  Can only be done once per delivered order. Items must be the same product type.
  REQUIRED: Get explicit customer confirmation before calling. Communicate the
  new items, any price difference, return instructions, and expected timeline.

  Args:
    ctx: Pydantic AI run context with dependencies.
    order_id: The order ID (e.g. '#W0001').
    item_ids: Item IDs to exchange (list of strings).
    new_item_ids: Replacement item IDs (must be same product type, same count).
    payment_method_id: Payment method ID for any price difference.

  Returns:
    JSON string of updated order with new items, or error message starting with
    'Error:' (e.g. 'Error: Can only exchange once per order', 'Error: Items must be same type').
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'exchange_delivered_order_items',
      'args': {
        'order_id': order_id,
        'item_ids': item_ids,
        'new_item_ids': new_item_ids,
        'payment_method_id': payment_method_id,
      },
    }
  )
  try:
    order = ctx.deps.db.exchange_order_items(
      order_id,
      item_ids,
      new_item_ids,
      payment_method_id,
    )
    return json.dumps(order, indent=2)
  except ValueError as exc:
    return f'Error: Cannot exchange items: {exc}'


def find_user_id_by_email(ctx: 'RunContext[HarnessDeps]', email: str) -> str:
  """Find a user ID by their email address. Use this as the first step to verify customer identity.

  Always ask the customer to confirm their email address matches before proceeding
  to access their account. This is the preferred method of customer identification.

  Args:
    ctx: Pydantic AI run context with dependencies.
    email: The user's email address (e.g. 'customer@example.com').

  Returns:
    The user ID string (e.g. 'sara_doe_496') if found, or error message starting
    with 'Error:' if the email is not in the system.
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'find_user_id_by_email',
      'args': {'email': email},
    }
  )
  user_id = ctx.deps.db.find_user_by_email(email)
  if user_id is None:
    return f'Error: No account found with email {email}. Customer may have misspelled email or not have an account yet.'
  return user_id


def find_user_id_by_name_zip(
  ctx: 'RunContext[HarnessDeps]',
  first_name: str,
  last_name: str,
  zip: str,
) -> str:
  """Find a user ID by first name, last name, and zip code.

  Use this ONLY when the customer cannot provide their email address. Always
  confirm with the customer that their name and zip code match before proceeding.

  Args:
    ctx: Pydantic AI run context with dependencies.
    first_name: Customer's first name.
    last_name: Customer's last name.
    zip: Customer's zip code (5 digits, e.g. '12345').

  Returns:
    The user ID string (e.g. 'sara_doe_496') if found, or error message starting
    with 'Error:' if no match found for the combination.
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'find_user_id_by_name_zip',
      'args': {'first_name': first_name, 'last_name': last_name, 'zip': zip},
    }
  )
  user_id = ctx.deps.db.find_user_by_name_zip(first_name, last_name, zip)
  if user_id is None:
    return f'Error: No account found matching {first_name} {last_name} with zip {zip}. Ask customer to verify spelling or provide email address instead.'
  return user_id


def get_order_details(ctx: 'RunContext[HarnessDeps]', order_id: str) -> str:
  """Get the status and details of an order.

  Use this to look up any order information. Always communicate the order status,
  items, total price, and estimated delivery date (if applicable) to the customer.

  Args:
    ctx: Pydantic AI run context with dependencies.
    order_id: The order ID (e.g. '#W0001').

  Returns:
    JSON string containing order details (status, items, price, tracking),
    or error message starting with 'Error:' if order not found.
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'get_order_details',
      'args': {'order_id': order_id},
    }
  )
  try:
    order = ctx.deps.db.get_order(order_id)
    return json.dumps(order, indent=2)
  except ValueError as exc:
    return f'Error: Order {order_id} not found: {exc}'


def get_product_details(ctx: 'RunContext[HarnessDeps]', product_id: str) -> str:
  """Get the inventory details of a product including all variants.

  Use this when you need to see product options or availability (e.g. for exchanges,
  modifications, or customer inquiries). Communicate available variants and prices to the customer.

  Args:
    ctx: Pydantic AI run context with dependencies.
    product_id: The product ID (different from item/variant ID, e.g. 'PROD_001').

  Returns:
    JSON string containing product name, description, and all available variants
    with prices and inventory, or error message starting with 'Error:' if not found.
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'get_product_details',
      'args': {'product_id': product_id},
    }
  )
  try:
    product = ctx.deps.db.get_product(product_id)
    return json.dumps(product, indent=2)
  except ValueError as exc:
    return f'Error: Product not found: {exc}'


def get_user_details(ctx: 'RunContext[HarnessDeps]', user_id: str) -> str:
  """Get the details of a user, including their orders and payment methods.

  Use this after identifying the customer to retrieve their full account information.
  Communicate relevant details to the customer (name, address, recent orders, payment methods).

  Args:
    ctx: Pydantic AI run context with dependencies.
    user_id: The user ID (e.g. 'sara_doe_496').

  Returns:
    JSON string containing user details (name, address, orders, payment methods),
    or error message starting with 'Error:' if user not found.
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'get_user_details',
      'args': {'user_id': user_id},
    }
  )
  try:
    user = ctx.deps.db.get_user(user_id)
    return json.dumps(user, indent=2)
  except ValueError as exc:
    return f'Error: User not found: {exc}'


def list_all_product_types(ctx: 'RunContext[HarnessDeps]') -> str:
  """List the name and product ID of all product types in the store.

  Args:
    ctx: Pydantic AI run context with dependencies.

  Returns:
    JSON string mapping product names to product IDs, sorted alphabetically.
  """
  ctx.deps.tool_log.append({'tool': 'list_all_product_types', 'args': {}})
  product_dict = ctx.deps.db.list_all_product_types()
  return json.dumps(product_dict, sort_keys=True)


def modify_pending_order_address(
  ctx: 'RunContext[HarnessDeps]',
  order_id: str,
  address1: str,
  address2: str,
  city: str,
  state: str,
  country: str,
  zip: str,
) -> str:
  """Modify the shipping address of a pending order.

  REQUIRED: Get explicit customer confirmation before calling this tool. Tell the
  customer the new address will be used for this order. Communicate the updated
  address and new delivery estimate (if changed) after modification.

  Args:
    ctx: Pydantic AI run context with dependencies.
    order_id: The order ID (e.g. '#W0001').
    address1: Primary address line.
    address2: Secondary address line (can be empty string).
    city: City name.
    state: State or province abbreviation.
    country: Country name.
    zip: Postal code.

  Returns:
    JSON string of updated order with new address, or error message starting with
    'Error:' (e.g. 'Error: Order not found', 'Error: Cannot modify delivered order').
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'modify_pending_order_address',
      'args': {
        'order_id': order_id,
        'address1': address1,
        'address2': address2,
        'city': city,
        'state': state,
        'country': country,
        'zip': zip,
      },
    }
  )
  try:
    order = ctx.deps.db.modify_order_address(
      order_id,
      address1,
      address2,
      city,
      state,
      country,
      zip,
    )
    return json.dumps(order, indent=2)
  except ValueError as exc:
    return f'Error: Cannot modify address: {exc}'


def modify_pending_order_items(
  ctx: 'RunContext[HarnessDeps]',
  order_id: str,
  item_ids: list[str],
  new_item_ids: list[str],
  payment_method_id: str,
) -> str:
  """Modify items in a pending order to new items of the same product type.

  Can only be called once per pending order. REQUIRED: Get explicit customer
  confirmation before calling. Items must be the same product type. Communicate
  the new items, any price difference, and payment method to the customer.

  Args:
    ctx: Pydantic AI run context with dependencies.
    order_id: The order ID (e.g. '#W0001').
    item_ids: Current item IDs being replaced (list of strings).
    new_item_ids: Replacement item IDs (must be same product type, same count).
    payment_method_id: Payment method ID for any price difference.

  Returns:
    JSON string of updated order with new items, or error message starting with
    'Error:' (e.g. 'Error: Items must be same product type', 'Error: Order not found').
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'modify_pending_order_items',
      'args': {
        'order_id': order_id,
        'item_ids': item_ids,
        'new_item_ids': new_item_ids,
        'payment_method_id': payment_method_id,
      },
    }
  )
  try:
    order = ctx.deps.db.modify_order_items(
      order_id,
      item_ids,
      new_item_ids,
      payment_method_id,
    )
    return json.dumps(order, indent=2)
  except ValueError as exc:
    return f'Error: Cannot modify items: {exc}'


def modify_pending_order_payment(
  ctx: 'RunContext[HarnessDeps]',
  order_id: str,
  payment_method_id: str,
) -> str:
  """Modify the payment method of a pending order.

  REQUIRED: Get explicit customer confirmation before calling. The original
  payment method is refunded and the new method is charged. Communicate the
  payment method change and any timing implications to the customer.

  Args:
    ctx: Pydantic AI run context with dependencies.
    order_id: The order ID (e.g. '#W0001').
    payment_method_id: New payment method ID (from customer's account).

  Returns:
    JSON string of updated order with new payment method, or error message starting
    with 'Error:' (e.g. 'Error: Invalid payment method', 'Error: Order not found').
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'modify_pending_order_payment',
      'args': {
        'order_id': order_id,
        'payment_method_id': payment_method_id,
      },
    }
  )
  try:
    order = ctx.deps.db.modify_order_payment(order_id, payment_method_id)
    return json.dumps(order, indent=2)
  except ValueError as exc:
    return f'Error: Cannot modify payment: {exc}'


def modify_user_address(
  ctx: 'RunContext[HarnessDeps]',
  user_id: str,
  address1: str,
  address2: str,
  city: str,
  state: str,
  country: str,
  zip: str,
) -> str:
  """Modify the default address of a user's account (not a specific order).

  This changes the user's profile default address. Use modify_pending_order_address
  to change address for a specific order. REQUIRED: Get explicit customer confirmation
  before calling. Communicate the new default address after modification.

  Args:
    ctx: Pydantic AI run context with dependencies.
    user_id: The user ID (e.g. 'sara_doe_496').
    address1: Primary address line.
    address2: Secondary address line (can be empty string).
    city: City name.
    state: State or province abbreviation.
    country: Country name.
    zip: Postal code.

  Returns:
    JSON string of updated user profile with new default address, or error message
    starting with 'Error:' if user not found or update fails.
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'modify_user_address',
      'args': {
        'user_id': user_id,
        'address1': address1,
        'address2': address2,
        'city': city,
        'state': state,
        'country': country,
        'zip': zip,
      },
    }
  )
  try:
    user = ctx.deps.db.modify_user_address(
      user_id,
      address1,
      address2,
      city,
      state,
      country,
      zip,
    )
    return json.dumps(user, indent=2)
  except ValueError as exc:
    return f'Error: Cannot update address: {exc}'


def return_delivered_order_items(
  ctx: 'RunContext[HarnessDeps]',
  order_id: str,
  item_ids: list[str],
  payment_method_id: str,
) -> str:
  """Return items from a delivered order.

  The order status will change to 'return requested' and the customer receives
  a follow-up email with return instructions. REQUIRED: Get explicit customer
  confirmation before calling. Communicate the return address, instructions,
  and refund timing to the customer.

  Args:
    ctx: Pydantic AI run context with dependencies.
    order_id: The order ID (e.g. '#W0001').
    item_ids: Item IDs to return (list of strings).
    payment_method_id: Payment method ID for the refund.

  Returns:
    JSON string of updated order with return request, or error message starting
    with 'Error:' (e.g. 'Error: Can only return once per order', 'Error: Order not found').
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'return_delivered_order_items',
      'args': {
        'order_id': order_id,
        'item_ids': item_ids,
        'payment_method_id': payment_method_id,
      },
    }
  )
  try:
    order = ctx.deps.db.return_order_items(order_id, item_ids, payment_method_id)
    return json.dumps(order, indent=2)
  except ValueError as exc:
    return f'Error: Cannot process return: {exc}'


def think(ctx: 'RunContext[HarnessDeps]', thought: str) -> str:
  """Use this tool to think about something before acting.

  Does not obtain new information or change the database; simply logs the
  thought. Useful for complex reasoning or caching intermediate conclusions.

  Args:
    ctx: Pydantic AI run context with dependencies.
    thought: A thought to reason about.

  Returns:
    Empty string.
  """
  ctx.deps.tool_log.append({'tool': 'think', 'args': {'thought': thought}})
  return ''


def transfer_to_human_agents(ctx: 'RunContext[HarnessDeps]', summary: str) -> str:
  """Transfer the user to a human agent with a summary of the issue.

  ONLY use this when: (1) the customer explicitly requests a human agent, or
  (2) the issue cannot be resolved with available tools. Always provide a clear
  summary of the customer's issue, what tools have been tried, and what couldn't be resolved.

  Args:
    ctx: Pydantic AI run context with dependencies.
    summary: Detailed summary of customer's issue, attempted resolutions, and why
             human assistance is needed (e.g. 'Customer wants to return order outside
             30-day window. Policy only allows returns within 30 days.').

  Returns:
    Confirmation message that transfer was initiated.
  """
  ctx.deps.tool_log.append(
    {
      'tool': 'transfer_to_human_agents',
      'args': {'summary': summary},
    }
  )
  return 'Transfer successful. A human agent will assist you shortly with your issue.'
