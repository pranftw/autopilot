"""ModuleCallOperator -- autograd boundary for Module.__call__.

Lives in its own file (not in ops.py) because it is Module-specific: it
distinguishes leaf vs container modules, wires parameters as implicit graph
inputs, and handles backward_transform dispatch.
"""

from autopilot.core.graph import (
  get_current_graph,
  is_grad_enabled,
)
from autopilot.core.operator import (
  AccumulateGrad,
  Context,
  Operator,
  OperatorNode,
  collect_input_nodes,
)
from autopilot.core.types import Datum
from typing import Any


class ModuleCallOperator(Operator):
  """Autograd boundary for Module.__call__.

  Reimplements graph wiring (does NOT call Operator.apply()) because it has
  fundamentally different input semantics: module params are implicit inputs,
  and leaf vs container detection drives wiring strategy.

  Behaviors:
    Leaf module (output.grad_fn is None):
      Wire ALL parameters recursively via module.parameters() + AccumulateGrad.
    Container module (output.grad_fn is not None, has direct _parameters):
      Preserve inner graph; only wire direct _parameters plus inner grad_fn
      and input nodes.
    Container, no direct params:
      No wrapper node -- inner graph is authoritative.
    Non-Datum output:
      No grad_fn, no new graph node (silent).
    Tuple/list output:
      No validation (unlike base Operator.apply); non-Datum outputs get no grad_fn.
  """

  @staticmethod
  def forward(ctx, module, *args, **kwargs):
    """Run ``module.forward`` and stash call inputs for backward.

    Returns:
      Whatever ``module.forward`` returns (typically a ``Datum``).
    """
    ctx.save_for_backward(module, args, kwargs)
    return module.forward(*args, **kwargs)

  @staticmethod
  def backward(ctx: Context, *grads: Any) -> Any:
    """Split or broadcast ``grad_output`` across ``ctx.n_next_functions`` edges.

    Returns:
      Tuple of gradients aligned with downstream ``next_functions`` fan-in.
    """
    grad_output = grads[0] if grads else None
    module = ctx.saved[0]
    total = ctx.n_next_functions

    transformed = module.backward_transform(ctx, grad_output)
    if transformed is not None:
      if isinstance(transformed, tuple) and len(transformed) == total:
        return transformed
      grad = transformed[0] if isinstance(transformed, tuple) else transformed
      return (grad,) * total

    return (grad_output,) * total

  @classmethod
  def apply(cls, module, *args, **kwargs):
    """Record a module call on the graph when appropriate.

    Returns:
      Forward output from ``module``, optionally wired with ``grad_fn``.
    """
    ctx = Context()
    output = cls.forward(ctx, module, *args, **kwargs)

    if is_grad_enabled() and isinstance(output, Datum):
      graph = get_current_graph()
      inner_grad_fn = output.grad_fn

      if inner_grad_fn is not None:
        direct_params = [p for _, p in module.named_parameters(recurse=False)]
        if not direct_params:
          return output
        if direct_params:
          prev_nodes = [(inner_grad_fn, 0)]
          seen_ids = {id(inner_grad_fn)}
          input_nodes = collect_input_nodes(args, kwargs, graph)
          for n, i in input_nodes:
            if n is not None and id(n) not in seen_ids:
              prev_nodes.append((n, i))
              seen_ids.add(id(n))
          for p in direct_params:
            acc = AccumulateGrad.get_or_create(p, graph)
            if id(acc) not in seen_ids:
              prev_nodes.append((acc, 0))
              seen_ids.add(id(acc))
          ctx.n_next_functions = len(prev_nodes)
          node = OperatorNode(
            operator_cls=cls,
            ctx=ctx,
            next_functions=tuple(prev_nodes),
            sequence_nr=graph.next_sequence_nr(),
          )
          graph.add_node(node)
          output.grad_fn = node
      else:
        prev_nodes = collect_input_nodes(args, kwargs, graph)
        seen_ids = {id(n) for n, _ in prev_nodes if n is not None}
        for param in module.parameters():
          acc = AccumulateGrad.get_or_create(param, graph)
          if id(acc) not in seen_ids:
            prev_nodes.append((acc, 0))
            seen_ids.add(id(acc))
        ctx.n_next_functions = len(prev_nodes)
        node = OperatorNode(
          operator_cls=cls,
          ctx=ctx,
          next_functions=tuple(prev_nodes),
          sequence_nr=graph.next_sequence_nr(),
        )
        graph.add_node(node)
        output.grad_fn = node
    return output
