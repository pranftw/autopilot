"""Computation graph: PyTorch-style autograd for AutoPilot.

Node records a Module.__call__ invocation. Graph is an explicit DAG of Nodes.
backward() traverses via dependency-counting (PyTorch engine algorithm).
no_grad()/enable_grad() control recording via ContextVar (graph ON by default).
"""

from autopilot.core.models import Datum
from autopilot.core.parameter import Parameter
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any, Iterator
import heapq
import weakref

_grad_enabled: ContextVar[bool] = ContextVar('_grad_enabled', default=True)
_current_graph: ContextVar['Graph | None'] = ContextVar('_current_graph', default=None)


def is_grad_enabled() -> bool:
  """Like torch.is_grad_enabled(). True by default."""
  return _grad_enabled.get(True)


def get_current_graph() -> 'Graph':
  """Get or lazily create the current computation graph."""
  graph = _current_graph.get(None)
  if graph is None:
    graph = Graph()
    _current_graph.set(graph)
  return graph


class no_grad:
  """Disable graph recording. Like torch.no_grad().

  Use for inference or any context where you don't need the computation
  graph (no backward pass planned).
  """

  def __enter__(self) -> 'no_grad':
    self._prev = _grad_enabled.get(True)
    _grad_enabled.set(False)
    return self

  def __exit__(self, *exc: object) -> None:
    _grad_enabled.set(self._prev)


class enable_grad:
  """Re-enable graph recording. Like torch.enable_grad().

  Useful inside a no_grad() block to selectively re-enable recording.
  """

  def __enter__(self) -> 'enable_grad':
    self._prev = _grad_enabled.get(True)
    _grad_enabled.set(True)
    return self

  def __exit__(self, *exc: object) -> None:
    _grad_enabled.set(self._prev)


class RemovableHandle:
  """Hook registration handle. remove() detaches the hook."""

  _next_id: int = 0

  def __init__(self, hooks_dict: dict) -> None:
    self.id = RemovableHandle._next_id
    RemovableHandle._next_id += 1
    self._hooks_dict_ref = weakref.ref(hooks_dict)

  def remove(self) -> None:
    hooks_dict = self._hooks_dict_ref()
    if hooks_dict is not None and self.id in hooks_dict:
      del hooks_dict[self.id]


class Node:
  """Records a single Module.__call__ invocation. Like torch.autograd.graph.Node."""

  def __init__(
    self,
    module: Any = None,
    next_functions: tuple[tuple['Node | None', int], ...] = (),
    sequence_nr: int = 0,
  ) -> None:
    self._module = module
    self._next_functions = next_functions
    self._hooks: OrderedDict[int, Any] = OrderedDict()
    self._prehooks: OrderedDict[int, Any] = OrderedDict()
    self._sequence_nr = sequence_nr

  def name(self) -> str:
    if self._module is not None:
      return type(self._module).__name__
    return 'Node'

  @property
  def next_functions(self) -> tuple[tuple['Node | None', int], ...]:
    return self._next_functions

  def register_hook(self, fn: Any) -> RemovableHandle:
    handle = RemovableHandle(self._hooks)
    self._hooks[handle.id] = fn
    return handle

  def register_prehook(self, fn: Any) -> RemovableHandle:
    handle = RemovableHandle(self._prehooks)
    self._prehooks[handle.id] = fn
    return handle

  def __call__(self, *grads: Any) -> tuple:
    for hook in self._prehooks.values():
      result = hook(self, grads)
      if result is not None:
        grads = result

    output = self.apply(*grads)

    for hook in self._hooks.values():
      result = hook(self, grads, output)
      if result is not None:
        output = result

    return output

  def apply(self, *grads: Any) -> tuple:
    return grads

  def __repr__(self) -> str:
    return f'{self.name()}(seq={self._sequence_nr})'


class AccumulateGrad(Node):
  """Leaf node for Parameters. Accumulates into Parameter.grad."""

  def __init__(self, parameter: Any, sequence_nr: int) -> None:
    super().__init__(module=None, next_functions=(), sequence_nr=sequence_nr)
    self._parameter = parameter

  def name(self) -> str:
    return 'AccumulateGrad'

  def apply(self, *grads: Any) -> tuple[()]:
    if grads and grads[0] is not None:
      if self._parameter.grad is None:
        self._parameter.grad = grads[0]
      else:
        self._parameter.grad = self._parameter.grad + grads[0]
    return ()


class Graph:
  """Explicit computation graph. Container for Nodes."""

  def __init__(self) -> None:
    self._nodes: list[Node] = []
    self._sequence_nr: int = 0

  def _next_sequence_nr(self) -> int:
    nr = self._sequence_nr
    self._sequence_nr += 1
    return nr

  def record(
    self,
    module: Any,
    inputs: Any,
    output: Any,
    prev_nodes: list[tuple[Node | None, int]],
  ) -> Node:
    """Create a Node for this operation and add it to the graph."""
    node = Node(
      module=module,
      next_functions=tuple(prev_nodes),
      sequence_nr=self._next_sequence_nr(),
    )
    self._nodes.append(node)
    return node

  def nodes(self) -> Iterator[Node]:
    yield from self._nodes

  def backward(
    self,
    target: Node,
    grad: Any = None,
    retain_graph: bool = False,
  ) -> None:
    """Dependency-counting backward traversal. Like PyTorch engine."""
    deps: dict[int, int] = {}
    visited: set[int] = set()
    queue = [target]
    while queue:
      node = queue.pop()
      nid = id(node)
      if nid in visited:
        continue
      visited.add(nid)
      for prev_node, _ in node.next_functions:
        if prev_node is not None:
          pid = id(prev_node)
          deps[pid] = deps.get(pid, 0) + 1
          queue.append(prev_node)

    ready: list[tuple[int, int, Node, Any]] = []
    counter = 0
    heapq.heappush(ready, (-target._sequence_nr, counter, target, grad))
    counter += 1

    processed: set[int] = set()

    while ready:
      _, _, node, node_grad = heapq.heappop(ready)
      nid = id(node)
      if nid in processed:
        continue
      processed.add(nid)

      output_grads = node(node_grad)

      for i, (prev_node, _) in enumerate(node.next_functions):
        if prev_node is None:
          continue
        pid = id(prev_node)
        deps[pid] = deps.get(pid, 1) - 1
        prev_grad = output_grads[i] if i < len(output_grads) else None
        if deps.get(pid, 0) <= 0:
          heapq.heappush(ready, (-prev_node._sequence_nr, counter, prev_node, prev_grad))
          counter += 1

    if not retain_graph:
      self.reset()

  def reset(self) -> None:
    """Clear all nodes. Called between epochs or after backward."""
    self._nodes.clear()
    self._sequence_nr = 0

  def __len__(self) -> int:
    return len(self._nodes)

  def __repr__(self) -> str:
    return f'Graph(nodes={len(self._nodes)})'


def _flatten(args: tuple, kwargs: dict) -> Iterator[Any]:
  """Recursively yield all leaf values from args and kwargs."""
  for arg in args:
    if isinstance(arg, (list, tuple)):
      yield from _flatten(tuple(arg), {})
    elif isinstance(arg, dict):
      yield from _flatten((), arg)
    else:
      yield arg
  for v in kwargs.values():
    if isinstance(v, (list, tuple)):
      yield from _flatten(tuple(v), {})
    elif isinstance(v, dict):
      yield from _flatten((), v)
    else:
      yield v


def _collect_input_nodes(
  args: tuple,
  kwargs: dict,
  graph: Graph,
) -> list[tuple[Node | None, int]]:
  """Extract grad_fn from any Datum in args/kwargs. Like collect_next_edges."""
  nodes: list[tuple[Node | None, int]] = []
  for arg in _flatten(args, kwargs):
    if not isinstance(arg, Datum):
      continue
    if hasattr(arg, 'grad_fn') and arg.grad_fn is not None:
      nodes.append((arg.grad_fn, 0))
    elif isinstance(arg, Parameter) and arg.requires_grad:
      acc = getattr(arg, '_grad_accumulator', None)
      if acc is None:
        acc = AccumulateGrad(arg, graph._next_sequence_nr())
        object.__setattr__(arg, '_grad_accumulator', acc)
        graph._nodes.append(acc)
      nodes.append((acc, 0))
  return nodes


def _create_gradient_edge(output: Any, node: Node) -> None:
  """Attach producer Node to output Datum. Like create_gradient_edge."""
  if isinstance(output, Datum):
    object.__setattr__(output, 'grad_fn', node)
