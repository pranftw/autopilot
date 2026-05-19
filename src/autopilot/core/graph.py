"""Computation graph: OperatorNode-based autograd engine for AutoPilot.

Graph is an explicit DAG container driven by OperatorNode (from core/operator.py).
backward() implements a dependency-counting engine with:
  Phase 1: DFS with 3-state coloring for cycle detection + dependency counting.
  Phase 2: Heap-ordered (by -sequence_nr) fan-in processing with gradient accumulation.

Note:
  This is a leaf module with zero ``autopilot`` imports. All nodes are duck-typed.

backward() calls node(combined_grad), reads node.next_functions and node.sequence_nr
without importing OperatorNode.

Key classes:
  Graph              -- explicit DAG container. add_node() registers, backward() traverses.
  RemovableHandle    -- hook registration handle with remove().
  no_grad/enable_grad -- context managers for recording control.
"""

from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any
import heapq
import weakref

_grad_enabled: ContextVar[bool] = ContextVar('_grad_enabled', default=True)
current_graph: ContextVar['Graph | None'] = ContextVar('current_graph', default=None)


def is_grad_enabled() -> bool:
  """Like torch.is_grad_enabled(). True by default.

  Returns:
    Whether autograd recording is enabled in the current context.
  """
  return _grad_enabled.get(True)


def get_current_graph() -> 'Graph':
  """Get or lazily create the current computation graph.

  Returns:
    The active ``Graph`` for this context, creating one if needed.
  """
  graph = current_graph.get(None)
  if graph is None:
    graph = Graph()
    current_graph.set(graph)
  return graph


class no_grad:
  """Disable graph recording. Like torch.no_grad().

  Use for inference or any context where you don't need the computation
  graph (no backward pass planned).
  """

  def __enter__(self) -> 'no_grad':
    """Disable grad recording for the duration of the context.

    Returns:
      This context manager instance.
    """
    self._prev = _grad_enabled.get(True)
    _grad_enabled.set(False)
    return self

  def __exit__(self, *exc: object) -> None:
    """Restore the previous grad-enabled flag."""
    _grad_enabled.set(self._prev)


class enable_grad:
  """Re-enable graph recording. Like torch.enable_grad().

  Useful inside a no_grad() block to selectively re-enable recording.
  """

  def __enter__(self) -> 'enable_grad':
    """Force grad recording on for the duration of the context.

    Returns:
      This context manager instance.
    """
    self._prev = _grad_enabled.get(True)
    _grad_enabled.set(True)
    return self

  def __exit__(self, *exc: object) -> None:
    """Restore the previous grad-enabled flag."""
    _grad_enabled.set(self._prev)


class RemovableHandle:
  """Hook registration handle. remove() detaches the hook."""

  _next_id: int = 0

  def __init__(self, hooks_dict: dict) -> None:
    """Register this handle against a weak-referenced hooks mapping.

    Args:
      hooks_dict: Mutable dict of hook id -> callable; weakly held.
    """
    self.id = RemovableHandle._next_id
    RemovableHandle._next_id += 1
    self._hooks_dict_ref = weakref.ref(hooks_dict)

  def remove(self) -> None:
    """Drop this hook from the parent dict if it is still alive."""
    hooks_dict = self._hooks_dict_ref()
    if hooks_dict is not None and self.id in hooks_dict:
      del hooks_dict[self.id]


class Graph:
  """Explicit computation graph driven by OperatorNode.

  Nodes are duck-typed: backward() calls node(combined_grad), reads
  node.next_functions and node.sequence_nr without importing any type.
  """

  def __init__(self) -> None:
    """Create an empty graph with sequence counter zero."""
    self._nodes: list = []
    self._sequence_nr: int = 0
    self._freed: bool = False

  def next_sequence_nr(self) -> int:
    """Return and increment the monotonic sequence counter."""
    nr = self._sequence_nr
    self._sequence_nr += 1
    return nr

  def add_node(self, node: Any) -> None:
    """Register a node (duck-typed) in the graph."""
    self._nodes.append(node)
    self._freed = False

  def nodes(self) -> Iterator[Any]:
    """Iterate over registered nodes.

    Yields:
      Each duck-typed node in registration order.
    """
    yield from self._nodes

  def _count_dependencies(self, root: Any) -> dict[int, int]:
    """Phase 1: DFS cycle detection and fan-in counts per predecessor id.

    Args:
      root: Root duck-typed node with ``next_functions``.

    Returns:
      Map from ``id(previous_node)`` to how many successors still reference it.

    Raises:
      RuntimeError: If a cycle is detected in the node dependency graph.
    """
    # iterative DFS with explicit post-order: counts each edge once and flags
    # back-edges (in_stack) so we fail before backward runs on a cyclic graph.
    unvisited, in_stack, done = 0, 1, 2
    deps: dict[int, int] = {}
    state: dict[int, int] = {}
    stack = [(root, False)]
    while stack:
      node, returning = stack.pop()
      nid = id(node)
      if returning:
        state[nid] = done
        continue
      visit_state = state.get(nid, unvisited)
      if visit_state == done:
        continue
      if visit_state == in_stack:
        msg = 'cycle detected in computation graph'
        raise RuntimeError(msg)
      state[nid] = in_stack
      stack.append((node, True))
      for prev_node, _ in node.next_functions:
        if prev_node is None:
          continue
        prev_id = id(prev_node)
        deps[prev_id] = deps.get(prev_id, 0) + 1
        stack.append((prev_node, False))
    return deps

  def _propagate_grads(
    self,
    node: Any,
    combined_grad: Any,
    deps: dict[int, int],
    pending_grads: dict[int, Any],
    ready: list[tuple[int, int, Any, Any]],
    counter: int,
  ) -> int:
    """Run ``node(combined_grad)`` and enqueue predecessors when fan-in is satisfied.

    Args:
      node: Current duck-typed operator node.
      combined_grad: Accumulated gradient for this node.
      deps: Mutable fan-in counts (decremented as gradients flow backward).
      pending_grads: Partial gradients keyed by ``id(predecessor)``.
      ready: Min-heap of ``(-sequence_nr, counter, node, grad)`` tuples.
      counter: Monotonic tie-breaker for heap-stable ordering.

    Returns:
      Updated counter after any ``heappush`` operations.
    """
    # deps mirrors "pending successors" per predecessor; reaching zero means
    # all upstream grads for that predecessor were accumulated in pending_grads.
    output_grads = node(combined_grad)
    next_counter = counter
    for idx, (prev_node, _) in enumerate(node.next_functions):
      if prev_node is None:
        continue
      prev_id = id(prev_node)
      prev_grad = output_grads[idx] if idx < len(output_grads) else None
      if prev_grad is None:
        continue
      deps[prev_id] = deps.get(prev_id, 1) - 1

      existing = pending_grads.get(prev_id)
      if existing is not None:
        pending_grads[prev_id] = existing.accumulate(prev_grad)
      else:
        pending_grads[prev_id] = prev_grad

      if deps.get(prev_id, 0) <= 0:
        final_grad = pending_grads.pop(prev_id, None)
        heapq.heappush(
          ready,
          (-prev_node.sequence_nr, next_counter, prev_node, final_grad),
        )
        next_counter += 1
    return next_counter

  def _process_backward_queue(
    self,
    root: Any,
    grad: Any,
    deps: dict[int, int],
  ) -> None:
    """Phase 2: heap-ordered fan-in backward from ``root`` using ``deps``.

    Args:
      root: Root node.
      grad: Initial gradient on ``root``.
      deps: Fan-in counts from `_count_dependencies` (mutated during traversal).
    """
    # max-heap via negated sequence_nr: pop highest forward order index first,
    # i.e. reverse forward schedule so gradients flow from outputs toward inputs.
    ready: list[tuple[int, int, Any, Any]] = []
    counter = 0
    heapq.heappush(ready, (-root.sequence_nr, counter, root, grad))
    counter += 1
    processed: set[int] = set()
    pending_grads: dict[int, Any] = {}

    while ready:
      _, _, node, node_grad = heapq.heappop(ready)
      nid = id(node)
      # duplicate heap entries can appear before fan-in completes; first visit wins.
      if nid in processed:
        continue
      processed.add(nid)

      combined_grad = pending_grads.pop(nid, None)
      if combined_grad is not None and node_grad is not None:
        combined_grad = combined_grad.accumulate(node_grad)
      elif node_grad is not None:
        combined_grad = node_grad
      if combined_grad is None:
        continue

      counter = self._propagate_grads(node, combined_grad, deps, pending_grads, ready, counter)

  def backward(self, root: Any, grad: Any, retain_graph: bool = False) -> None:
    """Dependency-counting backward traversal with cycle detection.

    Phase 1: DFS with 3-state coloring for cycle detection + dependency counting.
    Phase 2: Heap-ordered (by -sequence_nr) fan-in processing.

    Raises:
      RuntimeError: When the graph was freed without ``retain_graph=True``, or when
        a cycle is detected in the node dependency graph.
    """
    if self._freed:
      msg = 'graph has been freed; use retain_graph=True'
      raise RuntimeError(msg)

    # phase 1: derive per-node fan-in counts (and reject cycles) without running
    # backward callables, so phase 2 knows when a predecessor accumulated full grad.
    deps = self._count_dependencies(root)
    # phase 2: schedule ``backward`` in reverse forward order with heap tie-breaks.
    self._process_backward_queue(root, grad, deps)

    if not retain_graph:
      self.reset()

  def reset(self) -> None:
    """Clear all nodes. Called between epochs or after backward."""
    self._nodes.clear()
    self._sequence_nr = 0
    self._freed = True

  def __len__(self) -> int:
    """Return the number of registered nodes."""
    return len(self._nodes)

  def __repr__(self) -> str:
    """Return a debug string with the node count."""
    return f'Graph(nodes={len(self._nodes)})'
