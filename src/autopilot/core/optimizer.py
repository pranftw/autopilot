"""Optimizer base class with param_groups. Like torch.optim.Optimizer (NOT a Module).

Optimizer is not a Module -- it does not participate in the module tree
or auto-registration. It is instantiated separately and passed to the
Trainer via AutoPilotModule.configure_optimizers().

Constructor accepts either a flat list of parameters (auto-wrapped into a
single group) or a list of group dicts, each containing at least
``'params': list[Parameter]`` plus optional per-group hyperparameters.
Hyperparameters missing from a group are filled from ``self.defaults``.

Checkpoint schema (``state_dict`` / ``load_state_dict``)::

    {
      'param_groups': [{'params': [<Parameter.id str>, ...], 'lr': ..., ...}, ...],
      'defaults': {'lr': ..., ...},
      'blocked_strategies': [<str>, ...],
      'state': {<Parameter.id str>: {<key>: <value>, ...}, ...},
    }

Runtime ``self.state`` keys are ``id(param)`` (int) for object-identity
lookup. Checkpoints serialize state under ``Parameter.id`` (stable hex)
because JSON cannot preserve memory addresses.
"""

from autopilot.core.parameter import Parameter
from typing import Any, cast


class Optimizer:
  """Base optimizer with param_groups, per-parameter state, and strategy blocklist.

  Gradients arrive via graph propagation: Loss.backward() seeds the computation
  graph, and AccumulateGrad leaf nodes materialize param.grad during backward
  traversal. Optimizers never assign param.grad directly.

  Attributes:
    defaults: Default hyperparameters for new groups (at least ``'lr'``).
    param_groups: List of dicts, each with ``'params'`` and per-group hparams.
    state: Per-parameter optimizer state keyed by ``id(param)`` at runtime.

  Public iteration surface:
    parameters   -- read-only property returning flattened managed parameters

  Core protocol:
    step()       -- read param.grad, apply changes (subclass implements)
    zero_grad()  -- clear param.grad AND param.grad_accumulator on all parameters
    owns_step_gradient_context -- True when optimizer journals gradients itself (default False)

  Two-phase cleanup contract (per training step):
    Phase 1: graph.reset() clears graph state (automatic at end of backward).
    Phase 2: optimizer.zero_grad() clears parameter state (param.grad and
             param.grad_accumulator) so the next forward creates fresh
             AccumulateGrad nodes.

  Strategy blocklist API (used to prevent retrying failed strategies):
    block_strategy(name)          -- add to blocklist
    unblock_strategy(name)        -- remove from blocklist
    is_strategy_blocked(name)     -- check membership
    blocked_strategies            -- frozenset property of all blocked names

  Checkpoint hooks:
    state_dict()             -- serialize groups, defaults, state, blocklist
    load_state_dict(state)   -- restore from a previously serialized dict

  Built-in subclass: AgentOptimizer (ai/optimizer.py) uses an Agent to apply
  LLM-driven changes. Deterministic subclasses (e.g. RuleOptimizer) skip the LLM.

  Example:
    >>> from autopilot.core.optimizer import Optimizer
    >>> from autopilot.core.parameter import ScalarParameter
    >>>
    >>> class NoOpStep(Optimizer):
    ...   def step(self):
    ...     return None
    >>>
    >>> weight = ScalarParameter(value=1.0)
    >>> optimizer = NoOpStep([weight], lr=0.1)
    >>> optimizer.param_groups[0]['lr']
    0.1
  """

  def __init__(
    self,
    params: list[Parameter] | list[dict[str, Any]],
    *,
    lr: float = 1.0,
    **defaults: Any,
  ) -> None:
    """Initialize with parameters (or param groups) and default hyperparameters.

    Args:
      params: Either a flat list of Parameter objects (auto-wrapped into one
        group) or a list of dicts, each containing ``'params': list[Parameter]``
        plus optional per-group hyperparameters. An empty list yields an
        optimizer with no groups (valid for degenerate tests).
      lr: Default learning rate metadata for all groups (default 1.0).
      **defaults: Additional default hyperparameters seeded into every group.
    """
    self.defaults: dict[str, Any] = {'lr': lr, **defaults}
    self.param_groups: list[dict[str, Any]] = []
    self.state: dict[int, dict[str, Any]] = {}
    self._blocked_strategies: set[str] = set()
    if params and isinstance(params[0], dict):
      for group in cast(list[dict[str, Any]], params):
        self.add_param_group(group)
    elif params:
      self.add_param_group({'params': list(params)})

  def add_param_group(self, group: dict[str, Any]) -> None:
    """Add a parameter group with default hyperparameters filled in.

    Args:
      group: Dict containing at least ``'params': list[Parameter]``.
        Missing hyperparameters are filled from ``self.defaults``.

    Raises:
      TypeError: When ``group`` is missing the ``'params'`` key or
        ``group['params']`` is not a list.
    """
    if 'params' not in group:
      msg = (
        f'parameter group dict must contain a "params" key, '
        f'got keys {sorted(group.keys())}. '
        f'Pass {{"params": [param1, param2, ...], "lr": 0.1}} to add_param_group().'
      )
      raise TypeError(msg)
    if not isinstance(group['params'], list):
      msg = (
        f'parameter group "params" must be a list, '
        f'got {type(group["params"]).__name__}. '
        f'Wrap parameters in a list: {{"params": [param]}}.'
      )
      raise TypeError(msg)
    for key, default in self.defaults.items():
      group.setdefault(key, default)
    self.param_groups.append(group)

  @property
  def parameters(self) -> list[Parameter]:
    """Flattened parameters managed by this optimizer (snapshot order).

    Returns a new list each call; mutating the returned list does not
    affect the optimizer's internal state.

    Returns:
      Shallow copy of all parameters across all groups.
    """
    result: list[Parameter] = []
    for group in self.param_groups:
      result.extend(group['params'])
    return result

  @property
  def owns_step_gradient_context(self) -> bool:
    """Whether this optimizer emits gradient context entries during step().

    When True, Trainer skips its own gradient capture and completion-time
    gradient journal emission, deferring to the optimizer's per-step
    context entries. Default is False (Trainer handles gradient journaling).

    Returns:
      True if the optimizer journals gradients in its step() call.
    """
    return False

  def block_strategy(self, name: str) -> None:
    """Prevent a named strategy from being retried.

    Args:
      name: Strategy identifier to add to the blocklist.
    """
    self._blocked_strategies.add(name)

  def unblock_strategy(self, name: str) -> None:
    """Allow a named strategy to be used again.

    Args:
      name: Strategy identifier to remove from the blocklist.
    """
    self._blocked_strategies.discard(name)

  def is_strategy_blocked(self, name: str) -> bool:
    """Return whether the strategy is currently blocked.

    Args:
      name: Strategy identifier to check.

    Returns:
      True if name is in the blocklist.
    """
    return name in self._blocked_strategies

  @property
  def blocked_strategies(self) -> frozenset[str]:
    """Frozen copy of all blocked strategy names.

    Returns:
      Blocklist as an immutable set.
    """
    return frozenset(self._blocked_strategies)

  def step(self) -> None:
    """Apply one optimizer step using current parameter gradients.

    Subclasses read ``param.grad`` on each managed parameter. There is no
    separate ``grads`` iterable -- gradients are always accessed via
    ``param.grad``.

    Raises:
      NotImplementedError: Subclasses must implement parameter updates.
    """
    raise NotImplementedError

  def zero_grad(self) -> None:
    """Clear gradients and stale AccumulateGrad handles on all managed parameters.

    Clears both param.grad (the gradient value) and param.grad_accumulator
    (the AccumulateGrad node reference from the previous forward graph), including
    for frozen parameters (requires_grad=False). Fresh AccumulateGrad nodes are
    created on the next forward pass via AccumulateGrad.get_or_create(param, graph).
    """
    for param in self.parameters:
      param.grad = None
      param.grad_accumulator = None

  def state_dict(self) -> dict[str, Any]:
    """Serialize optimizer state for checkpointing.

    Returns:
      Dict with ``param_groups`` (params replaced by ``Parameter.id`` lists),
      ``defaults``, ``blocked_strategies``, and per-parameter ``state``
      keyed by ``Parameter.id`` strings.
    """
    serialized_groups = []
    for group in self.param_groups:
      serialized = {k: v for k, v in group.items() if k != 'params'}
      serialized['params'] = [p.id for p in group['params']]
      serialized_groups.append(serialized)

    serialized_state: dict[str, Any] = {}
    for param in self.parameters:
      runtime_key = id(param)
      if runtime_key in self.state:
        serialized_state[param.id] = dict(self.state[runtime_key])

    return {
      'param_groups': serialized_groups,
      'defaults': dict(self.defaults),
      'blocked_strategies': sorted(self._blocked_strategies),
      'state': serialized_state,
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    """Restore optimizer state from a checkpoint dict.

    Validates that group count and per-group parameter counts match the
    current optimizer configuration. Restores ``defaults``, per-group
    hyperparameters, blocked strategies, and per-parameter state.

    Args:
      state: Dict previously returned by :meth:`state_dict`. Must contain
        ``param_groups``, ``defaults``, and ``blocked_strategies``. The
        ``state`` key is optional (defaults to empty dict if absent).

    Raises:
      ValueError: When group count or parameter counts do not match the
        current optimizer, with counts and guidance.
    """
    saved_groups = state['param_groups']
    if len(saved_groups) != len(self.param_groups):
      msg = (
        f'checkpoint has {len(saved_groups)} param group(s) but optimizer has '
        f'{len(self.param_groups)}; checkpoint incompatible with current parameter list'
      )
      raise ValueError(msg)

    for idx, (saved_group, current_group) in enumerate(
      zip(saved_groups, self.param_groups, strict=True)
    ):
      saved_param_ids = saved_group['params']
      current_params = current_group['params']
      if len(saved_param_ids) != len(current_params):
        msg = (
          f'param group {idx}: checkpoint has {len(saved_param_ids)} parameter(s) '
          f'but optimizer has {len(current_params)}; '
          f'checkpoint incompatible with current parameter list'
        )
        raise ValueError(msg)

    self.defaults = dict(state['defaults'])
    self._blocked_strategies = set(state['blocked_strategies'])

    for saved_group, current_group in zip(saved_groups, self.param_groups, strict=True):
      for key, value in saved_group.items():
        if key != 'params':
          current_group[key] = value

    saved_state = state.get('state', {})
    self.state = {}
    saved_id_order: list[str] = []
    for saved_group in saved_groups:
      saved_id_order.extend(saved_group['params'])
    current_params = self.parameters
    for saved_id, param in zip(saved_id_order, current_params, strict=True):
      if saved_id in saved_state:
        self.state[id(param)] = dict(saved_state[saved_id])
