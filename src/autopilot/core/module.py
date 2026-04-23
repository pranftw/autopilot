"""Experiment module base class. Like nn.Module.

Two-class hierarchy:
  Module -- forward() only, core building block
  AutoPilotModule(Module) -- step methods + lifecycle hooks for Trainer
"""

from autopilot.core.graph import (
  RemovableHandle,
  _collect_input_nodes,
  _create_gradient_edge,
  get_current_graph,
  is_grad_enabled,
)
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from collections import OrderedDict
from typing import Any, Iterator


class Module:
  """Base class for experiment modules. Like nn.Module.

  Registration semantics:
    __setattr__ auto-registers: Parameter -> _parameters, child Module -> _modules.
    There is no separate _metrics dict. Metric(Module) and Loss(Module) live in
    _modules as regular child modules.

  Tree traversal API (mirrors nn.Module):
    children(), named_children()             -- immediate children
    modules(), named_modules()               -- full tree (self + descendants)
    parameters(), named_parameters()         -- all Parameter leaves
    train(mode), eval()                      -- toggle training flag recursively
    apply(fn)                                -- post-order fn application
    state_dict(), load_state_dict()          -- checkpoint serialization

  Computation:
    Override forward(*args, **kwargs) -> Datum. Invoked through __call__,
    which also runs pre/post hooks and records to the computation graph
    when grad mode is enabled.

  Hooks:
    register_forward_pre_hook(fn)  -- called before forward()
    register_forward_hook(fn)      -- called after forward()
    Returns a RemovableHandle for later detachment.

  Example::

    class MyModule(Module):
      def __init__(self):
        super().__init__()
        self.encoder = EncoderModule(...)
        self.decoder = DecoderModule(...)

      def forward(self, batch):
        return self.decoder(self.encoder(batch))
  """

  def __init__(self) -> None:
    object.__setattr__(self, '_modules', {})
    object.__setattr__(self, '_parameters', {})
    object.__setattr__(self, '_forward_pre_hooks', OrderedDict())
    object.__setattr__(self, '_forward_hooks', OrderedDict())
    object.__setattr__(self, 'training', True)

  def __setattr__(self, name: str, value: object) -> None:
    """Auto-register child Modules and Parameters. Like nn.Module.__setattr__.

    Competing-store cleanup: remove name from other internal dicts before adding.
    """
    params = self.__dict__.get('_parameters')
    if params is None:
      raise AttributeError('cannot assign before Module.__init__() call')
    modules = self.__dict__.get('_modules')
    if modules is None:
      raise AttributeError('cannot assign before Module.__init__() call')

    if isinstance(value, Parameter):
      modules.pop(name, None)
      params[name] = value
    elif isinstance(value, Module):
      params.pop(name, None)
      modules[name] = value
    else:
      params.pop(name, None)
      modules.pop(name, None)

    object.__setattr__(self, name, value)

  def __getattr__(self, name: str) -> Any:
    """Fallback lookup in _parameters, _modules. Like nn.Module.__getattr__."""
    _parameters = self.__dict__.get('_parameters')
    if _parameters is not None and name in _parameters:
      return _parameters[name]
    _modules = self.__dict__.get('_modules')
    if _modules is not None and name in _modules:
      return _modules[name]
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    """Primary computation. Like nn.Module.forward().

    Override in subclasses. Signature is flexible -- takes runtime data only.
    """
    raise NotImplementedError

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Hook orchestration + graph recording. Like nn.Module._call_impl."""
    for hook in self._forward_pre_hooks.values():
      result = hook(self, args, kwargs)
      if result is not None and isinstance(result, tuple) and len(result) == 2:
        args, kwargs = result

    output = self.forward(*args, **kwargs)

    for hook in self._forward_hooks.values():
      result = hook(self, args, output)
      if result is not None:
        output = result

    if is_grad_enabled():
      graph = get_current_graph()
      prev_nodes = _collect_input_nodes(args, kwargs, graph)
      node = graph.record(self, (args, kwargs), output, prev_nodes)
      _create_gradient_edge(output, node)

    return output

  def register_forward_pre_hook(self, fn: Any) -> RemovableHandle:
    handle = RemovableHandle(self._forward_pre_hooks)
    self._forward_pre_hooks[handle.id] = fn
    return handle

  def register_forward_hook(self, fn: Any) -> RemovableHandle:
    handle = RemovableHandle(self._forward_hooks)
    self._forward_hooks[handle.id] = fn
    return handle

  # tree traversal

  def children(self) -> Iterator['Module']:
    """Immediate child modules. Like nn.Module.children()."""
    yield from self._modules.values()

  def named_children(self) -> Iterator[tuple[str, 'Module']]:
    """Immediate child modules with names. Like nn.Module.named_children()."""
    yield from self._modules.items()

  def modules(self) -> Iterator['Module']:
    """All modules in the tree (self + all descendants). Like nn.Module.modules()."""
    yield self
    for child in self._modules.values():
      yield from child.modules()

  def named_modules(self, prefix: str = '') -> Iterator[tuple[str, 'Module']]:
    """All modules with dotted names. Like nn.Module.named_modules()."""
    yield prefix, self
    for name, child in self._modules.items():
      child_prefix = f'{prefix}.{name}' if prefix else name
      yield from child.named_modules(child_prefix)

  def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    """All parameters. Like nn.Module.parameters()."""
    yield from self._parameters.values()
    if recurse:
      for child in self._modules.values():
        yield from child.parameters(recurse=True)

  def named_parameters(
    self, prefix: str = '', recurse: bool = True
  ) -> Iterator[tuple[str, Parameter]]:
    """All parameters with dotted names. Like nn.Module.named_parameters()."""
    for name, param in self._parameters.items():
      full_name = f'{prefix}.{name}' if prefix else name
      yield full_name, param
    if recurse:
      for mod_name, child in self._modules.items():
        child_prefix = f'{prefix}.{mod_name}' if prefix else mod_name
        yield from child.named_parameters(child_prefix, recurse=True)

  # train/eval mode

  def train(self, mode: bool = True) -> 'Module':
    """Set training mode. Propagates to all children. Like nn.Module.train()."""
    self.training = mode
    for child in self._modules.values():
      child.train(mode)
    return self

  def eval(self) -> 'Module':
    """Set evaluation mode. Like nn.Module.eval()."""
    return self.train(False)

  # apply

  def apply(self, fn: Any) -> 'Module':
    """Apply fn to all children (post-order), then self. Like nn.Module.apply()."""
    for child in self._modules.values():
      child.apply(fn)
    fn(self)
    return self

  # state dict

  def state_dict(self) -> dict[str, Any]:
    """Return module state for checkpointing. Like nn.Module.state_dict()."""
    state: dict[str, Any] = {}
    for name, param in self._parameters.items():
      state[name] = param.to_dict()
    for name, child in self._modules.items():
      child_state = child.state_dict()
      for key, value in child_state.items():
        state[f'{name}.{key}'] = value
    return state

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    """Load module state from checkpoint. Like nn.Module.load_state_dict()."""
    for name, param in self._parameters.items():
      if name in state_dict:
        loaded = Parameter.from_dict(state_dict[name])
        param.requires_grad = loaded.requires_grad
        param.grad = loaded.grad
    for name, child in self._modules.items():
      child_state = {}
      prefix = f'{name}.'
      for key, value in state_dict.items():
        if key.startswith(prefix):
          child_state[key[len(prefix) :]] = value
      if child_state:
        child.load_state_dict(child_state)

  def extra_repr(self) -> str:
    """Override for custom __repr__ content. Like nn.Module.extra_repr()."""
    return ''

  def __repr__(self) -> str:
    """Tree structure repr. Like nn.Module.__repr__."""
    extra = self.extra_repr()
    lines = [f'{type(self).__name__}(']
    if extra:
      lines[0] = f'{type(self).__name__}({extra}'
      if not self._modules and not self._parameters:
        return f'{type(self).__name__}({extra})'
    for name, child in self._modules.items():
      child_repr = repr(child).replace('\n', '\n  ')
      lines.append(f'  ({name}): {child_repr}')
    for name in self._parameters:
      lines.append(f'  ({name}): Parameter')
    if len(lines) == 1:
      return f'{type(self).__name__}()'
    lines.append(')')
    return '\n'.join(lines)


class AutoPilotModule(Module):
  """Module with step methods and lifecycle hooks. Like LightningModule.

  Extends Module with the Trainer integration surface. The Trainer calls
  these methods during fit(); users override them.

  Step methods (override for custom behavior):
    training_step(batch)       -- called per train batch
    validation_step(batch)     -- called per validation batch
    test_step(batch)           -- called per test batch
    configure_optimizers()     -- return an Optimizer (or dict with 'optimizer' key)

  Lifecycle hooks (called by Trainer/EpochLoop):
    setup()                    -- before training starts
    teardown()                 -- after training ends
    on_train_start()           -- before first train batch
    on_train_end()             -- after last train batch
    on_validation_start()      -- before first val batch
    on_validation_end()        -- after last val batch

  Properties:
    trainer                    -- reference to the Trainer, set by Trainer.fit()

  Example::

    class MyModule(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.backend = BackendModule(...)
        self.loss = JudgeLoss(judge=MyJudge(), collator=ConcatCollator())
        self.metrics = AccuracyMetric()

      def forward(self, batch):
        return self.backend(batch)

      def training_step(self, batch):
        return self.forward(batch)

      def configure_optimizers(self):
        return AgentOptimizer(agent=ClaudeCodeAgent(), parameters=self.parameters())
  """

  def __init__(self) -> None:
    super().__init__()
    object.__setattr__(self, '_trainer', None)

  @property
  def trainer(self) -> Any:
    """Reference to the Trainer. Set by Trainer.fit()."""
    return self._trainer

  def training_step(self, batch: Any) -> Any:
    raise NotImplementedError

  def validation_step(self, batch: Any) -> Any:
    raise NotImplementedError

  def test_step(self, batch: Any) -> Any:
    raise NotImplementedError

  def configure_optimizers(self) -> Any:
    raise NotImplementedError

  # lifecycle hooks

  def setup(self) -> None:
    pass

  def teardown(self) -> None:
    pass

  def on_train_start(self) -> None:
    pass

  def on_train_end(self) -> None:
    pass

  def on_validation_start(self) -> None:
    pass

  def on_validation_end(self) -> None:
    pass
