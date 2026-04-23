# AutoPilot

PyTorch/Lightning-inspired optimization framework for non-differentiable systems.
forward -> loss -> backward -> optimizer.step() for prompts, configs, code, pipelines.

## Architecture

- **core**: Module, Parameter, Gradient, Loss, Optimizer, Metric, Trainer, Experiment, Store, Memory, Graph, Callbacks, Loops
- **ai**: PathParameter, FileStore, AgentOptimizer, JudgeLoss, ClaudeCodeAgent, TextGradient, GeneratorAgent, JudgeAgent
- **data**: Dataset, DataLoader, DataModule
- **policy**: Policy, Gate hierarchy
- **tracking**: manifest, events, I/O primitives
- **cli**: workspace/project/experiment/optimize/ai commands

## Design principles

- Usability over performance. Simple over easy. Progressive disclosure.
- isinstance on core classes only (Module, Parameter, Gradient, Datum), never concrete leaves.
- Store interacts with parameters only through snapshot()/restore().
- All customization hooks are public methods (never underscore-prefixed).
- Components are Python objects, not string-key lookups. No registries.
- Workflows in code, not config files.

## Style rules

- Google Python Style Guide baseline
- 2-space indentation (ruff enforced)
- Single quotes everywhere
- Absolute imports only, from-imports first, no blank lines between
- No relative imports, no dynamic imports, no deferred imports
- No `if TYPE_CHECKING:` blocks
- No `__init__.py` files -- all imports from terminal files directly
- Line length: 100

## Prohibited patterns

- No `getattr(args, 'x', default)` on declared argparse arguments
- No fake fallback objects on precondition failure
- No inline file-content strings for templates
- No module-level constants caching function calls
- No `str = ''` defaults (use `str | None = None`)
- No `.get('key', '')` (use `d['key']` or `.get('key')`)
- No `except Exception: pass`
- No `# noqa` comments
- No os.environ/os.getenv/load_dotenv inside src/autopilot/

## DRY rules

- One canonical implementation per concern
- I/O goes through tracking/io.py
- Path computation goes through core/paths.py
- Serialization uses DictMixin from core/serialization.py
- Command handlers orchestrate only; no duplicated backend logic

## Testing

- Unit tests for every function: normal, edge, error cases
- Dataclass round-trips: to_dict -> from_dict -> assert equal
- Integration tests for end-to-end flows
- Use tmp_path fixture, not filesystem mocking

## Safety

Never commit .env, API tokens, raw execution logs, or large generated outputs.
