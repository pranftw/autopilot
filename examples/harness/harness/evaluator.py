"""Conversation evaluator for tau-bench-style scenario scoring.

Provides ``EvaluationResult`` (structured metric container) and
``ConversationEvaluator`` (static scorer).

Metric definitions:

- **tool_recall**: fraction of expected actions matched in agent tool calls
  (vacuous ``1.0`` when no expected actions).
- **tool_precision**: fraction of agent tool calls (excluding skip names) that
  match an expected action (vacuous ``1.0`` when denominator is zero).
- **tool_argument_accuracy**: fraction of expected actions whose matched agent
  call has correct argument subset (vacuous ``1.0`` when no expected actions).
- **communication_recall**: fraction of required ``communicate_info`` strings
  found (case-insensitive) in assistant messages (vacuous ``1.0``).
- **policy_compliance**: fraction of ``nl_assertions`` satisfied by keyword
  heuristic against evidence text (vacuous ``1.0``).
- **task_success**: ``True`` only when not errored and every active dimension
  (tool recall, tool argument accuracy, tool precision, communication recall,
  policy compliance) is at full score.

Skipped tool names (excluded from precision denominator):
``TOOL_PRECISION_SKIP_NAMES`` -- at least ``'think'`` and ``'calculate'``.

Metadata round-trip key: ``metadata['eval_result']`` via
``EvaluationResult.from_metadata``.
"""

from dataclasses import dataclass, field
from harness.agent import ConversationResult
from harness.database import RetailDB
import re

TOOL_PRECISION_SKIP_NAMES = frozenset({'think', 'calculate'})

MIN_TOKEN_LEN = 4

FLOAT_EPSILON = 1e-9

NEGATION_PREFIXES = frozenset({'not ', "don't ", 'never '})


@dataclass
class EvaluationResult:
  """Structured evaluation of a completed conversation.

  Attributes:
    task_success: Whether all active dimensions scored perfectly.
    tool_recall: Fraction of expected tools matched.
    tool_precision: Fraction of non-skip agent calls that match expected.
    tool_argument_accuracy: Fraction of expected tools with correct args.
    communication_recall: Fraction of required info communicated.
    policy_compliance: Fraction of NL assertions satisfied.
    turns: Number of conversation turns.
    errored: Whether the conversation errored.
    details: Per-action match info, spurious calls, assertion breakdowns.
  """

  task_success: bool
  tool_recall: float
  tool_precision: float
  tool_argument_accuracy: float
  communication_recall: float
  policy_compliance: float
  turns: int
  errored: bool
  details: dict = field(default_factory=dict)

  @classmethod
  def error(cls) -> 'EvaluationResult':
    """Create an error result (all zeros, errored=True).

    Returns:
      An ``EvaluationResult`` with all metrics at zero and ``errored=True``.
    """
    return cls(
      task_success=False,
      tool_recall=0.0,
      tool_precision=0.0,
      tool_argument_accuracy=0.0,
      communication_recall=0.0,
      policy_compliance=0.0,
      turns=0,
      errored=True,
    )

  def to_dict(self) -> dict:
    """Serialize for ``EvalDatum.metadata`` storage.

    Returns:
      JSON-friendly dict with all fields.
    """
    return {
      'task_success': self.task_success,
      'tool_recall': self.tool_recall,
      'tool_precision': self.tool_precision,
      'tool_argument_accuracy': self.tool_argument_accuracy,
      'communication_recall': self.communication_recall,
      'policy_compliance': self.policy_compliance,
      'turns': self.turns,
      'errored': self.errored,
      'details': self.details,
    }

  @classmethod
  def from_dict(cls, data: dict) -> 'EvaluationResult':
    """Deserialize from dict.

    Returns ``error()`` when the dict is empty, missing ``task_success``,
    or missing any other required key (``KeyError``).

    Args:
      data: Dict previously produced by ``to_dict()``.

    Returns:
      Reconstructed ``EvaluationResult``, or ``error()`` on missing keys.
    """
    if not data or 'task_success' not in data:
      return cls.error()
    try:
      return cls(
        task_success=data['task_success'],
        tool_recall=data['tool_recall'],
        tool_precision=data['tool_precision'],
        tool_argument_accuracy=data['tool_argument_accuracy'],
        communication_recall=data['communication_recall'],
        policy_compliance=data['policy_compliance'],
        turns=data['turns'],
        errored=data['errored'],
        details=data.get('details', {}),
      )
    except KeyError:
      return cls.error()

  @classmethod
  def from_metadata(cls, metadata: dict) -> 'EvaluationResult':
    """Extract from ``EvalDatum.metadata['eval_result']``.

    Args:
      metadata: The full metadata dict (with an ``'eval_result'`` key).

    Returns:
      Deserialized ``EvaluationResult``.
    """
    return cls.from_dict(metadata.get('eval_result', {}))


class ConversationEvaluator:
  """Evaluate conversations against tau-bench criteria.

  All scoring is deterministic and heuristic-based.  Dimensions:

  - Tool matching: order-independent scan of expected vs actual tool calls
  - Communication: case-insensitive substring match for required info
  - NL assertions: keyword/phrase heuristic matching
  - Error detection: conversation failures or timeouts
  """

  @staticmethod
  def evaluate(
    scenario: dict,
    conv_result: ConversationResult,
    db: RetailDB,
  ) -> EvaluationResult:
    """Evaluate a completed conversation.

    Args:
      scenario: Scenario dict (``EvalDatum.metadata`` shape from plan 01).
      conv_result: Outcome of ``HarnessAgent.run_conversation`` (plan 02).
      db: Retail database (reserved for deeper checks; may be unused in
        Phase A).

    Returns:
      Structured ``EvaluationResult`` including ``task_success`` and
      breakdown details.
    """
    if conv_result.error is not None:
      return EvaluationResult.error()

    criteria = scenario.get('evaluation_criteria', {})
    expected_actions = criteria.get('expected_actions', [])
    communicate_info = criteria.get('communicate_info', [])
    nl_assertions = criteria.get('nl_assertions', [])

    tool_result = _evaluate_tools(expected_actions, conv_result.tool_calls)
    comm_recall, comm_found = _evaluate_communication(communicate_info, conv_result.trajectory)
    policy, nl_satisfied = _evaluate_nl_assertions(nl_assertions, conv_result)

    task_success = _compute_task_success(
      tool_result=tool_result,
      comm_recall=comm_recall,
      policy=policy,
      total_expected=len(expected_actions),
      total_communicate=len(communicate_info),
      total_nl=len(nl_assertions),
    )

    return EvaluationResult(
      task_success=task_success,
      tool_recall=tool_result['recall'],
      tool_precision=tool_result['precision'],
      tool_argument_accuracy=tool_result['argument_accuracy'],
      communication_recall=comm_recall,
      policy_compliance=policy,
      turns=conv_result.turns,
      errored=False,
      details={
        'tool_matches': tool_result['matches'],
        'tool_unmatched': tool_result['unmatched'],
        'tool_spurious': tool_result['spurious'],
        'communication_found': comm_found,
        'nl_satisfied': nl_satisfied,
      },
    )


def _match_expected_to_actual(
  expected_actions: list[dict],
  tool_calls: list[dict],
) -> tuple[int, int, list[dict], list[dict], set[int]]:
  """Match expected tool actions against actual agent tool calls.

  Order-independent scan: each expected action matches at most one actual
  call (first match wins, consumed indices tracked).

  Args:
    expected_actions: List of ``{'tool': str, 'args': dict}``.
    tool_calls: List of ``{'name': str, 'arguments': dict}``.

  Returns:
    Tuple of (matched_count, correct_args_count, matches_list,
    unmatched_list, used_indices_set).
  """
  used_indices: set[int] = set()
  matched_expected = 0
  correct_args = 0
  matches = []
  unmatched = []

  for expected in expected_actions:
    exp_tool = expected.get('tool', '')
    exp_args = expected.get('args', {})
    found = False
    for idx, tc in enumerate(tool_calls):
      if idx in used_indices:
        continue
      if tc.get('name', '') == exp_tool:
        used_indices.add(idx)
        matched_expected += 1
        args_ok = _args_subset_match(exp_args, tc.get('arguments', {}))
        if args_ok:
          correct_args += 1
        matches.append(
          {
            'expected_tool': exp_tool,
            'expected_args': exp_args,
            'actual_args': tc.get('arguments', {}),
            'args_match': args_ok,
          }
        )
        found = True
        break
    if not found:
      unmatched.append({'expected_tool': exp_tool, 'expected_args': exp_args})

  return matched_expected, correct_args, matches, unmatched, used_indices


def _compute_precision_spurious(
  expected_actions: list[dict],
  tool_calls: list[dict],
) -> tuple[float, list[str]]:
  """Compute tool precision and spurious call list.

  Precision = matched non-skip calls / total non-skip calls. Skip names
  (``TOOL_PRECISION_SKIP_NAMES``) are excluded from both numerator and
  denominator.

  Args:
    expected_actions: List of ``{'tool': str, 'args': dict}``.
    tool_calls: List of ``{'name': str, 'arguments': dict}``.

  Returns:
    Tuple of (precision_float, spurious_tool_names_list).
  """
  non_skip_calls = [tc for tc in tool_calls if tc.get('name', '') not in TOOL_PRECISION_SKIP_NAMES]
  matched_for_precision = 0
  for tc in non_skip_calls:
    if any(ea.get('tool', '') == tc.get('name', '') for ea in expected_actions):
      matched_for_precision += 1

  spurious = [
    tc.get('name', '')
    for tc in non_skip_calls
    if not any(ea.get('tool', '') == tc.get('name', '') for ea in expected_actions)
  ]

  total_non_skip = len(non_skip_calls)
  precision = matched_for_precision / total_non_skip if total_non_skip > 0 else 1.0
  return precision, spurious


def _evaluate_tools(
  expected_actions: list[dict],
  tool_calls: list[dict],
) -> dict:
  """Score tool matching: recall, precision, argument accuracy.

  Args:
    expected_actions: List of ``{'tool': str, 'args': dict}``.
    tool_calls: List of ``{'name': str, 'arguments': dict}``.

  Returns:
    Dict with ``recall``, ``precision``, ``argument_accuracy``, ``matches``,
    ``unmatched``, ``spurious``.
  """
  total_expected = len(expected_actions)
  if total_expected == 0:
    precision, spurious = _compute_precision_spurious(expected_actions, tool_calls)
    return {
      'recall': 1.0,
      'precision': precision,
      'argument_accuracy': 1.0,
      'matches': [],
      'unmatched': [],
      'spurious': spurious,
    }

  matched, correct_args, matches, unmatched, _ = _match_expected_to_actual(
    expected_actions,
    tool_calls,
  )
  precision, spurious = _compute_precision_spurious(expected_actions, tool_calls)
  recall = matched / total_expected
  argument_accuracy = correct_args / total_expected

  return {
    'recall': recall,
    'precision': precision,
    'argument_accuracy': argument_accuracy,
    'matches': matches,
    'unmatched': unmatched,
    'spurious': spurious,
  }


def _args_subset_match(expected: dict, actual: dict) -> bool:
  """Check that expected args are a subset of actual args (deep equality).

  Every key in ``expected`` must exist in ``actual`` with an equal value.

  Args:
    expected: Expected argument subset.
    actual: Actual arguments from agent tool call.

  Returns:
    ``True`` when every expected key-value pair matches.
  """
  for key, value in expected.items():
    if key not in actual:
      return False
    if actual[key] != value:
      return False
  return True


def _evaluate_communication(
  communicate_info: list[str],
  trajectory: list[dict],
) -> tuple[float, list[str]]:
  """Score communication recall via case-insensitive substring match.

  Args:
    communicate_info: Required info strings to communicate.
    trajectory: Conversation trajectory dicts.

  Returns:
    Tuple of (recall fraction, list of found info strings).
  """
  total = len(communicate_info)
  if total == 0:
    return 1.0, []

  assistant_text = ' '.join(
    entry.get('content', '') for entry in trajectory if entry.get('role') == 'assistant'
  ).lower()

  found = [info for info in communicate_info if info.lower() in assistant_text]
  return len(found) / total, found


def _evaluate_nl_assertions(
  nl_assertions: list[str],
  conv_result: ConversationResult,
) -> tuple[float, list[str]]:
  """Score NL assertion compliance via keyword heuristic.

  Args:
    nl_assertions: List of NL assertion strings.
    conv_result: The conversation result for evidence extraction.

  Returns:
    Tuple of (compliance fraction, list of satisfied assertion strings).
  """
  total = len(nl_assertions)
  if total == 0:
    return 1.0, []

  evidence = _build_evidence(conv_result)
  satisfied = [a for a in nl_assertions if _check_assertion(a, evidence)]
  return len(satisfied) / total, satisfied


def _build_evidence(conv_result: ConversationResult) -> str:
  """Build evidence text from assistant messages and tool calls.

  Args:
    conv_result: The conversation result.

  Returns:
    Lowercased evidence string for keyword matching.
  """
  parts = []
  for entry in conv_result.trajectory:
    if entry.get('role') == 'assistant':
      parts.append(entry.get('content', ''))
  for tc in conv_result.tool_calls:
    parts.append(f'{tc.get("name", "")} {tc.get("arguments", {})}')
  return ' '.join(parts).lower()


def _tokenize_assertion(assertion: str) -> list[str]:
  """Split assertion into key phrase tokens.

  Splits on punctuation, lowercases, filters by ``MIN_TOKEN_LEN``.

  Args:
    assertion: The assertion text.

  Returns:
    List of lowercase token strings.
  """
  raw_tokens = re.split(r'[^\w]+', assertion.lower())
  return [t for t in raw_tokens if len(t) >= MIN_TOKEN_LEN]


def _check_assertion(assertion: str, evidence: str) -> bool:
  """Check a single NL assertion against evidence.

  Positive assertions are satisfied when a majority of key tokens appear
  in the evidence.  Negative assertions (prefixed with ``not`` / ``don't``
  / ``never``) are satisfied when a majority of key tokens do NOT appear.

  Args:
    assertion: The assertion text.
    evidence: Lowercased evidence string.

  Returns:
    ``True`` when the assertion is satisfied.
  """
  is_negative = _is_negative_assertion(assertion)
  tokens = _tokenize_assertion(assertion)
  if not tokens:
    return True

  hits = sum(1 for t in tokens if t in evidence)
  majority = len(tokens) / 2

  if is_negative:
    return hits < majority
  return hits >= majority


def _is_negative_assertion(assertion: str) -> bool:
  """Detect negative polarity from leading negation keywords.

  Args:
    assertion: The assertion text.

  Returns:
    ``True`` when the assertion starts with a negation prefix.
  """
  lower = assertion.lower().lstrip()
  return any(lower.startswith(prefix) for prefix in NEGATION_PREFIXES)


def _compute_task_success(
  tool_result: dict,
  comm_recall: float,
  policy: float,
  total_expected: int,
  total_communicate: int,
  total_nl: int,
) -> bool:
  """Determine overall task success from individual dimensions.

  ``True`` only when every active dimension is at full score:
  tool recall, tool argument accuracy, tool precision (when expected
  actions exist), communication recall (when communicate_info exists),
  and policy compliance (when nl_assertions exist).

  Args:
    tool_result: Dict with ``recall``, ``argument_accuracy``, ``precision``.
    comm_recall: Communication recall score.
    policy: Policy compliance score.
    total_expected: Number of expected tool actions.
    total_communicate: Number of required communication strings.
    total_nl: Number of NL assertions.

  Returns:
    ``True`` when all active dimensions are perfect.
  """
  if total_expected > 0:
    if tool_result['recall'] < 1.0 - FLOAT_EPSILON:
      return False
    if tool_result['argument_accuracy'] < 1.0 - FLOAT_EPSILON:
      return False
    if tool_result['precision'] < 1.0 - FLOAT_EPSILON:
      return False
  if total_communicate > 0:
    if comm_recall < 1.0 - FLOAT_EPSILON:
      return False
  if total_nl > 0:
    if policy < 1.0 - FLOAT_EPSILON:
      return False
  return True
