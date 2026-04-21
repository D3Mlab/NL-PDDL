"""Extract forward-executable plans from regression subgoals.

The :class:`NLFOLRegressionPlanner` returns a list of
``(subgoal_formula, reversed_actions, substitution)`` tuples. Each tuple describes
a node in the regression search tree:

* ``subgoal_formula`` is a :class:`DisjunctiveFormula` of conjuncts; any conjunct
  that the world state satisfies is a valid starting point.
* ``reversed_actions`` are the actions that, executed in reverse, take us from
  that starting point to the original goal.
* ``substitution`` carries the variable bindings discovered during regression.

Given a concrete initial state, this module finds every subgoal whose
preconditions the initial state satisfies (in the **subset** sense — the
initial state may contain additional facts) and returns the corresponding
forward plan to the goal. The shortest such plan is the canonical output.

Matching is delegated to **pyDatalog**: initial-state facts are asserted, and
each subgoal conjunct is issued as a conjunctive Datalog query. Free variables
in the subgoal are solved existentially by the Datalog engine; any binding the
engine returns is a valid grounding. The planner's already-resolved
entailment substitution is applied to the subgoal *before* the query runs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from pddl_planner.logic.formula import (
    ConjunctiveFormula,
    Constant,
    DisjunctiveFormula,
    Equality,
    FalseFormula,
    Formula,
    Predicate,
    Substitution,
    Term,
    Variable,
)
from pddl_planner.logic.nl_formula import NLPredicate
from pddl_planner.logic.nl_parser import NLParser
from pddl_planner.pddl_core.action import Action

try:
    from pyDatalog import pyDatalog
except ImportError:  # pragma: no cover
    pyDatalog = None  # type: ignore

logger = logging.getLogger("pddl_planner.plan_extractor")

InitialStateInput = Union[
    Sequence[Tuple[str, Dict[str, str]]],
    Sequence[NLPredicate],
]
PlanTuple = Tuple[Formula, List[Action], Substitution]


@dataclass
class SubgoalMatch:
    """A subgoal whose preconditions are satisfied by the initial state.

    Attributes:
        subgoal_index: Position in the planner's regression order
            (``S0`` is the original goal, ``S1`` is one regression step back, ...).
        depth: Length of the backward-regression path from the original goal.
        subgoal: The original subgoal formula (full DNF) from the planner's
            output for this index — useful for showing the user exactly which
            regression node was selected.
        subgoal_formula: The fully substituted conjunct that matched the
            initial state (one disjunct of :attr:`subgoal`).
        actions: Forward-executable grounded :class:`Action` list — executing
            these in order from the initial state reaches the goal.
        binding: Additional variable-to-constant binding needed to satisfy the
            initial state (on top of the planner's recorded substitution).
        planner_substitution: The substitution recorded by the planner for this
            subgoal.
    """

    subgoal_index: int
    depth: int
    subgoal: Formula
    subgoal_formula: Formula
    actions: List[Action]
    binding: Substitution
    planner_substitution: Substitution

    @property
    def plan_length(self) -> int:
        return len(self.actions)

    @property
    def grounded_actions(self) -> List[str]:
        """Human-readable action strings in forward-execution order."""
        return [str(a) for a in self.actions]

    def __str__(self) -> str:
        return (
            f"SubgoalMatch(S{self.subgoal_index}, depth={self.depth}, "
            f"plan_length={self.plan_length}, actions={self.grounded_actions})"
        )


@dataclass
class PlanExtractionResult:
    """All subgoals satisfied by the initial state and the shortest derived plan.

    ``matches`` is sorted by plan length ascending so ``matches[0]`` is always the
    shortest viable plan; it is also exposed as :attr:`best` for readability.
    """

    initial_state: List[NLPredicate]
    matches: List[SubgoalMatch] = field(default_factory=list)
    goal: Optional[Formula] = None

    @property
    def best(self) -> Optional[SubgoalMatch]:
        return self.matches[0] if self.matches else None

    @property
    def longest(self) -> Optional[SubgoalMatch]:
        return max(self.matches, key=lambda m: m.plan_length) if self.matches else None

    @property
    def matched_subgoals(self) -> List[Formula]:
        """The original subgoal formula for each match, in best-first order."""
        return [m.subgoal for m in self.matches]

    def __bool__(self) -> bool:
        return bool(self.matches)

    def summary(self) -> str:
        lines: List[str] = []
        if self.initial_state:
            lines.append("Initial state:")
            for p in self.initial_state:
                lines.append(f"  - {p}")
        if self.goal is not None:
            lines.append(f"Goal: {self.goal}")
        if lines:
            lines.append("")  # blank line before verdict

        if not self.matches:
            lines.append(
                "No subgoal matches the initial state. "
                "The regression search may not have reached a subgoal that is a "
                "subset of your initial state — try increasing `max_depth` on the "
                "planner and re-running `regress_plan`."
            )
            return "\n".join(lines)

        lines.append(f"Found {len(self.matches)} matching subgoal(s). Shortest plan:")
        for i, act in enumerate(self.best.grounded_actions, start=1):
            lines.append(f"  {i}. {act}")
        return "\n".join(lines)


class InitialStatePlanExtractor:
    """Pick the subgoal the initial state satisfies and return the forward plan.

    Matching is done with pyDatalog: the initial state is asserted as facts and
    each subgoal conjunct is issued as a conjunctive Datalog query. Free
    variables in the subgoal are bound existentially by the Datalog engine.
    Entailment aliases discovered by the planner's LLM during regression
    propagate into the initial-state facts — each fact is asserted under every
    subgoal predicate name it is known to entail.
    """

    def __init__(self, planner: Optional["NLFOLRegressionPlanner"] = None,
                 *, parser: Optional[NLParser] = None) -> None:
        if pyDatalog is None:
            raise ImportError(
                "pyDatalog is required for plan extraction. "
                "Install it with `pip install pyDatalog` and re-run."
            )
        self._planner = planner
        if parser is not None:
            self._parser = parser
        elif planner is not None and getattr(planner, "_instance", None) is not None:
            self._parser = planner._instance._parser
        else:
            self._parser = NLParser()
        # The LLM and domain predicate list are used to realign initial-state
        # predicate names to their entailment-equivalent domain predicates so
        # pyDatalog queries over subgoals (which use domain names) hit.
        self._llm = getattr(planner, "_llm", None) if planner is not None else None
        self._domain_predicates = None
        if planner is not None and hasattr(planner, "_domain"):
            self._domain_predicates = getattr(planner._domain, "predicates", None)

    # ------------------------------ API ------------------------------ #

    def extract_plan(
        self,
        initial_state: InitialStateInput,
        plans: Sequence[PlanTuple],
    ) -> PlanExtractionResult:
        """Find every subgoal satisfied by ``initial_state`` and return plans.

        Args:
            initial_state: Either ``[(nl_text, type_tags), ...]`` tuples in the
                same format used for domains and goals, or pre-parsed
                :class:`NLPredicate` objects.
            plans: The list returned by ``NLFOLRegressionPlanner.regress_plan``.

        Returns:
            A :class:`PlanExtractionResult` with all matches sorted by plan
            length; ``result.best`` is the shortest plan.
        """
        init_preds = self._parse_initial_state(initial_state)
        # Rename each init predicate to its entailment-equivalent domain
        # predicate (LLM-resolved) so pyDatalog facts and subgoal queries share
        # a name vocabulary. Without this step a user-written "X is clear" in
        # the init would never match a subgoal's "X has no block above it".
        init_preds = self._align_init_to_domain(init_preds)
        logger.debug("Extracting plan over %d initial-state predicates and %d subgoals",
                     len(init_preds), len(plans))

        # Collect every predicate name that appears in any subgoal conjunct so
        # we can assert each init fact under all entailment-equivalent names.
        subgoal_names = self._collect_subgoal_predicate_names(plans)
        matcher = _PyDatalogMatcher(init_preds, subgoal_names)

        matches: List[SubgoalMatch] = []
        for idx, (sub_goal, reversed_actions, substitution) in enumerate(plans):
            grounded_formula = _safe_substitute(sub_goal, substitution)
            matched_conjunct: Optional[ConjunctiveFormula] = None
            binding: Optional[Substitution] = None
            for conjunct in _iter_conjuncts(grounded_formula):
                found = matcher.match(conjunct)
                if found is not None:
                    matched_conjunct = conjunct
                    binding = found
                    break
            if matched_conjunct is None:
                continue

            merged = Substitution({**substitution, **(binding or {})})
            forward_actions = [
                act.substitute(merged) for act in reversed(list(reversed_actions))
            ]
            grounded_conj = _safe_substitute(matched_conjunct, binding)
            matches.append(SubgoalMatch(
                subgoal_index=idx,
                depth=len(reversed_actions),
                subgoal=sub_goal,
                subgoal_formula=grounded_conj,
                actions=forward_actions,
                binding=binding or Substitution(),
                planner_substitution=substitution,
            ))

        matches.sort(key=lambda m: (len(m.actions), m.subgoal_index))
        logger.info("Plan extraction: %d/%d subgoals satisfied by the initial state",
                    len(matches), len(plans))
        goal_formula: Optional[Formula] = plans[0][0] if plans else None
        return PlanExtractionResult(
            initial_state=init_preds,
            matches=matches,
            goal=goal_formula,
        )

    # ---------------------------- internal ---------------------------- #

    def _parse_initial_state(self, initial_state: InitialStateInput) -> List[NLPredicate]:
        if initial_state is None:
            return []
        if not isinstance(initial_state, (list, tuple)):
            raise TypeError(
                "initial_state must be a list or tuple of (nl_text, type_tags) entries "
                f"or pre-parsed NLPredicate objects; got {type(initial_state).__name__}."
            )
        if len(initial_state) == 0:
            raise ValueError(
                "initial_state is empty — provide at least one ground fact describing "
                "the starting world state."
            )
        parsed: List[NLPredicate] = []
        for i, entry in enumerate(initial_state):
            if isinstance(entry, Predicate):
                parsed.append(entry)
                continue
            try:
                parsed.append(self._parser.parse_predicate(entry))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"initial_state[{i}] is malformed: {exc}\n"
                    f"Expected each entry to be either an NLPredicate or a "
                    f"(\"nl text\", {{\"term\": \"type\", ...}}) tuple — same format "
                    f"as the domain/goal predicates."
                ) from exc
        return parsed

    def _align_init_to_domain(self, init_preds: List[NLPredicate]) -> List[NLPredicate]:
        """Replace each init predicate's name with its entailed domain name.

        Calls :meth:`LLM.entailment` from :mod:`pddl_planner.llm.llm` for each
        initial-state predicate not already using a domain name, and rewrites
        the predicate via :meth:`LLM.replace_predicate_name`. The predicate's
        terms are preserved; only the name (and associated `term_type_dict`)
        is swapped. Cached entailment results make repeat calls fast.

        When the extractor has no LLM reference (e.g., instantiated without a
        planner), the input list is returned unchanged.
        """
        print("Aligning initial state predicates to domain predicate names...")
        if self._llm is None or not self._domain_predicates:
            return init_preds
        domain_names = {d.name for d in self._domain_predicates}
        aligned: List[NLPredicate] = []
        for ip in init_preds:
            if ip.name in domain_names:
                aligned.append(ip)
                continue
            try:
                entailed_pred = self._llm.entailment(
                    ip,
                    self._domain_predicates,
                    domain_predicates=True,
                )
            except Exception as exc:
                logger.warning("Entailment alignment failed for %s: %s", ip, exc)
                aligned.append(ip)
                continue

            if entailed_pred is None or getattr(entailed_pred, "entailed", None) is None:
                logger.info("No domain-predicate entailment for init predicate %r; "
                            "keeping original name", ip.name)
                aligned.append(ip)
                continue

            target = entailed_pred.entailed
            if isinstance(target, list):
                if not target:
                    aligned.append(ip)
                    continue
                target = target[0]

            try:
                renamed = self._llm.replace_predicate_name(ip, target)
            except Exception as exc:
                logger.warning("replace_predicate_name failed for %s → %s: %s",
                               ip, target, exc)
                aligned.append(ip)
                continue
            logger.info("Aligned init predicate: %r → %r", ip.name, renamed.name)
            aligned.append(renamed)
        return aligned

    @staticmethod
    def _collect_subgoal_predicate_names(plans: Sequence[PlanTuple]) -> Set[str]:
        names: Set[str] = set()

        def visit(formula: Formula) -> None:
            if isinstance(formula, Predicate):
                names.add(formula.name)
                return
            if hasattr(formula, "clauses"):
                for c in getattr(formula, "clauses", []):
                    visit(c)

        for sub_goal, _, substitution in plans:
            visit(_safe_substitute(sub_goal, substitution))
        return names


# --------------------------- module helpers --------------------------- #

def _safe_substitute(formula: Formula, substitution: Optional[Substitution]) -> Formula:
    if not substitution:
        return formula
    try:
        return formula.substitute(substitution)
    except Exception:
        logger.debug("Substitute failed on %s; returning original formula", formula)
        return formula


def _iter_conjuncts(formula: Formula) -> Iterable[ConjunctiveFormula]:
    """Yield each ConjunctiveFormula disjunct in ``formula``."""
    if isinstance(formula, FalseFormula):
        return
    if isinstance(formula, ConjunctiveFormula):
        yield formula
        return
    if isinstance(formula, DisjunctiveFormula):
        for clause in formula.clauses:
            if isinstance(clause, ConjunctiveFormula):
                yield clause
            elif isinstance(clause, (Predicate, Equality)):
                yield ConjunctiveFormula(clause)
        return
    if isinstance(formula, (Predicate, Equality)):
        yield ConjunctiveFormula(formula)


_NAME_SANITIZE_RE = re.compile(r"[^a-z0-9_]+")
_COLLAPSE_UNDER = re.compile(r"_+")


def _sanitize_name(name: str) -> str:
    """Map a natural-language predicate name to a pyDatalog-safe identifier."""
    s = _NAME_SANITIZE_RE.sub("_", name.strip().lower())
    s = _COLLAPSE_UNDER.sub("_", s).strip("_")
    if not s:
        return "p"
    if not s[0].isalpha():
        s = "p_" + s
    return s


def _pyd_var_name(var: Variable) -> str:
    """Normalize an NL variable name (e.g. ``?V63``, ``?b1``) for pyDatalog.

    pyDatalog treats lowercase identifiers in string queries as constants, so
    the result is forced to start with an uppercase letter.
    """
    raw = str(var.name).lstrip("?")
    raw = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    if not raw:
        raw = "V"
    if not raw[0].isalpha() or raw[0].islower():
        raw = "V_" + raw
    return raw


class _PyDatalogMatcher:
    """Per-extraction pyDatalog session.

    Asserts every initial-state fact under each name it is known to entail via
    the planner's LLM-resolved :meth:`NLPredicate.entailed_names`; then offers
    :meth:`match` which issues a conjunctive Datalog query for a subgoal and
    returns a variable binding (or ``None`` if unsatisfiable).
    """

    def __init__(self, init_preds: List[NLPredicate], subgoal_names: Set[str]) -> None:
        pyDatalog.clear()
        self._defined: Set[str] = set()
        self._declared_vars: Set[str] = set()
        alias_map = self._build_alias_map(init_preds, subgoal_names)
        for ip in init_preds:
            if getattr(ip, "is_neg", False):
                # Closed-world: we don't assert negative facts; negated subgoal
                # literals are verified by asking that the positive fact fails.
                continue
            args = [self._constant_name(t) for t in ip.terms if t is not None]
            if any(a is None for a in args):
                logger.debug("Skipping init predicate %s — non-constant term", ip)
                continue
            for alias in alias_map.get(ip.name, {ip.name}):
                san = _sanitize_name(alias)
                self._declare_term(san)
                if args:
                    pyDatalog.assert_fact(san, *args)
                else:
                    pyDatalog.assert_fact(san)
                self._defined.add(san)
        logger.debug("pyDatalog session initialized with %d predicate names defined",
                     len(self._defined))

    # ---------------- name/term helpers ---------------- #
    @staticmethod
    def _build_alias_map(init_preds: List[NLPredicate],
                         subgoal_names: Set[str]) -> Dict[str, Set[str]]:
        """For each init predicate, the set of names to assert its fact under.

        Always includes the predicate's own name, plus any subgoal predicate
        name that is entailment-equivalent (so LLM-resolved aliases propagate).
        """
        alias: Dict[str, Set[str]] = {}
        for ip in init_preds:
            aliases = {ip.name}
            try:
                entailed = set(ip.entailed_names() or [])
            except Exception:
                entailed = set()
            aliases.update(entailed)
            for sn in subgoal_names:
                if sn == ip.name or sn in entailed:
                    aliases.add(sn)
            alias[ip.name] = aliases
        return alias

    @staticmethod
    def _constant_name(term: Term) -> Optional[str]:
        if isinstance(term, Constant):
            return str(term.name)
        return None

    def _declare_term(self, name: str) -> None:
        pyDatalog.create_terms(name)

    def _declare_var(self, pyd_name: str) -> None:
        if pyd_name in self._declared_vars:
            return
        pyDatalog.create_terms(pyd_name)
        self._declared_vars.add(pyd_name)

    # ------------------- main entry -------------------- #
    def match(self, conjunct: ConjunctiveFormula) -> Optional[Substitution]:
        """Return a variable binding satisfying ``conjunct``, or ``None``.

        Returns an empty :class:`Substitution` if the conjunct is already
        ground-and-true. Negated literals are checked under a closed-world
        assumption against the candidate binding produced for the positive
        literals.
        """
        positive_atoms: List[str] = []
        negated_atoms: List[str] = []
        neq_constraints: List[Tuple[str, str]] = []
        eq_constraints: List[Tuple[str, str]] = []
        var_order: List[str] = []
        var_obj: Dict[str, Variable] = {}

        def fmt_term(t: Term) -> Optional[str]:
            if isinstance(t, Variable):
                vn = _pyd_var_name(t)
                if vn not in var_obj:
                    var_obj[vn] = t
                    var_order.append(vn)
                self._declare_var(vn)
                return vn
            if isinstance(t, Constant):
                return repr(str(t.name))
            return None

        for lit in conjunct.clauses:
            if isinstance(lit, Predicate):
                san = _sanitize_name(lit.name)
                # Predicate unknown to pyDatalog session
                if san not in self._defined:
                    if lit.is_neg:
                        # Closed world: ~p(...) trivially holds when p is undefined.
                        continue
                    return None
                args: List[str] = []
                valid = True
                for t in lit.terms:
                    s = fmt_term(t)
                    if s is None:
                        valid = False
                        break
                    args.append(s)
                if not valid:
                    return None
                atom = f"{san}({', '.join(args)})" if args else f"{san}()"
                if lit.is_neg:
                    negated_atoms.append(atom)
                else:
                    positive_atoms.append(atom)
            elif isinstance(lit, Equality):
                s1 = fmt_term(lit.term1)
                s2 = fmt_term(lit.term2)
                if s1 is None or s2 is None:
                    return None
                if lit.is_neq:
                    neq_constraints.append((s1, s2))
                else:
                    eq_constraints.append((s1, s2))
            else:
                # Unknown literal type — be conservative.
                logger.debug("Skipping unsupported literal type %s", type(lit).__name__)

        if not positive_atoms and not eq_constraints and not neq_constraints:
            # Only negated literals (or fully trivial). Verify each holds.
            for na in negated_atoms:
                if self._safe_ask(na) is not None:
                    return None
            return Substitution()

        query_parts: List[str] = list(positive_atoms)
        query_parts += [f"({a} == {b})" for a, b in eq_constraints]
        query_parts += [f"({a} != {b})" for a, b in neq_constraints]
        query = " & ".join(query_parts)

        ans = self._safe_ask(query)
        if ans is None or not getattr(ans, "answers", None):
            return None

        # Try each candidate answer; first one whose negated literals hold wins.
        for row in ans.answers:
            if var_order:
                if len(row) < len(var_order):
                    continue
                candidate: Dict[Variable, Constant] = {}
                for vn, val in zip(var_order, row):
                    v = var_obj[vn]
                    candidate[v] = Constant(str(val), v.type)
            else:
                candidate = {}

            if self._negated_all_hold(negated_atoms, var_order, row):
                return Substitution(candidate)

        return None

    # ----------------- query primitives ---------------- #
    def _safe_ask(self, query: str):
        try:
            return pyDatalog.ask(query)
        except Exception as exc:
            logger.debug("pyDatalog.ask failed on %r: %s", query, exc)
            return None

    def _negated_all_hold(self, negated_atoms: List[str],
                          var_order: List[str], row: Tuple) -> bool:
        """Check every negated literal fails (closed-world) under the binding."""
        for na in negated_atoms:
            bound = na
            for vn, val in zip(var_order, row):
                bound = re.sub(rf"\b{re.escape(vn)}\b", repr(str(val)), bound)
            # If the positive form of the negated literal still has free vars
            # (rare — would mean the negated literal introduced a var not in
            # positive atoms) we fall back to requiring NO solution.
            if self._safe_ask(bound) is not None:
                return False
        return True
