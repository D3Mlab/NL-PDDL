import copy
import time
import os
import sys
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from pddl_planner.pddl_core.nl_domain import NLDomain
from pddl_planner.pddl_core.nl_instance import NLInstance
from pddl_planner.logic.operation import Operations
from pddl_planner.logic.nl_formula import NLPredicate
from pddl_planner.logic.formula import Substitution, Formula, Predicate, DisjunctiveFormula, ConjunctiveFormula, Term, Equality, FalseFormula
from pddl_planner.pddl_core.action import Action
from pddl_planner.llm.llm import LLM
import dotenv

logger = logging.getLogger("pddl_planner.planner")
dotenv.load_dotenv()

class NLPlanner():
    def __init__(self, nl_domain: str, nl_problem: str, nl_init: str|None) -> None:
        """
        Initializes a Planner instance.

        Args:
            nl_domain (str): The domain PDDL file path.
            nl_problem (str): The problem PDDL file path.

        Returns:
            None
        """
        self._domain = NLDomain(nl_domain)
        self._instance = NLInstance(nl_problem, nl_init, self._domain)
        self._operations = Operations()

    def plan(self):
        """
        Abstract method to generate a plan.

        Returns:
            None
        """
        pass

class NLFOLRegressionPlanner(NLPlanner):
    def __init__(self, nl_domain: str, nl_problem: str, nl_init: str|None, max_depth: int = 16,
    llm_model: str = "gpt-4o-mini", llm_api_key: str = None, verbose: bool = True, llm_verbose: bool = False,
    log_path: str|None = None, time_limit: int|None = None, cache_path: str|None = None,
    llm_backend: str|None = None, llm_backend_options: dict|None = None,
    auto_match: bool = True) -> None:
        """
        Initialize a FOL-RegressionPlanner based on First-Order Logic (FOL) and uses SSA from Situation Calculus.

        Args:
            nl_domain (str): The NL domain file path.
            nl_problem (str): The NL problem file path.
            max_depth (int, optional): The maximum depth of the plan tree. Defaults to 10.
            llm_model (str, optional): The model name of the LLM. Defaults to "gpt-4o-mini".
            llm_api_key (str, optional): The API key of the LLM. Defaults to os.getenv("OPENAI_API_KEY").
            verbose (bool, optional): Whether to print planner logs (depth, frontier, etc.). Defaults to True.
            llm_verbose (bool, optional): Whether to print LLM entailment logs (cache, substitutions, responses). Defaults to False.
            cache_path (str, optional): The path to the cache file. Defaults to None.
            log_path (str, optional): The path to the log file. Defaults to None.
            time_limit (int, optional): The time limit for the planner. Defaults to None.
            llm_backend (str, optional): LLM backend — ``"openai"`` or ``"gemma"``.
                When ``None``, it is inferred from ``llm_model`` (names starting
                with ``"gemma"`` route to the local Gemma backend).
            llm_backend_options (dict, optional): Extra kwargs forwarded to the
                chosen backend (e.g. for Gemma: ``device``, ``dtype``,
                ``max_new_tokens``, ``hf_token``, ``cache_dir``).
            auto_match (bool, optional): When ``True`` (default) and an
                ``nl_init`` was provided, :meth:`regress_plan` will automatically
                run :meth:`match` on the generated plan using that initial state,
                storing the result on :attr:`match_result` and logging a short
                summary. Set to ``False`` to opt out and call :meth:`match`
                manually.
        """
        super().__init__(nl_domain, nl_problem, nl_init)
        # Keep the raw initial-state input so `match()` receives the exact same
        # (nl_text, type_tags) format the user originally passed. `nl_init` on
        # NLInstance is already parsed into a ConjunctiveFormula, which isn't
        # directly re-usable by the extractor's parse path.
        self._raw_nl_init = copy.deepcopy(nl_init) if nl_init is not None else None
        self._auto_match = auto_match
        # Populated by regress_plan when auto-match runs (or by explicit match calls).
        self.match_result = None
        self._max_depth = max_depth
        self._time_limit = time_limit
        self._verbose = verbose

        # Configure planner logger level based on verbose flag
        from pddl_planner import make_colored_handler
        llm_logger = logging.getLogger("pddl_planner.llm")
        self._log_path = log_path

        if log_path is not None:
            # When log_path is set, send all logs to the file only (no console output).
            # In Jupyter/Colab the root logger has its own stream handler that writes to
            # the cell output, so we must also disable propagation — otherwise records
            # reach the root handler even though we only attached a FileHandler here.
            # Any console handlers added by a previous planner instantiation are dropped
            # so a second cell run doesn't keep printing.
            file_handler = logging.FileHandler(log_path, mode="w")
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
            for _lg in (logger, llm_logger):
                for _h in list(_lg.handlers):
                    if not isinstance(_h, logging.FileHandler):
                        _lg.removeHandler(_h)
                if not any(isinstance(h, logging.FileHandler) for h in _lg.handlers):
                    _lg.addHandler(file_handler)
                _lg.setLevel(logging.DEBUG)
                _lg.propagate = False
        else:
            if verbose and not logger.handlers:
                logger.addHandler(make_colored_handler())
                logger.setLevel(logging.DEBUG)
            elif not verbose:
                logger.setLevel(logging.WARNING)

            if llm_verbose and not llm_logger.handlers:
                llm_logger.addHandler(make_colored_handler())
                llm_logger.setLevel(logging.DEBUG)
            elif not llm_verbose:
                llm_logger.setLevel(logging.WARNING)

        self._ssa = self.create_SSA()

        # Track number of times we fail to directly find a predicate in domain predicates
        self._missing_name_count = 0
        # The Gemma backend is fully local and does not need an OpenAI key.
        resolved_backend = (llm_backend or LLM._detect_backend(llm_model)).lower()
        if llm_api_key is None and resolved_backend == "openai":
            try:
                llm_api_key = os.getenv("OPENAI_API_KEY")
            except:
                logger.error("OPENAI_API_KEY is not set in environment")
        self._llm = LLM(
            model_name=llm_model,
            api_key=llm_api_key,
            verbose=llm_verbose,
            cache_path=cache_path,
            backend=llm_backend,
            backend_options=llm_backend_options,
        )

    @dataclass
    class SSA_Node:
        """
        A node in representing the SSA.

        Attributes:
            predicate_name (str): The name of the predicate.
            predicate_params (List[Term]): The parameters of the predicate.
            action_name (str): The name of the action.
            action_params (List[Term]): The parameters of the action.
            ssa (Union[Predicate, DisjunctiveFormula]): The SSA formula.
        """
        predicate_name: str
        predicate_params: List[Term]
        action_name: str
        action_params: List[Term]
        substitutions: List[Substitution]
        ssa: Union[Predicate, DisjunctiveFormula]

    class PlanNode():
        def __init__(self, action: Action, sub_goal: Formula, parent: Optional["NLFOLRegressionPlanner.PlanNode"] = None, depth: int = 0, substitution: Substitution = Substitution()) -> None:
            """
            Initializes a PlanNode. PlanNode is used to represent the planning tree.

            Args:
                planner (RegressionPlanner): The planner that this node belongs to.
                action (Action): The action leading to this node.
                sub_goal (Formula): The sub-goal for this node.
                parent (Optional[PlanNode], optional): The parent node. Defaults to None.
                depth (int, optional): The depth in the plan tree. Defaults to 0.

            Returns:
                None
            """

            self.action = action
            self.sub_goal = copy.deepcopy(sub_goal)
            self.parent = parent
            self.children: List["NLFOLRegressionPlanner.PlanNode"] = []
            self.depth = depth
            self.substitution = substitution

        def add_child(self, child_node: "NLFOLRegressionPlanner.PlanNode") -> None:
            """
            Adds a child node.

            Args:
                child_node (PlanNode): The child node to add.

            Returns:
                None
            """
            self.children.append(child_node)

    def _extract_plan_from_node(self, node: "NLFOLRegressionPlanner.PlanNode") -> List[Action]:
        """
        Walk a PlanNode back to the root and return the regression-ordered action list.

        Returns:
            List[Action]: Actions in regression order (root-first). Reverse to get
            forward execution order from the subgoal to the original goal.
        """
        plan: List[Action] = []
        while node.parent is not None:
            plan.append(node.action)
            node = node.parent
        plan.reverse()
        return plan

    def match(self, initial_state, plans):
        """
        Pick the subgoal satisfied by ``initial_state`` and return the forward plan.

        Args:
            initial_state: Either a list of ``(nl_text, type_tags)`` tuples (same
                format used for goals/domains) or a list of pre-parsed
                :class:`NLPredicate` objects.
            plans: The list returned by :meth:`regress_plan` — a sequence of
                ``(subgoal_formula, reversed_actions, substitution)`` tuples.

        Returns:
            :class:`pddl_planner.planner.plan_extractor.PlanExtractionResult`
            containing every matching subgoal sorted by plan length ascending;
            ``result.best`` is the shortest forward-executable plan.
        """
        from pddl_planner.planner.plan_extractor import InitialStatePlanExtractor
        return InitialStatePlanExtractor(self).extract_plan(initial_state, plans)


    def create_SSA(self, predicates: List[Predicate] = None) -> Dict[str, Dict[str, SSA_Node]]:
        """
        Construct SSA from the domain's action schema for each predicate.

        This function returns a dictionary mapping each predicate (from domain.predicates)
        to a dictionary that maps an action name to a tuple. The tuple contains:
            1. The standardized action's parameters (List[Term]).
            2. The SSA value, which is either:
                - A DisjunctiveFormula computed from the positive or negative effect, or
                - The predicate itself when the effect does not include the predicate.

        Returns:
            Dict[Predicate, Dict[str, Tuple[List[Term], Union[Predicate, DisjunctiveFormula]]]]:
                The constructed SSA mapping.
        """

        def get_positive_effect_axiom(action: Action, predicate: Predicate) -> Tuple[Optional[DisjunctiveFormula], Substitution]:
            axioms = []
            substitution = Substitution()
            for clause in action.effects.clauses:
                if isinstance(clause, Predicate) and clause.name == predicate.name and not clause.is_neg:
                    sub = self._operations.unify(clause, predicate, Substitution())
                    if sub is not None:
                        for var, term in sub.items():
                            axioms.append(Equality(var, term, is_neq=False))
                        axioms.append(action.preconditions)
                        substitution.update(sub)
            if axioms:
                return (ConjunctiveFormula(*axioms, term_type_dict=action.preconditions.term_type_dict), substitution)
            return (None, Substitution())

        def get_negative_effect_axiom(action: Action, predicate: Predicate) -> Tuple[Optional[ConjunctiveFormula], Substitution]:
            axioms = []
            substitution = Substitution()
            for clause in action.effects.clauses:
                if isinstance(clause, Predicate) and clause.name == predicate.name and clause.is_neg:
                    sub = self._operations.unify(clause, predicate.get_negation(), Substitution())
                    if sub is not None:
                        for var, term in sub.items():
                            axioms.append(Equality(var, term, is_neq=False))
                        axioms.append(action.preconditions)
                        substitution.update(sub)
            if axioms:
                negative_effect_axiom = ConjunctiveFormula(*axioms)
                return (negative_effect_axiom, substitution)
            return (None, Substitution())

        all_ssa: Dict[str, Dict[str, NLFOLRegressionPlanner.SSA_Node]] = {}
        if predicates is None:
            predicates = self._domain.predicates
        for pred in predicates:
            logger.debug("Building SSA for predicate: %s", pred.name)
            pred_ssa: Dict[str, NLFOLRegressionPlanner.SSA_Node] = {}
            for action in self._domain.actions:
                standardized_action = action.standardize(self._operations)
                (positive_effect_axiom, substitution) = get_positive_effect_axiom(standardized_action, pred)
                (negative_effect_axiom, _) = get_negative_effect_axiom(standardized_action, pred)

                if positive_effect_axiom is not None and negative_effect_axiom is not None:
                    # Both positive and negative effects exist, ssa takes the form
                    # SSA = (positive_effect) | (pred & ~negative_effect)
                    ssa = DisjunctiveFormula(positive_effect_axiom, ConjunctiveFormula(pred, negative_effect_axiom.get_negation())).simplify().distribute_and_over_or()
                elif positive_effect_axiom is not None:
                    # Only positive effect exists
                    # SSA = positive_effect | pred
                    ssa = DisjunctiveFormula(positive_effect_axiom, pred).simplify().distribute_and_over_or()
                elif negative_effect_axiom is not None:
                    # Only negative effect exists
                    # SSA = pred & ~negative_effect
                    ssa = ConjunctiveFormula(negative_effect_axiom.get_negation(), pred).simplify().distribute_and_over_or()
                else:
                    # No effect exists
                    ssa = DisjunctiveFormula(pred).distribute_and_over_or()

                pred_ssa[standardized_action.name] = NLFOLRegressionPlanner.SSA_Node(
                        pred.name,
                        pred.terms,
                        standardized_action.name,
                        standardized_action.parameters,
                        substitution,
                        ssa)
            all_ssa[pred.name] = pred_ssa
        return all_ssa

    def create_SSA_as_itself(self, predicate: Predicate) -> Dict[str, SSA_Node]:
        """
        Create SSA as itself.
        """
        pred_ssa: Dict[str, NLFOLRegressionPlanner.SSA_Node] = {}
        for action in self._domain.actions:
            standardized_action = action.standardize(self._operations)
            ssa = DisjunctiveFormula(predicate).distribute_and_over_or()
            pred_ssa[standardized_action.name] = NLFOLRegressionPlanner.SSA_Node(
                predicate.name,
                predicate.terms,
                standardized_action.name,
                standardized_action.parameters,
                Substitution(), ssa)
        return pred_ssa


    def regress_pred(self, predicate: NLPredicate, action: Action) -> DisjunctiveFormula:
        """
        Regress a predicate through an action via the stored SSA substitution.

        This function retrieves the corresponding SSA_Node for the given predicate and action.
        It then builds a substitution mapping that maps:
        - Each stored predicate parameter to the corresponding variable in the provided predicate.
        - Each stored action parameter to the corresponding variable in the provided action.
        Finally, it applies this substitution to the stored SSA formula and returns the resulting
        DisjunctiveFormula.

        Args:
            predicate (Predicate): The predicate whose variables are to be substituted.
            action (Action): The action used for regression whose parameters are mapped.

        Returns:
            DisjunctiveFormula: The regressed formula with variables substituted according to the SSA_Node.
        """
        # check if the predicate is in domain predicates
        if predicate.name in self._ssa:
            ssa_node = self._ssa[predicate.name][action.name]
        else:
            # check if the predicate can be entailed as a domain predicate
            # count this missing-name event
            try:
                self._missing_name_count += 1
            except Exception:
                self._missing_name_count = 1
            logger.info('Predicate "%s" (name="%s") not in domain — attempting LLM entailment', predicate.nl_description, predicate.name)
            background_predicates = (copy.deepcopy(action), [clause for clause in self._instance.goal.clauses if isinstance(clause, NLPredicate)])
            entailed_pred = self._llm.entailment(predicate, self._domain.predicates,
                                                    background_predicates=background_predicates, domain_predicates=True)

            if entailed_pred is not None:
                if isinstance(entailed_pred.entailed, list):
                    ssa_node = []
                    for pred in entailed_pred.entailed:
                        ssa_node.append(self._ssa[pred.name][action.name])
                else:
                    ssa_node = self._ssa[entailed_pred.entailed.name][action.name]
            else:
                # create a new ssa node with postive and negative effects as none
                logger.warning('Entailment failed for "%s" — creating identity SSA node', predicate.name)
                self._ssa[predicate.name] = self.create_SSA_as_itself(predicate)
                ssa_node = self._ssa[predicate.name][action.name]
        # Build a substitution:
        # Map the stored predicate parameters to the input predicate's terms.
        if not isinstance(ssa_node, List):
            substitution = Substitution()
            # Honor recorded entailment permutation (if any) between predicate vars
            recorded = predicate.entailed_substitutions.get(ssa_node.predicate_name)
            inv_name_map = {}
            if recorded is not None:
                for k, v in recorded.items():
                    inv_name_map[v.name] = k.name
            target_name_to_term = {getattr(t, 'name', str(t)): t for t in predicate.terms}
            for idx, stored_pred_var in enumerate(ssa_node.predicate_params):
                mapped_target_name = inv_name_map.get(getattr(stored_pred_var, 'name', str(stored_pred_var)))
                if mapped_target_name is not None and mapped_target_name in target_name_to_term:
                    substitution[stored_pred_var] = target_name_to_term[mapped_target_name]
                else:
                    if idx < len(predicate.terms):
                        substitution[stored_pred_var] = predicate.terms[idx]
            # Map the stored action parameters to the input action's parameters.
            for stored_act_var, input_act_var in zip(ssa_node.action_params, action.parameters):
                substitution[stored_act_var] = input_act_var

            returned_ssa = copy.deepcopy(ssa_node.ssa)
            return returned_ssa.substitute(substitution)
        else:
            # ssa_node is a list of SSA_Nodes (entailed to multiple domain predicates)
            logger.info('Multiple entailment: "%s" maps to %d domain predicates', predicate.name, len(ssa_node))
            substituted_ssas: List[Formula] = []
            for node in ssa_node:
                node_sub = Substitution()
                # Honor recorded entailment permutation per entailed predicate name
                recorded = None
                recorded = predicate.entailed_substitutions.get(node.predicate_name)
                inv_name_map = {}
                if recorded is not None:
                    for k, v in recorded.items():
                        inv_name_map[v.name] = k.name
                target_name_to_term = {getattr(t, 'name', str(t)): t for t in predicate.terms}
                for idx, stored_pred_var in enumerate(node.predicate_params):
                    mapped_target_name = inv_name_map.get(getattr(stored_pred_var, 'name', str(stored_pred_var)))
                    if mapped_target_name is not None and mapped_target_name in target_name_to_term:
                        node_sub[stored_pred_var] = target_name_to_term[mapped_target_name]
                    else:
                        if idx < len(predicate.terms):
                            node_sub[stored_pred_var] = predicate.terms[idx]
                for stored_act_var, input_act_var in zip(node.action_params, action.parameters):
                    node_sub[stored_act_var] = input_act_var
                node_ssa = copy.deepcopy(node.ssa).substitute(node_sub)
                substituted_ssas.append(node_ssa)

            if not predicate.is_neg:
                combined = DisjunctiveFormula(*substituted_ssas).distribute_and_over_or()
            else:
                combined = ConjunctiveFormula(*substituted_ssas).distribute_and_over_or()
            return combined

    def regress(self, goal: DisjunctiveFormula, action: Action) -> DisjunctiveFormula:
        """
        Regress the goal formula through the given action.

        This function takes a goal formula in Disjunctive Normal Form (DNF) and regresses each
        conjunctive component of the formula with respect to the given action. For each conjunct,
        it iterates over the clauses and uses 'regress_pred' on each predicate clause, retaining other
        clauses unchanged. After regressing all conjuncts, it recombines them into a new DNF formula
        and returns that as the regressed goal.

        Args:
            goal (DisjunctiveFormula): The goal formula in DNF to be regressed.
            action (Action): The action used for regression.

        Returns:
            DisjunctiveFormula: The regressed goal formula in Disjunctive Normal Form.
        """
        if not isinstance(goal, DisjunctiveFormula):
            raise ValueError(f"Goal must be a DisjunctiveFormula, but got {type(goal)}")
        regressed_disjunct_list = []

        for conjunct in goal.clauses:
            if not isinstance(conjunct, ConjunctiveFormula):
                raise ValueError(f"Each conjunct must be a ConjunctiveFormula, but got {type(conjunct)}")

            regressed_conjunct_list = []
            for clause in conjunct.clauses:
                if isinstance(clause, NLPredicate):
                    # Regress the predicate clause using regress_pred
                    regressed_clause = self.regress_pred(clause, action)
                else:
                    regressed_clause = clause
                regressed_conjunct_list.append(regressed_clause)
            # Combine the regressed clauses and convert to DN
            regressed_disjunct_list.append(ConjunctiveFormula(*regressed_conjunct_list).distribute_and_over_or())
        # Return a flattened regressed goal  in DNF
        flattened_regressed_goal = DisjunctiveFormula(*regressed_disjunct_list).distribute_and_over_or()
        return flattened_regressed_goal

    def regress_plan(self, simplify_equality: bool = True, simplify_contradiction: bool = True,
    simplify_typing: bool = True, simplify_dnf: bool = True, dup_detection: bool = True, save_file_path: Optional[str] = None) -> List[Tuple[Formula, List[Action]]]:
        """
        Generate a regressed plan by iteratively regressing the goal through applicable actions.

        This method starts with the instance goal (converted to Disjunctive Normal Form if needed)
        and then iteratively regresses it using the available actions up to a maximum depth.
        At each regression step, it creates new plan tree nodes and tracks visited subgoals to avoid duplication.

        Returns:
            List[Tuple[Formula, List[Action]]]: A list of tuples where each tuple contains:
                - A subgoal (Formula) that represents a regressed goal state.
                - A list of actions (List[Action]) that form the plan to achieve that subgoal.
        """
        # Pre-compute goal predicate names for entailment gating
        goal_predicate_names = set()
        def _collect_goal_predicates(formula: Formula) -> None:
            if isinstance(formula, NLPredicate):
                goal_predicate_names.add(formula.name)
                return
            if hasattr(formula, 'clauses') and isinstance(getattr(formula, 'clauses'), list):
                for cl in formula.clauses:
                    _collect_goal_predicates(cl)

        _collect_goal_predicates(self._instance.goal)
        # A LLM-backed entailment checker for predciates in the actions back to the goal
        def _entailment_checker(target: NLPredicate, pred: NLPredicate) -> bool:
            try:
                # Only attempt entailment if the candidate predicate name appears in the goal
                if pred.name not in goal_predicate_names:
                    return False
                # Do not need to check entailment if the target predicate is already in the goal
                if target.name in goal_predicate_names:
                    return True
                entailed_predicate = self._llm.entailment(copy.deepcopy(pred), [copy.deepcopy(target)], flag=False)
                if entailed_predicate is not None and entailed_predicate.entailed.name == target.name:
                    return True
            except Exception:
                return False
        NLPredicate.set_entailment_checker(_entailment_checker)

        plan = []
        goal = self._instance.goal.distribute_and_over_or()
        if not isinstance(goal, DisjunctiveFormula):
            raise ValueError(f"Goal must be a DisjunctiveFormula, but got {type(goal)}")
        frontier = [NLFOLRegressionPlanner.PlanNode(None, goal)]
        start_time = time.time()
        plan.append((frontier[0].sub_goal, [], Substitution()))

        def save_plan(plan: List[Tuple[Formula, List[Action], Substitution]], save_file_path: str, count: int = 0):
            last_plan = plan[-1]
            with open(save_file_path, 'a') as f:
                f.write(f"Subgoal S{count}:\n")
                # `str(...)` recurses into ConjunctiveFormula/DisjunctiveFormula
                # which delegate to each NLPredicate's `__str__`, which uses
                # display_name so the subgoal prints in its readable form.
                f.write(str(last_plan[0]) + '\n')
                reversed_plan = copy.deepcopy(last_plan[1])
                reversed_plan.reverse()
                actions = [p.substitute(last_plan[2]) for p in reversed_plan]
                f.write(str(actions) + '\n')
                f.write(str(last_plan[2]) + '\n')
                f.write("--------------------\n")
            count += 1
            return count

        plan_counter = 0
        plan_counter = save_plan(plan, save_file_path, plan_counter)

        visited_goal = []

        for clause in goal.clauses:
            if isinstance(clause, ConjunctiveFormula):
                visited_goal.append(clause)

        if self._log_path:
            print(f"Starting regression | max_depth={self._max_depth} | "
                  f"time_limit={self._time_limit} | {len(self._domain.actions)} actions in domain")
            print(f"Logs are being written to: {self._log_path}")
        logger.info("Starting regression | max_depth=%d | time_limit=%s | %d actions in domain",
                     self._max_depth, self._time_limit, len(self._domain.actions))

        while frontier:
            current_node: NLFOLRegressionPlanner.PlanNode = frontier.pop(0)
            current_goal: Formula = current_node.sub_goal

            elapsed = time.time() - start_time
            bar_len = 20
            filled = int((current_node.depth / max(1, self._max_depth)) * bar_len)
            bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
            logger.info("Depth %d/%d %s | %.2fs elapsed | %d nodes in frontier",
                        current_node.depth, self._max_depth, bar, elapsed, len(frontier))

            if current_node.depth >= self._max_depth:
                logger.debug("Max depth reached at depth %d — skipping node", current_node.depth)
                continue
            if self._time_limit is not None and time.time() - start_time > self._time_limit:
                logger.warning("Time limit reached (%.2fs > %ds) — skipping remaining nodes", time.time() - start_time, self._time_limit)
                continue
            for action in self._domain.actions:
                standardized_action = action.standardize(self._operations)
                regressed_goal = self.regress(current_goal, standardized_action)
                if simplify_contradiction:
                    regressed_goal = regressed_goal.simplify()

                if isinstance(regressed_goal, Predicate):
                    continue

                if simplify_equality:
                    per_conjunct_results = []
                    subst_map: Dict[str, Substitution] = {}
                    for clause in regressed_goal.clauses:
                        if isinstance(clause, ConjunctiveFormula):
                            # Build substitution from equality for this conjunct only
                            clause_simplified, clause_sub = clause.simplify_equality(current_goal)
                            per_conj = (
                                DisjunctiveFormula(clause_simplified)
                                .substitute(clause_sub)
                            )
                            per_conj = self._operations.replace_domain_with_goal_fluents(per_conj, self._instance.goal)
                            if simplify_contradiction:
                                per_conj = per_conj.simplify_plan().distribute_and_over_or()
                            else:
                                per_conj = per_conj.distribute_and_over_or()
                            per_conjunct_results.append(per_conj)
                            # Record substitution for each resulting conjunct
                            for conj in per_conj.clauses:
                                if isinstance(conj, ConjunctiveFormula):
                                    subst_map[str(conj)] = clause_sub
                        else:
                            df = clause if isinstance(clause, DisjunctiveFormula) else DisjunctiveFormula(clause)
                            per_conjunct_results.append(df)
                            # Map empty substitution for non-processed clauses
                            for conj in df.clauses if isinstance(df, DisjunctiveFormula) else [df]:
                                if isinstance(conj, ConjunctiveFormula):
                                    subst_map[str(conj)] = Substitution()

                    # Recombine per-conjunct processed results
                    regressed_goal = DisjunctiveFormula(*per_conjunct_results).distribute_and_over_or()

                else:
                    # No equality processing; create an empty mapping for child substitutions
                    subst_map = {}
                regressed_goal = self._operations.replace_domain_with_goal_fluents(regressed_goal, self._instance.goal)
                regressed_goal = self._operations.simplify_by_domain_axiom(regressed_goal, self._instance.init)

                if (simplify_typing and self._domain.has_type_conflict(regressed_goal)) or isinstance(regressed_goal, FalseFormula):
                    # skip if there is a type conflict or the formula simplifes to false
                    continue
                # remove any conjuncts in the regressed goal that implies the seen subgoal
                regressed_goal_list = []
                if simplify_dnf or dup_detection:
                    for conjunct in regressed_goal.clauses:
                        if isinstance(conjunct, ConjunctiveFormula):
                            implies_found = any(conjunct.implies(formula) for formula in visited_goal) if simplify_dnf else False
                            duplicate_found = any(conjunct.is_duplicate(formula) for formula in visited_goal) if dup_detection else False
                            if not implies_found and not duplicate_found:
                                regressed_goal_list.append(conjunct)
                                visited_goal.append(conjunct)
                    regressed_goal = DisjunctiveFormula(*regressed_goal_list).simplify().distribute_and_over_or() if simplify_contradiction else DisjunctiveFormula(*regressed_goal_list).distribute_and_over_or()

                # If regressed_goal contains multiple conjuncts, split only if there are non-empty per-conjunct substitutions
                regressed_conjuncts = [c for c in regressed_goal.clauses if isinstance(c, ConjunctiveFormula)]

                has_any_subst = any(
                bool(subst_map.get(str(conj), Substitution()))
                for conj in regressed_conjuncts
                    ) if 'subst_map' in locals() else False

                if len(regressed_conjuncts) > 1 and has_any_subst:
                    # Additional dup detection per conjunct when splitting
                    for conj in regressed_conjuncts:
                        split_goal = DisjunctiveFormula(conj).distribute_and_over_or()
                        conj_sub = subst_map.get(str(conj), Substitution())
                        child_subst = {**current_node.substitution, **conj_sub}
                        child_node = NLFOLRegressionPlanner.PlanNode(standardized_action, split_goal, current_node, current_node.depth + 1, child_subst)
                        if not isinstance(child_node.sub_goal, FalseFormula):
                            for c in child_node.sub_goal.clauses:
                                if isinstance(c, ConjunctiveFormula):
                                    visited_goal.append(c)
                            frontier.append(child_node)
                            plan.append((child_node.sub_goal, self._extract_plan_from_node(child_node), child_node.substitution))

                            if save_file_path is not None and len(plan) > 0:
                                plan_counter = save_plan(plan, save_file_path, plan_counter)
                else:
                    conj = regressed_conjuncts[0] if regressed_conjuncts else None
                    conj_sub = subst_map.get(str(conj), Substitution()) if conj is not None and 'subst_map' in locals() else Substitution()
                    child_subst = {**current_node.substitution, **conj_sub}
                    child_node = NLFOLRegressionPlanner.PlanNode(standardized_action, regressed_goal, current_node, current_node.depth + 1, child_subst)
                    # add to the frontier and plan if the subgoal hasn't visited before
                    if not isinstance(child_node.sub_goal, FalseFormula):
                        for conjunct in child_node.sub_goal.clauses:
                            if isinstance(conjunct, ConjunctiveFormula):
                                visited_goal.append(conjunct)
                            else:
                                logger.debug("Non-conjunctive clause in subgoal: %s", conjunct)
                        frontier.append(child_node)
                        plan.append((child_node.sub_goal, self._extract_plan_from_node(child_node), child_node.substitution))

                        if save_file_path is not None and len(plan) > 0:
                            plan_counter = save_plan(plan, save_file_path, plan_counter)

        # Final summary
        elapsed = time.time() - start_time
        missing = getattr(self, '_missing_name_count', 0)
        logger.info("Regression complete | %.2fs | %d subgoals | %d missing-predicate lookups", elapsed, len(plan), missing)
        lat_sum = getattr(self._llm, 'api_latency_sum', 0.0)
        lat_cnt = getattr(self._llm, 'api_latency_count', 0)
        if lat_cnt > 0:
            avg_lat = lat_sum / max(1, lat_cnt)
            logger.info("LLM stats: %d API calls | avg latency %.4fs | %d cache hits",
                        getattr(self._llm, 'api_call_count', 0), avg_lat, getattr(self._llm, 'cache_call_count', 0))
        if self._log_path:
            print(f"Regression complete | {elapsed:.2f}s | {len(plan)} subgoals | "
                  f"{missing} missing-predicate lookups")
            if lat_cnt > 0:
                print(f"LLM stats: {getattr(self._llm, 'api_call_count', 0)} API calls | "
                      f"avg latency {lat_sum / max(1, lat_cnt):.4f}s | "
                      f"{getattr(self._llm, 'cache_call_count', 0)} cache hits")
            print(f"Full log saved to: {self._log_path}")

        # Auto-match: if the user supplied an initial state to the constructor,
        # run match() automatically so they get an executable plan out of the
        # box. Stored on `self.match_result` and also returned alongside `plan`
        # via that attribute — `regress_plan` itself still returns `plan` for
        # backward compatibility with existing callers.
        if self._auto_match and self._raw_nl_init is not None and len(plan) > 0:
            try:
                self.match_result = self.match(self._raw_nl_init, plan)
                n_matches = len(self.match_result.matches) if self.match_result is not None else 0
                logger.info("Auto-match complete: %d subgoal(s) satisfied by the initial state", n_matches)
                if self._log_path:
                    print(f"Auto-match complete: {n_matches} matching subgoal(s); "
                          f"see `planner.match_result`.")
            except Exception as exc:
                logger.warning("Auto-match failed: %s", exc)
        return plan
