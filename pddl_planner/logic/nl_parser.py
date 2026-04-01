from typing import Any, Union, List, Dict
from pddl_planner.logic.formula import ConjunctiveFormula, DisjunctiveFormula, Predicate, Variable, Constant, Term
from pddl_planner.logic.nl_formula import NLPredicate

# Parser class to convert NL to logic fromulas
class NLParser():
    def parse_predicate(self, NL_predicate: tuple[str, Dict[str, str]]) -> NLPredicate:
        """
        Parses a NL predicate into a logic predicate. Example of NL predicate: "("goal_obj is an tomato", {"goal_obj": "object"})"
        
        This method extracts the name and terms from a NL predicate and creates a corresponding
        logic predicate. The predicate may be marked as negated based on the is_neg flag.
        
        Args:
            NL_predicate (str): The NL predicate to be parsed.
        
        Returns:
            Predicate: The logic predicate corresponding to the NL predicate.
        """
        # check if the predicate is negated by looking for "not" in the predicate (ingore the case)
        if "not" in NL_predicate[0].lower():
            is_neg = True
            NL_predicate[0] = NL_predicate[0].replace("not ", "")
        else:
            is_neg = False

        # pareser 
        #1. extract possible terms and type tags from the predicate's dictionary
        type_tags = NL_predicate[1]
        predicate_name = NL_predicate[0].strip().lower()
        # 2. parse the terms (constants and variables) within the predicate's dictionary
        if not type_tags == {}: # check if the type tags is empty (no variables or constants)
            terms = tuple(self.parse_term(term, type_tags) for term in type_tags.keys())
            # Create term_type_dict using the original term names as keys
            term_type_dict = {term: set([type_tags[str(term)]]) for term in terms}
            # 3. remove the variable terms from the predicate name
            for term in terms:
                predicate_name = predicate_name.replace(f' {term}', ' ').replace(f'{term} ', ' ').strip()
            predicate_name = predicate_name.strip()
        else: 
            terms = tuple()
            term_type_dict = {}
        return NLPredicate(predicate_name, NL_predicate[0].strip().lower(), is_neg, *terms, term_type_dict=term_type_dict)

    def parse_term(self, NL_term: str, type_tags: Dict[str, str] = None) -> Union[Variable, Constant]:
        """
        Parses a NL term into a logic term.
        
        This function converts a NL term (either a variable or a constant) into its logic representation.
        For a NL variable, a corresponding Variable instance is returned; similarly, a Constant instance is 
        returned for a NL constant.
    
        Args:
            NL_term (str): The NL term to parse.
            type_tags (Dict[str, str]): The type tags for all terms in the predicate
        
        Returns:
            Union[Variable, Constant]: The corresponding logic term.
        
        Raises:
            NotImplementedError: If the term type is not supported.
        """
        try:
            term_type = type_tags[NL_term] if type_tags is not None else None

            if NL_term.startswith("?"):
                return Variable(NL_term.replace("?","").strip().lower(), term_type)
            else:
                return Constant(NL_term.strip().lower(), term_type)

        except KeyError:
            raise NotImplementedError(f"Term is type {type(NL_term)}, which is not available in the provided type tags")

    def parse_goal(self, goal: Any) -> ConjunctiveFormula:
        """
        Parses a PDDL goal into a logic formula.
        
        This function converts a goal condition defined in PDDL into a conjunction of logic predicates.
        It accepts both a single predicate and a composite And expression.
        
        Args:
            goal (Any): The PDDL goal to parse.
        
        Returns:
            ConjunctiveFormula: The logic formula that represents the goal.
        
        Raises:
            AssertionError: If the parsed goal results in an empty operand list.
        """
        operands: List[Union[ConjunctiveFormula, DisjunctiveFormula, Predicate]] = []
        if isinstance(goal, tuple) or (isinstance(goal, list) and not (isinstance(goal[0], list) or isinstance(goal[0], tuple))):
            operands = [self.parse_predicate(goal)]
        elif isinstance(goal, List):
            operands = [self.parse_formula(operand) for operand in goal]
        else:
            raise NotImplementedError(f"Goal is type {type(goal)}, parser not implemented")
        assert len(operands) > 0, "Goal is empty"
        return ConjunctiveFormula(*operands)


    def parse_formula(self, nl_formula: tuple|List[tuple], term_type_dict: Dict[Term, List[str]] = None) -> Union[ConjunctiveFormula, DisjunctiveFormula, Predicate]:
        """
        Parses a NL formula into a corresponding logic formula.
        
        This method recursively inspects the input formula type and converts it into a logic 
        representation.  We assume all NL fromula are connected in a conjunction (no disjunctions)
        
        Args:
            formula (Any): The NL formula to parse.
        
        Returns:
            Union[ConjunctiveFormula, DisjunctiveFormula, Predicate]: The logic formula representation.
        
        Raises:
            NotImplementedError: If the formula type is not supported.
        """
        # base case: if the formula is a tuple, it is a predicate
        if isinstance(nl_formula, tuple) or (isinstance(nl_formula, list) and not (isinstance(nl_formula[0], list) or isinstance(nl_formula[0], tuple))):
            return self.parse_predicate(nl_formula)
        
        # recursive case: if the formula is a list of strings, it is a conjunction of predicates
        operands: List[Union[ConjunctiveFormula, DisjunctiveFormula, Predicate]] = [
            self.parse_formula(operand) for operand in nl_formula
        ]
        terms = [term for operand in operands for term in operand.collect_terms()]
        relevant_type_term_dict = {term: set([term_type_dict[term]]) for term in terms if term in term_type_dict} if term_type_dict is not None else None

        return ConjunctiveFormula(*operands, term_type_dict=relevant_type_term_dict)

