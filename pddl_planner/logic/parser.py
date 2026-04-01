from typing import Any, Union, List, Dict
from pddl.logic.terms import Variable as PDDLVariable
from pddl.logic.terms import Constant as PDDLConstant
from pddl.logic.terms import Term as PDDLTerm
from pddl.logic.predicates import Predicate as PDDLPredicate
from pddl.logic.base import And, Or, Not
from pddl.logic.effects import AndEffect
from pddl_planner.logic.formula import ConjunctiveFormula, DisjunctiveFormula, Predicate, Variable, Constant, Term

class Parser:
    """Parser class to convert PDDL formulas to logic formulas."""
    
    def parse_formula(self, formula: Any, term_type_dict: Dict[Term, List[str]] = None) -> Union[ConjunctiveFormula, DisjunctiveFormula, Predicate]:
        """
        Parses a PDDL formula into a corresponding logic formula.
        
        This method recursively inspects the input formula type and converts it into a logic 
        representation. If it is a compound formula (And, Or, or Not) it processes the 
        operands accordingly. If the formula type is unfamiliar, it raises an error.
        
        Args:
            formula (Any): The PDDL formula to parse.
        
        Returns:
            Union[ConjunctiveFormula, DisjunctiveFormula, Predicate]: The logic formula representation.
        
        Raises:
            NotImplementedError: If the formula type is not supported.
        """
        if isinstance(formula, (And, AndEffect)):
            operands: List[Union[ConjunctiveFormula, DisjunctiveFormula, Predicate]] = [
                self.parse_formula(operand) for operand in formula.operands
            ]
            terms = [term for operand in operands for term in operand.collect_terms()]
            relevant_type_term_dict = {term: set(term_type_dict[term]) for term in terms if term in term_type_dict} if term_type_dict is not None else None
            return ConjunctiveFormula(*operands, term_type_dict=relevant_type_term_dict)
        elif isinstance(formula, Or):
            operands = [self.parse_formula(operand) for operand in formula.operands]
            terms = [term for operand in operands for term in operand.collect_terms()]
            relevant_type_term_dict = {term: set(term_type_dict[term]) for term in terms if term in term_type_dict} if term_type_dict is not None else None
            return DisjunctiveFormula(*operands, term_type_dict=relevant_type_term_dict)
        elif isinstance(formula, PDDLPredicate) or (isinstance(formula, Not) and isinstance(formula.argument, PDDLPredicate)):
            return self.parse_predicate(formula)
        else:
            raise NotImplementedError(f"Formula is type {type(formula)}, parser not implemented")
        
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
        if isinstance(goal, PDDLPredicate):
            operands = [self.parse_predicate(goal)]
        elif isinstance(goal, And):
            operands = [self.parse_formula(operand) for operand in goal.operands]
        else:
            raise NotImplementedError(f"Goal is type {type(goal)}, parser not implemented")
        assert len(operands) > 0, "Goal is empty"
        return ConjunctiveFormula(*operands)
        
    def parse_init(self, init: Any) -> ConjunctiveFormula:
        """
        Parses a PDDL initialization block into a logic formula.
        
        This method converts the initial conditions of a PDDL problem into a conjunction
        of logic formulas. It supports both a list of initial formulas and a single predicate.
        
        Args:
            init (Any): The PDDL init block, which can be a list or a single predicate.
        
        Returns:
            ConjunctiveFormula: The logic representation of the initial state.
        """
        operands: List[Union[ConjunctiveFormula, DisjunctiveFormula, Predicate]] = []
        if isinstance(init, list):
            operands = [self.parse_formula(operand) for operand in init]
        elif isinstance(init, PDDLPredicate):
            operands = [self.parse_formula(init)]
        return ConjunctiveFormula(*operands)
        
    def parse_predicate(self, PDDL_predicate: PDDLPredicate) -> Predicate:
        """
        Parses a PDDL predicate into a logic predicate.
        
        This method extracts the name and terms from a PDDL predicate and creates a corresponding
        logic predicate. The predicate may be marked as negated based on the is_neg flag.
        
        Args:
            PDDL_predicate (PDDLPredicate): The PDDL predicate to be parsed.
        
        Returns:
            Predicate: The logic predicate corresponding to the PDDL predicate.
        """
        if isinstance(PDDL_predicate, Not):
            is_neg = True
            PDDL_predicate = PDDL_predicate.argument
        else:
            is_neg = False
        terms = tuple(self.parse_term(term) for term in PDDL_predicate.terms)
        return Predicate(PDDL_predicate.name, is_neg, *terms)
    
    def parse_term(self, PDDL_term: PDDLTerm) -> Union[Variable, Constant]:
        """
        Parses a PDDL term into a logic term.
        
        This function converts a PDDL term (either a variable or a constant) into its logic representation.
        For a PDDL variable, a corresponding Variable instance is returned; similarly, a Constant instance is 
        returned for a PDDL constant.
        
        Args:
            PDDL_term (PDDLTerm): The PDDL term to parse.
        
        Returns:
            Union[Variable, Constant]: The corresponding logic term.
        
        Raises:
            NotImplementedError: If the term type is not supported.
        """
        term_type = set(PDDL_term.type_tags) if PDDL_term.type_tags else None
        if isinstance(PDDL_term, PDDLVariable):
            return Variable(PDDL_term.name, term_type)
        if isinstance(PDDL_term, PDDLConstant):
            return Constant(PDDL_term.name, term_type)
        raise NotImplementedError(f"Term is type {type(PDDL_term)}, parser not implemented")