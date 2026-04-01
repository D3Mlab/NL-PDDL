from calendar import c
from typing import List, Dict, Any
from pddl_planner.pddl_core.action import Action
from pddl_planner.logic.nl_parser import NLParser
from pddl_planner.logic.formula import Predicate, Formula
from pddl_planner.pddl_core.domain import Domain

class NLDomain(Domain):
    """
    A Domain represents a NL domain and provides parsed actions, predicates and types.
    
    Attributes:
        _ppdl_domain (Any): The parsed PDDL domain object.
        _parser (Parser): The parser used to convert PDDL elements into logic representations.
        _actions (List[Action]): The list of actions parsed from the domain.
        _predicates (List[Predicate]): The list of predicates parsed from the domain.
        _types (Dict[str, List[str]]): The mapping of types as returned by the PDDL parser.
            (Keys are type names; values are lists of supertypes for that type.)
    """
    
    def __init__(self, nl_domain: dict) -> None:
        """
        Initialize the Domain with its NL representation provided in a json file and loaded into a dictionary.
        
        Args:
            nl_domain (dict): The NL domain object loaded from a json file.
        """
        self._nl_domain = nl_domain
        self._parser = NLParser()
        self._predicates = self._parse_predicates()
        self._types = self._parse_types()
        self._actions = self._parse_actions()
    
    def _parse_actions(self) -> List[Action]:
        """
        Parse and construct the list of actions in the domain.
        
        Returns:
            List[Action]: The actions parsed from the PDDL domain.
        """
        actions: List[Action] = []
        nl_actions = [items for items in self._nl_domain if 'Action' in items.keys()]
        for nl_action in nl_actions:
            type_tags = nl_action['Parameters']
            parameters = [self._parser.parse_term(param, type_tags) for param in type_tags.keys()]
            term_type_dict = {param: param.type for param in parameters}

            #parse the preconditions
            preconditions = self._parser.parse_formula(nl_action['Preconditions'], term_type_dict=term_type_dict)

            #parse the positive effects
            effects = Formula()
            if "Positive" in nl_action["Effects"].keys():
                effects =  self._parser.parse_formula(nl_action["Effects"]["Positive"], term_type_dict=term_type_dict)
            #for negative effects, we need to negate the formula
            if "Negative" in nl_action["Effects"].keys():
                negative_effects =  self._parser.parse_formula(nl_action["Effects"]["Negative"], term_type_dict=term_type_dict).get_negation()
                #add the negative effects to the positive effects
                for clause in negative_effects.clauses:
                    effects.add_clause(clause)
            #initialize the action
            action = Action(
                nl_action['Action name'][0],
                parameters,
                preconditions,
                effects
            )
            actions.append(action)
        return actions
        
    def _parse_predicates(self) -> List[Predicate]:
        """
        Parse and construct the list of predicates in the domain.
        
        Returns:
            List[Predicate]: The predicates parsed from the PDDL domain.
        """
        predicates: List[Predicate] = []
        nl_predicates = [nl_predicate for items in self._nl_domain if 'Predicate' in items.keys() for nl_predicate in items['Predicate']]
        for nl_predicate in nl_predicates:
            predicate = self._parser.parse_predicate(nl_predicate)
            predicates.append(predicate)
        return predicates
    
    def _parse_types(self) -> Dict[str, str]:
        """
        Return the types mapping as parsed from the NL domain text.
        
        Returns:
            Dict[str, str]: The types mapping, where keys are type names and values
            are string of supertypes (as returned by the PDDL parser).
            Note: each type has a maximum of one supertype, if the type has no supertype, the value is None.
        """
        nl_predicates = [nl_predicate for items in self._nl_domain if 'Predicate' in items.keys() for nl_predicate in items['Predicate']]
        type_tags = {}
        for nl_predicate in nl_predicates:
            for term in nl_predicate[1].keys():
                # ensure exist once per term
                type_tags[term] = nl_predicate[1][term]
        return type_tags

    @property
    def name(self) -> str:
        """
        Get the domain name.
        
        Returns:
            str: The name of the domain.
        """
        return self._nl_domain