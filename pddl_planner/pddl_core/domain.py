import logging
from typing import List, Dict, Any
from pddl_planner.pddl_core.action import Action
from pddl_planner.logic.parser import Parser
from pddl_planner.logic.formula import Predicate, Formula

logger = logging.getLogger("pddl_planner.domain")

class Domain:
    """
    A Domain represents a PDDL domain and provides parsed actions, predicates and types.
    
    Attributes:
        _ppdl_domain (Any): The parsed PDDL domain object.
        _parser (Parser): The parser used to convert PDDL elements into logic representations.
        _actions (List[Action]): The list of actions parsed from the domain.
        _predicates (List[Predicate]): The list of predicates parsed from the domain.
        _types (Dict[str, List[str]]): The mapping of types as returned by the PDDL parser.
            (Keys are type names; values are lists of supertypes for that type.)
    """
    
    def __init__(self, ppdl_domain: Any) -> None:
        """
        Initialize the Domain with its PDDL representation.
        
        Args:
            ppdl_domain (Any): The parsed PDDL domain object.
        """
        self._ppdl_domain = ppdl_domain
        self._parser = Parser()
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
        pddl_actions = list(list(self._ppdl_domain.actions))
        for pddl_action in pddl_actions:
            parameters = [self._parser.parse_term(param) for param in pddl_action.parameters]
            term_type_dict = {param: param.type for param in parameters}
            action = Action(
                pddl_action.name,
                parameters,
                self._parser.parse_formula(pddl_action.precondition, term_type_dict=term_type_dict),
                self._parser.parse_formula(pddl_action.effect, term_type_dict=term_type_dict)
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
        pddl_predicates = [pddl_predicate for pddl_predicate in list(self._ppdl_domain.predicates)]
        for pddl_predicate in pddl_predicates:
            predicate = self._parser.parse_predicate(pddl_predicate)
            predicates.append(predicate)
        return predicates
    
    def _parse_types(self) -> Dict[str, str]:
        """
        Return the types mapping as parsed from the PDDL domain.
        
        Returns:
            Dict[str, str]: The types mapping, where keys are type names and values
            are string of supertypes (as returned by the PDDL parser).
            Note: each type has a maximum of one supertype, if the type has no supertype, the value is None.
        """
        return self._ppdl_domain.types
    
    def is_subtype_of(self, type1: str, type2: str) -> bool:
        """
        Check whether 'type1' is a subtype of 'type2'.
        
        The check proceeds as follows:
            1. If type1 and type2 are equal, return True.
            2. Otherwise, if type2 is directly listed among the supertypes of type1 in _types, return True.
            3. Otherwise, recursively check each immediate supertype of type1.
        
        Args:
            type1 (str): The type to check.
            type2 (str): The potential supertype.
        
        Returns:
            bool: True if type1 is a subtype of type2, False otherwise.
        """
        if type1 == type2:
            return True
        
        # If type1 is not in our types mapping, we have no supertypes for it.
        if type1 not in self._types:
            return False
        
        # If type2 is an immediate supertype of type1, return True.
        if self._types[type1] is not None and type2 in self._types[type1]:
            return True

        # Otherwise, recursively check each immediate supertype of type1.
        return self.is_subtype_of(self._types.get(type1, ""), type2)
    
    def has_type_conflict(self, formula: Formula) -> bool:
        """
        Check if the formula has a type conflict.
        
        A type conflict occurs when a predicate in the formula has arguments that are not of the correct type.
        
        Args:    
            formula (Formula): The formula to check.
        
        Returns:
            bool: True if the formula has a type conflict, False otherwise.
        """
        if formula.term_type_dict is None:
            # no type information, cannot make any checks
            return False
        
        for term, types in formula.term_type_dict.items():
            if len(types) > 1:
                connected_types = []
                for term_type in types:
                    for other_type in types:
                        if term_type == other_type:
                            continue
                        if term_type != other_type and self.is_subtype_of(term_type, other_type):
                            connected_types.append(term_type)
                            connected_types.append(other_type)
                if len(connected_types) != len(types):
                    logger.debug("Type conflict detected: %s", formula.term_type_dict)
                    return True
        return False

    @property
    def actions(self) -> List[Action]:
        """
        Get the domain actions.
        
        Returns:
            List[Action]: The actions of the domain.
        """
        return self._actions
    
    @property
    def name(self) -> str:
        """
        Get the domain name.
        
        Returns:
            str: The name of the domain.
        """
        return self._ppdl_domain.name
    
    @property
    def predicates(self) -> List[Predicate]:
        """
        Get the domain predicates.
        
        Returns:
            List[Predicate]: The predicates of the domain.
        """
        return self._predicates
    
    @property
    def types(self) -> Dict[str, List[str]]:
        """
        Get the domain types hierarchy.
        
        Returns:
            Dict[str, str]: The types mapping of the domain, the values are supertypes.
        """
        return self._types