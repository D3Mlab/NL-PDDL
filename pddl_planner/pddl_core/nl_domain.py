import copy
from difflib import get_close_matches
from typing import List, Dict, Any
from pddl_planner.pddl_core.action import Action
from pddl_planner.logic.nl_parser import NLParser
from pddl_planner.logic.formula import Predicate, Formula
from pddl_planner.pddl_core.domain import Domain

# Valid keys for each entry type in the domain JSON
_VALID_PREDICATE_ENTRY_KEYS = {"Predicate"}
_VALID_ACTION_ENTRY_KEYS = {"Action", "Action name", "Parameters", "Preconditions", "Effects"}
_REQUIRED_ACTION_KEYS = {"Action", "Action name", "Parameters", "Preconditions", "Effects"}
_VALID_EFFECTS_KEYS = {"Positive", "Negative"}

def _suggest_correction(invalid_key: str, valid_keys: set) -> str:
    """Return a suggestion string if a close match is found for the invalid key."""
    matches = get_close_matches(invalid_key, valid_keys, n=1, cutoff=0.5)
    if matches:
        return f" Did you mean \"{matches[0]}\"?"
    return f" Valid keys are: {sorted(valid_keys)}"

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
        if not isinstance(nl_domain, list):
            raise TypeError(
                f"NL domain must be a list of dicts, got {type(nl_domain).__name__}. "
                f"Expected format: [{{\"Predicate\": [...]}}, {{\"Action\": ..., ...}}, ...]"
            )
        # Defensive deep copy so parsing/mutation cannot leak back into the caller's input.
        self._nl_domain = copy.deepcopy(nl_domain)
        self._parser = NLParser()
        self._validate_domain_entries()
        self._predicates = self._parse_predicates()
        self._types = self._parse_types()
        self._actions = self._parse_actions()
    
    def _validate_domain_entries(self) -> None:
        """Validate all entries in the domain list have recognized keys."""
        for i, entry in enumerate(self._nl_domain):
            if not isinstance(entry, dict):
                raise TypeError(
                    f"Domain entry at index {i} must be a dict, got {type(entry).__name__}: {entry!r}"
                )
            keys = set(entry.keys())
            is_predicate_entry = "Predicate" in keys
            is_action_entry = "Action" in keys

            if not is_predicate_entry and not is_action_entry:
                # Try to detect misspelled "Predicate" or "Action"
                all_known = _VALID_PREDICATE_ENTRY_KEYS | _VALID_ACTION_ENTRY_KEYS
                suggestions = []
                for key in keys:
                    hint = _suggest_correction(key, all_known)
                    suggestions.append(f"  - \"{key}\"{hint}")
                raise ValueError(
                    f"Domain entry at index {i} has no recognized entry type key "
                    f"(expected \"Predicate\" or \"Action\").\n"
                    f"Found keys:\n" + "\n".join(suggestions)
                )

            if is_predicate_entry:
                unexpected = keys - _VALID_PREDICATE_ENTRY_KEYS
                if unexpected:
                    hints = [f"  - \"{k}\"{_suggest_correction(k, _VALID_ACTION_ENTRY_KEYS)}" for k in unexpected]
                    raise ValueError(
                        f"Predicate entry at index {i} has unexpected key(s):\n" +
                        "\n".join(hints) +
                        f"\nPredicate entries should only contain the key \"Predicate\"."
                    )

            if is_action_entry:
                # Check for unexpected keys
                unexpected = keys - _VALID_ACTION_ENTRY_KEYS
                if unexpected:
                    hints = [f"  - \"{k}\"{_suggest_correction(k, _VALID_ACTION_ENTRY_KEYS)}" for k in unexpected]
                    action_label = entry.get("Action", f"(index {i})")
                    raise ValueError(
                        f"Action \"{action_label}\" has unexpected key(s):\n" +
                        "\n".join(hints)
                    )
                # Check for missing required keys
                missing = _REQUIRED_ACTION_KEYS - keys
                if missing:
                    action_label = entry.get("Action", f"(index {i})")
                    raise ValueError(
                        f"Action \"{action_label}\" is missing required key(s): {sorted(missing)}. "
                        f"Required keys are: {sorted(_REQUIRED_ACTION_KEYS)}"
                    )
                # Validate Effects sub-keys
                effects = entry["Effects"]
                if not isinstance(effects, dict):
                    action_label = entry.get("Action", f"(index {i})")
                    raise TypeError(
                        f"\"Effects\" in action \"{action_label}\" must be a dict with "
                        f"\"Positive\" and/or \"Negative\" keys, got {type(effects).__name__}"
                    )
                unexpected_effects = set(effects.keys()) - _VALID_EFFECTS_KEYS
                if unexpected_effects:
                    action_label = entry.get("Action", f"(index {i})")
                    hints = [f"  - \"{k}\"{_suggest_correction(k, _VALID_EFFECTS_KEYS)}" for k in unexpected_effects]
                    raise ValueError(
                        f"\"Effects\" in action \"{action_label}\" has unexpected key(s):\n" +
                        "\n".join(hints) +
                        f"\nValid keys are: {sorted(_VALID_EFFECTS_KEYS)}"
                    )

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