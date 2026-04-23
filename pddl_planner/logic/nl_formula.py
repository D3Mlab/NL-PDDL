from typing import Set, Dict, List, Any, Optional, Callable
from pddl_planner.logic.formula import Predicate, Term, Substitution, DisjunctiveFormula, ConjunctiveFormula
import re
import copy
class NLPredicate(Predicate):
    def __init__(self, name: str, str_representation: str, is_neg: bool, *terms: "Term",
     term_type_dict: Dict["Term", Set[str]] = None, entailed_by: "NLPredicate" = None, inital_terms: List["Term"] = None, inital_str: str = None,
     display_name: str = None) -> None:
        """
        A class that represents a NL predicate.

        Attributes:
            name (str): The name of the predicate.
            str_representation (str): The string representation of the predicate.
            is_neg (bool): Whether the predicate is negative.
            terms (List[Term]): The terms of the predicate.
            term_type_dict (Dict[Term, Set[str]]): The type of the terms.
            entailed_by (NLPredicate|List[NLPredicate]): The predicate that the predicate is entailed by.
            display_name (str): Human-readable schema name that keeps ?variables
                intact (e.g. "block ?b1 is directly above block ?b2"). Used only
                for display in the final regression-plan output — all internal
                logic (equality, hashing, SSA lookup, caching) still keys off
                :attr:`name`. Defaults to :attr:`name` when not provided.
        """

        super().__init__(name, is_neg, *terms, term_type_dict=term_type_dict)
        self._str_represntation = str_representation
        # keep track of the initial terms and original string for string replacement
        if inital_terms is None:
            self._inital_terms = copy.deepcopy(terms)
        else:
            self._inital_terms = copy.deepcopy(inital_terms)
        self._inital_str_represntation = copy.deepcopy(inital_str if inital_str is not None else self._str_represntation)
        self._entailed_by: "NLPredicate"|List["NLPredicate"] = entailed_by if entailed_by is not None else self
        # Map from entailed predicate name -> Substitution used for entailment (target_var -> candidate_var)
        self._entailed_substitutions: Dict[str, Substitution] = {}
        # Display-only schema name (keeps ?vars); fall back to `name` when
        # no dedicated display form was supplied by the caller.
        self._display_name: str = display_name if display_name is not None else name
        
    # Optional, planner-injected entailment checker: (a, b) -> bool if a is entailed by b or vice versa
    _entailment_checker: Optional[Callable[["NLPredicate", "NLPredicate"], bool]] = None

    @classmethod
    def set_entailment_checker(cls, checker: Callable[["NLPredicate", "NLPredicate"], bool]) -> None:
        cls._entailment_checker = checker

    def __str__(self) -> str:
        """Human-readable rendering that keeps ``?variable`` placeholders.

        Uses :attr:`display_name` (which preserves ``?b1``, ``?b2``, …) instead
        of :attr:`name` (which strips both variables and constants). The
        display form is strictly for presentation — equality, hashing, SSA
        lookup and entailment caching still key off :attr:`name` via the base
        :class:`Predicate` methods.

        Because :meth:`ConjunctiveFormula.__str__` and
        :meth:`DisjunctiveFormula.__str__` recurse with ``str(clause)``, the
        readable form automatically propagates when you print a subgoal or
        any formula that contains :class:`NLPredicate` leaves.
        """
        terms_str = ", ".join(str(term) for term in self._terms)
        head = self.display_name
        return f"{head}({terms_str})" if not self._is_neg else f"not {head}({terms_str})"

    def substitute(self, substitution: "Substitution") -> "NLPredicate":
        """Substitute the variables in the predicate using the provided substitution.

        Args:
            substitution (Substitution): A mapping of variables to terms.

        Returns:
            Predicate: A new Predicate with substitutions applied.
        """
        # combine the term_type_dict from the substitution with the term_type_dict of the predicate
        if self.term_type_dict is not None:
            for term1, term2 in substitution.items():
                if term1 in self.term_type_dict and term2 in self.term_type_dict:
                    self.term_type_dict[term2].update(self.term_type_dict[term1])
        # return a new predicate with substituted terms while preserving initial terms and original string
        return NLPredicate(
            self.name,
            self._str_represntation,
            self._is_neg,
            *[substitution.get(term, term) for term in self.terms],
            term_type_dict={substitution.get(term, term): types for term, types in self.term_type_dict.items()} if self.term_type_dict is not None else None,
            inital_terms=self._inital_terms,
            inital_str=self._inital_str_represntation,
            display_name=self._display_name,
        )

    def get_negation(self) -> "NLPredicate":
        """Get the negation of the predicate.

        Returns:
            Predicate: A new Predicate with the negation flag toggled.
        """
        return NLPredicate(self.name, self._str_represntation, not self._is_neg, *self.terms, inital_terms=self._inital_terms, inital_str=self._inital_str_represntation, display_name=self._display_name)

    def _equals_helper_with_entailment(self, other: "NLPredicate") -> bool:
        """Predicate equality that is aware of entailment computed by the LLM.

        Two predicates are considered equal if:
        - One's entailed set (from LLM) contains the other's name and negations match
        """
        # type check
        if not isinstance(other, NLPredicate):
            return False
        # get the entailed names of the predicate

        # Negation must match to be considered equal via entailment
        if self._is_neg != other.is_neg:
            return False

        # If other's name is in self's entailed names, consider equal (debug-safe print)
        if other.name in self.entailed_names():
            return True

        if NLPredicate._entailment_checker is not None:
            if NLPredicate._entailment_checker(self, other):
                return True
        return False

    def __hash__(self) -> int:
        """Use structural hash so instances remain hashable for sets/dicts.

        Note: Hash is intentionally structural and does not incorporate dynamic
        LLM entailment results to keep hashing stable.
        """
        return super().__hash__()

    @property
    def display_name(self) -> str:
        """Readable schema name that keeps ``?variable`` placeholders.

        ``display_name`` is intended for *presentation only* — e.g. formatting
        the final regression-plan output. All internal operations (equality,
        hashing, SSA lookup, entailment caching) must continue to use
        :attr:`name`, which strips both variables and constants so that
        paraphrased user predicates can be compared to their domain schemas.
        """
        return self._display_name

    @property
    def entailed(self) -> "NLPredicate":
        """Get the predicate that the predicate is entailed by.

        Returns:
            Predicate: The predicate that the predicate is entailed by.
        """
        if isinstance(self._entailed_by, List):
            #TODO: currently only support single entailed predicate, need to support multiple entailed predicates
            return copy.deepcopy(self._entailed_by[-1])
        return copy.deepcopy(self._entailed_by)
        
    @entailed.setter
    def entailed(self, entailed_predicate: "NLPredicate") -> None:
        """Set the predicate that the predicate is entailed by.

        Args:
            entailed_predicate (Predicate): The predicate that the predicate is entailed by.
        """
        if isinstance(self._entailed_by, List):
            self._entailed_by.append(entailed_predicate)
        elif isinstance(entailed_predicate, List):
            self._entailed_by = [self._entailed_by]
            self._entailed_by.extend(entailed_predicate)
        else:
            self._entailed_by = [self._entailed_by, entailed_predicate]

    def entailed_names(self) -> Set[str]:
        """
        Get the names of the predicates that the predicate is entailed by.

        Returns:
            Set[str]: The names of the predicates that the predicate is entailed by.
        """
        entailed_list: List[NLPredicate] = []
        if isinstance(self._entailed_by, List):
            entailed_list = [p for p in self._entailed_by if isinstance(p, NLPredicate)]
        elif isinstance(self._entailed_by, NLPredicate):
            entailed_list = [self._entailed_by]
        # Exclude self-name to avoid trivial equality
        return {p.name for p in entailed_list if p.name != self.name}
    
    @property
    def nl_description(self) -> str:
        """Get the NL description of the predicate.

        Returns:
            str: The NL description of the predicate.
        """
        # Safely replace initial terms with current terms in the original string using token-aware regex
        base = copy.deepcopy(self._inital_str_represntation)
        # Build pairs and replace longer tokens first to avoid partial overlaps
        pairs: List[tuple[str, str]] = [(str(old), str(new)) for new, old in zip(self.terms, self._inital_terms)]
        pairs.sort(key=lambda p: len(p[0]), reverse=True)

        def safe_replace(text: str, old: str, new: str) -> str:
            token_char = r"[A-Za-z0-9_?]"
            pattern = rf"(?<!{token_char}){re.escape(old)}(?!{token_char})"
            return re.sub(pattern, new, text)

        result = base
        for old, new in pairs:
            result = safe_replace(result, old, new)
        # Strip any pipe markers if present
        return result.replace('|', '')

    @nl_description.setter
    def nl_description(self, nl_description: str) -> None:
        """Set the NL description of the predicate.

        Args:
            nl_description (str): The NL description of the predicate.
        """
        self._str_represntation = nl_description

    # getter for _entailed_substitutions
    @property
    def entailed_substitutions(self) -> Dict[str, Substitution]:
        """Get the entailed substitutions of the predicate.

        Returns:
            Dict[str, Substitution]: The entailed substitutions of the predicate.
        """
        return self._entailed_substitutions

