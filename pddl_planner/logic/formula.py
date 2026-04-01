import textwrap, warnings, re, copy
from typing import List, Set, Dict, Union, Any, Tuple
import pddl_planner.logic.formula as formula
import numpy as np
class Logic:
    """Class capturing logical utilities and global variable generation.

    Class Attributes:
        _used_vars (Set[Variable]): Global set of variables that have been created.
        _free_var_idx (int): Counter for generating new variable names.
    """
    _used_vars: Set["Variable"] = set()
    _free_var_idx: int = 0

    @classmethod
    def get_new_var(cls) -> "Variable":
        """Get a new unused variable.

        The variable is named in the form 'V<int>' where <int> is the index, then incremented,
        and added to the global used-variable registry.

        Returns:
            Variable: A new unused variable.
        """
        var = Variable(f"V{cls._free_var_idx}")
        cls._used_vars.add(var)
        cls._free_var_idx += 1
        return var

    @classmethod
    def add_var(cls, var: "Variable") -> None:
        """Add a variable to the global set of used variables.

        Args:
            var (Variable): The variable to add.
        """
        cls._used_vars.add(var)

class Formula(Logic):
    """Abstract class representing a formula in logic.

    Attributes:
        _clauses (List["Formula"]): List of clauses that compose the formula.
    """

    def __init__(self, *clauses: "Formula", term_type_dict: Dict["Term", Set[str]] = None) -> None:
        """Initialize a Formula with a set of clauses.

        Args:
            *clauses (Formula): A variable number of Formula objects representing the clauses.
        """
        self._clauses: List["Formula"] = sorted(set(clauses), key=lambda c: repr(c))
        self._check_clauses_are_formulas()
        self._term_type_dict = self._get_combined_type_dict(term_type_dict)

    def add_clause(self, clause: "Formula") -> None:
        """Add a clause to the formula.

        Args:
            clause (Formula): A Formula object representing the clause to add.
        """
        self._clauses.append(clause)
        self._clauses = sorted(set(self._clauses), key=lambda c: repr(c))

    def remove_clause(self, clause: "Formula") -> None:
        """Remove a clause from the formula.

        Args:
            clause (Formula): A Formula object representing the clause to remove.
        """
        if clause in self._clauses:
            self._clauses.remove(clause)

    def collect_terms(self) -> Set["Term"]:
        """Collect all terms present in the formula.

        Returns:
            Set[Term]: A set of terms found in all clauses.
        """
        terms: Set[Term] = set()
        for clause in self._clauses:
            terms.update(clause.collect_terms())
        return terms

    def collect_vars(self) -> Set["Variable"]:
        """Collect all variables present in the formula.

        Returns:
            Set[Variable]: A set of variables found in all clauses.
        """
        vars: Set[Variable] = set()
        for clause in self._clauses:
            vars.update(clause.collect_vars())
        return vars

    def collect_preds(self) -> Set["Predicate"]:
        """Collect all predicates present in the formula.

        Returns:
            Set[Predicate]: A set of predicates found in all clauses.
        """
        preds: Set["Predicate"] = set()
        for clause in self._clauses:
            preds.update(clause.collect_preds())
        return preds

    def has_contradiction(self, other: "Formula") -> bool:
        """Check if the formula contradicts another formula.

        The method checks if any predicate in this formula has its negation in the other formula.

        Args:
            other (Formula): Another formula to check against.

        Returns:
            bool: True if a contradiction is found, False otherwise.
        """
        if isinstance(other, Formula):
            this_preds = self.collect_preds()
            other_preds = other.collect_preds()
            for pred in this_preds:
                if pred.get_negation() in other_preds:
                    return True
        return False

    def _check_clauses_are_formulas(self) -> None:
        """Ensure that all clauses are instances of Formula.

        Raises:
            ValueError: If a clause is not an instance of Formula.
        """
        for clause in self._clauses:
            if not isinstance(clause, Formula):
                raise ValueError(f"Clause {clause} is not a formula")

    def _get_combined_type_dict(self, type_dict: Dict["Term", Set[str]]) -> Dict["Term", str]:
        """Combined all the type_dict from the clauses in the formula.

        Returns:
            Dict[Term, Set[str]]: The combined type_dict.
        """
        if type_dict is not None:
            # type_dict is present, propagate the types down to the clauses
            for clause in self._clauses:
                if clause.term_type_dict is None:
                    clause.term_type_dict = {}
                for term, types in type_dict.items():
                    if term not in clause.collect_terms():
                        continue
                    if term in clause.term_type_dict:
                        clause.term_type_dict[term].update(types)
                    else:
                        clause.term_type_dict[term] = types

        else:
            # type_dict is not present, create a new one
            type_dict = {}


        # combine the types from the clauses in the formula
        for clause in self._clauses:
            if clause.term_type_dict is not None:
                for term, types in clause.term_type_dict.items():
                    if term in type_dict:
                        type_dict[term].update(types)
                    else:
                        type_dict[term] = types

        formula_terms = self.collect_terms()
        relevant_type_dict = {}
        # Remove any terms that are not present in the formula.
        for term, value in type_dict.items():
            if term in formula_terms:
                relevant_type_dict[term] = value
        return relevant_type_dict
    
    def get_num_preds(self) -> int:
        """Get the number of predicates in the formula.

        Returns:
            int: The number of predicates.
        """
        num = 0
        for clause in self._clauses:
            if isinstance(clause, Predicate) or isinstance(clause, Equality):
                num += 1
            else:
                num += clause.get_num_preds()
        return num

    @property
    def clauses(self) -> List["Formula"]:
        """Get the list of clauses in the formula.

        Returns:
            List[Formula]: The list of clauses.
        """
        return self._clauses

    @property
    def term_type_dict(self) -> Dict["Term", Set[str]]:
        """Get the dictionary mapping terms to their types."""
        return self._term_type_dict if hasattr(self, '_term_type_dict') else None
    
    @term_type_dict.setter
    def term_type_dict(self, value: Dict["Term", Set[str]]):
        self._term_type_dict = value

    def substitute(self, substitution: "Substitution") -> "Formula":
        """Substitute variables in the formula using the provided substitution.

        This is a stub that should be overridden by subclasses.

        Args:
            substitution (Substitution): The substitution mapping variables to terms.

        Returns:
            Formula: A new formula with variables replaced.
        """
        raise NotImplementedError(f'no implementation of substitute for {self}')
        pass

    def get_negation(self) -> "Formula":
        """Get the negation of the formula.

        This is a stub that should be overridden by subclasses.

        Returns:
            Formula: The negated formula.
        """
        pass

    def  distribute_and_over_or(self) -> "DisjunctiveFormula":
        """Recursively distribute all conjunctions over disjunctions.

        The resulting formula is a DisjunctiveFormula whose clauses are ConjunctiveFormulas,
        or a single ConjunctiveFormula for atomic formulas.

        Returns:
            DisjunctiveFormula: A DisjunctiveFormula in DNF.
        """
        # Helper: Given a formula in disjunctive form, return its conjunction parts.
        def to_conj_list(disj_formula: "DisjunctiveFormula") -> List[ConjunctiveFormula]:
            result: List[ConjunctiveFormula] = []
            for clause in disj_formula.clauses:
                if isinstance(clause, ConjunctiveFormula):
                    result.append(clause)
                else:
                    result.append(ConjunctiveFormula(clause))
            return result

        # Helper: Distribute two lists of ConjunctiveFormulas.
        def distribute_lists(list1: List[ConjunctiveFormula],
                             list2: List[ConjunctiveFormula]) -> List[ConjunctiveFormula]:
            combined: List[ConjunctiveFormula] = []
            for f1 in list1:
                for f2 in list2:
                    # Combine the clauses from both conjunctive formulas.
                    combined.append(ConjunctiveFormula(*(f1.clauses + f2.clauses)))
            return combined

        # Base case: atomic formulas are wrapped as a ConjunctiveFormula in a DisjunctiveFormula.
        if isinstance(self, FalseFormula):
            return FalseFormula()
        if isinstance(self, (Atomic, Equality, Predicate)):
            return DisjunctiveFormula(ConjunctiveFormula(self))  # type: ignore

        ret_formula = DisjunctiveFormula()
        # If the formula is a disjunction, distribute each clause recursively and flatten.
        if isinstance(self, DisjunctiveFormula):
            distributed = [cl.distribute_and_over_or() for cl in self.clauses]
            new_disjuncts: List[ConjunctiveFormula] = []
            for d in distributed:
                new_disjuncts.extend(d.clauses)  # Each d is expected to be a DisjunctiveFormula.
            ret_formula = DisjunctiveFormula(*new_disjuncts)

        # If the formula is a conjunction, recursively distribute the children and combine.
        if isinstance(self, ConjunctiveFormula):
            distributed_children = [child.distribute_and_over_or() for child in self.clauses]
            # Start with the DNF (list of conjunctive pieces) from the first child.
            result = to_conj_list(distributed_children[0])
            # For each remaining child, combine the current result with its pieces.
            for disj in distributed_children[1:]:
                result = distribute_lists(result, to_conj_list(disj))
            ret_formula = DisjunctiveFormula(*result)

        self._combine_and_propagate_type_dict(ret_formula)
        return ret_formula
    
    def to_latex(self, indent: int = 0, max_line_chars: int = 80) -> str:
        """
        Return a LaTeX representation of the formula.
        The final output is wrapped in a single align* environment.
        Each line is further wrapped so that no line exceeds max_line_chars,
        and each wrapped line is explicitly ended with '\\\\' to force a newline.
        """
        # Build the expression without wrapping each part in its own align* block.
        content = self._to_latex_recursive(indent, max_line_chars)
        # Split content into individual lines.
        lines = content.splitlines()
        wrapped_lines = []
        for line in lines:
            # Wrap each line using textwrap.wrap to get a list of wrapped chunks.
            chunks = textwrap.wrap(line, width=max_line_chars, break_long_words=False)
            # Join the chunks with an explicit LaTeX line break.
            wrapped_line = " \\\\ \n".join(chunks)
            # Optionally add the original indent.
            wrapped_lines.append(" " * indent + wrapped_line)
        content_wrapped = " \n".join(wrapped_lines)
        return "\\begin{align*}\n" + content_wrapped + "\n\\end{align*}"
    
    def _to_latex_recursive(self, indent: int, max_line_chars: int) -> str:
        indent_str = " " * indent  # 4-space indent
        # Example for a Disjunction
        if isinstance(self, DisjunctiveFormula):
            parts = [clause._to_latex_recursive(indent + 1, max_line_chars) for clause in self.clauses]
            # Use line breaks with proper indentation between disjuncts.
            return indent_str + "(" + (" \\vee " + indent_str).join(parts) + ")"
        elif isinstance(self, ConjunctiveFormula):
            parts = [clause._to_latex_recursive(indent + 1, max_line_chars) for clause in self.clauses]
            return indent_str + "(" + " \\wedge ".join(parts) + ")"
        elif isinstance(self, Equality):
            op = " = " if not self.is_neq else " \\neq "
            term1 = self._term1.to_latex() if hasattr(self._term1, "to_latex") else str(self._term1)
            term2 = self._term2.to_latex() if hasattr(self._term2, "to_latex") else str(self._term2)
            return indent_str + f"{term1}{op}{term2}"
        elif isinstance(self, Predicate):
            # Return a plain string for an atomic predicate.
            if hasattr(self, "name") and hasattr(self, "terms"):
                terms_latex = ", ".join(
                    [t.to_latex() if hasattr(t, "to_latex") else str(t) for t in self.terms]
                )
                return indent_str + f"{self.name}({terms_latex})"
            else:
                return indent_str + str(self)
        else:
            return indent_str + str(self)
    
    def __eq__(self, other: Any) -> bool:
        """Check if two formulas are equal.

        Args:
            other (Any): The formula to compare with.

        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(other, Formula) or type(self) != type(other):
            return False
        if len(self._clauses) != len(other._clauses):
            return False
        for clause, other_clause in zip(self._clauses, other._clauses):
            if not isinstance(clause, type(other_clause)):
                return False
            if clause != other_clause:
                return False
        return True

    def is_duplicate(self, other: Any) -> bool:
        """Check if two formulas are equal.

        Args:
            other (Any): The formula to compare with.

        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(other, Formula):
            return False
        if type(self) != type(other):
            return False

        mapping: Dict["Term", "Term"] = {}
        return self._equals_helper(other, mapping)

    def _equals_helper(self, other: "Formula", mapping: Dict["Term", "Term"], check_var_constant_consistency: bool = True) -> bool:
        """Helper method to check equality between formulas using a mapping for variables.

        Args:
            other (Formula): The formula with which to compare.
            mapping (Dict[Term, Term]): A mapping from terms in self to terms in other.
            check_var_constant_consistency (bool): Whether to check if the mapping is consistent with the variable-constant consistency.

        Returns:
            bool: True if formulas are equivalent, False otherwise.
        """

        if isinstance(self, Predicate) and isinstance(other, Predicate):
            if len(self.terms) != len(other.terms) or self.is_neg != other.is_neg:
                return False
            if self.name != other.name:
                # if the entailment checker is injected, use it to check if the predicates are equal
                if hasattr(self, '_equals_helper_with_entailment'):
                    if not self._equals_helper_with_entailment(other):
                        return False
                else:
                    return False
            for term1, term2 in zip(self.terms, other.terms):
                if isinstance(term1, Variable) and isinstance(term2, Variable):
                    if term1 in mapping:
                        if mapping[term1] != term2:
                            return False
                    else:
                        mapping[term1] = term2
                elif isinstance(term1, Constant) and isinstance(term2, Constant):
                    if term1.name != term2.name:
                        return False
                elif check_var_constant_consistency:
                    return False
            if len(mapping) != len(set(mapping.values())):
                return False
            return True


        elif isinstance(self, Equality) and isinstance(other, Equality):
            if self.is_neq != other.is_neq:
                return False
            for t1, t2 in ((self.term1, other.term1), (self.term2, other.term2)):
                if isinstance(t1, Variable) and isinstance(t2, Variable):
                    if t1 in mapping:
                        if mapping[t1] != t2:
                            return False
                    else:
                        mapping[t1] = t2
                elif isinstance(t1, Constant) and isinstance(t2, Constant):
                    if t1 != t2:
                        return False
            if len(mapping) != len(set(mapping.values())):
                return False
            return True
    
        elif ((isinstance(self, ConjunctiveFormula) and isinstance(other, ConjunctiveFormula))
              or (isinstance(self, DisjunctiveFormula) and isinstance(other, DisjunctiveFormula))):
            if len(self.clauses) != len(other.clauses):
                return False
            for clause1, clause2 in zip(sorted(self.clauses, key=str), sorted(other.clauses, key=str)):
                if not clause1._equals_helper(clause2, mapping):
                    return False
            return True
        else:
            return False

    def __hash__(self) -> int:
        """Compute the hash of the formula.

        Returns:
            int: The hash value.
        """
        return hash(tuple(self._clauses) + (type(self),))
    
    def __len__(self):
        return len(str(self))

    def _combine_and_propagate_type_dict(self, result_formula: "Formula") -> None:
        """
        Combine the parent's type dictionary with the type dictionaries of all the clauses in result_formula,
        and then propagate the combined dictionary to result_formula and each of its clauses.
        
        Args:
            result_formula (Formula): The formula (Disjunctive or Conjunctive) whose type dictionaries are to be unified.
        """
        combined: Dict["Term", Set[str]] = {}
        # If the parent's type dictionary exists, add it first.
        if self.term_type_dict is not None:
            for term, types in self.term_type_dict.items():
                combined[term] = set(types)
        # For each clause in the result_formula, union in its type dictionary.
        for clause in result_formula.clauses:
            if clause.term_type_dict is not None:
                for term, types in clause.term_type_dict.items():
                    if term in combined:
                        combined[term].update(types)
                    else:
                        combined[term] = set(types)
        terms = result_formula.collect_terms()
        relevant_combined = {}
        # Remove any terms that are not present in the formula.
        for term, value in combined.items():
            if term in terms:
                relevant_combined[term] = value

        # Set the combined dictionary in the result_formula with a deep copy.
        result_formula.term_type_dict = copy.deepcopy(relevant_combined)
        # Propagate a deep copy of the combined dictionary to each clause.
        for clause in result_formula.clauses:
            clause.term_type_dict = copy.deepcopy(relevant_combined)


class ConjunctiveFormula(Formula):
    """Conjunctive formula (AND)."""

    def substitute(self, substitution: "Substitution") -> "ConjunctiveFormula":
        """Substitute the variables in the formula using the provided substitution.

        Args:
            substitution (Substitution): A mapping of variables to terms.

        Returns:
            ConjunctiveFormula: A new ConjunctiveFormula with substitutions applied.
        """
        # combine the term_type_dict from the substitution with the term_type_dict of the predicate
        if self.term_type_dict is not None:
            for term1, term2 in substitution.items():
                if term1 in self.term_type_dict and term2 in self.term_type_dict:
                    self.term_type_dict[term2].update(self.term_type_dict[term1])
        return ConjunctiveFormula(*[clause.substitute(substitution) for clause in self.clauses], term_type_dict={substitution.get(term, term): types for term, types in self.term_type_dict.items()})

    def __repr__(self) -> str:
        """Return the canonical string representation of the ConjunctiveFormula.

        Returns:
            str: The string representation.
        """
        clauses_str = " ∧ ".join(repr(clause) for clause in self._clauses)
        return f"({clauses_str})"

    def __str__(self) -> str:
        """Return the informal string representation of the ConjunctiveFormula.

        Returns:
            str: The string representation.
        """
        clauses_str = " ∧ ".join(str(clause) for clause in self._clauses)
        return f"({clauses_str})"

    def get_negation(self) -> "DisjunctiveFormula":
        """Get the negation of the conjunctive formula.

        Returns:
            DisjunctiveFormula: A DisjunctiveFormula with each clause negated.
        """
        #print(f'get_negation dnf: {self._clauses}' if np.any([isinstance(clause, Equality) for clause in self.clauses]) else None)
        return DisjunctiveFormula(*[clause.get_negation() for clause in self.clauses])


    def simplify_plan(self) -> Formula:
        """Simplify the conjunction by simplifying each clause and removing contradictions in the regression planning stage.
        
        If any two clauses contradict, the result is False.
        
        Returns:
            Formula: The simplified conjunctive formula, or FalseFormula if a contradiction was found.
        """
        # Simplify each clause if possible
            # simplified_clauses = [clause.simplify() if hasattr(clause, 'simplify') else clause 
            #                         for clause in self._clauses]

        simplified_clauses = []
        for disj in self._clauses:
            s = disj.simplify() if hasattr(disj, 'simplify') else disj
            # Exclude disjuncts that are false.
            if isinstance(s, Equality):
                if s.is_neq and s.term1.name != s.term2.name and isinstance(s.term1, Constant) and isinstance(s.term2, Constant):
                        continue
                if not s.is_neq and s.term1.name == s.term2.name and isinstance(s.term1, Constant) and isinstance(s.term2, Constant):
                        continue
            simplified_clauses.append(s)
        
        
        # Check pairwise for contradiction among simplified clauses
        for i in range(len(simplified_clauses)):
            if isinstance(simplified_clauses[i], Equality):
                if simplified_clauses[i].is_neq and simplified_clauses[i].term1 == simplified_clauses[i].term2:
                    return FalseFormula()
                if simplified_clauses[i].term1.name != simplified_clauses[i].term2.name and (isinstance(simplified_clauses[i].term1, Constant) and isinstance(simplified_clauses[i].term2, Constant)):
                    return FalseFormula()
            for j in range(i + 1, len(simplified_clauses)):
                if simplified_clauses[i].has_contradiction(simplified_clauses[j]):
                    return FalseFormula()
        # Remove duplicates (optional) and return single clause if only one remains
        unique = list({str(clause): clause for clause in simplified_clauses}.values())
        if not unique:
            return FalseFormula()
        # if len(unique) == 1:
        #     return unique[0]

        ret_formula = ConjunctiveFormula(*unique)
        self._combine_and_propagate_type_dict(ret_formula)
        return ret_formula
    
    def simplify(self) -> Formula:
        """Simplify the conjunction by simplifying each clause and removing contradictions.
        
        If any two clauses contradict, the result is False.
        
        Returns:
            Formula: The simplified conjunctive formula, or FalseFormula if a contradiction was found.
        """
        # Simplify each clause if possible
            # simplified_clauses = [clause.simplify() if hasattr(clause, 'simplify') else clause 
            #                         for clause in self._clauses]
        simplified_clauses = [clause.simplify() if hasattr(clause, 'simplify') else clause 
                                for clause in self._clauses]
    
        # Check pairwise for contradiction among simplified clauses
        for i in range(len(simplified_clauses)):
            if isinstance(simplified_clauses[i], Equality):
                if simplified_clauses[i].is_neq and simplified_clauses[i].term1 == simplified_clauses[i].term2:
                    return FalseFormula()
                if simplified_clauses[i].term1.name != simplified_clauses[i].term2.name and (isinstance(simplified_clauses[i].term1, Constant) and isinstance(simplified_clauses[i].term2, Constant)):
                    return FalseFormula()
                # for j in range(i + 1, len(simplified_clauses)):
                #     if simplified_clauses[i].has_contradiction(simplified_clauses[j]):
                #         return FalseFormula()
            for j in range(i + 1, len(simplified_clauses)):
                if simplified_clauses[i].has_contradiction(simplified_clauses[j]):
                    return FalseFormula()
        # Remove duplicates (optional) and return single clause if only one remains
        unique = list({str(clause): clause for clause in simplified_clauses}.values())
        if not unique:
            return FalseFormula()
        # if len(unique) == 1:
        #     return unique[0]

        ret_formula = ConjunctiveFormula(*unique)
        self._combine_and_propagate_type_dict(ret_formula)
        return ret_formula

    def simplify_equality_variables_only(self, current_goal: "Formula") -> Tuple["Formula", "Substitution"]:
        """
        Simplify equalities in this conjunctive formula.
        
        This function processes equality clauses when only one of the two terms is a variable.
        For each such equality, it selects the representative variable (the one with the
        lower alphabetical value), adds a substitution mapping (using Substitution class),
        and then applies the substitution to every clause in the formula.
        Equality clauses used for substitution are then removed.
        
        Returns:
            Tuple[Formula, Substitution]: The simplified conjunctive formula with substitutions
            applied, and the substitution mapping.
        """
        simplified_clauses = [clause.simplify() if hasattr(clause, 'simplify') else clause 
                                for clause in self._clauses]
        
        # Build the substitution from equality clauses (only when both terms are variables).
        equality_subst = Substitution()
        for clause in simplified_clauses:
            if isinstance(clause, Equality) and not clause.is_neq:
                if isinstance(clause.term1, Variable) and isinstance(clause.term2, Variable):
                    equality_subst[clause.term2] = clause.term1
        
        substituted_clauses = set()
        for clause in simplified_clauses:
            # Apply the substitution to every clause.
            substituted_clauses.add(clause.substitute(equality_subst))
        
        # Remove equality clauses (only those where both terms are Variables) from the result.
        final_clauses = []
        for clause in substituted_clauses:
            if isinstance(clause, Equality) and not clause.is_neq:
                if isinstance(clause.term1, Variable) and isinstance(clause.term2, Variable):
                    continue
            final_clauses.append(clause)
        
        # Remove duplicates
        unique = list({str(clause): clause for clause in final_clauses}.values())
        if not unique:
            return (FalseFormula(), equality_subst)
        # if len(unique) == 1:
        #     return (unique[0], equality_subst)
        return (ConjunctiveFormula(*unique), equality_subst)
    
    def simplify_equality(self, current_goal: "Formula") -> Tuple["Formula", "Substitution"]:
        """
        Simplify equalities in this conjunctive formula.
        
        For each such equality, it selects the representative variable (the one with the
        lower alphabetical value), adds a substitution mapping (using Substitution class),
        and then applies the substitution to every clause in the formula.
        Equality clauses used for substitution are then removed.
        
        Returns:
            Tuple[Formula, Substitution]: The simplified conjunctive formula with substitutions
            applied, and the substitution mapping.
        """

        def build_equality_graph(eqs: List["Equality"]) -> Dict["Variable", List["Term"]]:
            graph: Dict[Variable, List[Term]] = {}
            def add_edge(v: Variable, t: Term):
                if v not in graph:
                    graph[v] = []
                if t not in graph[v]:
                    graph[v].append(t)
            for eq in eqs:
                if eq.is_neq:
                    continue
                t1, t2 = eq.term1, eq.term2
                if isinstance(t1, Variable) and isinstance(t2, Variable):
                    add_edge(t1, t2)
                    add_edge(t2, t1)
                elif isinstance(t1, Variable) and isinstance(t2, Constant):
                    add_edge(t1, t2)
                elif isinstance(t2, Variable) and isinstance(t1, Constant):
                    add_edge(t2, t1)
            return graph

        def resolve_graph_to_substitution(graph: Dict["Variable", List["Term"]]) -> Tuple["Substitution", bool]:
            substitution = Substitution()
            inconsistent = False
            visited: Set[Variable] = set()

            # helper to compare variables by numeric index if present (V12 < V2 handled properly)
            def variable_sort_key(v: Variable):
                parts = re.split(r'(\d+)', v.name)
                return [int(p) if p.isdigit() else p for p in parts]

            for start in sorted(graph.keys(), key=variable_sort_key):
                if start in visited:
                    continue
                # BFS to collect connected component of variables, plus any constants they connect to
                queue: List[Variable] = [start]
                component_vars: Set[Variable] = set()
                component_consts: Set[Constant] = set()
                visited.add(start)

                while queue:
                    v = queue.pop(0)
                    component_vars.add(v)
                    for t in graph.get(v, []):
                        if isinstance(t, Constant):
                            component_consts.add(t)
                        elif isinstance(t, Variable):
                            if t not in visited:
                                visited.add(t)
                                queue.append(t)

                # Resolution rule:
                # 1) If multiple distinct constants appear in the component, it's inconsistent.
                if len({c.name for c in component_consts}) > 1:
                    inconsistent = True
                    # No need to compute further; keep scanning to mark visited but final result will be False
                # 2) If any constants present, map all vars in component to that constant.
                representative_term: Term
                if component_consts:
                    # choose one constant deterministically by name
                    representative_term = sorted(component_consts, key=lambda c: c.name)[0]
                else:
                    # 3) Otherwise choose the smallest variable by our sort key
                    representative_term = sorted(component_vars, key=variable_sort_key)[0]

                # Map every variable in component to the representative (skip self-map)
                for v in component_vars:
                    if v != representative_term:
                        substitution[v] = representative_term  # type: ignore[arg-type]

                # Propagate constant choice: if representative is a constant, then any variable
                # that has this component's variables in its adjacency should also map to the constant
                if isinstance(representative_term, Constant):
                    for other_var, neighbors in graph.items():
                        if other_var in component_vars:
                            continue
                        if any(isinstance(n, Variable) and n in component_vars for n in neighbors):
                            substitution[other_var] = representative_term

            return (substitution, inconsistent)

        # First, simplify each clause if possible.
        simplified_clauses = [clause.simplify() if hasattr(clause, 'simplify') else clause 
                                for clause in self._clauses]
        
        # Build equality graph from all positive equalities in the conjunction
        eqs: List[Equality] = [c for c in simplified_clauses if isinstance(c, Equality) and not c.is_neq]
        equality_subst, inconsistent = resolve_graph_to_substitution(build_equality_graph(eqs))
        if inconsistent:
            return (FalseFormula(), Substitution())
        substituted_clauses = set()
        for clause in simplified_clauses:
            # Apply the substitution to every clause.
            substituted_clauses.add(clause.substitute(equality_subst))
        
        # Remove equality clauses (only those where both terms are Variables) from the result.
        final_clauses = []
        for clause in substituted_clauses:
            if isinstance(clause, Equality):
                if not clause.is_neq:
                    #if isinstance(clause.term1, Variable) and isinstance(clause.term2, Variable):
                    continue
                elif isinstance(clause.term1, Constant) and isinstance(clause.term2, Constant):
                    if clause.term1.name != clause.term2.name:
                        continue # remove due to unique name axiom
            final_clauses.append(clause)
        
        # Remove duplicates
        unique = list({str(clause): clause for clause in final_clauses}.values())
        if not unique:
            return (FalseFormula(), equality_subst)
        # if len(unique) == 1:
        #     return (unique[0], equality_subst)
        return (ConjunctiveFormula(*unique), equality_subst)
        
    def implies(self, other: "ConjunctiveFormula") -> bool:
        """Check if this formula implies another formula.

        Args:
            other (ConjunctiveFormula): The formula to check against.

        Returns:
            bool: True if this formula implies the other, False otherwise.
        """

        if not isinstance(other, ConjunctiveFormula):
            warnings.warn("Implication is only defined for ConjunctiveFormulas.", UserWarning)
            return False
        
        # If any clause in the other formula contradicts a clause in this formula, return False.
        for other_clause in other.clauses:
            flag = False
            for self_clause in self.clauses:
                if other_clause == self_clause:
                    # if other_clause can be found in self
                    flag = True
                    break
            if not flag:
                # if other_clause is not a duplicate of any clause in self, return False
                return False
        return True


class DisjunctiveFormula(Formula):
    """Disjunctive formula (OR)."""


    def simplify_plan(self) -> "Formula":
        """
        Planner-aware simplification.

        - Applies standard simplify() on each disjunct
        - Removes duplicate/superset disjuncts
        - Additionally removes tautological variable-inequalities (X != Y) under the Unique Name Axiom context
          when they do not constrain any other clause in the disjunct.

        Returns:
            Formula: The simplified disjunctive formula.
        """
        simplified_disjuncts: List[Formula] = []
        for disj in self._clauses:
            if hasattr(disj, 'simplify_plan'):
                s = disj.simplify_plan()
            elif hasattr(disj, 'simplify'):
                s = disj.simplify()
            else:
                s = disj
            if isinstance(s, FalseFormula):
                continue
            if isinstance(s, ConjunctiveFormula):
                kept = []
                for c in s.clauses:
                    if (isinstance(c, Equality) and c.is_neq and isinstance(c.term1, Constant) and isinstance(c.term2, Constant) and c.term1.name != c.term2.name):
                        # Drop pure variable-inequalities as they add no constraint under UNA in planning
                        continue
                    kept.append(c)
                s = ConjunctiveFormula(*kept)
                if isinstance(s, FalseFormula):
                    continue
            simplified_disjuncts.append(s)

        # Remove duplicate clauses.
        unique_disjuncts: List[Formula] = []
        for clause in simplified_disjuncts:
            is_dup = False
            for u in unique_disjuncts:
                if clause.is_duplicate(u):
                    is_dup = True
                    break
            if not is_dup:
                unique_disjuncts.append(clause)

        if not unique_disjuncts:
            return FalseFormula()

        # Remove any disjunct that is a superset of another disjunct.
        minimal_disjuncts: List[Formula] = []
        for i, di in enumerate(unique_disjuncts):
            di_clauses = di.clauses if isinstance(di, ConjunctiveFormula) else [di]
            di_set = set(str(cl) for cl in di_clauses)
            subsumed = False
            for j, dj in enumerate(unique_disjuncts):
                if i == j:
                    continue
                dj_clauses = dj.clauses if isinstance(dj, ConjunctiveFormula) else [dj]
                dj_set = set(str(cl) for cl in dj_clauses)
                if dj_set.issubset(di_set) and dj_set != di_set:
                    subsumed = True
                    break
            if not subsumed:
                minimal_disjuncts.append(di)

        ret_formula = DisjunctiveFormula(*minimal_disjuncts)
        self._combine_and_propagate_type_dict(ret_formula)
        return ret_formula

    def simplify(self) -> Formula:
        """Simplify the disjunction by simplifying each disjunct, eliminating false
        ones, and removing duplicate clauses.
        
        If all disjuncts simplify to false then the result is false.
        
        Returns:
            Formula: The simplified disjunctive formula, or a single disjunct if only one remains.
        """
        simplified_disjuncts = []
        for disj in self._clauses:
            s = disj.simplify() if hasattr(disj, 'simplify') else disj
            # Exclude disjuncts that are false.
            # if isinstance(s, Equality) and s.is_neq and s.term1.name != s.term2.name and isinstance(s.term1, Variable) and isinstance(s.term2, Variable):
            #         continue
            if not isinstance(s, FalseFormula):
                simplified_disjuncts.append(s)
        
        # Remove duplicate clauses.
        unique_disjuncts = []
        for clause in simplified_disjuncts:
            is_dup = False
            for u in unique_disjuncts:
                if clause.is_duplicate(u):
                    is_dup = True
                    break
            if not is_dup:
                unique_disjuncts.append(clause)

        if not unique_disjuncts:
            return FalseFormula()
        
        # Remove any disjunct that is a superset of another disjunct.
        minimal_disjuncts = []
        for i, di in enumerate(unique_disjuncts):
            # Get the set of conjuncts for di.
            # If di is not a ConjunctiveFormula, consider it as a singleton list.
            di_clauses = di.clauses if isinstance(di, ConjunctiveFormula) else [di]
            # Use string representations as a proxy for equality.
            di_set = set(str(cl) for cl in di_clauses)
            subsumed = False
            for j, dj in enumerate(unique_disjuncts):
                if i == j:
                    continue
                dj_clauses = dj.clauses if isinstance(dj, ConjunctiveFormula) else [dj]
                dj_set = set(str(cl) for cl in dj_clauses)
                # If dj_set is a proper subset of di_set, then di is redundant.
                if dj_set.issubset(di_set) and dj_set != di_set:
                    subsumed = True
                    break
            if not subsumed:
                minimal_disjuncts.append(di)
        
        # if len(minimal_disjuncts) == 1:
        #     return (DisjunctiveFormula(minimal_disjuncts[0])

        ret_formula = DisjunctiveFormula(*minimal_disjuncts)
        self._combine_and_propagate_type_dict(ret_formula)
        return ret_formula

    def substitute(self, substitution: "Substitution") -> "DisjunctiveFormula":
        """Substitute the variables in the formula using the provided substitution.

        Args:
            substitution (Substitution): A mapping of variables to terms.

        Returns:
            DisjunctiveFormula: A new DisjunctiveFormula with substitutions applied.
        """
        # combine the term_type_dict from the substitution with the term_type_dict of the predicate
        if self.term_type_dict is not None:
            for term1, term2 in substitution.items():
                if term1 in self.term_type_dict and term2 in self.term_type_dict:
                    self.term_type_dict[term2].update(self.term_type_dict[term1])
        term_type_dict={substitution.get(term, term): types for term, types in self.term_type_dict.items()}
        return DisjunctiveFormula(*[clause.substitute(substitution) for clause in self.clauses], term_type_dict=term_type_dict)

    def __repr__(self) -> str:
        """Return the canonical string representation of the DisjunctiveFormula.

        Returns:
            str: The string representation.
        """
        clauses_str = " ∨ ".join(repr(clause) for clause in self._clauses)
        return f"({clauses_str})"

    def __str__(self) -> str:
        """Return the informal string representation of the DisjunctiveFormula.

        Returns:
            str: The string representation.
        """
        clauses_str = "\n ∨ ".join(str(clause) for clause in self._clauses)
        return f"({clauses_str})"

    def get_negation(self) -> "ConjunctiveFormula":
        """Get the negation of the disjunctive formula.

        Returns:
            ConjunctiveFormula: A ConjunctiveFormula with each clause negated.
        """
        #print(f'get_negation cnf: {self._clauses}' if np.any([isinstance(clause, Equality) for clause in self.clauses]) else None)
        return ConjunctiveFormula(*[clause.get_negation() for clause in self.clauses])
    
class Operator(Formula):
    # def __init__(self, *clauses: Formula) -> None:
       """Initialize an Operator object.

        Args:
            *clauses (Formula): A variable number of Formula objects.
        """
        # super().__init__(*clauses)
    
class Equality(Operator):
    def __init__(self, term1: "Term", term2: "Term", is_neq: bool = False, term_type_dict=None) -> None:
        """Initialize an Equality object.

        Args:
            term1 (Term): The first term in the equality.
            term2 (Term): The second term in the equality.
            is_neq (bool): Whether the equality is a negated equality (i.e., !=).
        """
        sorted_terms = sorted([term1, term2], key=lambda t: [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r'(\d+)', str(t))])
        self._term1: Term = sorted_terms[0]
        self._term2: Term = sorted_terms[1]
        self._clauses: List["Formula"] = []  
        self._is_neq: bool = is_neq
        self._term_type_dict = term_type_dict
    def collect_terms(self) -> Set["Term"]:
        """Collect all terms present in the formula.

        Returns:
            Set[Term]: A set of terms found in all clauses.
        """
        return {self._term1, self._term2}
    
    def get_negation(self):
        return Equality(self._term1, self._term2, not self._is_neq)

    def substitute(self, substitution: "Substitution") -> "Equality":
        new_term1 = substitution.get(self._term1, self._term1)
        new_term2 = substitution.get(self._term2, self._term2)
        #print(f'substitute equality: {self._term1} {self._term2} {self._is_neq} {new_term1} {new_term2}')
        return Equality(new_term1, new_term2, self._is_neq)
    
    @property
    def is_neq(self) -> bool:
        return self._is_neq
    
    @property
    def term1(self) -> "Term":
        return self._term1
    
    @property
    def term2(self) -> "Term":
        return self._term2
    
    def __str__(self) -> str:
        """Return the string representation of the equality.

        Returns:
            str: A string in the form 'term1 == term2' or 'term1 != term2'.
        """
        op = "!=" if self._is_neq else "=="
        return f"{self._term1} {op} {self._term2}"
    
    def __repr__(self) -> str:
        """Return the canonical string representation of the equality.

        Returns:
            str: A canonical string showing the Equality with its terms and operator.
        """
        op = "!=" if self._is_neq else "=="
        return f"{repr(self._term1)} {op} {repr(self._term2)}"
    
    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Equality):
            return NotImplemented
        # For equality, sort by term names and the operator.
        return (str(self._term1), str(self._term2), self._is_neq) < \
            (str(other._term1), str(other._term2), other._is_neq)
    
    def __hash__(self) -> int:
        """Compute the hash of the equality.

        Returns:
            int: The hash value.
        """
        return hash((self._term1, self._term2, self._is_neq))
    
    def __len__(self):
        return len(str(self))

    def has_contradiction(self, other:"Equality") -> bool:
        if isinstance(other, Equality):
            if self._term1 == other._term1 and self._term2 == other._term2 and self._is_neq != other._is_neq:
                # check if the have same type
                if type(self._term1) != type(other._term1) or type(self._term2) != type(other._term2):
                    return False
                return True
            if self._term1 == other._term2 and self._term2 == other._term1 and self._is_neq != other._is_neq:
                if type(self._term1) != type(other._term1) or type(self._term2) != type(other._term2):
                    return False
                return True
            if self._term1 == other._term1 and self._term2 != other._term2 and not self._is_neq and not other._is_neq:
                if isinstance(self._term1, Variable) and type(self._term1) == type(other._term1) and isinstance(self._term2, Constant) and type(self._term2) == type(other._term2):
                    return True
        return False
    
    @property
    def term_type_dict(self) -> Dict["Term", Set[str]]:
        return self._term_type_dict
    
    @term_type_dict.setter
    def term_type_dict(self, value: Dict["Term", Set[str]]):
        self._term_type_dict = value


class Atomic(Formula):
    """Atomic formula: a formula that cannot be decomposed further."""
    pass

class FalseFormula(Atomic):
    """Represents a formula that is always false."""
    def __str__(self) -> str:
        return "FalseFormula"

    def __repr__(self) -> str:
        return "FalseFormula"

    def simplify(self) -> "FalseFormula":
        return self

    def get_negation(self) -> "Formula":
        # Depending on your system, you might wish to implement a TrueFormula.
        # For now, simply return self or raise an error.
        raise NotImplementedError("Negation of FalseFormula is not implemented.")
    
    def __eq__(self, other):
        return isinstance(other, FalseFormula)
    
    def __hash__(self):
        return hash(FalseFormula)
    
    def substitute(self, substitution: Dict["Term", "Term"]) -> "FalseFormula":
        return self

class Term(Logic):
    """Term in a formula"""
    def __init__(self, name, term_type=None):
        self._name = name
        self._type = term_type

    @property
    def name(self) -> str:
        """Get the name of the term."""
        return self._name

    @property   
    def type(self) -> str:
        """Get the type of the term."""
        return self._type

class Variable(Term):
    """Variable in a formula"""
    def __init__(self, name, term_type=None):
        super().__init__(name, term_type)
        
    def __str__(self) -> str:
        """Get the string representation."""
        return f"?{self._name}"
    
    def __repr__(self) -> str:
        """Get the string representation."""
        return f"?{self._name}"
    
    def __eq__(self, other) -> bool:
        """Compare with another object."""
        return (
            isinstance(other, Variable)
            and self._name.replace("?", "") == other._name.replace("?", "")
        )
    
    def __hash__(self):
        """Get the hash."""
        return hash((Variable, self._name))

class Constant(Term):
    "Constant in a formula"
    def __str__(self) -> str:
        """Get the string representation."""
        return self._name
    
    def __repr__(self) -> str:
        """Get the string representation."""
        return self._name
    
    def __eq__(self, other) -> bool:
        """Compare with another object."""
        return isinstance(other, Constant) and self.name == other.name
    
    def __hash__(self):
        """Get the hash."""
        return hash((Constant, self._name))

class Predicate(Atomic):
    """Predicate in a formula.

    Args:
        name (str): The predicate's name.
        is_neg (bool): True if the predicate is negated.
        *terms (Term): A list of terms in the predicate.
    """

    def __init__(self, name: str, is_neg: bool, *terms: "Term", term_type_dict: Dict["Term", Set[str]] = None) -> None:
        self._name: str = name
        self._terms: List["Term"] = list(terms)
        self._check_terms()
        self._vars: List[Variable] = self._get_vars()
        self._is_neg: bool = is_neg
        self._term_type_dict = term_type_dict

    def collect_terms(self) -> List["Term"]:
        """Return the list of terms in the predicate.

        Returns:
            List[Term]: The predicate's terms.
        """
        return self._terms

    def collect_vars(self) -> List["Variable"]:
        """Return the list of variables in the predicate.

        Returns:
            List[Variable]: The predicate's variables.
        """
        return self._vars

    def collect_preds(self) -> Set["Predicate"]:
        """Return a set containing this predicate.

        Returns:
            Set[Predicate]: A set with the predicate.
        """
        return {self}

    def collect_preds_name(self) -> Set[str]:
        """Return the predicate name in a set.

        Returns:
            Set[str]: A set with the predicate's name (prefixed with '¬' if negated).
        """
        return {self.name} if not self._is_neg else {f"¬{self.name}"}

    def substitute(self, substitution: "Substitution") -> "Predicate":
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
        # term_type_dict={substitution.get(term, term): types for term, types in self.term_type_dict.items()}
        return Predicate(self.name, self._is_neg, *[substitution.get(term, term) for term in self.terms], 
        term_type_dict={substitution.get(term, term): types for term, types in self.term_type_dict.items()} if self.term_type_dict is not None else None)

    def get_negation(self) -> "Predicate":
        """Get the negation of the predicate.

        Returns:
            Predicate: A new Predicate with the negation flag toggled.
        """
        return Predicate(self.name, not self._is_neg, *self.terms)

    def _get_vars(self) -> List["Variable"]:
        """Collect all variables from the predicate's terms.

        Returns:
            List[Variable]: A list of variables in the predicate.
        """
        vars: List[Variable] = []
        for term in self._terms:
            if isinstance(term, Variable):
                vars.append(term)
        return vars

    def _check_terms(self) -> None:
        """Check that the terms in the predicate are instances of Term.

        Raises:
            ValueError: If a term not an instance of Term.
        """
        for term in self._terms:
            # Check if it's a Variable or Constant (both inherit from Term)
            if not (isinstance(term, Term)):
                raise ValueError(f"Term {term} is not a Term, it is a {type(term)}")

    @property
    def vars(self) -> List["Variable"]:
        """Get the variables in the predicate.

        Returns:
            List[Variable]: A list of variables.
        """
        return self._vars

    @property
    def name(self) -> str:
        """Get the name of the predicate.

        Returns:
            str: The predicate's name.
        """
        return self._name

    @property
    def terms(self) -> List["Term"]:
        """Get the terms of the predicate.

        Returns:
            List[Term]: A list of terms.
        """
        return self._terms

    @property
    def arity(self) -> int:
        """Get the arity (number of arguments) of the predicate.

        Returns:
            int: The number of terms.
        """
        return len(self._terms)

    @property
    def is_neg(self) -> bool:
        """Get the negation flag of the predicate.

        Returns:
            bool: True if predicate is negated, False otherwise.
        """
        return self._is_neg
    
    @property
    def term_type_dict(self) -> Dict["Term", Set[str]]:
        return self._term_type_dict
    
    @term_type_dict.setter
    def term_type_dict(self, value: Dict["Term", Set[str]]):
        self._term_type_dict = value
    
    def __eq__(self, other: Any) -> bool:
        """Check if two predicates are equal.

        Args:
            other (Any): The predicate to compare with.

        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(other, Predicate):
            return False
        for term1, term2 in zip(self._terms, other.terms):
            if isinstance(term1, Variable) and isinstance(term2, Variable):
                if term1.name != term2.name:
                    return False
            elif isinstance(term1, Constant) and isinstance(term2, Constant):
                if term1.name != term2.name:
                    # if the terms are not the same, return False
                    return False
            else:
                # if the terms are not the same type, return False
                return False
        return (self._name == other.name) and (self._terms == other.terms) and (self._is_neg == other.is_neg)

    def is_duplicate(self, other: Any) -> bool:
        """Check if two predicates are equal.

        Args:
            other (Any): The predicate to compare with.

        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(other, Predicate):
            return False
        if self._name != other.name:
            return False
        if len(self._terms) != len(other.terms):
            return False
        if self._is_neg != other.is_neg:
            return False
        mapping: Dict["Term", "Term"] = {}
        for term1, term2 in zip(self._terms, other.terms):
            if isinstance(term1, Variable) and isinstance(term2, Variable):
                if term1 in mapping:
                    if mapping[term1] != term2:
                        return False
                else:
                    mapping[term1] = term2
            elif isinstance(term1, Constant) and isinstance(term2, Constant):
                if term1 != term2:
                    return False
        if len(mapping) != len(set(mapping.values())):
            return False
        return True

    def __repr__(self) -> str:
        """Return the canonical string representation of the predicate.

        Returns:
            str: The string representation.
        """
        terms_str = ", ".join(str(term) for term in self._terms)
        return f"{self.name}({terms_str})" if not self._is_neg else f"¬{self.name}({terms_str})"

    def __str__(self) -> str:
        """Return the informal string representation of the predicate.

        Returns:
            str: The string representation.
        """
        terms_str = ", ".join(str(term) for term in self._terms)

        return f"{self.name}({terms_str})" if not self._is_neg else f"not {self.name}({terms_str})"

    def __hash__(self) -> int:
        """Compute the hash of the predicate.

        Returns:
            int: The hash value.
        """
        return hash((self.name, tuple(self.terms), self._is_neg))

    def __len__(self):
        return len(str(self))

class   Substitution(dict):
    """Represents a set of substitutions mapping Variables to Terms.

    This class extends the built-in dictionary and performs an initial check to ensure all keys
    are Variables and all values are Terms.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Substitution.

        Args:
            *args: Positional arguments to pass to the dict constructor.
            **kwargs: Keyword arguments to pass to the dict constructor.
        """
        super().__init__(*args, **kwargs)
        self._check()

    def _check(self) -> None:
        """Check that the keys are Variables and the values are Terms.

        Raises:
            ValueError: If any key is not a Variable or any value is not a Term.
        """
        for key, value in self.items():
            # if not isinstance(key, Variable):
            #     raise ValueError(f"Key {key} is not a Variable")
            if not (isinstance(value, Variable) or isinstance(value, Constant)):
                raise ValueError(f"Value {value} is not a Term")

    def empty(self) -> bool:
        return len(self) == 0
            