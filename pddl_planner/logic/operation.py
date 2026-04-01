import copy
from typing import List
from pddl_planner.logic.formula import Logic, Substitution, Formula, Predicate, ConjunctiveFormula, DisjunctiveFormula, Variable, Constant
from pddl_planner.logic.nl_formula import NLPredicate

class Operations(Logic):
    """Operations performed to formulas"""

    def occur_check(self, var, x):
        """Check if variable occurs in formula
        """
        return var in x.collect_vars()
    
    def unify(self, x: "Formula", y: "Formula", substitution: "Substitution") -> "Substitution":
        """ method to unify two formulas
         """
        if substitution is None:
            # failure
            return substitution

        elif x == y:
            return substitution
        
        elif isinstance(x, Variable):
            return self.unify_var(x, y, substitution)
        
        elif isinstance(y, Variable):
            return self.unify_var(y, x, substitution)
            

        elif isinstance(x, Predicate) and isinstance(y, Predicate):
            return self.unify(x.collect_terms(), y.collect_terms(), self.unify(x.collect_preds_name(), y.collect_preds_name(), substitution))
        
        elif (isinstance(x, ConjunctiveFormula) and isinstance(y, ConjunctiveFormula)) or (isinstance(x, DisjunctiveFormula) and isinstance(y, DisjunctiveFormula)):
            return self.unify(x.clauses, y.clauses, substitution)

        elif (isinstance(x, set) and isinstance(y, set)) or (isinstance(x, list) and isinstance(y, list)):
            x_copy = copy.deepcopy(x)
            y_copy = copy.deepcopy(y)
            x_first = x_copy.pop()
            y_first = y_copy.pop()
            return self.unify(x_copy, y_copy, self.unify(x_first, y_first, substitution))
        
        else:
            return None

    def unify_with_different_name(self, x: "Formula", y: "Formula", 
    substitution: "Substitution") -> "Substitution":
        """Unify two formulas without requiring them to have the same name.
        This is specifically designed for entailment tasks where predicates with different names
        might have similar structures that can be unified.
        """
        if substitution is None:
            # failure
            return substitution

        elif x == y:
            return substitution
        
        elif isinstance(x, Variable) or isinstance(x, Constant):
            return self.unify_var_with_different_name(x, y, substitution)
        
        elif isinstance(y, Variable) or isinstance(y, Constant):
            return self.unify_var_with_different_name(y, x, substitution)
            
        elif isinstance(x, Predicate) and isinstance(y, Predicate):
            # Skip name unification and only unify terms
            return self.unify_with_different_name(x.collect_terms(), y.collect_terms(), substitution)
        
        elif (isinstance(x, ConjunctiveFormula) and isinstance(y, ConjunctiveFormula)) or (isinstance(x, DisjunctiveFormula) and isinstance(y, DisjunctiveFormula)):
            return self.unify_with_different_name(x.clauses, y.clauses, substitution)

        elif (isinstance(x, set) and isinstance(y, set)) or (isinstance(x, list) and isinstance(y, list)):
            x_copy = copy.deepcopy(x)
            y_copy = copy.deepcopy(y)
            # Handle empty lists/sets
            if len(x_copy) == 0 and len(y_copy) == 0:
                return substitution
            elif len(x_copy) == 0 or len(y_copy) == 0:
                return None
            x_first = x_copy.pop()
            y_first = y_copy.pop()
            return self.unify_with_different_name(x_copy, y_copy, self.unify_with_different_name(x_first, y_first, substitution))
        
        else:
            return None
    def unify_var_with_different_name(self, var, x, substitution):
        # if var in substitution:
        #     return self.unify_with_different_name(substitution[var], x, substitution)
        # elif x in substitution: 
        #     return self.unify_with_different_name(var, substitution[x], substitution)
        # elif isinstance(x, Formula) and self.occur_check(var, x):
        #     return None
        # else:
            substitution[var] = x
            return substitution

    def unify_var(self, var, x, substitution):
        if var in substitution:
            return self.unify(substitution[var], x, substitution)
        elif x in substitution: 
            return self.unify(var, substitution[x], substitution)
        elif isinstance(x, Formula) and self.occur_check(var, x):
            return None
        else:
            substitution[var] = x
            return substitution

    def standardize(self, *formulas):
        """Standardize the variables in a formula"""
        all_vars = set()
        for formula in formulas:
            all_vars.update(formula.collect_vars())  # Collect vars from all formulas

        substitution = Substitution()
        for var in all_vars:
            if var not in substitution:
                substitution[var] = self.get_new_var()

        # Apply the substitution to each formula and return the standardized formulas
        standardized_formulas = [formula.substitute(substitution) for formula in formulas]
        return standardized_formulas

    def replace_domain_with_goal_fluents(self, formula: "DisjunctiveFormula", goal: "Formula") -> "DisjunctiveFormula":
        """
        Replace predicate names in a formula with matching goal fluent names while preserving terms.

        Matching uses entailment-aware equality via Formula._equals_helper when available (e.g., NLPredicate).

        Args:
            formula (DisjunctiveFormula): The formula whose predicate names may be replaced.
            goal (Formula): The overall goal formula from which to collect goal fluents (predicates).

        Returns:
            DisjunctiveFormula: A new DisjunctiveFormula with predicate names replaced where matches are found.
        """

        # Gather all goal predicates
        goal_preds: List[Predicate] = []

        def _gather_preds(f: "Formula") -> None:
            if isinstance(f, Predicate):
                goal_preds.append(f)
                return
            if hasattr(f, 'clauses') and isinstance(getattr(f, 'clauses'), list):
                for cl in f.clauses:
                    _gather_preds(cl)

        _gather_preds(goal)

        def _match_and_replace(pred: Predicate) -> Predicate:
            # Try match against any goal predicate using entailment-aware equality when available.
            for gpred in goal_preds:
                if pred._equals_helper(gpred, {}, check_var_constant_consistency=False):
                    if pred.name == gpred.name:
                        return pred
                    gpred_copy = copy.deepcopy(gpred)
                    pred_copy = copy.deepcopy(pred)
                    # Unify the terms of the predicates, ignoring name
                    # substitution = self.unify_with_different_name(pred_copy, gpred_copy, Substitution())
                    # if substitution is not None:
                    #     #pred = pred.substitute(substitution)
                    #     gpred_copy = gpred_copy.substitute(substitution)
                    # Prefer preserving NLPredicate fields when present
                    return NLPredicate(
                        gpred_copy.name,
                        pred.nl_description,
                        pred.is_neg,
                        *pred.terms,
                        term_type_dict=pred.term_type_dict,
                        inital_terms=getattr(gpred_copy, '_inital_terms'),
                        inital_str=getattr(gpred_copy, '_inital_str_represntation'),
                    )
            return pred

        def _replace_in_formula(f: "Formula") -> "Formula":
            if isinstance(f, Predicate):
                return _match_and_replace(f)
            if isinstance(f, ConjunctiveFormula):
                replaced_children = [_replace_in_formula(cl) for cl in f.clauses]
                ret = ConjunctiveFormula(*replaced_children)
                formula._combine_and_propagate_type_dict(ret)  # reuse helper
                return ret
            if isinstance(f, DisjunctiveFormula):
                replaced_children = [_replace_in_formula(cl) for cl in f.clauses]
                ret = DisjunctiveFormula(*replaced_children)
                formula._combine_and_propagate_type_dict(ret)
                return ret
            return f

        replaced = _replace_in_formula(formula)
        if isinstance(replaced, DisjunctiveFormula):
            return replaced
        return DisjunctiveFormula(replaced)

    def simplify_by_domain_axiom(self, formula: "DisjunctiveFormula", init: Formula|None = None) -> "DisjunctiveFormula":
        """
        Domain-axiom simplification for regressed goals.

        Args:
            formula (DisjunctiveFormula): The regressed goal in DNF.

        Returns:
            DisjunctiveFormula: A filtered disjunction with invalid conjuncts removed.
        """
        if not isinstance(formula, DisjunctiveFormula):
            return formula
        kept_clauses: List[Formula] = []
        for clause in formula.clauses:
            if isinstance(clause, ConjunctiveFormula):
                count_dict = {}
                for c in clause.clauses:
                    if isinstance(c, Predicate):
                        name = getattr(c, 'name', '').lower()
                        if name == 'i am holding' or name == 'the hand is empty':
                            if name not in count_dict:
                                count_dict[name] = 0
                            count_dict[name] += 1
                if 'i am holding' in count_dict and count_dict['i am holding'] > 1:
                    # cannot hold more than one object at a time
                    # drop this conjunct
                    continue
                if 'the hand is empty' in count_dict and 'i am holding' in count_dict and count_dict['the hand is empty'] >= 1 and count_dict['i am holding'] >= 1:
                    # cannot hold and the hand is empty at the same time
                    # drop this conjunct
                    continue
                kept_clauses.append(clause)
            else:
                kept_clauses.append(clause)

        result = DisjunctiveFormula(*kept_clauses)
        # propagate type dicts
        formula._combine_and_propagate_type_dict(result)
        return result

    # Backward-compatibility alias for varied naming in call sites
    def _simplify_by_domian_axiom(self, formula: "DisjunctiveFormula") -> "DisjunctiveFormula":
        return self.simplify_by_domain_axiom(formula)
    def simpl_domain_axmoin(self, formula: "DisjunctiveFormula") -> "DisjunctiveFormula":
        return self.simplify_by_domain_axiom(formula)

